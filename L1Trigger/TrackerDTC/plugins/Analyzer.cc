#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "L1Trigger/TrackerDTC/interface/Settings.h"
#include "L1Trigger/TrackerDTC/interface/TTDTCConverter.h"

#include <TProfile.h>
#include <TProfile2D.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TEfficiency.h>

#include <vector>
#include <map>
#include <utility>
#include <set>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <array>
#include <initializer_list>
#include <sstream>

using namespace std;
using namespace edm;

namespace trackerDTC {

  // mc truth types
  typedef TTClusterAssociationMap<Ref_Phase2TrackerDigi_> TTClusterAssMap;
  typedef edm::Ptr<TrackingParticle> TPPtr;
  // stub resolution plots helper
  enum Resolution { R, Phi, Z, NumResolution };
  constexpr initializer_list<Resolution> AllResolution = {R, Phi, Z};
  constexpr auto NameResolution = {"R", "Phi", "Z"};
  inline string name(Resolution r) { return string(*(NameResolution.begin() + r)); }
  // max tracking efficiency plots helper
  enum Efficiency { Phi0, Pt, InvPt, D0, Z0, Eta, NumEfficiency };
  constexpr initializer_list<Efficiency> AllEfficiency = {Phi0, Pt, InvPt, D0, Z0, Eta};
  constexpr auto NameEfficiency = {"Phi0", "Pt", "InvPt", "D0", "Z0", "Eta"};
  inline string name(Efficiency e) { return string(*(NameEfficiency.begin() + e)); }

  /*! \class  trackerDTC::Analyzer
   *  \brief  Class to analyze hardware like structured TTStub Collection used by Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class Analyzer : public one::EDAnalyzer<one::WatchRuns, one::SharedResources> {
  public:
    Analyzer(const ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const Run& iEvent, const EventSetup& iSetup) override;
    void analyze(const Event& iEvent, const EventSetup& iSetup) override;
    void endRun(const Run& iEvent, const EventSetup& iSetup) override {}
    void endJob() override;

  private:
    // associate TPPtr with TTStubRef
    void assoc(const Handle<TTStubDetSetVec>&, const Handle<TTClusterAssMap>&, map<TPPtr, set<TTStubRef>>&);
    // organize reconstrucable TrackingParticles used for efficiency measurements
    void convert(const map<TPPtr, set<TTStubRef>>&, map<TTStubRef, set<TPPtr>>&);
    // checks if a stub selection is considered reconstructable
    bool reconstructable(const set<TTStubRef>& ttStubRefs) const;
    // checks if TrackingParticle is selected for efficiency measurements
    bool select(const TrackingParticle& tp) const;
    // fills kinematic tp histograms
    void fill(const TPPtr& tpPtr, const vector<TH1F*> th1fs) const;
    // analyze DTC products and find still reconstrucable TrackingParticles
    void analyzeStubs(const TTDTC*, const TTDTC*, const map<TTStubRef, set<TPPtr>>&, map<TPPtr, set<TTStubRef>>&);
    // fill stub related histograms
    void analyzeStream(const TTDTC::Stream& stream, int region, int& sum, TH2F* th2f);
    // returns global stub position
    GlobalPoint stubPos(const TTStubRef& ttStubRef) const;
    // handles 2 pi overflow
    double deltaPhi(double lhs, double rhs = 0.) { return reco::deltaPhi(lhs, rhs); }
    // analyze survived TPs
    void analyzeTPs(const map<TPPtr, set<TTStubRef>>& mapTPsStubs);
    // prints out MC summary
    void endJobMC();
    // prints out DTC summary
    void endJobDTC();

    // ed input tokens

    EDGetTokenT<TTDTC> getTokenTTDTCAccepted_;
    EDGetTokenT<TTDTC> getTokenTTDTCLost_;
    EDGetTokenT<TTStubDetSetVec> getTokenTTStubDetSetVec_;
    EDGetTokenT<TTClusterAssMap> getTokenTTClusterAssMap_;

    // es input tokens

    ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> getTokenTrackerGeometry_;
    ESGetToken<TrackerTopology, TrackerTopologyRcd> getTokenTrackerTopology_;

    // stores, calculates and provides run-time constants
    Settings settings_;
    // selector to partly select TPs for efficiency measurements
    TrackingParticleSelector tpSelector_;

    // Histograms

    TProfile* profMC_;
    TProfile* profDTC_;
    TProfile* profChannel_;
    TH1F* hisChannel_;
    TH2F* hisRZStubs_;
    TH2F* hisRZStubsLost_;
    TH2F* hisRZStubsEff_;
    vector<TH1F*> hisResolution_;
    vector<TProfile2D*> profResolution_;
    vector<TH1F*> hisEff_;
    vector<TH1F*> hisEffMC_;
    vector<TEfficiency*> eff_;

    // printout
    stringstream log_;
  };

  Analyzer::Analyzer(const ParameterSet& iConfig) : settings_(iConfig) {
    usesResource("TFileService");
    // book in- and output ED products
    getTokenTTDTCAccepted_ = consumes<TTDTC>(InputTag(settings_.producerLabel(), settings_.productBranchAccepted()));
    getTokenTTDTCLost_ = consumes<TTDTC>(InputTag(settings_.producerLabel(), settings_.productBranchLost()));
    if (settings_.useMCTruth()) {
      getTokenTTStubDetSetVec_ = consumes<TTStubDetSetVec>(settings_.inputTagTTStubDetSetVec());
      getTokenTTClusterAssMap_ = consumes<TTClusterAssMap>(settings_.inputTagTTClusterAssMap());
    }
    // book ES products
    getTokenTrackerGeometry_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, Transition::BeginRun>(
        settings_.inputTagTrackerGeometry());
    getTokenTrackerTopology_ =
        esConsumes<TrackerTopology, TrackerTopologyRcd, Transition::BeginRun>(settings_.inputTagTrackerTopology());
    // configuring track particle selector
    const double ptMin = settings_.tpMinPt();
    constexpr double ptMax = 9999999999.;
    const double etaMax = settings_.tpMaxEta();
    const double tip = settings_.tpMaxVertR();
    const double lip = settings_.tpMaxVertZ();
    constexpr int minHit = 0;
    constexpr bool signalOnly = true;
    constexpr bool intimeOnly = true;
    constexpr bool chargedOnly = true;
    constexpr bool stableOnly = false;
    tpSelector_ = TrackingParticleSelector(
        ptMin, ptMax, -etaMax, etaMax, tip, lip, minHit, signalOnly, intimeOnly, chargedOnly, stableOnly);
    // book histograms
    Service<TFileService> fs;
    TFileDirectory dir;
    // mc
    dir = fs->mkdir("MC");
    profMC_ = dir.make<TProfile>("Counts", ";", 4, 0.5, 4.5);
    profMC_->GetXaxis()->SetBinLabel(1, "Stubs");
    profMC_->GetXaxis()->SetBinLabel(2, "Matched Stubs");
    profMC_->GetXaxis()->SetBinLabel(3, "reco TPs");
    profMC_->GetXaxis()->SetBinLabel(4, "eff TPs");
    constexpr array<int, NumEfficiency> binsEff{{9 * 8, 10, 16, 10, 30, 24}};
    constexpr array<pair<double, double>, NumEfficiency> rangesEff{
        {{-M_PI, M_PI}, {0., 100.}, {-1. / 3., 1. / 3.}, {-5., 5.}, {-15., 15.}, {-2.4, 2.4}}};
    if (settings_.useMCTruth()) {
      hisEffMC_.reserve(NumEfficiency);
      for (Efficiency e : AllEfficiency)
        hisEffMC_.emplace_back(
            dir.make<TH1F>(("HisTP" + name(e)).c_str(), ";", binsEff[e], rangesEff[e].first, rangesEff[e].second));
    }
    // dtc
    dir = fs->mkdir("DTC");
    profDTC_ = dir.make<TProfile>("Counts", ";", 3, 0.5, 3.5);
    profDTC_->GetXaxis()->SetBinLabel(1, "Stubs");
    profDTC_->GetXaxis()->SetBinLabel(2, "Lost Stubs");
    profDTC_->GetXaxis()->SetBinLabel(3, "TPs");
    // channel occupancy
    constexpr int maxOcc = 180;
    const int numChannels = settings_.numDTCs();
    hisChannel_ = dir.make<TH1F>("Channel Occupancy", ";", maxOcc, -.5, maxOcc - .5);
    profChannel_ = dir.make<TProfile>("Channel Occupancy", ";", numChannels, -.5, numChannels - .5);
    // max tracking efficiencies
    if (settings_.useMCTruth()) {
      hisEff_.reserve(NumEfficiency);
      for (Efficiency e : AllEfficiency)
        hisEff_.emplace_back(
            dir.make<TH1F>(("HisTP" + name(e)).c_str(), ";", binsEff[e], rangesEff[e].first, rangesEff[e].second));
      dir = fs->mkdir("DTC/Effi");
      eff_.reserve(NumEfficiency);
      for (Efficiency e : AllEfficiency)
        eff_.emplace_back(
            dir.make<TEfficiency>(("Eff" + name(e)).c_str(), ";", binsEff[e], rangesEff[e].first, rangesEff[e].second));
    }
    // lost stub fraction in r-z
    dir = fs->mkdir("DTC/Loss");
    constexpr int bins = 400;
    constexpr double maxZ = 300.;
    constexpr double maxR = 120.;
    hisRZStubs_ = dir.make<TH2F>("RZ Stubs", ";;", bins, -maxZ, maxZ, bins, 0., maxR);
    hisRZStubsLost_ = dir.make<TH2F>("RZ Stubs Lost", ";;", bins, -maxZ, maxZ, bins, 0., maxR);
    hisRZStubsEff_ = dir.make<TH2F>("RZ Stubs Eff", ";;", bins, -maxZ, maxZ, bins, 0., maxR);
    // stub parameter resolutions
    dir = fs->mkdir("DTC/Res");
    constexpr array<double, NumResolution> ranges{{.2, .0001, .5}};
    constexpr int binsHis = 100;
    hisResolution_.reserve(NumResolution);
    profResolution_.reserve(NumResolution);
    for (Resolution r : AllResolution) {
      hisResolution_.emplace_back(dir.make<TH1F>(("HisRes" + name(r)).c_str(), ";", binsHis, -ranges[r], ranges[r]));
      profResolution_.emplace_back(
          dir.make<TProfile2D>(("ProfRes" + name(r)).c_str(), ";;", bins, -maxZ, maxZ, bins, 0., maxR));
    }
    // log config
    log_.setf(ios::fixed, ios::floatfield);
    log_.precision(4);
  }

  void Analyzer::beginRun(const Run& iEvent, const EventSetup& iSetup) {
    // read in detector parameter
    settings_.setTrackerGeometry(&iSetup.getData(getTokenTrackerGeometry_));
    settings_.setTrackerTopology(&iSetup.getData(getTokenTrackerTopology_));
  }

  void Analyzer::analyze(const Event& iEvent, const EventSetup& iSetup) {
    // read in TrackingParticle
    map<TTStubRef, set<TPPtr>> mapAllStubsTPs;
    if (settings_.useMCTruth()) {
      Handle<TTStubDetSetVec> handleTTStubDetSetVec;
      iEvent.getByToken<TTStubDetSetVec>(getTokenTTStubDetSetVec_, handleTTStubDetSetVec);
      Handle<TTClusterAssMap> handleTTClusterAssMap;
      iEvent.getByToken<TTClusterAssMap>(getTokenTTClusterAssMap_, handleTTClusterAssMap);
      // associate TPPtr with TTStubRef
      map<TPPtr, set<TTStubRef>> mapAllTPsAllStubs;
      assoc(handleTTStubDetSetVec, handleTTClusterAssMap, mapAllTPsAllStubs);
      // organize reconstrucable TrackingParticles used for efficiency measurements
      convert(mapAllTPsAllStubs, mapAllStubsTPs);
    }
    // read in dtc products
    Handle<TTDTC> handleTTDTCAccepted;
    iEvent.getByToken<TTDTC>(getTokenTTDTCAccepted_, handleTTDTCAccepted);
    Handle<TTDTC> handleTTDTCLost;
    iEvent.getByToken<TTDTC>(getTokenTTDTCLost_, handleTTDTCLost);
    map<TPPtr, set<TTStubRef>> mapTPsTTStubs;
    // analyze DTC products and find still reconstrucable TrackingParticles
    analyzeStubs(handleTTDTCAccepted.product(), handleTTDTCLost.product(), mapAllStubsTPs, mapTPsTTStubs);
    // analyze survived TPs
    analyzeTPs(mapTPsTTStubs);
  }

  void Analyzer::endJob() {
    // create r-z stub fraction plot
    TH2F th2f("", ";;", 400, -300, 300., 400, 0., 120.);
    th2f.Add(hisRZStubsLost_);
    th2f.Add(hisRZStubs_);
    hisRZStubsEff_->Add(hisRZStubsLost_);
    hisRZStubsEff_->Divide(&th2f);
    // create efficieny plots
    if (settings_.useMCTruth()) {
      for (Efficiency e : AllEfficiency) {
        eff_[e]->SetPassedHistogram(*hisEff_[e], "f");
        eff_[e]->SetTotalHistogram(*hisEffMC_[e], "f");
      }
    }
    // printout MC summary
    endJobMC();
    // printout DTC summary
    endJobDTC();
    log_ << "=============================================================" << endl;
    LogPrint("L1Trigger/TrackerDTC") << log_.str();
  }

  // associate TPPtr with TTStubRef
  void Analyzer::assoc(const Handle<TTStubDetSetVec>& handleTTStubDetSetVec,
                       const Handle<TTClusterAssMap>& handleTTClusterAssMap,
                       map<TPPtr, set<TTStubRef>>& mapTPsStubs) {
    int nStubs(0);
    int nStubsMatched(0);
    for (TTStubDetSetVec::const_iterator ttModule = handleTTStubDetSetVec->begin();
         ttModule != handleTTStubDetSetVec->end();
         ttModule++) {
      nStubs += ttModule->size();
      for (TTStubDetSet::const_iterator ttStub = ttModule->begin(); ttStub != ttModule->end(); ttStub++) {
        set<TPPtr> tpPtrs;
        for (unsigned int iClus = 0; iClus < 2; iClus++) {
          const vector<TPPtr>& assocPtrs = handleTTClusterAssMap->findTrackingParticlePtrs(ttStub->clusterRef(iClus));
          copy_if(assocPtrs.begin(), assocPtrs.end(), inserter(tpPtrs, tpPtrs.begin()), [](const TPPtr& tpPtr) {
            return tpPtr.isNonnull();
          });
        }
        for (const TPPtr& tpPtr : tpPtrs)
          mapTPsStubs[tpPtr].emplace(makeRefTo(handleTTStubDetSetVec, ttStub));
        if (!tpPtrs.empty())
          nStubsMatched++;
      }
    }
    profMC_->Fill(1, nStubs / (double)settings_.numRegions());
    profMC_->Fill(2, nStubsMatched / (double)settings_.numRegions());
  }

  // organize reconstrucable TrackingParticles used for efficiency measurements
  void Analyzer::convert(const map<TPPtr, set<TTStubRef>>& mapTPsStubs, map<TTStubRef, set<TPPtr>>& mapStubsTPs) {
    int nTPsReco(0);
    int nTPsEff(0);
    for (const auto& mapTPStubs : mapTPsStubs) {
      if (!reconstructable(mapTPStubs.second))
        continue;
      nTPsReco++;
      const bool useForAlgEff = select(*mapTPStubs.first.get());
      if (useForAlgEff) {
        nTPsEff++;
        fill(mapTPStubs.first, hisEffMC_);
        for (const TTStubRef& ttStubRef : mapTPStubs.second)
          mapStubsTPs[ttStubRef].insert(mapTPStubs.first);
      }
    }
    profMC_->Fill(3, nTPsReco);
    profMC_->Fill(4, nTPsEff);
  }

  // checks if a stub selection is considered reconstructable
  bool Analyzer::reconstructable(const set<TTStubRef>& ttStubRefs) const {
    const TrackerGeometry* trackerGeometry = settings_.trackerGeometry();
    const TrackerTopology* trackerTopology = settings_.trackerTopology();
    set<int> hitPattern;
    set<int> hitPatternPS;
    for (const TTStubRef& ttStubRef : ttStubRefs) {
      const DetId detId = ttStubRef->getDetId();
      const bool barrel = detId.subdetId() == StripSubdetector::TOB;
      const bool psModule = trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP;
      const int layerId = barrel ? trackerTopology->layer(detId) : trackerTopology->tidWheel(detId) + 10;
      hitPattern.insert(layerId);
      if (psModule)
        hitPatternPS.insert(layerId);
    }
    return (int)hitPattern.size() >= settings_.tpMinLayers() && (int)hitPatternPS.size() >= settings_.tpMinLayersPS();
  }

  // checks if TrackingParticle is selected for efficiency measurements
  bool Analyzer::select(const TrackingParticle& tp) const {
    const bool selected = tpSelector_(tp);
    const double cot = sinh(tp.eta());
    const double s = sin(tp.phi());
    const double c = cos(tp.phi());
    const TrackingParticle::Point& v = tp.vertex();
    const double z0 = v.z() - (v.x() * c + v.y() * s) * cot;
    const double d0 = v.x() * s - v.y() * c;
    return selected && (fabs(d0) < settings_.tpMaxD0()) && (fabs(z0) < settings_.tpMaxVertZ());
  }

  // fills kinematic tp histograms
  void Analyzer::fill(const TPPtr& tpPtr, const vector<TH1F*> th1fs) const {
    const double s = sin(tpPtr->phi());
    const double c = cos(tpPtr->phi());
    const TrackingParticle::Point& v = tpPtr->vertex();
    const vector<double> x = {tpPtr->phi(),
                              tpPtr->pt(),
                              tpPtr->charge() / tpPtr->pt(),
                              v.x() * s - v.y() * c,
                              v.z() - (v.x() * c + v.y() * s) * sinh(tpPtr->eta()),
                              tpPtr->eta()};
    for (Efficiency e : AllEfficiency)
      th1fs[e]->Fill(x[e]);
  }

  // analyze DTC products and find still reconstrucable TrackingParticles
  void Analyzer::analyzeStubs(const TTDTC* accepted,
                              const TTDTC* lost,
                              const map<TTStubRef, set<TPPtr>>& mapStubsTPs,
                              map<TPPtr, set<TTStubRef>>& mapTPsStubs) {
    for (int region = 0; region < settings_.numRegions(); region++) {
      int nStubs(0);
      int nLost(0);
      for (int channel = 0; channel < settings_.numDTCsPerTFP(); channel++) {
        const TTDTC::Stream& stream = accepted->stream(region, channel);
        hisChannel_->Fill(stream.size());
        profChannel_->Fill(region * settings_.numDTCsPerTFP() + channel, stream.size());
        for (const TTDTC::Frame& frame : stream) {
          if (frame.first.isNull())
            continue;
          const auto it = mapStubsTPs.find(frame.first);
          if (it == mapStubsTPs.end())
            continue;
          for (const TPPtr& tp : it->second)
            mapTPsStubs[tp].insert(frame.first);
        }
        analyzeStream(stream, region, nStubs, hisRZStubs_);
        analyzeStream(lost->stream(region, channel), region, nLost, hisRZStubsLost_);
      }
      profDTC_->Fill(1, nStubs);
      profDTC_->Fill(2, nLost);
    }
  }

  // fill stub related histograms
  void Analyzer::analyzeStream(const TTDTC::Stream& stream, int region, int& sum, TH2F* th2f) {
    for (const TTDTC::Frame& frame : stream) {
      if (frame.first.isNull())
        continue;
      sum++;
      const GlobalPoint& pos = TTDTCConverter(&settings_, frame, region);
      const GlobalPoint& ttPos = stubPos(frame.first);
      const vector<double> resolutions = {
          ttPos.perp() - pos.perp(), deltaPhi(ttPos.phi() - pos.phi()), ttPos.z() - pos.z()};
      for (Resolution r : AllResolution) {
        hisResolution_[r]->Fill(resolutions[r]);
        profResolution_[r]->Fill(ttPos.z(), ttPos.perp(), abs(resolutions[r]));
      }
      th2f->Fill(ttPos.z(), ttPos.perp());
    }
  }

  // returns global stub position
  GlobalPoint Analyzer::stubPos(const TTStubRef& ttStubRef) const {
    const TrackerGeometry* trackerGeometry = settings_.trackerGeometry();
    const DetId detId = ttStubRef->getDetId() + settings_.offsetDetIdDSV();
    const GeomDetUnit* det = trackerGeometry->idToDetUnit(detId);
    const PixelTopology* topol =
        dynamic_cast<const PixelTopology*>(&(dynamic_cast<const PixelGeomDetUnit*>(det)->specificTopology()));
    const Plane& plane = dynamic_cast<const PixelGeomDetUnit*>(det)->surface();
    const MeasurementPoint& mp = ttStubRef->clusterRef(0)->findAverageLocalCoordinatesCentered();
    return plane.toGlobal(topol->localPosition(mp));
  }

  // analyze survived TPs
  void Analyzer::analyzeTPs(const map<TPPtr, set<TTStubRef>>& mapTPsStubs) {
    int nTPs(0);
    for (const auto& mapTPStubs : mapTPsStubs) {
      if (!reconstructable(mapTPStubs.second))
        continue;
      nTPs++;
      fill(mapTPStubs.first, hisEff_);
    }
    profDTC_->Fill(3, nTPs);
  }

  // prints out MC summary
  void Analyzer::endJobMC() {
    const double numStubs = profMC_->GetBinContent(1);
    const double numStubsMatched = profMC_->GetBinContent(2);
    const double numTPsReco = profMC_->GetBinContent(3);
    const double numTPsEff = profMC_->GetBinContent(4);
    const double errStubs = profMC_->GetBinError(1);
    const double errStubsMatched = profMC_->GetBinError(2);
    const double errTPsReco = profMC_->GetBinError(3);
    const double errTPsEff = profMC_->GetBinError(4);
    const vector<double> nums = {numStubs, numStubsMatched, numTPsReco, numTPsEff};
    const vector<double> errs = {errStubs, errStubsMatched, errTPsReco, errTPsEff};
    const int wNums = ceil(log10(*max_element(nums.begin(), nums.end()))) + 5;
    const int wErrs = ceil(log10(*max_element(errs.begin(), errs.end()))) + 5;
    log_ << "=============================================================" << endl;
    log_ << "                         MC  SUMMARY                         " << endl;
    log_ << "number of stubs         per TFP = " << setw(wNums) << numStubs << " +- " << setw(wErrs) << errStubs
         << endl;
    log_ << "number of matched stubs per TFP = " << setw(wNums) << numStubsMatched << " +- " << setw(wErrs)
         << errStubsMatched << endl;
    log_ << "number of TPs           per TFP = " << setw(wNums) << numTPsReco << " +- " << setw(wErrs) << errTPsReco
         << endl;
    log_ << "number of TPs for eff   per TFP = " << setw(wNums) << numTPsEff << " +- " << setw(wErrs) << errTPsEff
         << endl;
  }

  // prints out DTC summary
  void Analyzer::endJobDTC() {
    const double numStubs = profDTC_->GetBinContent(1);
    const double numStubsLost = profDTC_->GetBinContent(2);
    const double numTPs = profDTC_->GetBinContent(3);
    const double errStubs = profDTC_->GetBinError(1);
    const double errStubsLost = profDTC_->GetBinError(2);
    const double totalTPs = profMC_->GetBinContent(4);
    const double eff = numTPs / totalTPs;
    const double errEff = sqrt(eff * (1. - eff) / totalTPs);
    const vector<double> nums = {numStubs, numStubsLost};
    const vector<double> errs = {errStubs, errStubsLost};
    const int wNums = ceil(log10(*max_element(nums.begin(), nums.end()))) + 5;
    const int wErrs = ceil(log10(*max_element(errs.begin(), errs.end()))) + 5;
    log_ << "=============================================================" << endl;
    log_ << "                         DTC SUMMARY                         " << endl;
    log_ << "number of stubs      per TFP = " << setw(wNums) << numStubs << " +- " << setw(wErrs) << errStubs << endl;
    log_ << "number of lost stubs per TFP = " << setw(wNums) << numStubsLost << " +- " << setw(wErrs) << errStubsLost
         << endl;
    log_ << "     max tracking efficiency = " << setw(wNums) << eff << " +- " << setw(wErrs) << errEff << endl;
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::Analyzer);
