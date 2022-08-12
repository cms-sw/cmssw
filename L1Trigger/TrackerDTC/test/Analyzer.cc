#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerDTC/interface/LayerEncoding.h"

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
using namespace tt;

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
   *  \brief  Class to analyze hardware like structured TTStub Collection used by Track Trigger emulators, runs DTC stub emulation, plots performance & stub occupancy
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
    // configuring track particle selector
    void configTPSelector();
    // book histograms
    void bookHistograms();
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
    void analyzeStream(const StreamStub& stream, int region, int channel, int& sum, TH2F* th2f);
    // returns layerId [1-6, 11-15] of stub
    int layerId(const TTStubRef& ttStubRef) const;
    // analyze survived TPs
    void analyzeTPs(const map<TPPtr, set<TTStubRef>>& mapTPsStubs);
    // prints out MC summary
    void endJobMC();
    // prints out DTC summary
    void endJobDTC();

    // ED input token of DTC stubs
    EDGetTokenT<TTDTC> getTokenTTDTCAccepted_;
    // ED input token of lost DTC stubs
    EDGetTokenT<TTDTC> getTokenTTDTCLost_;
    // ED input token of TT stubs
    EDGetTokenT<TTStubDetSetVec> getTokenTTStubDetSetVec_;
    // ED input token of TTClsuter
    EDGetTokenT<TTClusterDetSetVec> getTokenTTClusterDetSetVec_;
    // ED input token of TTCluster to TPPtr association
    EDGetTokenT<TTClusterAssMap> getTokenTTClusterAssMap_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetToken_;
    // stores, calculates and provides run-time constants
    const Setup* setup_ = nullptr;
    // selector to partly select TPs for efficiency measurements
    TrackingParticleSelector tpSelector_;
    //
    TrackingParticleSelector tpSelectorLoose_;
    // enables analyze of TPs
    bool useMCTruth_;
    // specifies used TT algorithm
    bool hybrid_;
    //
    int nEvents_ = 0;

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

  Analyzer::Analyzer(const ParameterSet& iConfig)
      : useMCTruth_(iConfig.getParameter<bool>("UseMCTruth")), hybrid_(iConfig.getParameter<bool>("UseHybrid")) {
    usesResource("TFileService");
    // book in- and output ED products
    const auto& inputTagAccepted = iConfig.getParameter<InputTag>("InputTagAccepted");
    const auto& inputTagLost = iConfig.getParameter<InputTag>("InputTagLost");
    getTokenTTDTCAccepted_ = consumes<TTDTC>(inputTagAccepted);
    getTokenTTDTCLost_ = consumes<TTDTC>(inputTagLost);
    if (useMCTruth_) {
      const auto& inputTagTTStubDetSetVec = iConfig.getParameter<InputTag>("InputTagTTStubDetSetVec");
      const auto& inputTagTTClusterDetSetVec = iConfig.getParameter<InputTag>("InputTagTTClusterDetSetVec");
      const auto& inputTagTTClusterAssMap = iConfig.getParameter<InputTag>("InputTagTTClusterAssMap");
      getTokenTTStubDetSetVec_ = consumes<TTStubDetSetVec>(inputTagTTStubDetSetVec);
      getTokenTTClusterDetSetVec_ = consumes<TTClusterDetSetVec>(inputTagTTClusterDetSetVec);
      getTokenTTClusterAssMap_ = consumes<TTClusterAssMap>(inputTagTTClusterAssMap);
    }
    // book ES product
    esGetToken_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    // log config
    log_.setf(ios::fixed, ios::floatfield);
    log_.precision(4);
  }

  void Analyzer::beginRun(const Run& iEvent, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetToken_);
    // configuring track particle selector
    configTPSelector();
    // book histograms
    bookHistograms();
  }

  void Analyzer::analyze(const Event& iEvent, const EventSetup& iSetup) {
    // read in TrackingParticle
    map<TTStubRef, set<TPPtr>> mapAllStubsTPs;
    if (useMCTruth_) {
      Handle<TTStubDetSetVec> handleTTStubDetSetVec;
      iEvent.getByToken<TTStubDetSetVec>(getTokenTTStubDetSetVec_, handleTTStubDetSetVec);
      Handle<TTClusterAssMap> handleTTClusterAssMap;
      iEvent.getByToken<TTClusterAssMap>(getTokenTTClusterAssMap_, handleTTClusterAssMap);
      // associate TPPtr with TTStubRef
      map<TPPtr, set<TTStubRef>> mapAllTPsAllStubs;
      assoc(handleTTStubDetSetVec, handleTTClusterAssMap, mapAllTPsAllStubs);
      // organize reconstrucable TrackingParticles used for efficiency measurements
      convert(mapAllTPsAllStubs, mapAllStubsTPs);
      Handle<TTClusterDetSetVec> handleTTClusterDetSetVec;
      iEvent.getByToken<TTClusterDetSetVec>(getTokenTTClusterDetSetVec_, handleTTClusterDetSetVec);
      int nCluster(0);
      for (const auto& detSet : *handleTTClusterDetSetVec)
        nCluster += detSet.size();
      profMC_->Fill(6, nCluster / (double)setup_->numRegions());
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
    nEvents_++;
  }

  void Analyzer::endJob() {
    if (nEvents_ == 0)
      return;
    // create r-z stub fraction plot
    TH2F th2f("", ";;", 400, -300, 300., 400, 0., 120.);
    th2f.Add(hisRZStubsLost_);
    th2f.Add(hisRZStubs_);
    hisRZStubsEff_->Add(hisRZStubsLost_);
    hisRZStubsEff_->Divide(&th2f);
    // create efficieny plots
    if (useMCTruth_) {
      for (Efficiency e : AllEfficiency) {
        eff_[e]->SetPassedHistogram(*hisEff_[e], "f");
        eff_[e]->SetTotalHistogram(*hisEffMC_[e], "f");
      }
    }
    log_ << "'Lost' below refers to truncation losses" << endl;
    // printout MC summary
    endJobMC();
    // printout DTC summary
    endJobDTC();
    log_ << "=============================================================";
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
    profMC_->Fill(1, nStubs / (double)setup_->numRegions());
    profMC_->Fill(2, nStubsMatched / (double)setup_->numRegions());
  }

  // organize reconstrucable TrackingParticles used for efficiency measurements
  void Analyzer::convert(const map<TPPtr, set<TTStubRef>>& mapTPsStubs, map<TTStubRef, set<TPPtr>>& mapStubsTPs) {
    int nTPsReco(0);
    int nTPsEff(0);
    for (const auto& mapTPStubs : mapTPsStubs) {
      if (!tpSelectorLoose_(*mapTPStubs.first) || !reconstructable(mapTPStubs.second))
        continue;
      nTPsReco++;
      const bool useForAlgEff = select(*mapTPStubs.first);
      if (useForAlgEff) {
        nTPsEff++;
        fill(mapTPStubs.first, hisEffMC_);
        for (const TTStubRef& ttStubRef : mapTPStubs.second)
          mapStubsTPs[ttStubRef].insert(mapTPStubs.first);
      }
    }
    profMC_->Fill(3, nTPsReco / (double)setup_->numRegions());
    profMC_->Fill(4, nTPsEff / (double)setup_->numRegions());
    profMC_->Fill(5, nTPsEff);
  }

  // checks if a stub selection is considered reconstructable
  bool Analyzer::reconstructable(const set<TTStubRef>& ttStubRefs) const {
    const TrackerGeometry* trackerGeometry = setup_->trackerGeometry();
    const TrackerTopology* trackerTopology = setup_->trackerTopology();
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
    return (int)hitPattern.size() >= setup_->tpMinLayers() && (int)hitPatternPS.size() >= setup_->tpMinLayersPS();
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
    return selected && (fabs(d0) < setup_->tpMaxD0()) && (fabs(z0) < setup_->tpMaxVertZ());
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
    for (int region = 0; region < setup_->numRegions(); region++) {
      int nStubs(0);
      int nLost(0);
      for (int channel = 0; channel < setup_->numDTCsPerTFP(); channel++) {
        const StreamStub& stream = accepted->stream(region, channel);
        hisChannel_->Fill(stream.size());
        profChannel_->Fill(region * setup_->numDTCsPerTFP() + channel, stream.size());
        for (const FrameStub& frame : stream) {
          if (frame.first.isNull())
            continue;
          const auto it = mapStubsTPs.find(frame.first);
          if (it == mapStubsTPs.end())
            continue;
          for (const TPPtr& tp : it->second)
            mapTPsStubs[tp].insert(frame.first);
        }
        analyzeStream(stream, region, channel, nStubs, hisRZStubs_);
        analyzeStream(lost->stream(region, channel), region, channel, nLost, hisRZStubsLost_);
      }
      profDTC_->Fill(1, nStubs);
      profDTC_->Fill(2, nLost);
    }
  }

  // fill stub related histograms
  void Analyzer::analyzeStream(const StreamStub& stream, int region, int channel, int& sum, TH2F* th2f) {
    for (const FrameStub& frame : stream) {
      if (frame.first.isNull())
        continue;
      sum++;
      const GlobalPoint& pos = setup_->stubPos(hybrid_, frame, region);
      const GlobalPoint& ttPos = setup_->stubPos(frame.first);
      const vector<double> resolutions = {
          ttPos.perp() - pos.perp(), deltaPhi(ttPos.phi() - pos.phi()), ttPos.z() - pos.z()};
      for (Resolution r : AllResolution) {
        hisResolution_[r]->Fill(resolutions[r]);
        profResolution_[r]->Fill(ttPos.z(), ttPos.perp(), abs(resolutions[r]));
      }
      th2f->Fill(ttPos.z(), ttPos.perp());
    }
  }

  // returns layerId [1-6, 11-15] of stub
  int Analyzer::layerId(const TTStubRef& ttStubRef) const {
    const TrackerTopology* trackerTopology = setup_->trackerTopology();
    const DetId detId = ttStubRef->getDetId() + setup_->offsetDetIdDSV();
    const bool barrel = detId.subdetId() == StripSubdetector::TOB;
    return barrel ? trackerTopology->layer(detId) : trackerTopology->tidWheel(detId) + setup_->offsetLayerDisks();
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
    const double numCluster = profMC_->GetBinContent(6);
    const double errCluster = profMC_->GetBinError(6);
    const vector<double> nums = {numStubs, numStubsMatched, numTPsReco, numTPsEff, numCluster};
    const vector<double> errs = {errStubs, errStubsMatched, errTPsReco, errTPsEff, errCluster};
    const int wNums = ceil(log10(*max_element(nums.begin(), nums.end()))) + 5;
    const int wErrs = ceil(log10(*max_element(errs.begin(), errs.end()))) + 5;
    log_ << "=============================================================" << endl;
    log_ << "                         MC  SUMMARY                         " << endl;
    log_ << "number of cluster       per TFP = " << setw(wNums) << numCluster << " +- " << setw(wErrs) << errCluster
         << endl;
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
    const double totalTPs = profMC_->GetBinContent(5);
    const double eff = numTPs / totalTPs;
    const double errEff = sqrt(eff * (1. - eff) / totalTPs / nEvents_);
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

  // configuring track particle selector
  void Analyzer::configTPSelector() {
    const double ptMin = hybrid_ ? setup_->hybridMinPtStub() : setup_->minPt();
    constexpr double ptMax = 9999999999.;
    const double etaMax = setup_->tpMaxEta();
    const double tip = setup_->tpMaxVertR();
    const double lip = setup_->tpMaxVertZ();
    constexpr int minHit = 0;
    constexpr bool signalOnly = true;
    constexpr bool intimeOnly = true;
    constexpr bool chargedOnly = true;
    constexpr bool stableOnly = false;
    tpSelector_ = TrackingParticleSelector(
        ptMin, ptMax, -etaMax, etaMax, tip, lip, minHit, signalOnly, intimeOnly, chargedOnly, stableOnly);
    tpSelectorLoose_ =
        TrackingParticleSelector(ptMin, ptMax, -etaMax, etaMax, tip, lip, minHit, false, false, false, stableOnly);
  }

  // book histograms
  void Analyzer::bookHistograms() {
    Service<TFileService> fs;
    TFileDirectory dir;
    // mc
    dir = fs->mkdir("MC");
    profMC_ = dir.make<TProfile>("Counts", ";", 6, 0.5, 6.5);
    profMC_->GetXaxis()->SetBinLabel(1, "Stubs");
    profMC_->GetXaxis()->SetBinLabel(2, "Matched Stubs");
    profMC_->GetXaxis()->SetBinLabel(3, "reco TPs");
    profMC_->GetXaxis()->SetBinLabel(4, "eff TPs");
    profMC_->GetXaxis()->SetBinLabel(5, "total eff TPs");
    profMC_->GetXaxis()->SetBinLabel(6, "Cluster");
    constexpr array<int, NumEfficiency> binsEff{{9 * 8, 10, 16, 10, 30, 24}};
    constexpr array<pair<double, double>, NumEfficiency> rangesEff{
        {{-M_PI, M_PI}, {0., 100.}, {-1. / 3., 1. / 3.}, {-5., 5.}, {-15., 15.}, {-2.4, 2.4}}};
    if (useMCTruth_) {
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
    const int numChannels = setup_->numDTCs() * setup_->numOverlappingRegions();
    hisChannel_ = dir.make<TH1F>("His Channel Occupancy", ";", maxOcc, -.5, maxOcc - .5);
    profChannel_ = dir.make<TProfile>("Prof Channel Occupancy", ";", numChannels, -.5, numChannels - .5);
    // max tracking efficiencies
    if (useMCTruth_) {
      dir = fs->mkdir("DTC/Effi");
      hisEff_.reserve(NumEfficiency);
      for (Efficiency e : AllEfficiency)
        hisEff_.emplace_back(
            dir.make<TH1F>(("HisTP" + name(e)).c_str(), ";", binsEff[e], rangesEff[e].first, rangesEff[e].second));
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
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::Analyzer);
