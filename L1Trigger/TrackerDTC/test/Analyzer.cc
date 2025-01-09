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
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "SimTracker/TrackTriggerAssociation/interface/StubAssociation.h"
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
    // fills kinematic tp histograms
    void fill(const TPPtr& tpPtr, const vector<TH1F*> th1fs) const;
    // fill stub related histograms
    void fill(const StreamStub& stream, int region, int channel, int& sum, TH2F* th2f);
    // prints out MC summary
    void endJobMC();
    // prints out DTC summary
    void endJobDTC();

    // ED input token of DTC stubs
    EDGetTokenT<TTDTC> edGetTokenTTDTCAccepted_;
    // ED input token of lost DTC stubs
    EDGetTokenT<TTDTC> edGetTokenTTDTCLost_;
    // ED input token of TTStubRef to TPPtr association for tracking efficiency
    EDGetTokenT<StubAssociation> edGetTokenSelection_;
    // ED input token of TTStubRef to recontructable TPPtr association
    EDGetTokenT<StubAssociation> edGetTokenReconstructable_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetToken_;
    // stores, calculates and provides run-time constants
    const Setup* setup_;
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
    edGetTokenTTDTCAccepted_ = consumes<TTDTC>(inputTagAccepted);
    edGetTokenTTDTCLost_ = consumes<TTDTC>(inputTagLost);
    if (useMCTruth_) {
      const auto& inputTagSelection = iConfig.getParameter<InputTag>("InputTagSelection");
      const auto& inputTagReconstructable = iConfig.getParameter<InputTag>("InputTagReconstructable");
      edGetTokenSelection_ = consumes<StubAssociation>(inputTagSelection);
      edGetTokenReconstructable_ = consumes<StubAssociation>(inputTagReconstructable);
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
    // book histograms
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

  void Analyzer::analyze(const Event& iEvent, const EventSetup& iSetup) {
    // read in dtc products
    Handle<TTDTC> handleTTDTCAccepted;
    iEvent.getByToken<TTDTC>(edGetTokenTTDTCAccepted_, handleTTDTCAccepted);
    Handle<TTDTC> handleTTDTCLost;
    iEvent.getByToken<TTDTC>(edGetTokenTTDTCLost_, handleTTDTCLost);
    // read in MCTruth
    const StubAssociation* selection = nullptr;
    const StubAssociation* reconstructable = nullptr;
    if (useMCTruth_) {
      Handle<StubAssociation> handleSelection;
      iEvent.getByToken<StubAssociation>(edGetTokenSelection_, handleSelection);
      selection = handleSelection.product();
      Handle<StubAssociation> handleReconstructable;
      iEvent.getByToken<StubAssociation>(edGetTokenReconstructable_, handleReconstructable);
      reconstructable = handleReconstructable.product();
      profMC_->Fill(3, reconstructable->numTPs() / (double)setup_->numRegions());
      profMC_->Fill(4, selection->numTPs() / (double)setup_->numRegions());
      profMC_->Fill(5, selection->numTPs());
      for (const auto& p : selection->getTrackingParticleToTTStubsMap())
        fill(p.first, hisEffMC_);
    }
    // analyze dtc products and find still reconstrucable TrackingParticles
    set<TPPtr> tpPtrs;
    for (int region = 0; region < setup_->numRegions(); region++) {
      int nStubs(0);
      int nLost(0);
      map<TPPtr, vector<TTStubRef>> mapTPsTTStubs;
      for (int channel = 0; channel < setup_->numDTCsPerTFP(); channel++) {
        const StreamStub& accepted = handleTTDTCAccepted->stream(region, channel);
        const StreamStub& lost = handleTTDTCLost->stream(region, channel);
        hisChannel_->Fill(accepted.size());
        profChannel_->Fill(channel, accepted.size());
        fill(accepted, region, channel, nStubs, hisRZStubs_);
        fill(lost, region, channel, nLost, hisRZStubsLost_);
        if (!useMCTruth_)
          continue;
        for (const FrameStub& frame : accepted) {
          if (frame.first.isNull())
            continue;
          for (const TPPtr& tpPtr : selection->findTrackingParticlePtrs(frame.first)) {
            auto it = mapTPsTTStubs.find(tpPtr);
            if (it == mapTPsTTStubs.end()) {
              it = mapTPsTTStubs.emplace(tpPtr, vector<TTStubRef>()).first;
              it->second.reserve(selection->findTTStubRefs(tpPtr).size());
            }
            it->second.push_back(frame.first);
          }
        }
        for (const auto& p : mapTPsTTStubs)
          if (setup_->reconstructable(p.second))
            tpPtrs.insert(p.first);
      }
      profDTC_->Fill(1, nStubs);
      profDTC_->Fill(2, nLost);
    }
    for (const TPPtr& tpPtr : tpPtrs)
      fill(tpPtr, hisEff_);
    profDTC_->Fill(3, tpPtrs.size());
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
    LogPrint(moduleDescription().moduleName()) << log_.str();
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

  // fill stub related histograms
  void Analyzer::fill(const StreamStub& stream, int region, int channel, int& sum, TH2F* th2f) {
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

  // prints out Monte Carlo summary
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
    log_ << "                         Monte Carlo  SUMMARY                         " << endl;
    /*log_ << "number of cluster       per TFP = " << setw(wNums) << numCluster << " +- " << setw(wErrs) << errCluster
         << endl;
    log_ << "number of stubs         per TFP = " << setw(wNums) << numStubs << " +- " << setw(wErrs) << errStubs
         << endl;
    log_ << "number of matched stubs per TFP = " << setw(wNums) << numStubsMatched << " +- " << setw(wErrs)
         << errStubsMatched << endl;*/
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

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::Analyzer);
