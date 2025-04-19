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

namespace trackerDTC {

  // stub resolution plots helper
  enum Resolution { R, Phi, Z, NumResolution };
  constexpr std::initializer_list<Resolution> AllResolution = {R, Phi, Z};
  constexpr auto NameResolution = {"R", "Phi", "Z"};
  inline std::string name(Resolution r) { return std::string(*(NameResolution.begin() + r)); }
  // max tracking efficiency plots helper
  enum Efficiency { Phi0, Pt, InvPt, D0, Z0, Eta, NumEfficiency };
  constexpr std::initializer_list<Efficiency> AllEfficiency = {Phi0, Pt, InvPt, D0, Z0, Eta};
  constexpr auto NameEfficiency = {"Phi0", "Pt", "InvPt", "D0", "Z0", "Eta"};
  inline std::string name(Efficiency e) { return std::string(*(NameEfficiency.begin() + e)); }

  /*! \class  trackerDTC::Analyzer
   *  \brief  Class to analyze hardware like structured TTStub Collection used by Track Trigger emulators, runs DTC stub emulation, plots performance & stub occupancy
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class Analyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
  public:
    Analyzer(const edm::ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override;
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override {}
    void endJob() override;

  private:
    // fills kinematic tp histograms
    void fill(const TPPtr& tpPtr, const std::vector<TH1F*> th1fs) const;
    // fill stub related histograms
    void fill(const tt::StreamStub& stream, int region, int channel, int& sum, TH2F* th2f);
    // prints out MC summary
    void endJobMC();
    // prints out DTC summary
    void endJobDTC();

    // ED input token of DTC stubs
    edm::EDGetTokenT<TTDTC> edGetTokenTTDTCAccepted_;
    // ED input token of lost DTC stubs
    edm::EDGetTokenT<TTDTC> edGetTokenTTDTCLost_;
    // ED input token of TTStubRef to TPPtr association for tracking efficiency
    edm::EDGetTokenT<tt::StubAssociation> edGetTokenSelection_;
    // ED input token of TTStubRef to recontructable TPPtr association
    edm::EDGetTokenT<tt::StubAssociation> edGetTokenReconstructable_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetToken_;
    // stores, calculates and provides run-time constants
    const tt::Setup* setup_;
    // enables analyze of TPs
    bool useMCTruth_;
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
    std::vector<TH1F*> hisResolution_;
    std::vector<TProfile2D*> profResolution_;
    std::vector<TH1F*> hisEff_;
    std::vector<TH1F*> hisEffMC_;
    std::vector<TEfficiency*> eff_;

    // printout
    std::stringstream log_;
  };

  Analyzer::Analyzer(const edm::ParameterSet& iConfig) : useMCTruth_(iConfig.getParameter<bool>("UseMCTruth")) {
    usesResource("TFileService");
    // book in- and output ED products
    const auto& inputTagAccepted = iConfig.getParameter<edm::InputTag>("InputTagAccepted");
    const auto& inputTagLost = iConfig.getParameter<edm::InputTag>("InputTagLost");
    edGetTokenTTDTCAccepted_ = consumes<TTDTC>(inputTagAccepted);
    edGetTokenTTDTCLost_ = consumes<TTDTC>(inputTagLost);
    if (useMCTruth_) {
      const auto& inputTagSelection = iConfig.getParameter<edm::InputTag>("InputTagSelection");
      const auto& inputTagReconstructable = iConfig.getParameter<edm::InputTag>("InputTagReconstructable");
      edGetTokenSelection_ = consumes<tt::StubAssociation>(inputTagSelection);
      edGetTokenReconstructable_ = consumes<tt::StubAssociation>(inputTagReconstructable);
    }
    // book ES product
    esGetToken_ = esConsumes<edm::Transition::BeginRun>();
    // log config
    log_.setf(std::ios::fixed, std::ios::floatfield);
    log_.precision(4);
  }

  void Analyzer::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetToken_);
    // book histograms
    edm::Service<TFileService> fs;
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
    constexpr std::array<int, NumEfficiency> binsEff{{9 * 8, 10, 16, 10, 30, 24}};
    constexpr std::array<std::pair<double, double>, NumEfficiency> rangesEff{
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
    constexpr std::array<double, NumResolution> ranges{{.2, .0001, .5}};
    constexpr int binsHis = 100;
    hisResolution_.reserve(NumResolution);
    profResolution_.reserve(NumResolution);
    for (Resolution r : AllResolution) {
      hisResolution_.emplace_back(dir.make<TH1F>(("HisRes" + name(r)).c_str(), ";", binsHis, -ranges[r], ranges[r]));
      profResolution_.emplace_back(
          dir.make<TProfile2D>(("ProfRes" + name(r)).c_str(), ";;", bins, -maxZ, maxZ, bins, 0., maxR));
    }
  }

  void Analyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // read in dtc products
    edm::Handle<TTDTC> handleTTDTCAccepted;
    iEvent.getByToken<TTDTC>(edGetTokenTTDTCAccepted_, handleTTDTCAccepted);
    edm::Handle<TTDTC> handleTTDTCLost;
    iEvent.getByToken<TTDTC>(edGetTokenTTDTCLost_, handleTTDTCLost);
    // read in MCTruth
    const tt::StubAssociation* selection = nullptr;
    const tt::StubAssociation* reconstructable = nullptr;
    if (useMCTruth_) {
      edm::Handle<tt::StubAssociation> handleSelection;
      iEvent.getByToken<tt::StubAssociation>(edGetTokenSelection_, handleSelection);
      selection = handleSelection.product();
      edm::Handle<tt::StubAssociation> handleReconstructable;
      iEvent.getByToken<tt::StubAssociation>(edGetTokenReconstructable_, handleReconstructable);
      reconstructable = handleReconstructable.product();
      profMC_->Fill(3, reconstructable->numTPs() / (double)setup_->numRegions());
      profMC_->Fill(4, selection->numTPs() / (double)setup_->numRegions());
      profMC_->Fill(5, selection->numTPs());
      for (const auto& p : selection->getTrackingParticleToTTStubsMap())
        fill(p.first, hisEffMC_);
    }
    // analyze dtc products and find still reconstrucable TrackingParticles
    std::set<TPPtr> tpPtrs;
    for (int region = 0; region < setup_->numRegions(); region++) {
      int nStubs(0);
      int nLost(0);
      std::map<TPPtr, std::vector<TTStubRef>> mapTPsTTStubs;
      for (int channel = 0; channel < setup_->numDTCsPerTFP(); channel++) {
        const tt::StreamStub& accepted = handleTTDTCAccepted->stream(region, channel);
        const tt::StreamStub& lost = handleTTDTCLost->stream(region, channel);
        hisChannel_->Fill(accepted.size());
        profChannel_->Fill(channel, accepted.size());
        fill(accepted, region, channel, nStubs, hisRZStubs_);
        fill(lost, region, channel, nLost, hisRZStubsLost_);
        if (!useMCTruth_)
          continue;
        for (const tt::FrameStub& frame : accepted) {
          if (frame.first.isNull())
            continue;
          for (const TPPtr& tpPtr : selection->findTrackingParticlePtrs(frame.first)) {
            auto it = mapTPsTTStubs.find(tpPtr);
            if (it == mapTPsTTStubs.end()) {
              it = mapTPsTTStubs.emplace(tpPtr, std::vector<TTStubRef>()).first;
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
    log_ << "'Lost' below refers to truncation losses" << std::endl;
    // printout MC summary
    endJobMC();
    // printout DTC summary
    endJobDTC();
    log_ << "=============================================================";
    edm::LogPrint(moduleDescription().moduleName()) << log_.str();
  }

  // fills kinematic tp histograms
  void Analyzer::fill(const TPPtr& tpPtr, const std::vector<TH1F*> th1fs) const {
    const double s = sin(tpPtr->phi());
    const double c = cos(tpPtr->phi());
    const TrackingParticle::Point& v = tpPtr->vertex();
    const std::vector<double> x = {tpPtr->phi(),
                                   tpPtr->pt(),
                                   tpPtr->charge() / tpPtr->pt(),
                                   v.x() * s - v.y() * c,
                                   v.z() - (v.x() * c + v.y() * s) * sinh(tpPtr->eta()),
                                   tpPtr->eta()};
    for (Efficiency e : AllEfficiency)
      th1fs[e]->Fill(x[e]);
  }

  // fill stub related histograms
  void Analyzer::fill(const tt::StreamStub& stream, int region, int channel, int& sum, TH2F* th2f) {
    for (const tt::FrameStub& frame : stream) {
      if (frame.first.isNull())
        continue;
      sum++;
      const GlobalPoint& pos = setup_->stubPos(frame, region);
      const GlobalPoint& ttPos = setup_->stubPos(frame.first);
      const std::vector<double> resolutions = {
          ttPos.perp() - pos.perp(), tt::deltaPhi(ttPos.phi() - pos.phi()), ttPos.z() - pos.z()};
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
    const std::vector<double> nums = {numStubs, numStubsMatched, numTPsReco, numTPsEff, numCluster};
    const std::vector<double> errs = {errStubs, errStubsMatched, errTPsReco, errTPsEff, errCluster};
    const int wNums = std::ceil(std::log10(*std::max_element(nums.begin(), nums.end()))) + 5;
    const int wErrs = std::ceil(std::log10(*std::max_element(errs.begin(), errs.end()))) + 5;
    log_ << "=============================================================" << std::endl;
    log_ << "                         Monte Carlo  SUMMARY                         " << std::endl;
    log_ << "number of cluster       per TFP = " << std::setw(wNums) << numCluster << " +- " << std::setw(wErrs)
         << errCluster << std::endl;
    log_ << "number of stubs         per TFP = " << std::setw(wNums) << numStubs << " +- " << std::setw(wErrs)
         << errStubs << std::endl;
    log_ << "number of matched stubs per TFP = " << std::setw(wNums) << numStubsMatched << " +- " << std::setw(wErrs)
         << errStubsMatched << std::endl;
    log_ << "number of TPs           per TFP = " << std::setw(wNums) << numTPsReco << " +- " << std::setw(wErrs)
         << errTPsReco << std::endl;
    log_ << "number of TPs for eff   per TFP = " << std::setw(wNums) << numTPsEff << " +- " << std::setw(wErrs)
         << errTPsEff << std::endl;
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
    const std::vector<double> nums = {numStubs, numStubsLost};
    const std::vector<double> errs = {errStubs, errStubsLost};
    const int wNums = std::ceil(std::log10(*std::max_element(nums.begin(), nums.end()))) + 5;
    const int wErrs = std::ceil(std::log10(*std::max_element(errs.begin(), errs.end()))) + 5;
    log_ << "=============================================================" << std::endl;
    log_ << "                         DTC SUMMARY                         " << std::endl;
    log_ << "number of stubs      per TFP = " << std::setw(wNums) << numStubs << " +- " << std::setw(wErrs) << errStubs
         << std::endl;
    log_ << "number of lost stubs per TFP = " << std::setw(wNums) << numStubsLost << " +- " << std::setw(wErrs)
         << errStubsLost << std::endl;
    log_ << "     max tracking efficiency = " << std::setw(wNums) << eff << " +- " << std::setw(wErrs) << errEff
         << std::endl;
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::Analyzer);
