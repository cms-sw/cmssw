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
#include "DataFormats/Common/interface/Handle.h"

#include "SimTracker/TrackTriggerAssociation/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <TProfile.h>
#include <TH1F.h>
#include <TEfficiency.h>

#include <vector>
#include <deque>
#include <set>
#include <cmath>
#include <numeric>
#include <sstream>

namespace trackerTFP {

  /*! \class  trackerTFP::AnalyzerTFP
   *  \brief  Class to analyze TTTracks found by tfp
   *  \author Thomas Schuh
   *  \date   20204 Aug
   */
  class AnalyzerTFP : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
  public:
    AnalyzerTFP(const edm::ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override;
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override {}
    void endJob() override;

  private:
    // gets all TPs associated too any of the tracks & number of tracks matching at least one TP
    void associate(const std::vector<std::vector<TTStubRef>>& tracks,
                   const tt::StubAssociation* ass,
                   std::set<TPPtr>& tps,
                   int& nMatchTrk,
                   bool perfect = false) const;

    // ED input token of tracks
    edm::EDGetTokenT<tt::TTTracks> edGetToken_;
    // ED input token of TTStubRef to TPPtr association for tracking efficiency
    edm::EDGetTokenT<tt::StubAssociation> edGetTokenSelection_;
    // ED input token of TTStubRef to recontructable TPPtr association
    edm::EDGetTokenT<tt::StubAssociation> edGetTokenReconstructable_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // stores, calculates and provides run-time constants
    const tt::Setup* setup_ = nullptr;
    // enables analyze of TPs
    bool useMCTruth_;
    //
    int nEvents_ = 0;

    // Histograms

    // counts per TFP (processing nonant and event)
    TProfile* prof_;
    // no. of tracks per nonant
    TProfile* profChannel_;
    TH1F* hisChannel_;
    TH1F* hisEff_;
    TH1F* hisEffTotal_;
    TEfficiency* eff_;

    // printout
    std::stringstream log_;
  };

  AnalyzerTFP::AnalyzerTFP(const edm::ParameterSet& iConfig) : useMCTruth_(iConfig.getParameter<bool>("UseMCTruth")) {
    usesResource("TFileService");
    // book in- and output ED products
    const std::string& label = iConfig.getParameter<std::string>("OutputLabelTFP");
    const std::string& branch = iConfig.getParameter<std::string>("BranchTTTracks");
    edGetToken_ = consumes<tt::TTTracks>(edm::InputTag(label, branch));
    if (useMCTruth_) {
      const auto& inputTagSelecttion = iConfig.getParameter<edm::InputTag>("InputTagSelection");
      const auto& inputTagReconstructable = iConfig.getParameter<edm::InputTag>("InputTagReconstructable");
      edGetTokenSelection_ = consumes<tt::StubAssociation>(inputTagSelecttion);
      edGetTokenReconstructable_ = consumes<tt::StubAssociation>(inputTagReconstructable);
    }
    // book ES products
    esGetTokenSetup_ = esConsumes<edm::Transition::BeginRun>();
    // log config
    log_.setf(std::ios::fixed, std::ios::floatfield);
    log_.precision(4);
  }

  void AnalyzerTFP::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    // book histograms
    edm::Service<TFileService> fs;
    TFileDirectory dir;
    dir = fs->mkdir("TFP");
    prof_ = dir.make<TProfile>("Counts", ";", 10, 0.5, 10.5);
    prof_->GetXaxis()->SetBinLabel(1, "Stubs");
    prof_->GetXaxis()->SetBinLabel(2, "Tracks");
    prof_->GetXaxis()->SetBinLabel(3, "Lost Tracks");
    prof_->GetXaxis()->SetBinLabel(4, "Matched Tracks");
    prof_->GetXaxis()->SetBinLabel(5, "All Tracks");
    prof_->GetXaxis()->SetBinLabel(6, "Found TPs");
    prof_->GetXaxis()->SetBinLabel(7, "Found selected TPs");
    prof_->GetXaxis()->SetBinLabel(8, "Lost TPs");
    prof_->GetXaxis()->SetBinLabel(9, "All TPs");
    prof_->GetXaxis()->SetBinLabel(10, "Perfectly Found selected TPs");
    // channel occupancy
    constexpr int maxOcc = 180;
    const int numChannels = setup_->numRegions();
    hisChannel_ = dir.make<TH1F>("His Channel Occupancy", ";", maxOcc, -.5, maxOcc - .5);
    profChannel_ = dir.make<TProfile>("Prof Channel Occupancy", ";", numChannels, -.5, numChannels - .5);
    // Efficiencies
    hisEffTotal_ = dir.make<TH1F>("HisTPEtaTotal", ";", 128, -2.5, 2.5);
    hisEff_ = dir.make<TH1F>("HisTPEta", ";", 128, -2.5, 2.5);
    eff_ = dir.make<TEfficiency>("EffEta", ";", 128, -2.5, 2.5);
  }

  void AnalyzerTFP::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    auto fill = [](const TPPtr& tpPtr, TH1F* his) { his->Fill(tpPtr->eta()); };
    // read in tracklet products
    edm::Handle<tt::TTTracks> handle;
    iEvent.getByToken<tt::TTTracks>(edGetToken_, handle);
    // read in MCTruth
    const tt::StubAssociation* selection = nullptr;
    const tt::StubAssociation* reconstructable = nullptr;
    if (useMCTruth_) {
      edm::Handle<tt::StubAssociation> handleSelection;
      iEvent.getByToken<tt::StubAssociation>(edGetTokenSelection_, handleSelection);
      selection = handleSelection.product();
      prof_->Fill(9, selection->numTPs());
      edm::Handle<tt::StubAssociation> handleReconstructable;
      iEvent.getByToken<tt::StubAssociation>(edGetTokenReconstructable_, handleReconstructable);
      reconstructable = handleReconstructable.product();
      for (const auto& p : selection->getTrackingParticleToTTStubsMap())
        fill(p.first, hisEffTotal_);
    }
    //
    const tt::TTTracks& ttTracks = *handle.product();
    std::vector<std::vector<TTTrackRef>> ttTrackRefsRegions(setup_->numRegions());
    std::vector<int> nTTTracksRegions(setup_->numRegions(), 0);
    for (const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack : ttTracks)
      nTTTracksRegions[ttTrack.phiSector()]++;
    for (int region = 0; region < setup_->numRegions(); region++)
      ttTrackRefsRegions[region].reserve(nTTTracksRegions[region]);
    int i(0);
    for (const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack : ttTracks)
      ttTrackRefsRegions[ttTrack.phiSector()].emplace_back(TTTrackRef(handle, i++));
    for (int region = 0; region < setup_->numRegions(); region++) {
      const std::vector<TTTrackRef>& ttTrackRefs = ttTrackRefsRegions[region];
      const int nStubs =
          std::accumulate(ttTrackRefs.begin(), ttTrackRefs.end(), 0, [](int sum, const TTTrackRef& ttTrackRef) {
            return sum + ttTrackRef->getStubRefs().size();
          });
      const int nTracks = ttTrackRefs.size();
      prof_->Fill(1, nStubs);
      prof_->Fill(2, nTracks);
      // no access to lost tracks
      prof_->Fill(3, 0);
      hisChannel_->Fill(nTracks);
      profChannel_->Fill(region, nTracks);
    }
    // analyze tracklet products and associate found tracks with reconstrucable TrackingParticles
    std::set<TPPtr> tpPtrs;
    std::set<TPPtr> tpPtrsSelection;
    std::set<TPPtr> tpPtrsPerfect;
    int nAllMatched(0);
    // convert vector of tracks to vector of vector of associated stubs
    std::vector<std::vector<TTStubRef>> tracks;
    tracks.reserve(ttTracks.size());
    std::transform(ttTracks.begin(),
                   ttTracks.end(),
                   std::back_inserter(tracks),
                   [](const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack) { return ttTrack.getStubRefs(); });
    if (useMCTruth_) {
      int tmp(0);
      associate(tracks, selection, tpPtrsSelection, tmp);
      associate(tracks, selection, tpPtrsPerfect, tmp, true);
      associate(tracks, reconstructable, tpPtrs, nAllMatched);
    }
    for (const TPPtr& tpPtr : tpPtrsSelection)
      fill(tpPtr, hisEff_);
    prof_->Fill(4, nAllMatched);
    prof_->Fill(5, ttTracks.size());
    prof_->Fill(6, tpPtrs.size());
    prof_->Fill(7, tpPtrsSelection.size());
    // no access to lost tp
    prof_->Fill(8, 0);
    prof_->Fill(10, tpPtrsPerfect.size());
    nEvents_++;
  }

  void AnalyzerTFP::endJob() {
    if (nEvents_ == 0)
      return;
    // effi
    eff_->SetPassedHistogram(*hisEff_, "f");
    eff_->SetTotalHistogram(*hisEffTotal_, "f");
    // printout SF summary
    const double totalTPs = prof_->GetBinContent(9);
    const double numStubs = prof_->GetBinContent(1);
    const double numTracks = prof_->GetBinContent(2);
    const double totalTracks = prof_->GetBinContent(5);
    const double numTracksMatched = prof_->GetBinContent(4);
    const double numTPsAll = prof_->GetBinContent(6);
    const double numTPsEff = prof_->GetBinContent(7);
    const double numTPsEffPerfect = prof_->GetBinContent(10);
    const double errStubs = prof_->GetBinError(1);
    const double errTracks = prof_->GetBinError(2);
    const double fracFake = (totalTracks - numTracksMatched) / totalTracks;
    const double fracDup = (numTracksMatched - numTPsAll) / totalTracks;
    const double eff = numTPsEff / totalTPs;
    const double errEff = sqrt(eff * (1. - eff) / totalTPs / nEvents_);
    const double effPerfect = numTPsEffPerfect / totalTPs;
    const double errEffPerfect = sqrt(effPerfect * (1. - effPerfect) / totalTPs / nEvents_);
    const std::vector<double> nums = {numStubs, numTracks};
    const std::vector<double> errs = {errStubs, errTracks};
    const int wNums = std::ceil(std::log10(*std::max_element(nums.begin(), nums.end()))) + 5;
    const int wErrs = std::ceil(std::log10(*std::max_element(errs.begin(), errs.end()))) + 5;
    log_ << "                      TFP  SUMMARY                      " << std::endl;
    log_ << "number of stubs     per TFP = " << std::setw(wNums) << numStubs << " +- " << std::setw(wErrs) << errStubs
         << std::endl;
    log_ << "number of tracks    per TFP = " << std::setw(wNums) << numTracks << " +- " << std::setw(wErrs) << errTracks
         << std::endl;
    log_ << "current tracking efficiency = " << std::setw(wNums) << effPerfect << " +- " << std::setw(wErrs)
         << errEffPerfect << std::endl;
    log_ << "max     tracking efficiency = " << std::setw(wNums) << eff << " +- " << std::setw(wErrs) << errEff
         << std::endl;
    log_ << "                  fake rate = " << std::setw(wNums) << fracFake << std::endl;
    log_ << "             duplicate rate = " << std::setw(wNums) << fracDup << std::endl;
    log_ << "=============================================================";
    edm::LogPrint(moduleDescription().moduleName()) << log_.str();
  }

  // gets all TPs associated too any of the tracks & number of tracks matching at least one TP
  void AnalyzerTFP::associate(const std::vector<std::vector<TTStubRef>>& tracks,
                              const tt::StubAssociation* ass,
                              std::set<TPPtr>& tps,
                              int& nMatchTrk,
                              bool perfect) const {
    for (const std::vector<TTStubRef>& ttStubRefs : tracks) {
      const std::vector<TPPtr>& tpPtrs = perfect ? ass->associateFinal(ttStubRefs) : ass->associate(ttStubRefs);
      if (tpPtrs.empty())
        continue;
      nMatchTrk++;
      std::copy(tpPtrs.begin(), tpPtrs.end(), std::inserter(tps, tps.begin()));
    }
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::AnalyzerTFP);
