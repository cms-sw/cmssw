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
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"

#include <TProfile.h>
#include <TH1F.h>
#include <TEfficiency.h>

#include <vector>
#include <deque>
#include <set>
#include <cmath>
#include <numeric>
#include <sstream>

using namespace std;
using namespace edm;
using namespace trackerTFP;
using namespace tt;

namespace trklet {

  /*! \class  trklet::AnalyzerTracklet
   *  \brief  Class to analyze TTTracks found by tracklet pattern recognition
   *  \author Thomas Schuh
   *  \date   2020, Oct
   */
  class AnalyzerTracklet : public one::EDAnalyzer<one::WatchRuns, one::SharedResources> {
  public:
    AnalyzerTracklet(const ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const Run& iEvent, const EventSetup& iSetup) override;
    void analyze(const Event& iEvent, const EventSetup& iSetup) override;
    void endRun(const Run& iEvent, const EventSetup& iSetup) override {}
    void endJob() override;

  private:
    // gets all TPs associated too any of the tracks & number of tracks matching at least one TP
    void associate(const vector<vector<TTStubRef>>& tracks,
                   const StubAssociation* ass,
                   set<TPPtr>& tps,
                   int& nMatchTrk,
                   bool perfect = false) const;

    // ED input token of tracks
    EDGetTokenT<TTTracks> edGetToken_;
    // ED input token of TTStubRef to TPPtr association for tracking efficiency
    EDGetTokenT<StubAssociation> edGetTokenSelection_;
    // ED input token of TTStubRef to recontructable TPPtr association
    EDGetTokenT<StubAssociation> edGetTokenReconstructable_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // stores, calculates and provides run-time constants
    const Setup* setup_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
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
    stringstream log_;
  };

  AnalyzerTracklet::AnalyzerTracklet(const ParameterSet& iConfig)
      : useMCTruth_(iConfig.getParameter<bool>("UseMCTruth")) {
    usesResource("TFileService");
    // book in- and output ED products
    const InputTag& inputTag = iConfig.getParameter<InputTag>("InputTag");
    edGetToken_ = consumes<TTTracks>(inputTag);
    if (useMCTruth_) {
      const auto& inputTagSelecttion = iConfig.getParameter<InputTag>("InputTagSelection");
      const auto& inputTagReconstructable = iConfig.getParameter<InputTag>("InputTagReconstructable");
      edGetTokenSelection_ = consumes<StubAssociation>(inputTagSelecttion);
      edGetTokenReconstructable_ = consumes<StubAssociation>(inputTagReconstructable);
    }
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    // log config
    log_.setf(ios::fixed, ios::floatfield);
    log_.precision(4);
  }

  void AnalyzerTracklet::beginRun(const Run& iEvent, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // book histograms
    Service<TFileService> fs;
    TFileDirectory dir;
    dir = fs->mkdir("Tracklet");
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

  void AnalyzerTracklet::analyze(const Event& iEvent, const EventSetup& iSetup) {
    auto fill = [](const TPPtr& tpPtr, TH1F* his) { his->Fill(tpPtr->eta()); };
    // read in tracklet products
    Handle<TTTracks> handle;
    iEvent.getByToken<TTTracks>(edGetToken_, handle);
    // read in MCTruth
    const StubAssociation* selection = nullptr;
    const StubAssociation* reconstructable = nullptr;
    if (useMCTruth_) {
      Handle<StubAssociation> handleSelection;
      iEvent.getByToken<StubAssociation>(edGetTokenSelection_, handleSelection);
      selection = handleSelection.product();
      prof_->Fill(9, selection->numTPs());
      Handle<StubAssociation> handleReconstructable;
      iEvent.getByToken<StubAssociation>(edGetTokenReconstructable_, handleReconstructable);
      reconstructable = handleReconstructable.product();
      for (const auto& p : selection->getTrackingParticleToTTStubsMap())
        fill(p.first, hisEffTotal_);
    }
    //
    const TTTracks& ttTracks = *handle.product();
    vector<vector<TTTrackRef>> ttTrackRefsRegions(setup_->numRegions());
    vector<int> nTTTracksRegions(setup_->numRegions(), 0);
    for (const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack : ttTracks)
      nTTTracksRegions[ttTrack.phiSector()]++;
    for (int region = 0; region < setup_->numRegions(); region++)
      ttTrackRefsRegions[region].reserve(nTTTracksRegions[region]);
    int i(0);
    for (const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack : ttTracks)
      ttTrackRefsRegions[ttTrack.phiSector()].emplace_back(TTTrackRef(handle, i++));
    for (int region = 0; region < setup_->numRegions(); region++) {
      const vector<TTTrackRef>& ttTrackRefs = ttTrackRefsRegions[region];
      const int nStubs =
          accumulate(ttTrackRefs.begin(), ttTrackRefs.end(), 0, [](int& sum, const TTTrackRef& ttTrackRef) {
            return sum += ttTrackRef->getStubRefs().size();
          });
      const int nTracks = ttTrackRefs.size();
      hisChannel_->Fill(nTracks);
      profChannel_->Fill(region, nTracks);
      prof_->Fill(1, nStubs);
      prof_->Fill(2, nTracks);
      // no access to lost tracks
      prof_->Fill(3, 0);
    }
    // analyze tracklet products and associate found tracks with reconstrucable TrackingParticles
    set<TPPtr> tpPtrs;
    set<TPPtr> tpPtrsSelection;
    set<TPPtr> tpPtrsPerfect;
    int nAllMatched(0);
    // convert vector of tracks to vector of vector of associated stubs
    vector<vector<TTStubRef>> tracks;
    tracks.reserve(ttTracks.size());
    transform(
        ttTracks.begin(), ttTracks.end(), back_inserter(tracks), [](const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack) {
          return ttTrack.getStubRefs();
        });
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

  void AnalyzerTracklet::endJob() {
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
    const vector<double> nums = {numStubs, numTracks};
    const vector<double> errs = {errStubs, errTracks};
    const int wNums = ceil(log10(*max_element(nums.begin(), nums.end()))) + 5;
    const int wErrs = ceil(log10(*max_element(errs.begin(), errs.end()))) + 5;
    log_ << "                      Tracklet  SUMMARY                      " << endl;
    log_ << "number of stubs     per TFP = " << setw(wNums) << numStubs << " +- " << setw(wErrs) << errStubs << endl;
    log_ << "number of tracks    per TFP = " << setw(wNums) << numTracks << " +- " << setw(wErrs) << errTracks << endl;
    log_ << "current tracking efficiency = " << setw(wNums) << effPerfect << " +- " << setw(wErrs) << errEffPerfect
         << endl;
    log_ << "max     tracking efficiency = " << setw(wNums) << eff << " +- " << setw(wErrs) << errEff << endl;
    log_ << "                  fake rate = " << setw(wNums) << fracFake << endl;
    log_ << "             duplicate rate = " << setw(wNums) << fracDup << endl;
    log_ << "=============================================================";
    LogPrint("L1Trigger/TrackFindingTracklet") << log_.str();
  }

  // gets all TPs associated too any of the tracks & number of tracks matching at least one TP
  void AnalyzerTracklet::associate(const vector<vector<TTStubRef>>& tracks,
                                   const StubAssociation* ass,
                                   set<TPPtr>& tps,
                                   int& nMatchTrk,
                                   bool perfect) const {
    for (const vector<TTStubRef>& ttStubRefs : tracks) {
      const vector<TPPtr>& tpPtrs = perfect ? ass->associateFinal(ttStubRefs) : ass->associate(ttStubRefs);
      if (tpPtrs.empty())
        continue;
      nMatchTrk++;
      copy(tpPtrs.begin(), tpPtrs.end(), inserter(tps, tps.begin()));
    }
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::AnalyzerTracklet);