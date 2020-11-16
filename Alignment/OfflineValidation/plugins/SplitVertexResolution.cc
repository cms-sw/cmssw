// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Class:      SplitVertexResolution
//
/**\class SplitVertexResolution SplitVertexResolution.cc Alignment/OfflineValidation/plugins/SplitVertexResolution.cc

*/
//
// Original Author:  Marco Musich
//         Created:  Mon, 13 Jun 2016 15:07:11 GMT
//
//

// system include files
#include <memory>
#include <algorithm>  // std::sort
#include <vector>     // std::vector
#include <chrono>
#include <iostream>
#include <random>
#include <boost/range/adaptor/indexed.hpp>

// ROOT include files
#include "TTree.h"
#include "TProfile.h"
#include "TF1.h"
#include "TMath.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "Alignment/OfflineValidation/interface/PVValidationHelpers.h"
#include "Alignment/OfflineValidation/interface/pvTree.h"

//
// useful code
//

namespace statmode {
  using fitParams = std::pair<Measurement1D, Measurement1D>;
}

//
// class declaration
//

class SplitVertexResolution : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit SplitVertexResolution(const edm::ParameterSet&);
  ~SplitVertexResolution() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static bool mysorter(reco::Track i, reco::Track j) { return (i.pt() > j.pt()); }

private:
  void beginJob() override;
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  virtual void beginEvent() final;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  void endRun(edm::Run const&, edm::EventSetup const&) override{};

  template <std::size_t SIZE>
  bool checkBinOrdering(std::array<float, SIZE>& bins);
  std::vector<TH1F*> bookResidualsHistogram(TFileDirectory dir,
                                            unsigned int theNOfBins,
                                            TString resType,
                                            TString varType);

  void fillTrendPlotByIndex(TH1F* trendPlot, std::vector<TH1F*>& h, PVValHelper::estimator fitPar_);
  statmode::fitParams fitResiduals(TH1* hist, bool singleTime = false);
  statmode::fitParams fitResiduals_v0(TH1* hist);

  std::pair<long long, long long> getRunTime(const edm::EventSetup& iSetup) const;

  // counters
  int ievt;
  int itrks;

  // compression settings
  const int compressionSettings_;

  // switch to keep the ntuple
  bool storeNtuple_;

  // storing integrated luminosity
  double intLumi_;
  bool debug_;

  edm::InputTag pvsTag_;
  edm::EDGetTokenT<reco::VertexCollection> pvsToken_;

  edm::InputTag tracksTag_;
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;

  edm::InputTag triggerResultsTag_ = edm::InputTag("TriggerResults", "", "HLT");  //InputTag tag("TriggerResults");
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;

  // ES Tokens
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackBuilderToken_;
  edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;

  double minVtxNdf_;
  double minVtxWgt_;

  // checks on the run to be processed

  bool runControl_;
  std::vector<unsigned int> runControlNumbers_;

  static constexpr double cmToUm = 10000.;

  edm::Service<TFileService> outfile_;

  TH1F* h_lumiFromConfig;
  TH1I* h_runFromConfig;

  std::map<unsigned int, std::pair<long long, long long> > runNumbersTimesLog_;
  TH1I* h_runStartTimes;
  TH1I* h_runEndTimes;

  TH1F* h_diffX;
  TH1F* h_diffY;
  TH1F* h_diffZ;

  TH1F* h_OrigVertexErrX;
  TH1F* h_OrigVertexErrY;
  TH1F* h_OrigVertexErrZ;

  TH1F* h_errX;
  TH1F* h_errY;
  TH1F* h_errZ;

  TH1F* h_pullX;
  TH1F* h_pullY;
  TH1F* h_pullZ;

  TH1F* h_ntrks;
  TH1F* h_sumPt;
  TH1F* h_avgSumPt;

  TH1F* h_sumPt1;
  TH1F* h_sumPt2;

  TH1F* h_wTrks1;
  TH1F* h_wTrks2;

  TH1F* h_minWTrks1;
  TH1F* h_minWTrks2;

  TH1F* h_PVCL_subVtx1;
  TH1F* h_PVCL_subVtx2;

  TH1F* h_runNumber;

  TH1I* h_nOfflineVertices;
  TH1I* h_nVertices;
  TH1I* h_nNonFakeVertices;
  TH1I* h_nFinalVertices;

  // trigger results

  TH1D* tksByTrigger_;
  TH1D* evtsByTrigger_;

  // resolutions

  std::vector<TH1F*> h_resolX_sumPt_;
  std::vector<TH1F*> h_resolY_sumPt_;
  std::vector<TH1F*> h_resolZ_sumPt_;

  std::vector<TH1F*> h_resolX_Ntracks_;
  std::vector<TH1F*> h_resolY_Ntracks_;
  std::vector<TH1F*> h_resolZ_Ntracks_;

  std::vector<TH1F*> h_resolX_Nvtx_;
  std::vector<TH1F*> h_resolY_Nvtx_;
  std::vector<TH1F*> h_resolZ_Nvtx_;

  TH1F* p_resolX_vsSumPt;
  TH1F* p_resolY_vsSumPt;
  TH1F* p_resolZ_vsSumPt;

  TH1F* p_resolX_vsNtracks;
  TH1F* p_resolY_vsNtracks;
  TH1F* p_resolZ_vsNtracks;

  TH1F* p_resolX_vsNvtx;
  TH1F* p_resolY_vsNvtx;
  TH1F* p_resolZ_vsNvtx;

  // pulls
  std::vector<TH1F*> h_pullX_sumPt_;
  std::vector<TH1F*> h_pullY_sumPt_;
  std::vector<TH1F*> h_pullZ_sumPt_;

  std::vector<TH1F*> h_pullX_Ntracks_;
  std::vector<TH1F*> h_pullY_Ntracks_;
  std::vector<TH1F*> h_pullZ_Ntracks_;

  std::vector<TH1F*> h_pullX_Nvtx_;
  std::vector<TH1F*> h_pullY_Nvtx_;
  std::vector<TH1F*> h_pullZ_Nvtx_;

  TH1F* p_pullX_vsSumPt;
  TH1F* p_pullY_vsSumPt;
  TH1F* p_pullZ_vsSumPt;

  TH1F* p_pullX_vsNtracks;
  TH1F* p_pullY_vsNtracks;
  TH1F* p_pullZ_vsNtracks;

  TH1F* p_pullX_vsNvtx;
  TH1F* p_pullY_vsNvtx;
  TH1F* p_pullZ_vsNvtx;

  std::mt19937 engine_;

  pvEvent event_;
  TTree* tree_;

  // ----------member data ---------------------------
  static const int nPtBins_ = 30;
  std::array<float, nPtBins_ + 1> mypT_bins_ = PVValHelper::makeLogBins<float, nPtBins_>(1., 1e3);

  static const int nTrackBins_ = 60;
  std::array<float, nTrackBins_ + 1> myNTrack_bins_;

  static const int nVtxBins_ = 40;
  std::array<float, nVtxBins_ + 1> myNVtx_bins_;

  std::map<std::string, std::pair<int, int> > triggerMap_;
};

SplitVertexResolution::SplitVertexResolution(const edm::ParameterSet& iConfig)
    : compressionSettings_(iConfig.getUntrackedParameter<int>("compressionSettings", -1)),
      storeNtuple_(iConfig.getParameter<bool>("storeNtuple")),
      intLumi_(iConfig.getUntrackedParameter<double>("intLumi", 0.)),
      debug_(iConfig.getUntrackedParameter<bool>("Debug", false)),
      pvsTag_(iConfig.getParameter<edm::InputTag>("vtxCollection")),
      pvsToken_(consumes<reco::VertexCollection>(pvsTag_)),
      tracksTag_(iConfig.getParameter<edm::InputTag>("trackCollection")),
      tracksToken_(consumes<reco::TrackCollection>(tracksTag_)),
      triggerResultsToken_(consumes<edm::TriggerResults>(triggerResultsTag_)),
      transientTrackBuilderToken_(
          esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))),
      runInfoToken_(esConsumes<RunInfo, RunInfoRcd, edm::Transition::BeginRun>()),
      minVtxNdf_(iConfig.getUntrackedParameter<double>("minVertexNdf")),
      minVtxWgt_(iConfig.getUntrackedParameter<double>("minVertexMeanWeight")),
      runControl_(iConfig.getUntrackedParameter<bool>("runControl", false)) {
  usesResource(TFileService::kSharedResource);

  std::vector<unsigned int> defaultRuns;
  defaultRuns.push_back(0);
  runControlNumbers_ = iConfig.getUntrackedParameter<std::vector<unsigned int> >("runControlNumber", defaultRuns);

  std::vector<float> vect = PVValHelper::generateBins(nTrackBins_ + 1, -0.5, 120.);
  std::copy(vect.begin(), vect.begin() + nTrackBins_ + 1, myNTrack_bins_.begin());

  vect.clear();
  vect = PVValHelper::generateBins(nVtxBins_ + 1, 1., 40.);
  std::copy(vect.begin(), vect.begin() + nVtxBins_ + 1, myNVtx_bins_.begin());
}

SplitVertexResolution::~SplitVertexResolution() {}

// ------------ method called for each event  ------------
void SplitVertexResolution::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // deterministic seed from the event number
  // should not bias the result as the event number is already
  // assigned randomly-enough
  engine_.seed(iEvent.id().event() + (iEvent.id().luminosityBlock() << 10) + (iEvent.id().run() << 20));

  // first check if the event passes the run control
  bool passesRunControl = false;

  if (runControl_) {
    for (const auto& runControlNumber : runControlNumbers_) {
      if (iEvent.eventAuxiliary().run() == runControlNumber) {
        if (debug_) {
          edm::LogInfo("SplitVertexResolution")
              << " run number: " << iEvent.eventAuxiliary().run() << " keeping run:" << runControlNumber;
        }
        passesRunControl = true;
        break;
      }
    }
    if (!passesRunControl)
      return;
  }

  // Fill general info
  h_runNumber->Fill(iEvent.id().run());

  ievt++;
  edm::Handle<edm::TriggerResults> hltresults;
  iEvent.getByToken(triggerResultsToken_, hltresults);

  const edm::TriggerNames& triggerNames_ = iEvent.triggerNames(*hltresults);
  int ntrigs = hltresults->size();
  //const std::vector<std::string>& triggernames = triggerNames_.triggerNames();

  beginEvent();

  // Fill general info
  event_.runNumber = iEvent.id().run();
  event_.luminosityBlockNumber = iEvent.id().luminosityBlock();
  event_.eventNumber = iEvent.id().event();

  TransientTrackBuilder const& theB = iSetup.getData(transientTrackBuilderToken_);

  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(pvsToken_, vertices);
  const reco::VertexCollection pvtx = *(vertices.product());

  event_.nVtx = pvtx.size();
  int nOfflineVtx = pvtx.size();
  h_nOfflineVertices->Fill(nOfflineVtx);

  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(tracksToken_, tracks);
  itrks += tracks.product()->size();

  for (int itrig = 0; itrig != ntrigs; ++itrig) {
    const std::string& trigName = triggerNames_.triggerName(itrig);
    bool accept = hltresults->accept(itrig);
    if (accept == 1) {
      triggerMap_[trigName].first += 1;
      triggerMap_[trigName].second += tracks.product()->size();
      // triggerInfo.push_back(pair <string, int> (trigName, accept));
    }
  }

  int counter = 0;
  int noFakecounter = 0;
  int goodcounter = 0;

  for (auto pvIt = pvtx.cbegin(); pvIt != pvtx.cend(); ++pvIt) {
    reco::Vertex iPV = *pvIt;
    counter++;
    if (iPV.isFake())
      continue;
    noFakecounter++;

    // vertex selection as in bs code
    if (iPV.ndof() < minVtxNdf_ || (iPV.ndof() + 3.) / iPV.tracksSize() < 2 * minVtxWgt_)
      continue;

    goodcounter++;
    reco::TrackCollection allTracks;
    reco::TrackCollection groupOne, groupTwo;
    for (auto trki = iPV.tracks_begin(); trki != iPV.tracks_end(); ++trki) {
      if (trki->isNonnull()) {
        reco::TrackRef trk_now(tracks, (*trki).key());
        allTracks.push_back(*trk_now);
      }
    }

    if (goodcounter > 1)
      continue;

    // order with decreasing pt
    std::sort(allTracks.begin(), allTracks.end(), mysorter);

    int ntrks = allTracks.size();
    h_ntrks->Fill(ntrks);

    // discard lowest pt track
    uint even_ntrks;
    ntrks % 2 == 0 ? even_ntrks = ntrks : even_ntrks = ntrks - 1;

    // split into two sets equally populated
    for (uint tracksIt = 0; tracksIt < even_ntrks; tracksIt = tracksIt + 2) {
      reco::Track firstTrk = allTracks.at(tracksIt);
      reco::Track secondTrk = allTracks.at(tracksIt + 1);
      auto dis = std::uniform_int_distribution<>(0, 1);  // [0, 1]

      if (dis(engine_) > 0.5) {
        groupOne.push_back(firstTrk);
        groupTwo.push_back(secondTrk);
      } else {
        groupOne.push_back(secondTrk);
        groupTwo.push_back(firstTrk);
      }
    }

    if (!(groupOne.size() >= 2 && groupTwo.size() >= 2))
      continue;

    h_OrigVertexErrX->Fill(iPV.xError() * cmToUm);
    h_OrigVertexErrY->Fill(iPV.yError() * cmToUm);
    h_OrigVertexErrZ->Fill(iPV.zError() * cmToUm);

    float sumPt = 0, sumPt1 = 0, sumPt2 = 0, avgSumPt = 0;

    // refit the two sets of tracks
    std::vector<reco::TransientTrack> groupOne_ttks;
    groupOne_ttks.clear();
    for (auto itrk = groupOne.cbegin(); itrk != groupOne.cend(); itrk++) {
      reco::TransientTrack tmpTransientTrack = theB.build(*itrk);
      groupOne_ttks.push_back(tmpTransientTrack);
      sumPt1 += itrk->pt();
      sumPt += itrk->pt();
    }

    AdaptiveVertexFitter pvFitter;
    TransientVertex pvOne = pvFitter.vertex(groupOne_ttks);
    if (!pvOne.isValid())
      continue;

    reco::Vertex onePV = pvOne;

    std::vector<reco::TransientTrack> groupTwo_ttks;
    groupTwo_ttks.clear();
    for (auto itrk = groupTwo.cbegin(); itrk != groupTwo.cend(); itrk++) {
      reco::TransientTrack tmpTransientTrack = theB.build(*itrk);
      groupTwo_ttks.push_back(tmpTransientTrack);
      sumPt2 += itrk->pt();
      sumPt += itrk->pt();
    }

    // average sumPt
    avgSumPt = (sumPt1 + sumPt2) / 2.;
    h_avgSumPt->Fill(avgSumPt);

    TransientVertex pvTwo = pvFitter.vertex(groupTwo_ttks);
    if (!pvTwo.isValid())
      continue;

    reco::Vertex twoPV = pvTwo;

    float theminW1 = 1.;
    float theminW2 = 1.;
    for (auto otrk = pvOne.originalTracks().cbegin(); otrk != pvOne.originalTracks().cend(); ++otrk) {
      h_wTrks1->Fill(pvOne.trackWeight(*otrk));
      if (pvOne.trackWeight(*otrk) < theminW1) {
        theminW1 = pvOne.trackWeight(*otrk);
      }
    }
    for (auto otrk = pvTwo.originalTracks().cbegin(); otrk != pvTwo.originalTracks().end(); ++otrk) {
      h_wTrks2->Fill(pvTwo.trackWeight(*otrk));
      if (pvTwo.trackWeight(*otrk) < theminW2) {
        theminW2 = pvTwo.trackWeight(*otrk);
      }
    }

    h_sumPt->Fill(sumPt);

    int half_trks = twoPV.nTracks();

    const double invSqrt2 = 1. / std::sqrt(2.);

    double deltaX = (twoPV.x() - onePV.x());
    double deltaY = (twoPV.y() - onePV.y());
    double deltaZ = (twoPV.z() - onePV.z());

    double resX = deltaX * invSqrt2;
    double resY = deltaY * invSqrt2;
    double resZ = deltaZ * invSqrt2;

    h_diffX->Fill(resX * cmToUm);
    h_diffY->Fill(resY * cmToUm);
    h_diffZ->Fill(resZ * cmToUm);

    double errX = sqrt(pow(twoPV.xError(), 2) + pow(onePV.xError(), 2));
    double errY = sqrt(pow(twoPV.yError(), 2) + pow(onePV.yError(), 2));
    double errZ = sqrt(pow(twoPV.zError(), 2) + pow(onePV.zError(), 2));

    h_errX->Fill(errX * cmToUm);
    h_errY->Fill(errY * cmToUm);
    h_errZ->Fill(errZ * cmToUm);

    h_pullX->Fill(deltaX / errX);
    h_pullY->Fill(deltaY / errY);
    h_pullZ->Fill(deltaZ / errZ);

    // filling the pT-binned distributions

    for (int ipTBin = 0; ipTBin < nPtBins_; ipTBin++) {
      float pTF = mypT_bins_[ipTBin];
      float pTL = mypT_bins_[ipTBin + 1];

      if (avgSumPt >= pTF && avgSumPt < pTL) {
        PVValHelper::fillByIndex(h_resolX_sumPt_, ipTBin, resX * cmToUm, "1");
        PVValHelper::fillByIndex(h_resolY_sumPt_, ipTBin, resY * cmToUm, "2");
        PVValHelper::fillByIndex(h_resolZ_sumPt_, ipTBin, resZ * cmToUm, "3");

        PVValHelper::fillByIndex(h_pullX_sumPt_, ipTBin, deltaX / errX, "4");
        PVValHelper::fillByIndex(h_pullY_sumPt_, ipTBin, deltaY / errY, "5");
        PVValHelper::fillByIndex(h_pullZ_sumPt_, ipTBin, deltaZ / errZ, "6");
      }
    }

    // filling the track multeplicity binned distributions

    for (int inTrackBin = 0; inTrackBin < nTrackBins_; inTrackBin++) {
      float nTrackF = myNTrack_bins_[inTrackBin];
      float nTrackL = myNTrack_bins_[inTrackBin + 1];

      if (ntrks >= nTrackF && ntrks < nTrackL) {
        PVValHelper::fillByIndex(h_resolX_Ntracks_, inTrackBin, resX * cmToUm, "7");
        PVValHelper::fillByIndex(h_resolY_Ntracks_, inTrackBin, resY * cmToUm, "8");
        PVValHelper::fillByIndex(h_resolZ_Ntracks_, inTrackBin, resZ * cmToUm, "9");

        PVValHelper::fillByIndex(h_pullX_Ntracks_, inTrackBin, deltaX / errX, "10");
        PVValHelper::fillByIndex(h_pullY_Ntracks_, inTrackBin, deltaY / errY, "11");
        PVValHelper::fillByIndex(h_pullZ_Ntracks_, inTrackBin, deltaZ / errZ, "12");
      }
    }

    // filling the vertex multeplicity binned distributions

    for (int inVtxBin = 0; inVtxBin < nVtxBins_; inVtxBin++) {
      /*
	float nVtxF = myNVtx_bins_[inVtxBin];
	float nVtxL = myNVtx_bins_[inVtxBin+1];
	if(nOfflineVtx >= nVtxF && nOfflineVtx < nVtxL){
      */

      if (nOfflineVtx == inVtxBin) {
        PVValHelper::fillByIndex(h_resolX_Nvtx_, inVtxBin, deltaX * cmToUm, "7");
        PVValHelper::fillByIndex(h_resolY_Nvtx_, inVtxBin, deltaY * cmToUm, "8");
        PVValHelper::fillByIndex(h_resolZ_Nvtx_, inVtxBin, deltaZ * cmToUm, "9");

        PVValHelper::fillByIndex(h_pullX_Nvtx_, inVtxBin, deltaX / errX, "10");
        PVValHelper::fillByIndex(h_pullY_Nvtx_, inVtxBin, deltaY / errY, "11");
        PVValHelper::fillByIndex(h_pullZ_Nvtx_, inVtxBin, deltaZ / errZ, "12");
      }
    }

    h_sumPt1->Fill(sumPt1);
    h_sumPt2->Fill(sumPt2);

    h_minWTrks1->Fill(theminW1);
    h_minWTrks2->Fill(theminW2);

    h_PVCL_subVtx1->Fill(TMath::Prob(pvOne.totalChiSquared(), (int)(pvOne.degreesOfFreedom())));
    h_PVCL_subVtx2->Fill(TMath::Prob(pvTwo.totalChiSquared(), (int)(pvTwo.degreesOfFreedom())));

    // fill ntuples
    pvCand thePV;
    thePV.ipos = counter;
    thePV.nTrks = ntrks;

    thePV.x_origVtx = iPV.x();
    thePV.y_origVtx = iPV.y();
    thePV.z_origVtx = iPV.z();

    thePV.xErr_origVtx = iPV.xError();
    thePV.yErr_origVtx = iPV.yError();
    thePV.zErr_origVtx = iPV.zError();

    thePV.n_subVtx1 = half_trks;
    thePV.x_subVtx1 = onePV.x();
    thePV.y_subVtx1 = onePV.y();
    thePV.z_subVtx1 = onePV.z();

    thePV.xErr_subVtx1 = onePV.xError();
    thePV.yErr_subVtx1 = onePV.yError();
    thePV.zErr_subVtx1 = onePV.zError();
    thePV.sumPt_subVtx1 = sumPt1;

    thePV.n_subVtx2 = half_trks;
    thePV.x_subVtx2 = twoPV.x();
    thePV.y_subVtx2 = twoPV.y();
    thePV.z_subVtx2 = twoPV.z();

    thePV.xErr_subVtx2 = twoPV.xError();
    thePV.yErr_subVtx2 = twoPV.yError();
    thePV.zErr_subVtx2 = twoPV.zError();
    thePV.sumPt_subVtx2 = sumPt2;

    thePV.CL_subVtx1 = TMath::Prob(pvOne.totalChiSquared(), (int)(pvOne.degreesOfFreedom()));
    thePV.CL_subVtx2 = TMath::Prob(pvTwo.totalChiSquared(), (int)(pvTwo.degreesOfFreedom()));

    thePV.minW_subVtx1 = theminW1;
    thePV.minW_subVtx2 = theminW2;

    event_.pvs.push_back(thePV);

  }  // loop on the vertices

  // fill the histogram of vertices per event
  h_nVertices->Fill(counter);
  h_nNonFakeVertices->Fill(noFakecounter);
  h_nFinalVertices->Fill(goodcounter);

  if (storeNtuple_) {
    tree_->Fill();
  }
}

void SplitVertexResolution::beginEvent() {
  event_.pvs.clear();
  event_.nVtx = -1;
}

void SplitVertexResolution::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  unsigned int RunNumber_ = run.run();

  if (!runNumbersTimesLog_.count(RunNumber_)) {
    auto times = getRunTime(iSetup);

    if (debug_) {
      const time_t start_time = times.first / 1.0e+6;
      edm::LogInfo("SplitVertexResolution")
          << RunNumber_ << " has start time: " << times.first << " - " << times.second << std::endl;
      edm::LogInfo("SplitVertexResolution")
          << "human readable time: " << std::asctime(std::gmtime(&start_time)) << std::endl;
    }
    runNumbersTimesLog_[RunNumber_] = times;
  }
}

// ------------ method called once each job just before starting event loop  ------------
void SplitVertexResolution::beginJob() {
  ievt = 0;
  itrks = 0;

  if (compressionSettings_ > 0) {
    outfile_->file().SetCompressionSettings(compressionSettings_);
  }

  // luminosity histo
  TFileDirectory EventFeatures = outfile_->mkdir("EventFeatures");
  h_lumiFromConfig =
      EventFeatures.make<TH1F>("h_lumiFromConfig", "luminosity from config;;luminosity of present run", 1, -0.5, 0.5);
  h_lumiFromConfig->SetBinContent(1, intLumi_);

  h_runFromConfig = EventFeatures.make<TH1I>("h_runFromConfig",
                                             "run number from config;;run number (from configuration)",
                                             runControlNumbers_.size(),
                                             0.,
                                             runControlNumbers_.size());

  for (const auto& run : runControlNumbers_ | boost::adaptors::indexed(1)) {
    h_runFromConfig->SetBinContent(run.index(), run.value());
  }

  // resolutions

  if (!checkBinOrdering(mypT_bins_)) {
    edm::LogError("SplitVertexResolution") << " Warning - the vector of pT bins is not ordered " << std::endl;
  }

  if (!checkBinOrdering(myNTrack_bins_)) {
    edm::LogError("SplitVertexResolution") << " Warning -the vector of n. tracks bins is not ordered " << std::endl;
  }

  if (!checkBinOrdering(myNVtx_bins_)) {
    edm::LogError("SplitVertexResolution") << " Warning -the vector of n. vertices bins is not ordered " << std::endl;
  }

  TFileDirectory xResolSumPt = outfile_->mkdir("xResolSumPt");
  h_resolX_sumPt_ = bookResidualsHistogram(xResolSumPt, nPtBins_, "resolX", "sumPt");

  TFileDirectory yResolSumPt = outfile_->mkdir("yResolSumPt");
  h_resolY_sumPt_ = bookResidualsHistogram(yResolSumPt, nPtBins_, "resolY", "sumPt");

  TFileDirectory zResolSumPt = outfile_->mkdir("zResolSumPt");
  h_resolZ_sumPt_ = bookResidualsHistogram(zResolSumPt, nPtBins_, "resolZ", "sumPt");

  TFileDirectory xResolNtracks_ = outfile_->mkdir("xResolNtracks");
  h_resolX_Ntracks_ = bookResidualsHistogram(xResolNtracks_, nTrackBins_, "resolX", "Ntracks");

  TFileDirectory yResolNtracks_ = outfile_->mkdir("yResolNtracks");
  h_resolY_Ntracks_ = bookResidualsHistogram(yResolNtracks_, nTrackBins_, "resolY", "Ntracks");

  TFileDirectory zResolNtracks_ = outfile_->mkdir("zResolNtracks");
  h_resolZ_Ntracks_ = bookResidualsHistogram(zResolNtracks_, nTrackBins_, "resolZ", "Ntracks");

  TFileDirectory xResolNvtx_ = outfile_->mkdir("xResolNvtx");
  h_resolX_Nvtx_ = bookResidualsHistogram(xResolNvtx_, nVtxBins_, "resolX", "Nvtx");

  TFileDirectory yResolNvtx_ = outfile_->mkdir("yResolNvtx");
  h_resolY_Nvtx_ = bookResidualsHistogram(yResolNvtx_, nVtxBins_, "resolY", "Nvtx");

  TFileDirectory zResolNvtx_ = outfile_->mkdir("zResolNvtx");
  h_resolZ_Nvtx_ = bookResidualsHistogram(zResolNvtx_, nVtxBins_, "resolZ", "Nvtx");

  // pulls

  TFileDirectory xPullSumPt = outfile_->mkdir("xPullSumPt");
  h_pullX_sumPt_ = bookResidualsHistogram(xPullSumPt, nPtBins_, "pullX", "sumPt");

  TFileDirectory yPullSumPt = outfile_->mkdir("yPullSumPt");
  h_pullY_sumPt_ = bookResidualsHistogram(yPullSumPt, nPtBins_, "pullY", "sumPt");

  TFileDirectory zPullSumPt = outfile_->mkdir("zPullSumPt");
  h_pullZ_sumPt_ = bookResidualsHistogram(zPullSumPt, nPtBins_, "pullZ", "sumPt");

  TFileDirectory xPullNtracks_ = outfile_->mkdir("xPullNtracks");
  h_pullX_Ntracks_ = bookResidualsHistogram(xPullNtracks_, nTrackBins_, "pullX", "Ntracks");

  TFileDirectory yPullNtracks_ = outfile_->mkdir("yPullNtracks");
  h_pullY_Ntracks_ = bookResidualsHistogram(yPullNtracks_, nTrackBins_, "pullY", "Ntracks");

  TFileDirectory zPullNtracks_ = outfile_->mkdir("zPullNtracks");
  h_pullZ_Ntracks_ = bookResidualsHistogram(zPullNtracks_, nTrackBins_, "pullZ", "Ntracks");

  TFileDirectory xPullNvtx_ = outfile_->mkdir("xPullNvtx");
  h_pullX_Nvtx_ = bookResidualsHistogram(xPullNvtx_, nVtxBins_, "pullX", "Nvtx");

  TFileDirectory yPullNvtx_ = outfile_->mkdir("yPullNvtx");
  h_pullY_Nvtx_ = bookResidualsHistogram(yPullNvtx_, nVtxBins_, "pullY", "Nvtx");

  TFileDirectory zPullNvtx_ = outfile_->mkdir("zPullNvtx");
  h_pullZ_Nvtx_ = bookResidualsHistogram(zPullNvtx_, nVtxBins_, "pullZ", "Nvtx");

  // control plots
  h_runNumber = outfile_->make<TH1F>("h_runNumber", "run number;run number;n_{events}", 100000, 250000., 350000.);

  h_nOfflineVertices = outfile_->make<TH1I>("h_nOfflineVertices", "n. of vertices;n. vertices; events", 100, 0, 100);
  h_nVertices = outfile_->make<TH1I>("h_nVertices", "n. of vertices;n. vertices; events", 100, 0, 100);
  h_nNonFakeVertices =
      outfile_->make<TH1I>("h_nRealVertices", "n. of non-fake vertices;n. vertices; events", 100, 0, 100);
  h_nFinalVertices =
      outfile_->make<TH1I>("h_nSelectedVertices", "n. of selected vertices vertices;n. vertices; events", 100, 0, 100);

  h_diffX = outfile_->make<TH1F>(
      "h_diffX", "x-coordinate vertex resolution;vertex resolution (x) [#mum];vertices", 100, -300, 300.);
  h_diffY = outfile_->make<TH1F>(
      "h_diffY", "y-coordinate vertex resolution;vertex resolution (y) [#mum];vertices", 100, -300, 300.);
  h_diffZ = outfile_->make<TH1F>(
      "h_diffZ", "z-coordinate vertex resolution;vertex resolution (z) [#mum];vertices", 100, -500, 500.);

  h_OrigVertexErrX = outfile_->make<TH1F>(
      "h_OrigVertexErrX", "x-coordinate vertex error;vertex error (x) [#mum];vertices", 300, 0., 300.);
  h_OrigVertexErrY = outfile_->make<TH1F>(
      "h_OrigVertexErrY", "y-coordinate vertex error;vertex error (y) [#mum];vertices", 300, 0., 300.);
  h_OrigVertexErrZ = outfile_->make<TH1F>(
      "h_OrigVertexErrZ", "z-coordinate vertex error;vertex error (z) [#mum];vertices", 500, 0., 500.);

  h_errX = outfile_->make<TH1F>(
      "h_errX", "x-coordinate vertex resolution error;vertex resoltuion error (x) [#mum];vertices", 300, 0., 300.);
  h_errY = outfile_->make<TH1F>(
      "h_errY", "y-coordinate vertex resolution error;vertex resolution error (y) [#mum];vertices", 300, 0., 300.);
  h_errZ = outfile_->make<TH1F>(
      "h_errZ", "z-coordinate vertex resolution error;vertex resolution error (z) [#mum];vertices", 500, 0., 500.);

  h_pullX = outfile_->make<TH1F>("h_pullX", "x-coordinate vertex pull;vertex pull (x);vertices", 500, -10, 10.);
  h_pullY = outfile_->make<TH1F>("h_pullY", "y-coordinate vertex pull;vertex pull (y);vertices", 500, -10, 10.);
  h_pullZ = outfile_->make<TH1F>("h_pullZ", "z-coordinate vertex pull;vertex pull (z);vertices", 500, -10, 10.);

  h_ntrks = outfile_->make<TH1F>("h_ntrks",
                                 "number of tracks in vertex;vertex multeplicity;vertices",
                                 myNTrack_bins_.size() - 1,
                                 myNTrack_bins_.data());

  h_sumPt = outfile_->make<TH1F>(
      "h_sumPt", "#Sigma p_{T};#sum p_{T} [GeV];vertices", mypT_bins_.size() - 1, mypT_bins_.data());

  h_avgSumPt = outfile_->make<TH1F>(
      "h_avgSumPt", "#LT #Sigma p_{T} #GT;#LT #sum p_{T} #GT [GeV];vertices", mypT_bins_.size() - 1, mypT_bins_.data());

  h_sumPt1 = outfile_->make<TH1F>("h_sumPt1",
                                  "#Sigma p_{T} sub-vertex 1;#sum p_{T} sub-vertex 1 [GeV];subvertices",
                                  mypT_bins_.size() - 1,
                                  mypT_bins_.data());
  h_sumPt2 = outfile_->make<TH1F>("h_sumPt2",
                                  "#Sigma p_{T} sub-vertex 2;#sum p_{T} sub-vertex 2 [GeV];subvertices",
                                  mypT_bins_.size() - 1,
                                  mypT_bins_.data());

  h_wTrks1 = outfile_->make<TH1F>("h_wTrks1", "weight of track for sub-vertex 1;track weight;subvertices", 500, 0., 1.);
  h_wTrks2 = outfile_->make<TH1F>("h_wTrks2", "weithg of track for sub-vertex 2;track weight;subvertices", 500, 0., 1.);

  h_minWTrks1 = outfile_->make<TH1F>(
      "h_minWTrks1", "minimum weight of track for sub-vertex 1;minimum track weight;subvertices", 500, 0., 1.);
  h_minWTrks2 = outfile_->make<TH1F>(
      "h_minWTrks2", "minimum weithg of track for sub-vertex 2;minimum track weight;subvertices", 500, 0., 1.);

  h_PVCL_subVtx1 =
      outfile_->make<TH1F>("h_PVCL_subVtx1",
                           "#chi^{2} probability for sub-vertex 1;Prob(#chi^{2},ndof) for sub-vertex 1;subvertices",
                           100,
                           0.,
                           1);
  h_PVCL_subVtx2 =
      outfile_->make<TH1F>("h_PVCL_subVtx2",
                           "#chi^{2} probability for sub-vertex 2;Prob(#chi^{2},ndof) for sub-vertex 2;subvertices",
                           100,
                           0.,
                           1);

  // resolutions

  p_resolX_vsSumPt = outfile_->make<TH1F>("p_resolX_vsSumPt",
                                          "x-resolution vs #Sigma p_{T};#sum p_{T}; x vertex resolution [#mum]",
                                          mypT_bins_.size() - 1,
                                          mypT_bins_.data());
  p_resolY_vsSumPt = outfile_->make<TH1F>("p_resolY_vsSumPt",
                                          "y-resolution vs #Sigma p_{T};#sum p_{T}; y vertex resolution [#mum]",
                                          mypT_bins_.size() - 1,
                                          mypT_bins_.data());
  p_resolZ_vsSumPt = outfile_->make<TH1F>("p_resolZ_vsSumPt",
                                          "z-resolution vs #Sigma p_{T};#sum p_{T}; z vertex resolution [#mum]",
                                          mypT_bins_.size() - 1,
                                          mypT_bins_.data());

  p_resolX_vsNtracks = outfile_->make<TH1F>("p_resolX_vsNtracks",
                                            "x-resolution vs n_{tracks};n_{tracks}; x vertex resolution [#mum]",
                                            myNTrack_bins_.size() - 1,
                                            myNTrack_bins_.data());
  p_resolY_vsNtracks = outfile_->make<TH1F>("p_resolY_vsNtracks",
                                            "y-resolution vs n_{tracks};n_{tracks}; y vertex resolution [#mum]",
                                            myNTrack_bins_.size() - 1,
                                            myNTrack_bins_.data());
  p_resolZ_vsNtracks = outfile_->make<TH1F>("p_resolZ_vsNtracks",
                                            "z-resolution vs n_{tracks};n_{tracks}; z vertex resolution [#mum]",
                                            myNTrack_bins_.size() - 1,
                                            myNTrack_bins_.data());

  p_resolX_vsNvtx = outfile_->make<TH1F>("p_resolX_vsNvtx",
                                         "x-resolution vs n_{vertices};n_{vertices}; x vertex resolution [#mum]",
                                         myNVtx_bins_.size() - 1,
                                         myNVtx_bins_.data());
  p_resolY_vsNvtx = outfile_->make<TH1F>("p_resolY_vsNvtx",
                                         "y-resolution vs n_{vertices};n_{vertices}; y vertex resolution [#mum]",
                                         myNVtx_bins_.size() - 1,
                                         myNVtx_bins_.data());
  p_resolZ_vsNvtx = outfile_->make<TH1F>("p_resolZ_vsNvtx",
                                         "z-resolution vs n_{vertices};n_{vertices}; z vertex resolution [#mum]",
                                         myNVtx_bins_.size() - 1,
                                         myNVtx_bins_.data());

  // pulls

  p_pullX_vsSumPt = outfile_->make<TH1F>(
      "p_pullX_vsSumPt", "x-pull vs #Sigma p_{T};#sum p_{T}; x vertex pull", mypT_bins_.size() - 1, mypT_bins_.data());
  p_pullY_vsSumPt = outfile_->make<TH1F>(
      "p_pullY_vsSumPt", "y-pull vs #Sigma p_{T};#sum p_{T}; y vertex pull", mypT_bins_.size() - 1, mypT_bins_.data());
  p_pullZ_vsSumPt = outfile_->make<TH1F>(
      "p_pullZ_vsSumPt", "z-pull vs #Sigma p_{T};#sum p_{T}; z vertex pull", mypT_bins_.size() - 1, mypT_bins_.data());

  p_pullX_vsNtracks = outfile_->make<TH1F>("p_pullX_vsNtracks",
                                           "x-pull vs n_{tracks};n_{tracks}; x vertex pull",
                                           myNTrack_bins_.size() - 1,
                                           myNTrack_bins_.data());
  p_pullY_vsNtracks = outfile_->make<TH1F>("p_pullY_vsNtracks",
                                           "y-pull vs n_{tracks};n_{tracks}; y vertex pull",
                                           myNTrack_bins_.size() - 1,
                                           myNTrack_bins_.data());
  p_pullZ_vsNtracks = outfile_->make<TH1F>("p_pullZ_vsNtracks",
                                           "z-pull vs n_{tracks};n_{tracks}; z vertex pull",
                                           myNTrack_bins_.size() - 1,
                                           myNTrack_bins_.data());

  p_pullX_vsNvtx = outfile_->make<TH1F>("p_pullX_vsNvtx",
                                        "x-pull vs n_{vertices};n_{vertices}; x vertex pull",
                                        myNVtx_bins_.size() - 1,
                                        myNVtx_bins_.data());
  p_pullY_vsNvtx = outfile_->make<TH1F>("p_pullY_vsNvtx",
                                        "y-pull vs n_{vertices};n_{vertices}; y vertex pull",
                                        myNVtx_bins_.size() - 1,
                                        myNVtx_bins_.data());
  p_pullZ_vsNvtx = outfile_->make<TH1F>("p_pullZ_vsNvtx",
                                        "z-pull vs n_{vertices};n_{vertices}; z vertex pull",
                                        myNVtx_bins_.size() - 1,
                                        myNVtx_bins_.data());

  tree_ = outfile_->make<TTree>("pvTree", "pvTree");
  tree_->Branch("event", &event_, 64000, 2);
}

//*************************************************************
// Generic booker function
//*************************************************************
std::vector<TH1F*> SplitVertexResolution::bookResidualsHistogram(TFileDirectory dir,
                                                                 unsigned int theNOfBins,
                                                                 TString resType,
                                                                 TString varType) {
  TH1F::SetDefaultSumw2(kTRUE);

  double up = 500.;
  double down = -500.;

  if (resType.Contains("pull")) {
    up *= 0.01;
    down *= 0.01;
  }

  std::vector<TH1F*> h;
  h.reserve(theNOfBins);

  const char* auxResType = (resType.ReplaceAll("_", "")).Data();

  for (unsigned int i = 0; i < theNOfBins; i++) {
    TH1F* htemp = dir.make<TH1F>(Form("histo_%s_%s_plot%i", resType.Data(), varType.Data(), i),
                                 Form("%s vs %s - bin %i;%s;vertices", auxResType, varType.Data(), i, auxResType),
                                 250,
                                 down,
                                 up);
    h.push_back(htemp);
  }

  return h;
}

// ------------ method called once each job just after ending the event loop  ------------
void SplitVertexResolution::endJob() {
  edm::LogVerbatim("SplitVertexResolution") << "*******************************" << std::endl;
  edm::LogVerbatim("SplitVertexResolution") << "Events run in total: " << ievt << std::endl;
  edm::LogVerbatim("SplitVertexResolution") << "n. tracks: " << itrks << std::endl;
  edm::LogVerbatim("SplitVertexResolution") << "*******************************" << std::endl;

  int nFiringTriggers = triggerMap_.size();
  edm::LogVerbatim("SplitVertexResolution") << "firing triggers: " << nFiringTriggers << std::endl;
  edm::LogVerbatim("SplitVertexResolution") << "*******************************" << std::endl;

  tksByTrigger_ = outfile_->make<TH1D>(
      "tksByTrigger", "tracks by HLT path;;% of # traks", nFiringTriggers, -0.5, nFiringTriggers - 0.5);
  evtsByTrigger_ = outfile_->make<TH1D>(
      "evtsByTrigger", "events by HLT path;;% of # events", nFiringTriggers, -0.5, nFiringTriggers - 0.5);

  int i = 0;
  for (std::map<std::string, std::pair<int, int> >::iterator it = triggerMap_.begin(); it != triggerMap_.end(); ++it) {
    i++;

    double trkpercent = ((it->second).second) * 100. / double(itrks);
    double evtpercent = ((it->second).first) * 100. / double(ievt);

    edm::LogVerbatim("SplitVertexResolution")
        << "HLT path: " << std::setw(60) << std::left << it->first << " | events firing: " << std::right << std::setw(8)
        << (it->second).first << " (" << std::setw(8) << std::fixed << std::setprecision(4) << evtpercent << "%)"
        << " | tracks collected: " << std::setw(8) << (it->second).second << " (" << std::setw(8) << std::fixed
        << std::setprecision(4) << trkpercent << "%)";

    tksByTrigger_->SetBinContent(i, trkpercent);
    tksByTrigger_->GetXaxis()->SetBinLabel(i, (it->first).c_str());

    evtsByTrigger_->SetBinContent(i, evtpercent);
    evtsByTrigger_->GetXaxis()->SetBinLabel(i, (it->first).c_str());
  }

  TFileDirectory RunFeatures = outfile_->mkdir("RunFeatures");
  h_runStartTimes = RunFeatures.make<TH1I>(
      "runStartTimes", "run start times", runNumbersTimesLog_.size(), 0, runNumbersTimesLog_.size());
  h_runEndTimes =
      RunFeatures.make<TH1I>("runEndTimes", "run end times", runNumbersTimesLog_.size(), 0, runNumbersTimesLog_.size());

  unsigned int count = 1;
  for (const auto& run : runNumbersTimesLog_) {
    // strip down the microseconds
    h_runStartTimes->SetBinContent(count, run.second.first / 10e6);
    h_runStartTimes->GetXaxis()->SetBinLabel(count, (std::to_string(run.first)).c_str());

    h_runEndTimes->SetBinContent(count, run.second.second / 10e6);
    h_runEndTimes->GetXaxis()->SetBinLabel(count, (std::to_string(run.first)).c_str());

    count++;
  }

  // resolutions

  fillTrendPlotByIndex(p_resolX_vsSumPt, h_resolX_sumPt_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_resolY_vsSumPt, h_resolY_sumPt_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_resolZ_vsSumPt, h_resolZ_sumPt_, PVValHelper::WIDTH);

  fillTrendPlotByIndex(p_resolX_vsNtracks, h_resolX_Ntracks_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_resolY_vsNtracks, h_resolY_Ntracks_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_resolZ_vsNtracks, h_resolZ_Ntracks_, PVValHelper::WIDTH);

  fillTrendPlotByIndex(p_resolX_vsNvtx, h_resolX_Nvtx_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_resolY_vsNvtx, h_resolY_Nvtx_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_resolZ_vsNvtx, h_resolZ_Nvtx_, PVValHelper::WIDTH);

  // pulls

  fillTrendPlotByIndex(p_pullX_vsSumPt, h_pullX_sumPt_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_pullY_vsSumPt, h_pullY_sumPt_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_pullZ_vsSumPt, h_pullZ_sumPt_, PVValHelper::WIDTH);

  fillTrendPlotByIndex(p_pullX_vsNtracks, h_pullX_Ntracks_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_pullY_vsNtracks, h_pullY_Ntracks_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_pullZ_vsNtracks, h_pullZ_Ntracks_, PVValHelper::WIDTH);

  fillTrendPlotByIndex(p_pullX_vsNvtx, h_pullX_Nvtx_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_pullY_vsNvtx, h_pullY_Nvtx_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_pullZ_vsNvtx, h_pullZ_Nvtx_, PVValHelper::WIDTH);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SplitVertexResolution::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//*************************************************************
std::pair<long long, long long> SplitVertexResolution::getRunTime(const edm::EventSetup& iSetup) const
//*************************************************************
{
  const auto& runInfo = iSetup.getData(runInfoToken_);
  if (debug_) {
    edm::LogInfo("SplitVertexResolution") << runInfo.m_start_time_str << " " << runInfo.m_stop_time_str << std::endl;
  }
  return std::make_pair(runInfo.m_start_time_ll, runInfo.m_stop_time_ll);
}

//*************************************************************
void SplitVertexResolution::fillTrendPlotByIndex(TH1F* trendPlot, std::vector<TH1F*>& h, PVValHelper::estimator fitPar_)
//*************************************************************
{
  for (auto iterator = h.begin(); iterator != h.end(); iterator++) {
    unsigned int bin = std::distance(h.begin(), iterator) + 1;
    statmode::fitParams myFit = fitResiduals((*iterator));

    switch (fitPar_) {
      case PVValHelper::MEAN: {
        float mean_ = myFit.first.value();
        float meanErr_ = myFit.first.error();
        trendPlot->SetBinContent(bin, mean_);
        trendPlot->SetBinError(bin, meanErr_);
        break;
      }
      case PVValHelper::WIDTH: {
        float width_ = myFit.second.value();
        float widthErr_ = myFit.second.error();
        trendPlot->SetBinContent(bin, width_);
        trendPlot->SetBinError(bin, widthErr_);
        break;
      }
      case PVValHelper::MEDIAN: {
        float median_ = PVValHelper::getMedian((*iterator)).value();
        float medianErr_ = PVValHelper::getMedian((*iterator)).error();
        trendPlot->SetBinContent(bin, median_);
        trendPlot->SetBinError(bin, medianErr_);
        break;
      }
      case PVValHelper::MAD: {
        float mad_ = PVValHelper::getMAD((*iterator)).value();
        float madErr_ = PVValHelper::getMAD((*iterator)).error();
        trendPlot->SetBinContent(bin, mad_);
        trendPlot->SetBinError(bin, madErr_);
        break;
      }
      default:
        edm::LogWarning("SplitVertexResolution")
            << "fillTrendPlotByIndex() " << fitPar_ << " unknown estimator!" << std::endl;
        break;
    }
  }
}

//*************************************************************
statmode::fitParams SplitVertexResolution::fitResiduals(TH1* hist, bool singleTime)
//*************************************************************
{
  if (hist->GetEntries() < 10) {
    LogDebug("SplitVertexResolution") << "hist name: " << hist->GetName() << " has less than 10 entries" << std::endl;
    return std::make_pair(Measurement1D(0., 0.), Measurement1D(0., 0.));
  }

  float maxHist = hist->GetXaxis()->GetXmax();
  float minHist = hist->GetXaxis()->GetXmin();
  float mean = hist->GetMean();
  float sigma = hist->GetRMS();

  if (edm::isNotFinite(mean) || edm::isNotFinite(sigma)) {
    mean = 0;
    //sigma= - hist->GetXaxis()->GetBinLowEdge(1) + hist->GetXaxis()->GetBinLowEdge(hist->GetNbinsX()+1);
    sigma = -minHist + maxHist;
    edm::LogWarning("SplitVertexResolution")
        << "FitPVResiduals::fitResiduals(): histogram" << hist->GetName() << " mean or sigma are NaN!!" << std::endl;
  }

  TF1 func("tmp", "gaus", mean - 2. * sigma, mean + 2. * sigma);
  if (0 == hist->Fit(&func, "QNR")) {  // N: do not blow up file by storing fit!
    mean = func.GetParameter(1);
    sigma = func.GetParameter(2);

    if (!singleTime) {
      // second fit: three sigma of first fit around mean of first fit
      func.SetRange(std::max(mean - 3 * sigma, minHist), std::min(mean + 3 * sigma, maxHist));
      // I: integral gives more correct results if binning is too wide
      // L: Likelihood can treat empty bins correctly (if hist not weighted...)
      if (0 == hist->Fit(&func, "Q0LR")) {
        if (hist->GetFunction(func.GetName())) {  // Take care that it is later on drawn:
          hist->GetFunction(func.GetName())->ResetBit(TF1::kNotDraw);
        }
      }
    }
  }

  float res_mean = func.GetParameter(1);
  float res_width = func.GetParameter(2);

  float res_mean_err = func.GetParError(1);
  float res_width_err = func.GetParError(2);

  Measurement1D resultM(res_mean, res_mean_err);
  Measurement1D resultW(res_width, res_width_err);

  statmode::fitParams result = std::make_pair(resultM, resultW);
  return result;
}

//*************************************************************
statmode::fitParams SplitVertexResolution::fitResiduals_v0(TH1* hist)
//*************************************************************
{
  float mean = hist->GetMean();
  float sigma = hist->GetRMS();

  TF1 func("tmp", "gaus", mean - 1.5 * sigma, mean + 1.5 * sigma);
  if (0 == hist->Fit(&func, "QNR")) {  // N: do not blow up file by storing fit!
    mean = func.GetParameter(1);
    sigma = func.GetParameter(2);
    // second fit: three sigma of first fit around mean of first fit
    func.SetRange(mean - 2 * sigma, mean + 2 * sigma);
    // I: integral gives more correct results if binning is too wide
    // L: Likelihood can treat empty bins correctly (if hist not weighted...)
    if (0 == hist->Fit(&func, "Q0LR")) {
      if (hist->GetFunction(func.GetName())) {  // Take care that it is later on drawn:
        hist->GetFunction(func.GetName())->ResetBit(TF1::kNotDraw);
      }
    }
  }

  float res_mean = func.GetParameter(1);
  float res_width = func.GetParameter(2);

  float res_mean_err = func.GetParError(1);
  float res_width_err = func.GetParError(2);

  Measurement1D resultM(res_mean, res_mean_err);
  Measurement1D resultW(res_width, res_width_err);

  statmode::fitParams result = std::make_pair(resultM, resultW);
  return result;
}

//*************************************************************
template <std::size_t SIZE>
bool SplitVertexResolution::checkBinOrdering(std::array<float, SIZE>& bins)
//*************************************************************
{
  int i = 1;

  if (std::is_sorted(bins.begin(), bins.end())) {
    return true;
  } else {
    for (const auto& bin : bins) {
      edm::LogInfo("SplitVertexResolution") << "bin: " << i << " : " << bin << std::endl;
      i++;
    }
    edm::LogInfo("SplitVertexResolution") << "--------------------------------" << std::endl;
    return false;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SplitVertexResolution);
