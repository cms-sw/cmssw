#ifndef TRKANALYZER_H
#define TRKANALYZER_H

// system include files
#include <iostream>
#include <vector>

// CMSSW user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"

// Root include files
#include "TTree.h"

class TrackAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit TrackAnalyzer(const edm::ParameterSet&);
  ~TrackAnalyzer() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void fillVertices(const edm::Event& iEvent);
  void fillTracks(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void clearVectors();

  // ----------member data ---------------------------
  const bool doTrack_;
  const double trackPtMin_;
  const double trackEtaMax_;
  const bool applyTrackSelections_;

  const edm::EDGetTokenT<reco::VertexCollection> vertexSrc_;
  const edm::EDGetTokenT<reco::TrackCollection> trackSrc_;
  const edm::EDGetTokenT<std::vector<edm::Ptr<pat::PackedCandidate> > > track2pcSrc_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotProducer_;
  std::map<std::string, edm::EDGetTokenT<edm::ValueMap<reco::DeDxData>>> dedxEstimatorsSrc_;

  edm::Service<TFileService> fs;

  int iMaxPtSumVtx;

  // Root object
  TTree* trackTree_;

  //Branch entries
  int nRun;
  int nEv;
  int nLumi;

  int nVtx;
  std::vector<float> xVtx;
  std::vector<float> yVtx;
  std::vector<float> zVtx;
  std::vector<float> xErrVtx;
  std::vector<float> yErrVtx;
  std::vector<float> zErrVtx;
  std::vector<float> chi2Vtx;
  std::vector<float> ndofVtx;
  std::vector<bool> isFakeVtx;
  std::vector<int> nTracksVtx;
  std::vector<float> ptSumVtx;

  int nTrk;
  std::vector<float> trkPt;
  std::vector<float> trkPtError;
  std::vector<float> trkEta;
  std::vector<float> trkPhi;
  std::vector<char> trkCharge;
  std::vector<int> trkPDGId;
  std::vector<char> trkNHits;
  std::vector<char> trkNLostHits;
  std::vector<char> trkNPixHits;
  std::vector<char> trkNLayers;
  std::vector<bool> highPurity;
  std::vector<float> trkNormChi2;

  std::vector<float> pfEnergy;
  std::vector<float> pfEcal;
  std::vector<float> pfHcal;

  std::vector<int> trkAssociatedVtxIndx;
  std::vector<int> trkAssociatedVtxQuality;
  std::vector<float> trkDzAssociatedVtx;
  std::vector<float> trkDzErrAssociatedVtx;
  std::vector<float> trkDxyAssociatedVtx;
  std::vector<float> trkDxyErrAssociatedVtx;

  std::vector<int> trkFirstVtxQuality;
  std::vector<float> trkDzFirstVtx;
  std::vector<float> trkDzErrFirstVtx;
  std::vector<float> trkDxyFirstVtx;
  std::vector<float> trkDxyErrFirstVtx;
  std::map<std::string, std::vector<float>> trkDeDx;
};

inline void TrackAnalyzer::clearVectors() {
  nVtx = 0;
  xVtx.clear();
  yVtx.clear();
  zVtx.clear();
  xErrVtx.clear();
  yErrVtx.clear();
  zErrVtx.clear();
  chi2Vtx.clear();
  ndofVtx.clear();
  isFakeVtx.clear();
  nTracksVtx.clear();
  ptSumVtx.clear();

  nTrk = 0;
  trkPt.clear();
  trkPtError.clear();
  trkEta.clear();
  trkPhi.clear();
  trkCharge.clear();
  trkPDGId.clear();
  trkNHits.clear();
  trkNPixHits.clear();
  trkNLayers.clear();
  trkNormChi2.clear();
  highPurity.clear();

  pfEnergy.clear();
  pfEcal.clear();
  pfHcal.clear();

  trkAssociatedVtxIndx.clear();
  trkAssociatedVtxQuality.clear();
  trkDzAssociatedVtx.clear();
  trkDzErrAssociatedVtx.clear();
  trkDxyAssociatedVtx.clear();
  trkDxyErrAssociatedVtx.clear();

  trkFirstVtxQuality.clear();
  trkDzFirstVtx.clear();
  trkDzErrFirstVtx.clear();
  trkDxyFirstVtx.clear();
  trkDxyErrFirstVtx.clear();
  for (auto& d : trkDeDx)
    d.second.clear();
}

#endif
