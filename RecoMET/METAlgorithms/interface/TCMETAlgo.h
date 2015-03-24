// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      TCMETAlgo
//
//

/** \class TCMETAlgo

   Calculates TCMET based on detector response to charged paricles
   using the tracker to correct for the non-linearity of the
   calorimeter and the displacement of charged particles by the
   B-field. Given a track pt, eta the expected energy deposited in the
   calorimeter is obtained from a lookup table, removed from the
   calorimeter, and replaced with the track at the vertex.

*/
//
// Original Author:  F. Golf
//         Created:  March 24, 2009
//
//

//____________________________________________________________________________||
#ifndef TCMETAlgo_h
#define TCMETAlgo_h

//____________________________________________________________________________||
#include <vector>
#include <string>
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/ValueMap.h" 

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/MET.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "TH2D.h"
#include "TVector3.h"

//____________________________________________________________________________||
class TCMETAlgo 
{
public:
  typedef std::vector<const reco::Candidate> InputCollection;
  TCMETAlgo();
  virtual ~TCMETAlgo();
  reco::MET CalculateTCMET(edm::Event& event, const edm::EventSetup& setup);
  TH2D* getResponseFunction_fit ( );
  TH2D* getResponseFunction_mode ( );
  TH2D* getResponseFunction_shower ( );
  TH2D* getResponseFunction_noshower ( );
  void configure(const edm::ParameterSet &iConfig, edm::ConsumesCollector && iConsumesCollector);

private:
  double met_x_;
  double met_y_;
  double sumEt_;

  void initialize_MET_with_PFClusters(edm::Event& event);
  void initialize_MET_with_CaloMET(edm::Event& event);
  void correct_MET_for_Muons();
  void correct_MET_for_Tracks();

  edm::Handle<reco::MuonCollection> muonHandle_;
  edm::Handle<reco::GsfElectronCollection> electronHandle_;
  edm::Handle<reco::TrackCollection> trackHandle_;
  edm::Handle<reco::BeamSpot> beamSpotHandle_;
  edm::Handle<reco::VertexCollection> vertexHandle_;
  edm::Handle<edm::ValueMap<reco::MuonMETCorrectionData> > muonDepValueMapHandle_;
  edm::Handle<edm::ValueMap<reco::MuonMETCorrectionData> > tcmetDepValueMapHandle_;

  edm::ESHandle<MagneticField> magneticFieldHandle_;

  edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
  edm::EDGetTokenT<edm::View<reco::MET> > metToken_;
  edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  edm::EDGetTokenT<reco::PFClusterCollection> clustersECALToken_;
  edm::EDGetTokenT<reco::PFClusterCollection> clustersHCALToken_;
  edm::EDGetTokenT<reco::PFClusterCollection> clustersHFToken_;
  edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData> > muonDepValueMapToken_;
  edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData> > tcmetDepValueMapToken_;


  bool    usePFClusters_;
  int     nLayers_;
  int     nLayersTight_;
  int     vertexNdof_;
  double  vertexZ_;
  double  vertexRho_;
  double  vertexMaxDZ_;
  double  maxpt_eta25_;
  double  maxpt_eta20_;
  bool    vetoDuplicates_;
  double  dupMinPt_;
  double  dupDPhi_;
  double  dupDCotTh_;
  std::vector<int> duplicateTracks_;

  double  d0cuta_;
  double  d0cutb_;
  double  maxd0cut_;
  double  maxchi2_tight_;
  double  minhits_tight_;
  double  maxPtErr_tight_;
  int     nMinOuterHits_;
  std::vector<reco::TrackBase::TrackAlgorithm> trackAlgos_;
  double  usedeltaRRejection_;
  double  deltaRShower_;
  double  minpt_;
  double  maxpt_;
  double  maxeta_;
  double  maxchi2_;
  double  minhits_;
  double  maxPtErr_;
  double  radius_;
  double  zdist_;
  double  corner_;
  double  eVetoDeltaR_;
  double  eVetoDeltaPhi_;
  double  eVetoDeltaCotTheta_; 
  double  eVetoMinElectronPt_;
  double  hOverECut_;
  std::vector<int> trkQuality_;
  std::vector<reco::TrackBase::TrackAlgorithm> trkAlgos_;

  bool isCosmics_;
  bool correctShowerTracks_;
  bool electronVetoCone_;
  bool usePvtxd0_;
  bool checkTrackPropagation_;


  class TH2D* response_function_;
  class TH2D* showerRF_;
  bool hasValidVertex_;
  const reco::VertexCollection *vertexColl_;

  bool isMuon(const reco::TrackRef& trackRef);
  bool isElectron(const reco::TrackRef& trackRef);
  bool isGoodTrack(const reco::TrackRef trackRef);
  bool closeToElectron( const reco::TrackRef );
  void correctMETforMuon(const reco::TrackRef, reco::MuonRef& muonRef);
  void correctMETforMuon(reco::MuonRef& muonRef);
  void correctMETforTrack( const reco::TrackRef , TH2D* rf, const TVector3& );
  void correctSumEtForTrack( const reco::TrackRef , TH2D* rf, const TVector3& );
  class TVector3 propagateTrackToCalorimeterFace(const reco::TrackRef trackRef);
  void findGoodShowerTracks(std::vector<int>& goodShowerTracks);
  bool nearGoodShowerTrack( const reco::TrackRef , const std::vector<int>& goodShowerTracks );
  int nExpectedInnerHits(const reco::TrackRef);
  int nExpectedOuterHits(const reco::TrackRef);
  int nLayers(const reco::TrackRef);
  bool isValidVertex();
  void findDuplicateTracks();
  int vetoTrack( int i1 , int i2 );
};

//____________________________________________________________________________||
#endif // TCMETAlgo_h

