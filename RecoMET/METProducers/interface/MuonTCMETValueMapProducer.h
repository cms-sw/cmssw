// -*- C++ -*-
//
// Package:    METProducers
// Class:      MuonTCMETValueMapProducer
//

/**\class MuonTCMETValueMapProducer

*/
//
// Original Author:  Frank Golf
//         Created:  Sun Mar 15 11:33:20 CDT 2009
//
//

//____________________________________________________________________________||
#ifndef RecoMET_MuonTCMETValueMapProducer_h
#define RecoMET_MuonTCMETValueMapProducer_h

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TH2.h"
#include "TVector3.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

//____________________________________________________________________________||
class TCMETAlgo;

namespace cms
{

class MuonTCMETValueMapProducer : public edm::stream::EDProducer<>
{

public:
  explicit MuonTCMETValueMapProducer(const edm::ParameterSet&);
  ~MuonTCMETValueMapProducer();


private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
      
  edm::Handle<reco::MuonCollection>    muons_;
  edm::Handle<reco::BeamSpot>          beamSpot_;
  edm::Handle<reco::VertexCollection>  vertexHandle_;

  edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  const class MagneticField* bField;

  const reco::VertexCollection *vertices_;

  class TH2D* response_function;

  bool muonGlobal_;
  bool muonTracker_;
  bool useCaloMuons_;
  bool hasValidVertex;

  int     rfType_;
  int     nLayers_;
  int     nLayersTight_;
  int     vertexNdof_;
  double  vertexZ_;
  double  vertexRho_;
  double  vertexMaxDZ_;
  double  maxpt_eta25_;
  double  maxpt_eta20_;
  int     maxTrackAlgo_;
  double  minpt_;
  double  maxpt_;
  double  maxeta_;
  double  maxchi2_;
  double  minhits_;
  double  maxPtErr_;
  double  maxd0cut_;
  double  maxchi2_tight_;
  double  minhits_tight_;
  double  maxPtErr_tight_;
  double  d0cuta_;
  double  d0cutb_;
  bool    usePvtxd0_;
  std::vector<int> trkQuality_;
  std::vector<int> trkAlgos_;

  int     muonMinValidStaHits_;
  double  muonpt_;
  double  muoneta_;
  double  muonchi2_;
  double  muonhits_;
  double  muond0_;
  double  muonDeltaR_;
  double  muon_dptrel_;
  TCMETAlgo *tcmetAlgo_;

  bool isGoodMuon( const reco::Muon* );
  bool isGoodCaloMuon( const reco::Muon*, const unsigned int );
  bool isGoodTrack( const reco::Muon* );
  class TVector3 propagateTrack( const reco::Muon* );
  int nLayers(const reco::TrackRef);
  bool isValidVertex();
};

}

//____________________________________________________________________________||
#endif /* RecoMET_MuonTCMETValueMapProducer_h */


