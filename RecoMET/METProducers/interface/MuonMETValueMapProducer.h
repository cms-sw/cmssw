// -*- C++ -*-
//
// Package:    METProducers
// Class:      MuonMETValueMapProducer
//
//

/**\class MuonMETValueMapProducer

*/
//
// Original Author:  Puneeth Kalavase
//         Created:  Sun Mar 15 11:33:20 CDT 2009
//
//

//____________________________________________________________________________||
#ifndef RecoMET_MuonMETValueMapProducer_h
#define RecoMET_MuonMETValueMapProducer_h

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"


//____________________________________________________________________________||
namespace cms
{

class MuonMETValueMapProducer : public edm::stream::EDProducer<>
{
public:
  explicit MuonMETValueMapProducer(const edm::ParameterSet&);
  ~MuonMETValueMapProducer() { }

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  void determine_deltax_deltay(double& deltax, double& deltay, const reco::Muon& muon, double bfield, edm::Event& iEvent, const edm::EventSetup& iSetup);
  reco::MuonMETCorrectionData::Type decide_correction_type(const reco::Muon& muon, const math::XYZPoint &beamSpotPosition);
  bool should_type_MuonCandidateValuesUsed(const reco::Muon& muon, const math::XYZPoint &beamSpotPosition);
      
  double minPt_;
  double maxEta_;
  bool isAlsoTkMu_;
  double maxNormChi2_;
  double maxd0_;
  int minnHits_;
  int minnValidStaHits_;

  bool useTrackAssociatorPositions_;
  bool useHO_;
  double towerEtThreshold_;
  bool useRecHits_;

  edm::EDGetTokenT<edm::View<reco::Muon> > muonToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

  TrackAssociatorParameters trackAssociatorParameters_;
  TrackDetectorAssociator trackAssociator_;
};

}

//____________________________________________________________________________||
#endif /* RecoMET_MuonMETValueMapProducer_h */


