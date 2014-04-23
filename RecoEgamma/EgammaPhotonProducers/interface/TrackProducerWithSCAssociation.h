#ifndef TrackProducerWithSCAssociation_h
#define TrackProducerWithSCAssociation_h
/** \class  TrackProducerWithSCAssociation
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **   Modified version of TrackProducer by Giuseppe Cerati
 **   to have super cluster - conversion track association
 ** 
 ***/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCandidateCaloClusterAssociation.h"

class TrackProducerWithSCAssociation : public TrackProducerBase<reco::Track>, public edm::stream::EDProducer<> {
public:

  explicit TrackProducerWithSCAssociation(const edm::ParameterSet& iConfig);


  virtual void produce(edm::Event&, const edm::EventSetup&);

  std::vector<reco::TransientTrack> getTransient(edm::Event&, const edm::EventSetup&);

private:
  std::string myname_; 
  TrackProducerAlgorithm<reco::Track> theAlgo;
  std::string conversionTrackCandidateProducer_;
  std::string trackCSuperClusterAssociationCollection_;
  std::string trackSuperClusterAssociationCollection_;
  edm::EDGetTokenT<reco::TrackCandidateCaloClusterPtrAssociation> assoc_token;
  edm::OrphanHandle<reco::TrackCollection> rTracks_;
  bool myTrajectoryInEvent_;
  bool validTrackCandidateSCAssociationInput_;


  //Same recipe as Ursula's for electrons. Copy this from TrackProducerBase to get the OrphanHandle
  //ugly temporary solution!! I agree !
  void putInEvt(edm::Event& evt,
		const Propagator* thePropagator,
		const MeasurementTracker* theMeasTk,
		std::auto_ptr<TrackingRecHitCollection>& selHits,
		std::auto_ptr<reco::TrackCollection>& selTracks,
		std::auto_ptr<reco::TrackExtraCollection>& selTrackExtras,
		std::auto_ptr<std::vector<Trajectory> >&   selTrajectories,
		AlgoProductCollection& algoResults, TransientTrackingRecHitBuilder const * hitBuilder);
};

#endif
