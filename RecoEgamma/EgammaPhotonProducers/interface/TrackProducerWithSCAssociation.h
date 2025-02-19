#ifndef TrackProducerWithSCAssociation_h
#define TrackProducerWithSCAssociation_h
/** \class  TrackProducerWithSCAssociation
 **  
 **
 **  $Id: TrackProducerWithSCAssociation.h,v 1.5 2010/09/29 12:20:16 mangano Exp $ 
 **  $Date: 2010/09/29 12:20:16 $ 
 **  $Revision: 1.5 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **   Modified version of TrackProducer by Giuseppe Cerati
 **   to have super cluster - conversion track association
 ** 
 ***/

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class TrackProducerWithSCAssociation : public TrackProducerBase<reco::Track>, public edm::EDProducer {
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
		AlgoProductCollection& algoResults);
};

#endif
