/** \class MuonMillepedeTrackRefitter
*
*  This class produces a collection of TrackForAlignment using as input
*  Tracks and 4DSegments from AlcaReco.
*  Calculation of predicted states is performed here.
*
*  $Date: 2008/12/12 10:23:59 $
*  $Revision: 1.1 $
*  \author P. Martinez Ruiz del Arbol, IFCA (CSIC-UC)  <Pablo.Martinez@cern.ch>
*/


#ifndef Alignment_MuonMillepedeTrackRefitter_MuonMillepedeTrackRefitter_H
#define Alignment_MuonMillepedeTrackRefitter_MuonMillepedeTrackRefitter_H


// Base Class Headers
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/SegmentToTrackAssociator.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"





namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
  class InputTag;
}


using namespace edm;


typedef std::vector< std::vector<int> > intDVector;
typedef std::vector<TrackingRecHit *> RecHitVector;

class MuonMillepedeTrackRefitter: public edm::EDProducer {
public:

  typedef AlignmentAlgorithmBase::ConstTrajTrackPair ConstTrajTrackPair;
  typedef AlignmentAlgorithmBase::ConstTrajTrackPairCollection ConstTrajTrackPairCollection;

  /// Constructor
  MuonMillepedeTrackRefitter(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~MuonMillepedeTrackRefitter();

  // Operations

  virtual void produce(edm::Event & event, const edm::EventSetup& eventSetup);

protected:

private:
  
  
  edm::InputTag MuonCollectionTag;
 
  edm::InputTag TrackerCollectionTag;
  
  edm::InputTag SACollectionTag;

  std::string  TrackRefitterType;

  std::string propagatorSourceOpposite;  
  
  std::string propagatorSourceAlong;  

  //Propagator *thePropagator;

  SegmentToTrackAssociator *theSegmentsAssociator;
   
  enum TrackType {
    CosmicLike,
    LHCLike
  };
    
};
#endif




