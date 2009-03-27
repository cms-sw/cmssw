/** \class MuonMillepedeTrackRefitter
*
*  This class produces a collection of TrackForAlignment using as input
*  Tracks and 4DSegments from AlcaReco.
*  Calculation of predicted states is performed here.
*
*  $Date: 2008/12/12 18:02:14 $
*  $Revision: 1.1 $
*  \author P. Martinez Ruiz del Arbol, IFCA (CSIC-UC)  <Pablo.Martinez@cern.ch>
*/


#ifndef Alignment_MuonMillepedeTrackRefitter_MuonMillepedeTrackRefitter_H
#define Alignment_MuonMillepedeTrackRefitter_MuonMillepedeTrackRefitter_H


// Base Class Headers
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"

#include <string>
#include <vector>

class SegmentToTrackAssociator;
class TrackingRecHit;

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
  class InputTag;
}


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




