#ifndef TrackerSeedGenerator_PrimitiveMuonSeed_H
#define TrackerSeedGenerator_PrimitiveMuonSeed_H

/**  \class PrimitiveMuonSeed
 *
 *   Primitive tracker seed from muon which 
 *   consists of one measurement and a trajectory state
 *
 *   $Date: 2006/07/27 08:49:00 $
 *   $Revision: 1.4 $
 *
 *   \author   N. Neumeister   - Purdue University
 */

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/BasicTrajectorySeed.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

class FreeTrajectoryState;
class TrajectoryMeasurement;
class TransientTrackingRecHit;
//              ---------------------
//              -- Class Interface --
//              ---------------------

class PrimitiveMuonSeed : public TrajectorySeed {

  public:

    /// constructor
    PrimitiveMuonSeed(const PTrajectoryStateOnDet& state,
                      const PropagationDirection& direction,
		      const recHitContainer layerRecHits,
                      const TrajectoryMeasurement& meas);

    /// destructor
    virtual ~PrimitiveMuonSeed() {}

    //PropagationDirection direction() const;

    std::vector<TrajectoryMeasurement> measurements() const;

    virtual bool share(const BasicTrajectorySeed&) const;

    //virtual PrimitiveMuonSeed* clone() const;
    PrimitiveMuonSeed * clone() const {return new PrimitiveMuonSeed( * this); }

    //range recHits() const;

    //PTrajectoryStateOnDet startingState() const;

  private:

    //PTrajectoryStateOnDet         theState;
    //PropagationDirection          theDirection;  
    //TrajectorySeed                theTrajectorySeed;
    TrajectoryMeasurement         theMeasurement;

};

#endif 

