#ifndef TrackerSeedGenerator_PrimitiveMuonSeed_H
#define TrackerSeedGenerator_PrimitiveMuonSeed_H

/**  \class PrimitiveMuonSeed
 *
 *   Primitive tracker seed from muon which 
 *   consists of one measurement and a trajectory state
 *
 *   $Date: 2006/11/08 08:04:54 $
 *   $Revision: 1.5 $
 *
 *   \author   N. Neumeister   - Purdue University
 */

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
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

    virtual bool share(const TrajectorySeed&) const;

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

