#ifndef TrackerSeedGenerator_MuonSeedFromConsecutiveHits_H
#define TrackerSeedGenerator_MuonSeedFromConsecutiveHits_H

/**  \class MuonSeedFromConsecutiveHits
 * 
 *   Create muon seed from hits in two
 *   consecutive tracker layers
 * 
 *
 *   $Date: 2006/07/27 08:49:25 $
 *   $Revision: 1.4 $
 *
 *   \author   N. Neumeister            Purdue University
 *   \author porting C. Liu             Purdue University
 */

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include "FWCore/Framework/interface/EventSetup.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class MuonSeedFromConsecutiveHits : public TrajectorySeed {

  public:

    /// constructors
    MuonSeedFromConsecutiveHits(const TransientTrackingRecHit& outerHit,
                                const TransientTrackingRecHit& innerHit,
                                const PropagationDirection direction,
                                const GlobalPoint& vertexPos,
                                const GlobalError& vertexErr,
                                const edm::EventSetup& iSetup);
                                
    /// destructor
    virtual ~MuonSeedFromConsecutiveHits();

    virtual const FreeTrajectoryState& freeTrajectoryState() const;
    
    virtual PTrajectoryStateOnDet startingState() const;

    virtual PropagationDirection direction() const;

    virtual range recHits() const;

    virtual std::vector<TrajectoryMeasurement> measurements() const;

    virtual bool share( const BasicTrajectorySeed&) const;

    virtual MuonSeedFromConsecutiveHits* clone() const;

  private:

    void construct(const TransientTrackingRecHit& outerHit, 
                   const TransientTrackingRecHit& innerHit,
                   const GlobalPoint& vertexPos,
                   const GlobalError& vertexErr,
                   const edm::EventSetup& iSetup);

 private :

    bool status;
    PropagationDirection  theDirection;
    TrajectoryMeasurement theInnerMeas;
    TrajectoryMeasurement theOuterMeas;

};

#endif 

