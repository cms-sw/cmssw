#ifndef RecoEGAMMA_ConversionTrack_InOutConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_InOutConversionTrackFinder_h
/** \class InOutConversionTrackFinder
 **  
 **
 **  $Id: InOutConversionTrackFinder.h,v 1.9 2008/05/08 20:41:19 nancy Exp $ 
 **  $Date: 2008/05/08 20:41:19 $ 
 **  $Revision: 1.9 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

//
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
//
#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"

// C/C++ headers
#include <string>
#include <vector>

//

class MagneticField;
class TrajectoryCleanerBySharedHits;


class InOutConversionTrackFinder : public ConversionTrackFinder {
 public :
   
  InOutConversionTrackFinder(  const edm::EventSetup& es,
                               const edm::ParameterSet& config );

 
 virtual ~InOutConversionTrackFinder();
 virtual std::vector<Trajectory> tracks(const TrajectorySeedCollection& seeds, TrackCandidateCollection &candidate ) const ;

 
 private:
 
 TrajectoryCleanerBySharedHits* theTrajectoryCleaner_;
 RedundantSeedCleaner*  theSeedCleaner_;


};

#endif
