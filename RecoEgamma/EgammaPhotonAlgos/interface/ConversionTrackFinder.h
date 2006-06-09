#ifndef RecoEGAMMA_ConversionTrack_ConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_ConversionTrackFinder_h

/** \class ConversionTrackFinder
 **  
 **
 **  $Id: $ 
 **  $Date: $ 
 **  $Revision: $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"


// C/C++ headers
#include <string>
#include <vector>






class ConversionTrackFinder {

 public:
  
  ConversionTrackFinder()
    {
      std::cout << " ConversionTrackFinder CTOR " << std:: endl;      
    }

  
  virtual ~ConversionTrackFinder()
    {
    }

  
  virtual std::vector<const TrajectoryMeasurement*> tracks(const TrajectorySeedCollection seeds ) const {std::cout << " Returning tracks " << std::endl; return theTracks_;}

					       

 private: 
  
  std::vector<const TrajectoryMeasurement* > theTracks_;


};

#endif
