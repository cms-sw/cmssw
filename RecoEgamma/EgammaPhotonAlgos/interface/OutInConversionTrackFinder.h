#ifndef RecoEGAMMA_ConversionTrack_OutInConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_OutInConversionTrackFinder_h
/** \class OutInConversionTrackFinder
 **  
 **
 **  $Id: $ 
 **  $Date: $ 
 **  $Revision: $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"

// C/C++ headers
#include <string>
#include <vector>

//

class MagneticField;

class OutInConversionTrackFinder : public ConversionTrackFinder {
 public :
  
  OutInConversionTrackFinder(const MagneticField* field ) 
    {
      std::cout << " OutInConversionTrackFinder CTOR " << std:: endl;      
    }

  
  virtual ~OutInConversionTrackFinder()
    {
    }


  
  

 private: 

  std::vector<TrajectoryMeasurement> theTracks_;


};

#endif
