#ifndef RecoEGAMMA_ConversionTrack_InOutConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_InOutConversionTrackFinder_h
/** \class InOutConversionTrackFinder
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

class InOutConversionTrackFinder : public ConversionTrackFinder {
 public :
  
  InOutConversionTrackFinder(const MagneticField* field ) 
    {
      std::cout << " InOutConversionTrackFinder CTOR " << std:: endl;      
    }

  
  virtual ~InOutConversionTrackFinder()
    {
    }


  
  

 private: 

  std::vector<const TrajectoryMeasurement*> theTracks_;


};

#endif
