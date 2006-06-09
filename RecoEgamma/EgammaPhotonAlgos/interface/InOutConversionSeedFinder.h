#ifndef RecoEGAMMA_ConversionSeed_InOutConversionSeedFinder_h
#define RecoEGAMMA_ConversionSeed_InOutConversionSeedFinder_h

/** \class InOutConversionSeedFinder
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
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"

#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include <string>
#include <vector>



class MagneticField;
class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class TrajectoryMeasurement;

class InOutConversionSeedFinder : public ConversionSeedFinder {


 private:
 
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryStateOnSurface TSOS;


  public :
    
  
  
  InOutConversionSeedFinder( const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker) : ConversionSeedFinder( field, theInputMeasurementTracker  ) {
    std::cout << "  InOutConversionSeedFinder CTOR " << std::endl;      
    
  }
  
   
  virtual ~InOutConversionSeedFinder();
    
  virtual void  makeSeeds(const reco::BasicClusterCollection& allBc) const { ; }  
  void setTracks(std::vector<const TrajectoryMeasurement*> in) {theOutInTracks_.clear(); theOutInTracks_ = in;}


  private :

    std::vector<const TrajectoryMeasurement*> theOutInTracks_;
    
    mutable vector<TrajectoryMeasurement> theFirstMeasurements_;
   
    const LayerMeasurements*      theLayerMeasurements_;
    const NavigationSchool*       theNavigationSchool_;

};

#endif
