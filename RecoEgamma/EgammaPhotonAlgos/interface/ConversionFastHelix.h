#ifndef Egamma_ConversionFastHelix_H_
#define Egamma_ConversionFastHelix_H_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"


/**
   Generation of track parameters at a vertex using two hits and a vertex.
 */

class ConversionFastHelix {

private:
  
  typedef FreeTrajectoryState FTS;

public:

  ConversionFastHelix(const GlobalPoint& outerHit, 
	    const GlobalPoint& middleHit,
	    const GlobalPoint& aVertex,
	    const MagneticField* field
            );
  
  ~ConversionFastHelix() {}

  void makeHelix();
  

  bool isValid() {return validStateAtVertex;  }

  FTS stateAtVertex() ;

  FTS helixStateAtVertex() ;

  FTS straightLineStateAtVertex() ;

private:

  FTS theHelix_; 
  bool validStateAtVertex; 
  GlobalPoint theOuterHit;
  GlobalPoint theMiddleHit;
  GlobalPoint theVertex;
  FastCircle theCircle;
  const MagneticField* mField;
 
};

#endif //Egamma_ConversionFastHelix_H_




