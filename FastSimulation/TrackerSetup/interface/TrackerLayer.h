#ifndef _TrackerLayer_H_
#define _TrackerLayer_H_

#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

#include <vector>

/** A class that gives some properties of the Tracker Layers in FAMOS
 */

class TrackerLayer {
public:
  
  /// constructor from private members
  TrackerLayer(BoundSurface* theSurface,
	       bool isForward,
	       unsigned int theLayerNumber,
	       const std::vector<double>& theMinDim,
	       const std::vector<double>& theMaxDim,
	       const std::vector<double>& theFudge) :
    theSurface(theSurface), 
    isForward(isForward),
    theLayerNumber(theLayerNumber),
    theDimensionMinValues(theMinDim),
    theDimensionMaxValues(theMaxDim),
    theFudgeFactors(theFudge),
    theNumberOfFudgeFactors(theFudgeFactors.size())
   { 
     isSensitive = (theLayerNumber<100);
     if ( isForward ) { 
       theDisk = dynamic_cast<BoundDisk*>(theSurface);
       theDiskInnerRadius = theDisk->innerRadius();
       theDiskOuterRadius = theDisk->outerRadius();
       theCylinder = 0;
     } else {
       theCylinder = dynamic_cast<BoundCylinder*>(theSurface);
       theDisk = 0;
       theDiskInnerRadius = 0.;
       theDiskOuterRadius = 0.;
     }

   }

  TrackerLayer(BoundSurface* theSurface,
	       unsigned int theLayerNumber,
	       const std::vector<double>& theMinDim,
	       const std::vector<double>& theMaxDim,
	       const std::vector<double>& theFudge) :
    theSurface(theSurface), 
    theLayerNumber(theLayerNumber),
    theDimensionMinValues(theMinDim),
    theDimensionMaxValues(theMaxDim),
    theFudgeFactors(theFudge),
    theNumberOfFudgeFactors(theFudgeFactors.size())
   { 
     isSensitive = true;
     isForward = true;
     theDisk = dynamic_cast<BoundDisk*>(theSurface);
     theDiskInnerRadius = theDisk->innerRadius();
     theDiskOuterRadius = theDisk->outerRadius();
     theCylinder = 0;
   }

  /// Is the layer sensitive ?
  inline bool sensitive() const { return isSensitive; }

  /// Is the layer forward ?
  inline bool forward() const { return isForward; }

  /// Returns the surface
  inline const BoundSurface& surface() const { return *theSurface; }

  /// Returns the cylinder
  inline BoundCylinder const* cylinder() const { return theCylinder; }

  /// Returns the surface
  inline BoundDisk const* disk() const { return theDisk; }

  /// Returns the layer number  
  inline unsigned int layerNumber() const { return theLayerNumber; }

  /// Returns the inner radius of a disk
  inline double diskInnerRadius() const { return theDiskInnerRadius; }

  /// Returns the outer radius of a disk
  inline double diskOuterRadius() const { return theDiskOuterRadius; }

  /// Set a fudge factor for material inhomogeneities in this layer
  /*
  void setFudgeFactor(double min, double max, double f) { 
    ++theNumberOfFudgeFactors;
    theDimensionMinValues.push_back(min);
    theDimensionMaxValues.push_back(max);
    theFudgeFactors.push_back(f);
  }
  */

  /// Get the fudge factors back
  inline unsigned int fudgeNumber() const { return  theNumberOfFudgeFactors; }
  inline double fudgeMin(unsigned iFudge) const { 
    return (iFudge < theNumberOfFudgeFactors) ? theDimensionMinValues[iFudge] : 999.;
  }
  inline double fudgeMax(unsigned iFudge) const { 
    return (iFudge < theNumberOfFudgeFactors) ? theDimensionMaxValues[iFudge] : -999.;
  }
  inline double fudgeFactor(unsigned iFudge) const { 
    return (iFudge < theNumberOfFudgeFactors) ? theFudgeFactors[iFudge] : 0.;
  }

private:

  BoundSurface* theSurface;
  BoundDisk* theDisk;
  BoundCylinder* theCylinder;
  bool isForward;
  unsigned int theLayerNumber;
  bool isSensitive;
  double theDiskInnerRadius;
  double theDiskOuterRadius;

  /// These are fudges factors to account for the inhomogeneities of the material
  std::vector<double> theDimensionMinValues;
  std::vector<double> theDimensionMaxValues;
  std::vector<double> theFudgeFactors;
  unsigned int  theNumberOfFudgeFactors;  

};
#endif

