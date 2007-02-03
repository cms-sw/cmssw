#ifndef MagVolume_H
#define MagVolume_H

#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "MagneticField/VolumeGeometry/interface/VolumeSide.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <vector>

template <class T>
class MagneticFieldProvider;

class MagVolume : public GloballyPositioned<float>, public MagneticField {
public:

  typedef GloballyPositioned<float>    Base;
  typedef GloballyPositioned<float>::LocalPoint     LocalPoint;
  typedef GloballyPositioned<float>::LocalVector    LocalVector;
  typedef GloballyPositioned<float>::GlobalPoint    GlobalPoint;
  typedef GloballyPositioned<float>::GlobalVector   GlobalVector;

  MagVolume( const PositionType& pos, const RotationType& rot, 
	     DDSolidShape shape, const MagneticFieldProvider<float> * mfp) :
    Base(pos,rot), MagneticField(), theShape(shape), theProvider( mfp) {}

  virtual ~MagVolume() {}

  DDSolidShape shapeType() const {return theShape;}

  LocalVector fieldInTesla( const LocalPoint& lp) const;
  GlobalVector fieldInTesla( const GlobalPoint& lp) const;

  virtual bool inside( const GlobalPoint& gp, double tolerance=0.) const = 0;
  virtual bool inside( const LocalPoint& lp, double tolerance=0.) const {
    return inside( toGlobal(lp), tolerance);
  }

  const MagneticFieldProvider<float>* provider() const {return theProvider;}

  /// Access to volume faces
  virtual std::vector<VolumeSide> faces() const = 0;

  virtual ::GlobalVector inTesla ( const ::GlobalPoint& gp) const {
    return fieldInTesla( gp);
  }

private:

  DDSolidShape theShape;
  const MagneticFieldProvider<float> * theProvider;

};

#endif
