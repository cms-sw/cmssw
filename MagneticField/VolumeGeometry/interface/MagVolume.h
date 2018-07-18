#ifndef MagVolume_H
#define MagVolume_H

#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
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
	     const MagneticFieldProvider<float> * mfp,
	     double sf=1.) :
    Base(pos,rot), MagneticField(), theProvider(mfp), 
    theProviderOwned(false), theScalingFactor(sf), isIronFlag(false) {}

  ~MagVolume() override;

  LocalVector fieldInTesla( const LocalPoint& lp) const;
  GlobalVector fieldInTesla( const GlobalPoint& lp) const;

  virtual bool inside( const GlobalPoint& gp, double tolerance=0.) const = 0;
  virtual bool inside( const LocalPoint& lp, double tolerance=0.) const {
    return inside( toGlobal(lp), tolerance);
  }

  const MagneticFieldProvider<float>* provider() const {return theProvider;}

  /// Access to volume faces
  virtual const std::vector<VolumeSide>& faces() const = 0;

  ::GlobalVector inTesla ( const ::GlobalPoint& gp) const override {
    return fieldInTesla( gp);
  }

  /// Temporary hack to pass information on material. Will eventually be replaced!
  bool isIron() const {return isIronFlag;}
  void setIsIron(bool iron) {isIronFlag = iron;}
  void ownsFieldProvider(bool o) {theProviderOwned=o;}

private:

  const MagneticFieldProvider<float> * theProvider;
  bool theProviderOwned;
  double theScalingFactor;
  // Temporary hack to keep information on material. Will eventually be replaced!
  bool isIronFlag;

};

#endif
