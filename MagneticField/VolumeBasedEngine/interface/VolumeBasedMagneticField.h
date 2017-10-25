#ifndef MagneticField_VolumeBasedMagneticField_h
#define MagneticField_VolumeBasedMagneticField_h

/** \class VolumeBasedMagneticField
 *
 *  Field engine providing interpolation within the full CMS region.
 *
 *  \author N. Amapane - CERN
 */

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/VolumeBasedEngine/interface/MagGeometry.h"

// Class for testing VolumeBasedMagneticField
class testMagneticField;
class testMagGeometryAnalyzer;

class VolumeBasedMagneticField : public MagneticField {
  // For tests
  friend class testMagneticField;
  friend class testMagGeometryAnalyzer;

 public:
  //  VolumeBasedMagneticField(const DDCompactView & cpv);
  VolumeBasedMagneticField( int geomVersion,
			    const std::vector<MagBLayer *>& theBLayers,
			    const std::vector<MagESector *>& theESectors,
			    const std::vector<MagVolume6Faces*>& theBVolumes,
			    const std::vector<MagVolume6Faces*>& theEVolumes,
			    float rMax, float zMax,
			    const MagneticField* param=nullptr,
			    bool isParamFieldOwned=false);
  ~VolumeBasedMagneticField() override;

  /// Copy constructor implement a shallow copy (ie no ownership of actual engines)
  VolumeBasedMagneticField(const VolumeBasedMagneticField& vbf);

  /// Returns a shallow copy.
  MagneticField* clone() const override;

  GlobalVector inTesla ( const GlobalPoint& g) const override;

  GlobalVector inTeslaUnchecked ( const GlobalPoint& g) const override;

  const MagVolume * findVolume(const GlobalPoint & gp) const;

  bool isDefined(const GlobalPoint& gp) const override;

  bool isZSymmetric() const;


 private:
  const MagGeometry* field;
  float maxR;
  float maxZ;
  const MagneticField* paramField;
  bool magGeomOwned;
  bool paramFieldOwned;
};

#endif
