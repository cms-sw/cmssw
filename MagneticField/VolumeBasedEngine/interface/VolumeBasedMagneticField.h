#ifndef MagneticField_VolumeBasedMagneticField_h
#define MagneticField_VolumeBasedMagneticField_h

/** \class VolumeBasedMagneticField
 *
 *  Field engine providing interpolation within the full CMS region.
 *
 *  $Date: 2011/12/10 16:03:31 $
 *  $Revision: 1.10 $
 *  \author N. Amapane - CERN
 */

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/VolumeBasedEngine/interface/MagGeometry.h"

class VolumeBasedMagneticField : public MagneticField {
 public:
  //  VolumeBasedMagneticField(const DDCompactView & cpv);
  VolumeBasedMagneticField( const edm::ParameterSet& config,
			    std::vector<MagBLayer *> theBLayers,
			    std::vector<MagESector *> theESectors,
			    std::vector<MagVolume6Faces*> theBVolumes,
			    std::vector<MagVolume6Faces*> theEVolumes,
			    float rMax, float zMax,
			    const MagneticField* param=0,
			    bool isParamFieldOwned=false);
  virtual ~VolumeBasedMagneticField();

  /// Copy constructor implement a shallow copy (ie no ownership of actual engines)
  VolumeBasedMagneticField(const VolumeBasedMagneticField& vbf);

  /// Returns a shallow copy.
  virtual MagneticField* clone() const;

  GlobalVector inTesla ( const GlobalPoint& g) const;

  GlobalVector inTeslaUnchecked ( const GlobalPoint& g) const;

  const MagVolume * findVolume(const GlobalPoint & gp) const;

  bool isDefined(const GlobalPoint& gp) const;

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
