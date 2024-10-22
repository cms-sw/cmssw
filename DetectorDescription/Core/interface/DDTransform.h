#ifndef DETECTOR_DESCRIPTION_CORE_DDTRANSFORM_H
#define DETECTOR_DESCRIPTION_CORE_DDTRANSFORM_H

#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include <memory>

class DDRotation;

std::ostream &operator<<(std::ostream &, const DDRotation &);

//! Definition of a uniquely identifiable rotation matrix named by DDName \a name
/** DDrot() returns a reference-object DDRotation representing the rotation matrix \a rot.
    
    The user must not free memory allocated for \a rot!
*/
DDRotation DDrot(const DDName &name, std::unique_ptr<DDRotationMatrix> rot);

std::unique_ptr<DDRotation> DDrotPtr(const DDName &name, std::unique_ptr<DDRotationMatrix> rot);

//! Definition of a uniquely identifiable rotation matrix named by DDName \a name in the GEANT3 style
/** DDrot() returns a reference-object DDRotation representing the rotation matrix.
*/
DDRotation DDrot(const DDName &name, double thetaX, double phiX, double thetaY, double phiY, double thetaZ, double phiZ);

//! Defines a rotation-reflection in the Geant3 way.
/** The resulting matrix MUST be a LEFThanded orthonormal system, otherwise
    a DDException will be thrown!
*/
DDRotation DDrotReflect(
    const DDName &name, double thetaX, double phiX, double thetaY, double phiY, double thetaZ, double phiZ);

DDRotation DDrotReflect(const DDName &name, DDRotationMatrix *rot);

//! Defines a anonymous rotation or rotation-reflection matrix.
/** It can't be addressed by a unique DDName. Once created, it's the
    users responsibility to keep the reference object DDRotation!
    Will be mostly used by algorithmic positioning.
*/
DDRotation DDanonymousRot(std::unique_ptr<DDRotationMatrix> rot);

//! create a new DDRotationMatrix in the GEANT3 style.
/** The Matrix must be orthonormal - left or right handed - otherwise a DDException is thrown; 
    memory of the returned pointer belongs to the caller
*/
std::unique_ptr<DDRotationMatrix> DDcreateRotationMatrix(
    double thetaX, double phiX, double thetaY, double phiY, double thetaZ, double phiZ);

//! Represents a uniquely identifyable rotation matrix
/** An object of this class is a reference-object and thus leightweighted.
    It is uniquely identified by its DDName. Further details concerning
    reference-objects can be found in the documentation of DDLogicalPart.
    
    DDRotation encapsulates ROOT Rotation3D.
*/
class DDRotation : public DDBase<DDName, std::unique_ptr<DDRotationMatrix>> {
  friend DDRotation DDrot(const DDName &, std::unique_ptr<DDRotationMatrix>);
  friend std::unique_ptr<DDRotation> DDrotPtr(const DDName &, std::unique_ptr<DDRotationMatrix>);
  friend DDRotation DDrotReflect(const DDName &, double, double, double, double, double, double);
  friend DDRotation DDanonymousRot(std::unique_ptr<DDRotationMatrix>);

public:
  //! refers to the unit-rotation (no rotation at all)
  DDRotation();

  //! Creates a initialized reference-object or a reference to an allready defined rotation.
  /**
     A reference-object to a defined rotation is created if a rotation was already defined usind DDrot(). 
     Otherwise a (default) initialized reference-object named \a name is created. At any later stage the rotation matrix
     can be defined using DDrot(). All initialized-reference object referring to the same \a name will 
     then immidialtely refere to the matrix created by DDrot().
     
     DDRotation is a lightweighted reference-object. For further details concerning reference-object
     refere to the documentation of DDLogicalPart.
  */
  DDRotation(const DDName &name);

  DDRotation(const DDName &, std::unique_ptr<DDRotationMatrix>);
  //! Returns the read-only rotation-matrix
  const DDRotationMatrix &rotation() const { return rep(); }

  DDRotationMatrix &rotation() { return rep(); }

  DDRotationMatrix &matrix() { return rotation(); }

private:
  DDRotation(std::unique_ptr<DDRotationMatrix>);
};

#endif
