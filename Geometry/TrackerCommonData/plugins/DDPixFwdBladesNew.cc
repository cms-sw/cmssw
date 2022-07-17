/* 
   == CMS Forward Pixels Geometry ==
   Algorithm for placing one-per-blade components.
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Vector/RotationInterfaces.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

class DDPixFwdBladesNew : public DDAlgorithm {
public:
  DDPixFwdBladesNew() {}
  ~DDPixFwdBladesNew() override = default;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  double endcap_;             // +1 for Z Plus endcap disks, -1 for Z Minus endcap disks
  int nBlades_;               // Number of blades
  double bladeAngle_;         // Angle of blade rotation around axis perpendicular to beam
  double zPlane_;             // Common shift in Z for all blades (with respect to disk center plane)
  double bladeZShift_;        // Shift in Z between the axes of two adjacent blades
  double ancorRadius_;        // Distance from beam line to ancor point defining center of "blade frame"
  int nippleType_;            // Flag if it is called frm Nipple (1) or not (0)
  double jX_, jY_, jZ_;       // Coordinates of Nipple ancor points J in blade frame
  double kX_, kY_, kZ_;       // Coordinates of Nipple ancor points K in blade frame
  std::string flagString_;    // String of flags
  std::string flagSelector_;  // Character that means "yes" in flagString
  std::string childName_;     // Child volume name
  int startCopy_;             // First copy number
  std::vector<double> childTranslationVector_;  // Child translation with respect to "blade frame"
  std::string childRotationName_;               // Child rotation with respect to "blade frame"
  std::string idNameSpace_;                     //Namespace of this and ALL sub-parts

  CLHEP::Hep3Vector getTranslation();
  CLHEP::HepRotation getRotation();
};

void DDPixFwdBladesNew::initialize(const DDNumericArguments& nArgs,
                                   const DDVectorArguments& vArgs,
                                   const DDMapArguments&,
                                   const DDStringArguments& sArgs,
                                   const DDStringVectorArguments&) {
  endcap_ = nArgs["Endcap"];
  nBlades_ = static_cast<int>(nArgs["Blades"]);  // Number of blades
  bladeAngle_ = nArgs["BladeAngle"];             // Angle of blade rotation around its axis
  bladeZShift_ = nArgs["BladeZShift"];           // Shift in Z between the axes of two adjacent blades
  ancorRadius_ = nArgs["AncorRadius"];  // Distance from beam line to ancor point defining center of "blade frame"
  // Coordinates of Nipple ancor points J and K in "blade frame" :
  nippleType_ = static_cast<int>(nArgs["NippleType"]);
  jX_ = nArgs["JX"];
  jY_ = nArgs["JY"];
  jZ_ = nArgs["JZ"];
  kX_ = nArgs["KX"];
  kY_ = nArgs["KY"];
  kZ_ = nArgs["KZ"];

  flagString_ = sArgs["FlagString"];
  flagSelector_ = sArgs["FlagSelector"];
  childName_ = sArgs["Child"];
  startCopy_ = static_cast<int>(nArgs["StartCopy"]);
  childTranslationVector_ = vArgs["ChildTranslation"];
  childRotationName_ = sArgs["ChildRotation"];

  idNameSpace_ = DDCurrentNamespace::ns();

  edm::LogVerbatim("PixelGeom") << "DDPixFwdBladesNew: Initialize with endcap " << endcap_ << " FlagString "
                                << flagString_ << " FlagSelector " << flagSelector_ << " Child " << childName_
                                << " ChildTranslation " << childTranslationVector_[0] << ":"
                                << childTranslationVector_[1] << ":" << childTranslationVector_[2] << " ChildRotation "
                                << childRotationName_ << " NameSpace " << idNameSpace_ << "\n  nBlades " << nBlades_
                                << " bladeAngle " << bladeAngle_ << " zPlane " << zPlane_ << " bladeZShift "
                                << bladeZShift_ << " ancorRadius " << ancorRadius_ << " NippleType " << nippleType_
                                << " jX|jY|jZ " << jX_ << ":" << jY_ << ":" << jZ_ << " kX|kY|kZ " << kX_ << ":" << kY_
                                << ":" << kZ_;
}

void DDPixFwdBladesNew::execute(DDCompactView& cpv) {
  // -- Signed versions of blade angle and z-shift :

  double effBladeAngle = -endcap_ * bladeAngle_;
  double effBladeZShift = endcap_ * bladeZShift_;

  // -- Names of mother and child volumes :

  DDName mother = parent().name();
  DDName child(DDSplit(childName_).first, DDSplit(childName_).second);

  // -- Get translation and rotation from "blade frame" to "child frame", if any :

  CLHEP::HepRotation childRotMatrix = CLHEP::HepRotation();
  if (nippleType_ == 1) {
    childRotMatrix = getRotation();
  } else if (!childRotationName_.empty()) {
    DDRotation childRotation =
        DDRotation(DDName(DDSplit(childRotationName_).first, DDSplit(childRotationName_).second));
    // due to conversion to ROOT::Math::Rotation3D -- Michael Case
    DD3Vector x, y, z;
    childRotation.rotation().GetComponents(x, y, z);  // these are the orthonormal columns.
    CLHEP::HepRep3x3 tr(x.X(), y.X(), z.X(), x.Y(), y.Y(), z.Y(), x.Z(), y.Z(), z.Z());
    childRotMatrix = CLHEP::HepRotation(tr);
  }

  CLHEP::Hep3Vector childTranslation =
      (nippleType_ == 1)
          ? getTranslation()
          : CLHEP::Hep3Vector(childTranslationVector_[0], childTranslationVector_[1], childTranslationVector_[2]);

  // Create a matrix for rotation around blade axis (to "blade frame") :
  CLHEP::HepRotation bladeRotMatrix(CLHEP::Hep3Vector(0., 1., 0.), effBladeAngle);

  // Cycle over Phi positions, placing copies of the child volume :

  double deltaPhi = (360. / nBlades_) * CLHEP::deg;
  int nQuarter = nBlades_ / 4;
  double zShiftMax = effBladeZShift * ((nQuarter - 1) / 2.);
  int copy(startCopy_);

  for (int iBlade = 0; iBlade < nBlades_; iBlade++) {
    // check if this blade position should be skipped :

    if (flagString_[iBlade] != flagSelector_[0])
      continue;

    // calculate Phi and Z shift for this blade :

    double phi = (iBlade + 0.5) * deltaPhi - 90. * CLHEP::deg;
    int iQuarter = iBlade % nQuarter;
    double zShift = -zShiftMax + iQuarter * effBladeZShift;

    // compute rotation matrix from mother to blade frame :
    CLHEP::HepRotation rotMatrix(CLHEP::Hep3Vector(0., 0., 1.), phi);
    rotMatrix *= bladeRotMatrix;

    // convert translation vector from blade frame to mother frame, and add Z shift :
    CLHEP::Hep3Vector translation = rotMatrix(childTranslation + CLHEP::Hep3Vector(0., ancorRadius_, 0.));
    translation += CLHEP::Hep3Vector(0., 0., zShift + zPlane_);

    // create DDRotation for placing the child if not already existent :
    DDRotation rotation;
    std::string rotstr = mother.name() + DDSplit(childName_).first + std::to_string(copy);
    rotation = DDRotation(DDName(rotstr, idNameSpace_));
    edm::LogVerbatim("PixelGeom") << "DDPixFwdBlades: Rotation " << rotstr << " : " << rotation;

    if (!rotation) {
      rotMatrix *= childRotMatrix;
      rotation = DDrot(DDName(rotstr, idNameSpace_),
                       std::make_unique<DDRotationMatrix>(rotMatrix.xx(),
                                                          rotMatrix.xy(),
                                                          rotMatrix.xz(),
                                                          rotMatrix.yx(),
                                                          rotMatrix.yy(),
                                                          rotMatrix.yz(),
                                                          rotMatrix.zx(),
                                                          rotMatrix.zy(),
                                                          rotMatrix.zz()));
    }
    // position the child :

    DDTranslation ddtran(translation.x(), translation.y(), translation.z());
    cpv.position(child, mother, copy, ddtran, rotation);
    edm::LogVerbatim("PixelGeom") << "DDPixFwdBlades::Position " << child << " copy " << copy << " in " << mother
                                  << " with translation " << ddtran << " and rotation " << rotation;
    ++copy;
  }

  // End of cycle over Phi positions
}

// -- Calculating Nipple parameters :  ---------------------------------------------------

CLHEP::Hep3Vector DDPixFwdBladesNew::getTranslation() {
  double effBladeAngle = endcap_ * bladeAngle_;

  CLHEP::Hep3Vector jC =
      CLHEP::Hep3Vector(endcap_ * jX_, jY_ + ancorRadius_, jZ_);  // Point J in the "cover" blade frame
  CLHEP::Hep3Vector kB =
      CLHEP::Hep3Vector(endcap_ * kX_, kY_ + ancorRadius_, kZ_);  // Point K in the "body" blade frame

  // Z-shift from "cover" to "body" blade frame:
  CLHEP::Hep3Vector tCB(bladeZShift_ * sin(effBladeAngle), 0., bladeZShift_ * cos(effBladeAngle));

  // Rotation from "cover" blade frame into "body" blade frame :
  double deltaPhi = endcap_ * (360. / nBlades_) * CLHEP::deg;
  CLHEP::HepRotation rCB(CLHEP::Hep3Vector(1. * sin(effBladeAngle), 0., 1. * cos(effBladeAngle)), deltaPhi);

  // Transform vector k into "cover" blade frame :
  CLHEP::Hep3Vector kC = rCB * (kB + tCB);

  // Position of the center of a nipple in "cover" blade frame :
  CLHEP::Hep3Vector nippleTranslation((kC + jC) / 2. - CLHEP::Hep3Vector(0., ancorRadius_, 0.));
  edm::LogVerbatim("PixelGeom") << "Child translation : " << nippleTranslation;
  return nippleTranslation;
}

CLHEP::HepRotation DDPixFwdBladesNew::getRotation() {
  double effBladeAngle = endcap_ * bladeAngle_;

  CLHEP::Hep3Vector jC =
      CLHEP::Hep3Vector(endcap_ * jX_, jY_ + ancorRadius_, jZ_);  // Point J in the "cover" blade frame
  CLHEP::Hep3Vector kB =
      CLHEP::Hep3Vector(endcap_ * kX_, kY_ + ancorRadius_, kZ_);  // Point K in the "body" blade frame

  // Z-shift from "cover" to "body" blade frame:
  CLHEP::Hep3Vector tCB(bladeZShift_ * sin(effBladeAngle), 0., bladeZShift_ * cos(effBladeAngle));

  // Rotation from "cover" blade frame into "body" blade frame :
  double deltaPhi = endcap_ * (360. / nBlades_) * CLHEP::deg;
  CLHEP::HepRotation rCB(CLHEP::Hep3Vector(1. * sin(effBladeAngle), 0., 1. * cos(effBladeAngle)), deltaPhi);

  // Transform vector k into "cover" blade frame :
  CLHEP::Hep3Vector kC = rCB * (kB + tCB);
  CLHEP::Hep3Vector jkC = kC - jC;
  edm::LogVerbatim("PixelGeom") << "+++++++++++++++ DDPixFwdBlades: "
                                << "JK Length " << jkC.mag() * CLHEP::mm;

  // Rotations from nipple frame to "cover" blade frame and back :
  CLHEP::Hep3Vector vZ(0., 0., 1.);
  CLHEP::Hep3Vector axis = vZ.cross(jkC);
  double angleCover = vZ.angle(jkC);
  edm::LogVerbatim("PixelGeom") << " Angle to Cover: " << angleCover;
  CLHEP::HepRotation rpCN(axis, angleCover);
  return rpCN;
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDPixFwdBladesNew, "track:DDPixFwdBladesNew");
