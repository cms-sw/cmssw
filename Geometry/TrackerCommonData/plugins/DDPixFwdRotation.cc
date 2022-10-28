/* 
   == CMS Forward Pixels Geometry ==
   Algorithm for creating rotatuion matrix
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
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

class DDPixFwdRotation : public DDAlgorithm {
public:
  DDPixFwdRotation() {}
  ~DDPixFwdRotation() override = default;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  double endcap_;  // +1 for Z Plus endcap disks, -1 for Z Minus endcap disks
  std::string rotNameNippleToCover_;
  std::string rotNameCoverToNipple_;
  std::string rotNameNippleToBody_;
  int nBlades_;              // Number of blades
  double bladeAngle_;        // Angle of blade rotation around axis perpendicular to beam
  double bladeZShift_;       // Shift in Z between the axes of two adjacent blades
  double ancorRadius_;       // Distance from beam line to ancor point defining center of "blade frame"
  double jX_, jY_, jZ_;      // Coordinates of Nipple ancor points J in blade frame
  double kX_, kY_, kZ_;      // Coordinates of Nipple ancor points K in blade frame
  std::string rotNS_;        //Namespace of the rotation matrix
  std::string idNameSpace_;  //Namespace of this and ALL sub-parts
};

void DDPixFwdRotation::initialize(const DDNumericArguments& nArgs,
                                  const DDVectorArguments& vArgs,
                                  const DDMapArguments&,
                                  const DDStringArguments& sArgs,
                                  const DDStringVectorArguments&) {
  // -- Input geometry parameters :  -----------------------------------------------------
  endcap_ = nArgs["Endcap"];
  rotNameNippleToCover_ = sArgs["NippleToCover"];
  rotNameCoverToNipple_ = sArgs["CoverToNipple"];
  rotNameNippleToBody_ = sArgs["NippleToBody"];
  nBlades_ = static_cast<int>(nArgs["Blades"]);  // Number of blades
  bladeAngle_ = nArgs["BladeAngle"];             // Angle of blade rotation around its axis
  bladeZShift_ = nArgs["BladeZShift"];           // Shift in Z between the axes of two adjacent blades
  ancorRadius_ = nArgs["AncorRadius"];  // Distance from beam line to ancor point defining center of "blade frame"
  // Coordinates of Nipple ancor points J and K in "blade frame" :
  jX_ = nArgs["JX"];
  jY_ = nArgs["JY"];
  jZ_ = nArgs["JZ"];
  kX_ = nArgs["KX"];
  kY_ = nArgs["KY"];
  kZ_ = nArgs["KZ"];

  rotNS_ = sArgs["RotationNS"];
  idNameSpace_ = DDCurrentNamespace::ns();

  edm::LogVerbatim("PixelGeom") << "DDPixFwdRotation: Initialize with endcap " << endcap_ << " NameSpace "
                                << idNameSpace_ << ":" << rotNS_ << "\n  nBlades " << nBlades_ << " bladeAngle "
                                << bladeAngle_ << " bladeZShift " << bladeZShift_ << " ancorRadius " << ancorRadius_
                                << " jX|jY|jZ " << jX_ << ":" << jY_ << ":" << jZ_ << " kX|kY|kZ " << kX_ << ":" << kY_
                                << ":" << kZ_;
}

void DDPixFwdRotation::execute(DDCompactView&) {
  // -- Compute Nipple parameters if not already computed :
  double effBladeAngle = endcap_ * bladeAngle_;

  CLHEP::Hep3Vector jC = CLHEP::Hep3Vector(jX_ * endcap_, jY_ + ancorRadius_, jZ_);
  ;  // Point J in the "cover" blade frame
  CLHEP::Hep3Vector kB = CLHEP::Hep3Vector(kX_ * endcap_, kY_ + ancorRadius_, kZ_);
  ;  // PoinladeZShiftladeZShiftladeZShiftt K in the "body" blade frame

  // Z-shift from "cover" to "body" blade frame:
  CLHEP::Hep3Vector tCB(bladeZShift_ * sin(effBladeAngle), 0., bladeZShift_ * cos(effBladeAngle));

  // Rotation from "cover" blade frame into "body" blade frame :
  double deltaPhi = endcap_ * (360. / nBlades_) * CLHEP::deg;
  CLHEP::HepRotation rCB(CLHEP::Hep3Vector(1. * sin(effBladeAngle), 0., 1. * cos(effBladeAngle)), deltaPhi);

  // Transform vector k into "cover" blade frame :
  CLHEP::Hep3Vector kC = rCB * (kB + tCB);

  // Vector JK in the "cover" blade frame:
  CLHEP::Hep3Vector jkC = kC - jC;
  double jkLength = jkC.mag();
  DDConstant JK(DDName("JK", rotNS_), std::make_unique<double>(jkLength));
  edm::LogVerbatim("PixelGeom") << "+++++++++++++++ DDPixFwdRotation: JK Length " << jkLength * CLHEP::mm;

  // Position of the center of a nipple in "cover" blade frame :
  CLHEP::Hep3Vector nippleTranslation((kC + jC) / 2. - CLHEP::Hep3Vector(0., ancorRadius_, 0.));
  edm::LogVerbatim("PixelGeom") << "Child translation : " << nippleTranslation;

  // Rotations from nipple frame to "cover" blade frame and back :
  CLHEP::Hep3Vector vZ(0., 0., 1.);
  CLHEP::Hep3Vector axis = vZ.cross(jkC);
  double angleCover = vZ.angle(jkC);
  edm::LogVerbatim("PixelGeom") << " Angle to Cover: " << angleCover;
  CLHEP::HepRotation* rpCN = new CLHEP::HepRotation(axis, angleCover);

  DDrot(
      DDName(rotNameCoverToNipple_, rotNS_),
      std::make_unique<DDRotationMatrix>(
          rpCN->xx(), rpCN->xy(), rpCN->xz(), rpCN->yx(), rpCN->yy(), rpCN->yz(), rpCN->zx(), rpCN->zy(), rpCN->zz()));
  CLHEP::HepRotation rpNC(axis, -angleCover);
  edm::LogVerbatim("PixelGeom") << "DDPixFwdBlades::Defines " << DDName(rotNameCoverToNipple_, rotNS_) << " with "
                                << rpCN;

  DDrot(DDName(rotNameNippleToCover_, rotNS_),
        std::make_unique<DDRotationMatrix>(
            rpNC.xx(), rpNC.xy(), rpNC.xz(), rpNC.yx(), rpNC.yy(), rpNC.yz(), rpNC.zx(), rpNC.zy(), rpNC.zz()));
  edm::LogVerbatim("PixelGeom") << "DDPixFwdBlades::Defines " << DDName(rotNameNippleToCover_, rotNS_) << " with "
                                << rpNC;

  // Rotation from nipple frame to "body" blade frame :
  CLHEP::HepRotation rpNB(rpNC * rCB);
  DDrot(DDName(rotNameNippleToBody_, rotNS_),
        std::make_unique<DDRotationMatrix>(
            rpNB.xx(), rpNB.xy(), rpNB.xz(), rpNB.yx(), rpNB.yy(), rpNB.yz(), rpNB.zx(), rpNB.zy(), rpNB.zz()));
  edm::LogVerbatim("PixelGeom") << "DDPixFwdBlades::Defines " << DDName(rotNameNippleToBody_, rotNS_) << " with "
                                << rpNB;
  edm::LogVerbatim("PixelGeom") << " Angle to body : " << vZ.angle(rpNB * vZ);
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDPixFwdRotation, "track:DDPixFwdRotation");
