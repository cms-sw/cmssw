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

/* 

== CMS Forward Pixels Geometry ==

 @version 3.02.01 May 30, 2006
 @created Dmitry Onoprienko

== ALGORITHM DESCRIPTION: ==

  Algorithm for placing one-per-blade components
  Also computes parameters necessary for defining the "nipple" geometry.

== Parameters : ==

  "Endcap" - +1 if placing the child volume into +Z disk, -1 if placing into -Z disk.
  "Child" - name of a child volume being places (should be in the form "file:volume")
            In no child name is given, the algorithm simply calculates Nipple parameters.
  "ChildRotation" - rotation of the child volume with respect to the "blade frame". [OPTIONAL]
  "ChildTranslation" - vector defining translation of the child volume with respect to the 
                       "blade frame". [OPTIONAL]
  "FlagString" - string of 24 characters, used to indicate blades into which the child volume 
                 should be placed. [OPTIONAL]
  "FlagSelector" - 1 character string, key to interpreting "FlagString".
                   Positions in "BladeFlag" that have this character will get the child volume.
                   
  If "Child" parameter is omitted, the algorithm computes rotation needed for describing 
  coolant "nipples" but does not do any placements.
  
  If "Child" is "PixelForwardNippleZPlus" or "PixelForwardNippleZMinus" and no rotation or translation
  is supplied, correct rotations and translations are automatically computed.
  
  Blade frame: origin on the axis of the blade at a distance "ancorRadius" from the beam line
  (it therefore coincides with the ancor point of a blade). 
  Y along blade axis pointing away from beam line, Z perpendicular to blade plane and pointing away from IP.
  (That assumes the axes of ZPlus disk are aligned with CMS global reference frame, and ZMinus disk
  is rotated around Y by 180 degrees.)

== Example of use : ==

<Algorithm name="track:DDPixFwdBlades">
  <rParent name="pixfwdDisk:PixelForwardDiskZMinus"/>
  <Numeric name="Endcap"        value="-1." />
  <String  name="Child"         value="pixfwdPanel:PixelForwardPanel4Left"/>
  <Vector  name="ChildTranslation" type="numeric" nEntries="3"> 0., -[pixfwdPanel:AncorY], [zPanel] </Vector>
  <String  name="ChildRotation" value="pixfwdCommon:Y180"/>
  <String  name="FlagString"    value="LRRRRLRRRRRRLRRRRLRRRRRR" />  <!-- Panel Layout ZMinus 4  -->
  <String  name="FlagSelector"  value="L" />
</Algorithm>

*/

using namespace std;

class DDPixFwdBlades : public DDAlgorithm {
public:
  DDPixFwdBlades();
  ~DDPixFwdBlades() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  int nBlades;         // Number of blades
  double bladeAngle;   // Angle of blade rotation around axis perpendicular to beam
  double zPlane;       // Common shift in Z for all blades (with respect to disk center plane)
  double bladeZShift;  // Shift in Z between the axes of two adjacent blades

  double ancorRadius;  // Distance from beam line to ancor point defining center of "blade frame"

  // Coordinates of Nipple ancor points J and K in "blade frame" :

  double jX;
  double jY;
  double jZ;
  double kX;
  double kY;
  double kZ;

  double endcap;  // +1 for Z Plus endcap disks, -1 for Z Minus endcap disks

  string flagString;    // String of flags
  string flagSelector;  // Character that means "yes" in flagString

  string childName;  // Child volume name

  vector<double> childTranslationVector;  // Child translation with respect to "blade frame"
  string childRotationName;               // Child rotation with respect to "blade frame"
  string idNameSpace;                     //Namespace of this and ALL sub-parts

  map<string, int> copyNumbers;

  CLHEP::HepRotation* nippleRotationZPlus;
  CLHEP::HepRotation* nippleRotationZMinus;
  double nippleTranslationX, nippleTranslationY, nippleTranslationZ;

  int issueCopyNumber();
  void computeNippleParameters(double endcap);
};

DDPixFwdBlades::DDPixFwdBlades() {}
DDPixFwdBlades::~DDPixFwdBlades() {}

void DDPixFwdBlades::initialize(const DDNumericArguments& nArgs,
                                const DDVectorArguments& vArgs,
                                const DDMapArguments&,
                                const DDStringArguments& sArgs,
                                const DDStringVectorArguments&) {
  if (nArgs.find("Endcap") != nArgs.end()) {
    endcap = nArgs["Endcap"];
  } else {
    endcap = 1.;
  }

  if (sArgs.find("FlagString") != sArgs.end()) {
    flagString = sArgs["FlagString"];
    flagSelector = sArgs["FlagSelector"];
  } else {
    flagString = "YYYYYYYYYYYYYYYYYYYYYYYY";
    flagSelector = "Y";
  }

  if (sArgs.find("Child") != sArgs.end()) {
    childName = sArgs["Child"];
  } else {
    childName = "";
  }

  if (vArgs.find("ChildTranslation") != vArgs.end()) {
    childTranslationVector = vArgs["ChildTranslation"];
  } else {
    childTranslationVector = vector<double>(3, 0.);
  }

  if (sArgs.find("ChildRotation") != sArgs.end()) {
    childRotationName = sArgs["ChildRotation"];
  } else {
    childRotationName = "";
  }

  idNameSpace = DDCurrentNamespace::ns();

  // -- Input geometry parameters :  -----------------------------------------------------

  nBlades = 24;                   // Number of blades
  bladeAngle = 20. * CLHEP::deg;  // Angle of blade rotation around its axis
  zPlane = 0.;                    // Common shift in Z for all blades (with respect to disk center plane)
  bladeZShift = 6. * CLHEP::mm;   // Shift in Z between the axes of two adjacent blades

  ancorRadius = 54.631 * CLHEP::mm;  // Distance from beam line to ancor point defining center of "blade frame"

  // Coordinates of Nipple ancor points J and K in "blade frame" :

  jX = -16.25 * CLHEP::mm;
  jY = 96.50 * CLHEP::mm;
  jZ = 1.25 * CLHEP::mm;
  kX = 16.25 * CLHEP::mm;
  kY = 96.50 * CLHEP::mm;
  kZ = -1.25 * CLHEP::mm;

  // -- Static initialization :  -----------------------------------------------------------

  nippleRotationZPlus = nullptr;
  nippleRotationZMinus = nullptr;
  nippleTranslationX = 0.;
  nippleTranslationY = 0.;
  nippleTranslationZ = 0.;

  copyNumbers.clear();

  edm::LogVerbatim("PixelGeom") << "DDPixFwdBlades: Initialize with endcap " << endcap << " FlagString " << flagString
                                << " FlagSelector " << flagSelector << " Child " << childName << " ChildTranslation "
                                << childTranslationVector[0] << ":" << childTranslationVector[1] << ":"
                                << childTranslationVector[2] << " ChildRotation " << childRotationName << " NameSpace "
                                << idNameSpace << "\n  nBlades " << nBlades << " bladeAngle " << bladeAngle
                                << " zPlane " << zPlane << " bladeZShift " << bladeZShift << " ancorRadius "
                                << ancorRadius << " jX|jY|jZ " << jX << ":" << jY << ":" << jZ << " kX|kY|kZ " << kX
                                << ":" << kY << ":" << kZ;
}

void DDPixFwdBlades::execute(DDCompactView& cpv) {
  // -- Compute Nipple parameters if not already computed :

  if (!nippleRotationZPlus) {
    computeNippleParameters(1.);   // Z Plus endcap
    computeNippleParameters(-1.);  // Z Minus endcap
  }
  if (childName.empty())
    return;

  // -- Signed versions of blade angle and z-shift :

  double effBladeAngle = -endcap * bladeAngle;
  double effBladeZShift = endcap * bladeZShift;

  // -- Names of mother and child volumes :

  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);

  // -- Get translation and rotation from "blade frame" to "child frame", if any :

  CLHEP::HepRotation childRotMatrix = CLHEP::HepRotation();
  if (!childRotationName.empty()) {
    DDRotation childRotation = DDRotation(DDName(DDSplit(childRotationName).first, DDSplit(childRotationName).second));
    // due to conversion to ROOT::Math::Rotation3D -- Michael Case
    DD3Vector x, y, z;
    childRotation.rotation().GetComponents(x, y, z);  // these are the orthonormal columns.
    CLHEP::HepRep3x3 tr(x.X(), y.X(), z.X(), x.Y(), y.Y(), z.Y(), x.Z(), y.Z(), z.Z());
    childRotMatrix = CLHEP::HepRotation(tr);
  } else if (childName == "pixfwdNipple:PixelForwardNippleZPlus") {
    childRotMatrix = *nippleRotationZPlus;
  } else if (childName == "pixfwdNipple:PixelForwardNippleZMinus") {
    childRotMatrix = *nippleRotationZMinus;
  }

  CLHEP::Hep3Vector childTranslation;
  if (childName == "pixfwdNipple:PixelForwardNippleZPlus") {
    childTranslation = CLHEP::Hep3Vector(nippleTranslationX, nippleTranslationY, nippleTranslationZ);
  } else if (childName == "pixfwdNipple:PixelForwardNippleZMinus") {
    childTranslation = CLHEP::Hep3Vector(-nippleTranslationX, nippleTranslationY, nippleTranslationZ);
  } else {
    childTranslation =
        CLHEP::Hep3Vector(childTranslationVector[0], childTranslationVector[1], childTranslationVector[2]);
  }

  // Create a matrix for rotation around blade axis (to "blade frame") :

  CLHEP::HepRotation bladeRotMatrix(CLHEP::Hep3Vector(0., 1., 0.), effBladeAngle);

  // Cycle over Phi positions, placing copies of the child volume :

  double deltaPhi = (360. / nBlades) * CLHEP::deg;
  int nQuarter = nBlades / 4;
  double zShiftMax = effBladeZShift * ((nQuarter - 1) / 2.);

  for (int iBlade = 0; iBlade < nBlades; iBlade++) {
    // check if this blade position should be skipped :

    if (flagString[iBlade] != flagSelector[0])
      continue;
    int copy = issueCopyNumber();

    // calculate Phi and Z shift for this blade :

    double phi = (iBlade + 0.5) * deltaPhi - 90. * CLHEP::deg;
    int iQuarter = iBlade % nQuarter;
    double zShift = -zShiftMax + iQuarter * effBladeZShift;

    // compute rotation matrix from mother to blade frame :

    CLHEP::HepRotation rotMatrix(CLHEP::Hep3Vector(0., 0., 1.), phi);
    rotMatrix *= bladeRotMatrix;

    // convert translation vector from blade frame to mother frame, and add Z shift :

    CLHEP::Hep3Vector translation = rotMatrix(childTranslation + CLHEP::Hep3Vector(0., ancorRadius, 0.));
    translation += CLHEP::Hep3Vector(0., 0., zShift + zPlane);

    // create DDRotation for placing the child if not already existent :

    DDRotation rotation;
    string rotstr = mother.name() + DDSplit(childName).first + to_string(copy);
    rotation = DDRotation(DDName(rotstr, idNameSpace));
    edm::LogVerbatim("PixelGeom") << "DDPixFwdBlades: Rotation " << rotstr << " : " << rotation;

    if (!rotation) {
      rotMatrix *= childRotMatrix;
      rotation = DDrot(DDName(rotstr, idNameSpace),
                       make_unique<DDRotationMatrix>(rotMatrix.xx(),
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
  }

  // End of cycle over Phi positions
}

int DDPixFwdBlades::issueCopyNumber() {
  if (copyNumbers.count(childName) == 0)
    copyNumbers[childName] = 0;
  return ++copyNumbers[childName];
}

// -- Calculating Nipple parameters :  ---------------------------------------------------

void DDPixFwdBlades::computeNippleParameters(double endcap) {
  double effBladeAngle = endcap * bladeAngle;

  CLHEP::Hep3Vector jC;  // Point J in the "cover" blade frame
  CLHEP::Hep3Vector kB;  // Point K in the "body" blade frame
  string rotNameNippleToCover;
  string rotNameCoverToNipple;
  string rotNameNippleToBody;

  if (endcap > 0.) {
    jC = CLHEP::Hep3Vector(jX, jY + ancorRadius, jZ);
    kB = CLHEP::Hep3Vector(kX, kY + ancorRadius, kZ);
    rotNameNippleToCover = "NippleToCoverZPlus";
    rotNameCoverToNipple = "CoverToNippleZPlus";
    rotNameNippleToBody = "NippleToBodyZPlus";
  } else {
    jC = CLHEP::Hep3Vector(-jX, jY + ancorRadius, jZ);
    kB = CLHEP::Hep3Vector(-kX, kY + ancorRadius, kZ);
    rotNameNippleToCover = "NippleToCoverZMinus";
    rotNameCoverToNipple = "CoverToNippleZMinus";
    rotNameNippleToBody = "NippleToBodyZMinus";
  }

  // Z-shift from "cover" to "body" blade frame:

  CLHEP::Hep3Vector tCB(bladeZShift * sin(effBladeAngle), 0., bladeZShift * cos(effBladeAngle));

  // Rotation from "cover" blade frame into "body" blade frame :

  double deltaPhi = endcap * (360. / nBlades) * CLHEP::deg;
  CLHEP::HepRotation rCB(CLHEP::Hep3Vector(1. * sin(effBladeAngle), 0., 1. * cos(effBladeAngle)), deltaPhi);

  // Transform vector k into "cover" blade frame :

  CLHEP::Hep3Vector kC = rCB * (kB + tCB);

  // Vector JK in the "cover" blade frame:

  CLHEP::Hep3Vector jkC = kC - jC;
  double jkLength = jkC.mag();
  DDConstant JK(DDName("JK", "pixfwdNipple"), make_unique<double>(jkLength));
  edm::LogVerbatim("PixelGeom") << "+++++++++++++++ DDPixFwdBlades: "
                                << "JK Length " << jkLength * CLHEP::mm;

  // Position of the center of a nipple in "cover" blade frame :

  CLHEP::Hep3Vector nippleTranslation((kC + jC) / 2. - CLHEP::Hep3Vector(0., ancorRadius, 0.));
  if (endcap > 0) {
    nippleTranslationX = nippleTranslation.x();
    nippleTranslationY = nippleTranslation.y();
    nippleTranslationZ = nippleTranslation.z();
  }
  edm::LogVerbatim("PixelGeom") << "Child translation : " << nippleTranslation;

  // Rotations from nipple frame to "cover" blade frame and back :

  CLHEP::Hep3Vector vZ(0., 0., 1.);
  CLHEP::Hep3Vector axis = vZ.cross(jkC);
  double angleCover = vZ.angle(jkC);
  edm::LogVerbatim("PixelGeom") << " Angle to Cover: " << angleCover;
  CLHEP::HepRotation* rpCN = new CLHEP::HepRotation(axis, angleCover);
  if (endcap > 0.) {
    nippleRotationZPlus = rpCN;
  } else {
    nippleRotationZMinus = rpCN;
  }
  //( endcap > 0. ? nippleRotationZPlus : nippleRotationZMinus ) = rpCN;

  DDrot(
      DDName(rotNameCoverToNipple, "pixfwdNipple"),
      make_unique<DDRotationMatrix>(
          rpCN->xx(), rpCN->xy(), rpCN->xz(), rpCN->yx(), rpCN->yy(), rpCN->yz(), rpCN->zx(), rpCN->zy(), rpCN->zz()));
  CLHEP::HepRotation rpNC(axis, -angleCover);
  edm::LogVerbatim("PixelGeom") << "DDPixFwdBlades::Defines " << DDName(rotNameCoverToNipple, "pixfwdNipple")
                                << " with " << rpCN;
  DDrot(DDName(rotNameNippleToCover, "pixfwdNipple"),
        make_unique<DDRotationMatrix>(
            rpNC.xx(), rpNC.xy(), rpNC.xz(), rpNC.yx(), rpNC.yy(), rpNC.yz(), rpNC.zx(), rpNC.zy(), rpNC.zz()));
  edm::LogVerbatim("PixelGeom") << "DDPixFwdBlades::Defines " << DDName(rotNameNippleToCover, "pixfwdNipple")
                                << " with " << rpNC;

  // Rotation from nipple frame to "body" blade frame :

  CLHEP::HepRotation rpNB(rpNC * rCB);

  DDrot(DDName(rotNameNippleToBody, "pixfwdNipple"),
        make_unique<DDRotationMatrix>(
            rpNB.xx(), rpNB.xy(), rpNB.xz(), rpNB.yx(), rpNB.yy(), rpNB.yz(), rpNB.zx(), rpNB.zy(), rpNB.zz()));
  edm::LogVerbatim("PixelGeom") << "DDPixFwdBlades::Defines " << DDName(rotNameNippleToBody, "pixfwdNipple") << " with "
                                << rpNB;
  double angleBody = vZ.angle(rpNB * vZ);
  edm::LogVerbatim("PixelGeom") << " Angle to body : " << angleBody;
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDPixFwdBlades, "track:DDPixFwdBlades");
