/* 
   == CMS Forward Pixels Geometry ==
   Algorithm for placing one-per-blade components.
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DD4hep/Printout.h"

#include "DataFormats/Math/interface/CMSUnits.h"
#include "CLHEP/Vector/Rotation.h"

#include <sstream>

using namespace cms_units::operators;  // _deg and convertRadToDeg
using namespace std;

namespace {

  struct DDPixFwdBlades {
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

    int issueCopyNumber() {
      if (copyNumbers.count(childName) == 0)
        copyNumbers[childName] = 0;
      return ++copyNumbers[childName];
    }

    void computeNippleParameters(cms::DDNamespace& ns, double endcap) {
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
      // DDConstant JK(DDName("JK", "pixfwdNipple"), make_unique<double>(jkLength));
      LogDebug("PixelGeom") << "+++++++++++++++ DDPixFwdBlades: "
                            << "JK Length " << jkLength * CLHEP::mm;

      // Position of the center of a nipple in "cover" blade frame :

      CLHEP::Hep3Vector nippleTranslation((kC + jC) / 2. - CLHEP::Hep3Vector(0., ancorRadius, 0.));
      if (endcap > 0) {
        nippleTranslationX = nippleTranslation.x();
        nippleTranslationY = nippleTranslation.y();
        nippleTranslationZ = nippleTranslation.z();
      }
      LogDebug("PixelGeom") << "Child translation : " << nippleTranslation;

      // Rotations from nipple frame to "cover" blade frame and back :

      CLHEP::Hep3Vector vZ(0., 0., 1.);
      CLHEP::Hep3Vector axis = vZ.cross(jkC);
      double angleCover = vZ.angle(jkC);
      LogDebug("PixelGeom") << " Angle to Cover: " << angleCover;
      CLHEP::HepRotation* rpCN = new CLHEP::HepRotation(axis, angleCover);
      if (endcap > 0.) {
        nippleRotationZPlus = rpCN;
      } else {
        nippleRotationZMinus = rpCN;
      }
      //( endcap > 0. ? nippleRotationZPlus : nippleRotationZMinus ) = rpCN;

      ns.addRotation(
          rotNameCoverToNipple + "pixfwdNipple",
          dd4hep::Rotation3D
          // DDrot(DDName(rotNameCoverToNipple, "pixfwdNipple"),
          //       make_unique<DDRotationMatrix>
          (rpCN->xx(), rpCN->xy(), rpCN->xz(), rpCN->yx(), rpCN->yy(), rpCN->yz(), rpCN->zx(), rpCN->zy(), rpCN->zz()));
      CLHEP::HepRotation rpNC(axis, -angleCover);

      ns.addRotation(
          rotNameNippleToCover + "pixfwdNipple",
          dd4hep::Rotation3D
          // DDrot(DDName(rotNameNippleToCover, "pixfwdNipple"),
          //       make_unique<DDRotationMatrix>
          (rpNC.xx(), rpNC.xy(), rpNC.xz(), rpNC.yx(), rpNC.yy(), rpNC.yz(), rpNC.zx(), rpNC.zy(), rpNC.zz()));

      // Rotation from nipple frame to "body" blade frame :

      CLHEP::HepRotation rpNB(rpNC * rCB);

      ns.addRotation(
          rotNameNippleToBody + "pixfwdNipple",
          dd4hep::Rotation3D
          // DDrot(DDName(rotNameNippleToBody, "pixfwdNipple"),
          //       make_unique<DDRotationMatrix>
          (rpNB.xx(), rpNB.xy(), rpNB.xz(), rpNB.yx(), rpNB.yy(), rpNB.yz(), rpNB.zx(), rpNB.zy(), rpNB.zz()));
      double angleBody = vZ.angle(rpNB * vZ);
      LogDebug("PixelGeom") << " Angle to body : " << angleBody;
    }

    void initialize(cms::DDNamespace& ns, cms::DDAlgoArguments& args) {
      endcap = args.find("Endcap") ? args.value<int>("Endcap") : 1.;
      flagString = args.find("FlagString") ? args.value<std::string>("FlagString") : "YYYYYYYYYYYYYYYYYYYYYYYY";
      flagSelector = args.find("FlagSelector") ? args.value<std::string>("FlagSelector") : "Y";
      childName = args.find("Child") ? args.value<std::string>("Child") : "";
      childTranslationVector = args.find("ChildTranslation") ? args.vecDble("ChildTranslation") : vector<double>(3, 0.);
      childRotationName = args.find("ChildRotation") ? args.value<std::string>("ChildRotation") : "";
      idNameSpace = ns.name();

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
    }

    void execute(cms::DDNamespace& ns, cms::DDAlgoArguments& args, cms::DDParsingContext& ctxt) {
      // -- Compute Nipple parameters if not already computed :

      if (!nippleRotationZPlus) {
        computeNippleParameters(ns, 1.);   // Z Plus endcap
        computeNippleParameters(ns, -1.);  // Z Minus endcap
      }
      if (childName.empty())
        return;

      // -- Signed versions of blade angle and z-shift :

      double effBladeAngle = -endcap * bladeAngle;
      double effBladeZShift = endcap * bladeZShift;

      // -- Names of mother and child volumes :

      dd4hep::Volume mother = ns.volume(args.parentName());
      dd4hep::Volume child = ns.volume(childName);

      // -- Get translation and rotation from "blade frame" to "child frame", if any :

      CLHEP::HepRotation childRotMatrix = CLHEP::HepRotation();
      if (!childRotationName.empty()) {
        auto childRotation = ns.rotation(childRotationName);

        // due to conversion to ROOT::Math::Rotation3D -- Michael Case
        DD3Vector x, y, z;
        childRotation.GetComponents(x, y, z);  // these are the orthonormal columns.
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

        string rotstr = mother.name() + childName + to_string(copy);
        auto rotation = dd4hep::Rotation3D();
        auto nsrot = ctxt.rotations.find(ns.prepend(rotstr));
        if (nsrot == ctxt.rotations.end()) {
          rotMatrix *= childRotMatrix;
          rotation = rotMatrix;

          // makeRotation3D(DDrot(DDName(rotstr, idNameSpace),
          //                  make_unique<Rotation>(rotMatrix.xx(),
          //                                                rotMatrix.xy(),
          //                                                rotMatrix.xz(),
          //                                                rotMatrix.yx(),
          //                                                rotMatrix.yy(),
          //                                                rotMatrix.yz(),
          //                                                rotMatrix.zx(),
          //                                                rotMatrix.zy(),
          //                                                rotMatrix.zz()));
        }
        // position the child :

        dd4hep::Position ddtran(translation.x(), translation.y(), translation.z());
        mother.placeVolume(child, copy, dd4hep::Transform3D(rotation, ddtran));
      }

      // End of cycle over Phi positions
    }
  };
}  // namespace

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  DDPixFwdBlades blades;
  blades.initialize(ns, args);
  blades.execute(ns, args, ctxt);
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDPixFwdBlades, algorithm)
