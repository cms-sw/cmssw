/* 
== CMS Forward Pixels Geometry ==

 @version 3.02.01 May 30, 2006
 @created Dmitry Onoprienko

  Algorithm for placing one-per-blade components.
  See header file (DDPixFwdBlades.h) for a detailed description.
*/

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "Geometry/TrackerCommonData/interface/DDPixFwdBlades.h"
#include "CLHEP/Vector/RotationInterfaces.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

  // -- Input geometry parameters :  -----------------------------------------------------

const int DDPixFwdBlades::nBlades = 24;            // Number of blades
const double DDPixFwdBlades::bladeAngle = 20.*deg;    // Angle of blade rotation around its axis
const double DDPixFwdBlades::zPlane = 0.;             // Common shift in Z for all blades (with respect to disk center plane)
const double DDPixFwdBlades::bladeZShift = 6.*mm;     // Shift in Z between the axes of two adjacent blades
  
const double DDPixFwdBlades::ancorRadius = 54.631*mm; // Distance from beam line to ancor point defining center of "blade frame"
  
      // Coordinates of Nipple ancor points J and K in "blade frame" :

const double DDPixFwdBlades::jX = -16.25*mm;
const double DDPixFwdBlades::jY = 96.50*mm;
const double DDPixFwdBlades::jZ = 1.25*mm;
const double DDPixFwdBlades::kX = 16.25*mm;
const double DDPixFwdBlades::kY = 96.50*mm;
const double DDPixFwdBlades::kZ = -1.25*mm;
  
// -- Static initialization :  -----------------------------------------------------------

HepRotation* DDPixFwdBlades::nippleRotationZPlus = 0;
HepRotation* DDPixFwdBlades::nippleRotationZMinus = 0;
double DDPixFwdBlades::nippleTranslationX = 0.;
double DDPixFwdBlades::nippleTranslationY = 0.;
double DDPixFwdBlades::nippleTranslationZ = 0.;

std::map<std::string, int> DDPixFwdBlades::copyNumbers;

// -- Constructors & Destructor :  -------------------------------------------------------

DDPixFwdBlades::DDPixFwdBlades() {}
DDPixFwdBlades::~DDPixFwdBlades() {}

// Initialization :  ---------------------------------------------------------------------

void DDPixFwdBlades::initialize(const DDNumericArguments & nArgs,
				const DDVectorArguments & vArgs,
				const DDMapArguments & ,
				const DDStringArguments & sArgs,
				const DDStringVectorArguments & ) {
				  	
  try {
    endcap = nArgs["Endcap"];
  } catch (...) {
    endcap = 1.;
  }

  try {
    flagString = sArgs["FlagString"];
    flagSelector = sArgs["FlagSelector"];
  } catch (...) {
    flagString = "YYYYYYYYYYYYYYYYYYYYYYYY";
    flagSelector = "Y";
  }

  try {
    childName   = sArgs["Child"];
  } catch (...) {
    childName   = "";
  }

  try {
    childTranslationVector = vArgs["ChildTranslation"];
  } catch (...) {
    childTranslationVector = std::vector<double>(3, 0.);
  }

  try {
    childRotationName = sArgs["ChildRotation"];
  } catch (...) {
    childRotationName = "";
  }

  idNameSpace = DDCurrentNamespace::ns();
}
  
// Execution :  --------------------------------------------------------------------------

void DDPixFwdBlades::execute() {

  // -- Compute Nipple parameters if not already computed :
  
  if (!nippleRotationZPlus) {
  	computeNippleParameters(1.);   // Z Plus endcap
  	computeNippleParameters(-1.);  // Z Minus endcap
  }
  if (childName == "") return;
  
  // -- Signed versions of blade angle and z-shift :
  
  double effBladeAngle = - endcap * bladeAngle;
  double effBladeZShift = endcap * bladeZShift;
  
  // -- Names of mother and child volumes :

  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  
  // -- Get translation and rotation from "blade frame" to "child frame", if any :
  
  HepRotation childRotMatrix = HepRotation();
  if (childRotationName != "") {
    DDRotation childRotation = DDRotation(DDName(DDSplit(childRotationName).first, DDSplit(childRotationName).second));
    // due to conversion to ROOT::Math::Rotation3D -- Michael Case
    DD3Vector x, y, z;
    childRotation.rotation()->GetComponents(x, y, z); // these are the orthonormal columns.
    HepRep3x3 tr(x.X(), y.X(), z.X(), x.Y(), y.Y(), z.Y(), x.Z(), y.Z(), z.Z());
    childRotMatrix = HepRotation(tr);
  } else if (childName == "pixfwdNipple:PixelForwardNippleZPlus") {
    childRotMatrix = *nippleRotationZPlus;
  } else if (childName == "pixfwdNipple:PixelForwardNippleZMinus") {
    childRotMatrix = *nippleRotationZMinus;
  }
  
  Hep3Vector childTranslation;
  if (childName == "pixfwdNipple:PixelForwardNippleZPlus") {
  	childTranslation = Hep3Vector(nippleTranslationX, nippleTranslationY, nippleTranslationZ);
  } else if (childName == "pixfwdNipple:PixelForwardNippleZMinus") {
  	childTranslation = Hep3Vector(-nippleTranslationX, nippleTranslationY, nippleTranslationZ);
  } else {
  	childTranslation = Hep3Vector(childTranslationVector[0],childTranslationVector[1],childTranslationVector[2]);
  }
  
  // Create a matrix for rotation around blade axis (to "blade frame") :
  
  HepRotation bladeRotMatrix(Hep3Vector(0.,1.,0.), effBladeAngle);
  
  // Cycle over Phi positions, placing copies of the child volume :

  double deltaPhi = (360./nBlades)*deg;
  int nQuarter = nBlades/4;
  double zShiftMax = effBladeZShift*((nQuarter-1)/2.);

  for (int iBlade=0; iBlade < nBlades; iBlade++) {
    
    // check if this blade position should be skipped :
  	
  	if (flagString[iBlade] != flagSelector[0]) continue;
    int copy = issueCopyNumber();
    
    // calculate Phi and Z shift for this blade :

    double phi = (iBlade + 0.5) * deltaPhi - 90.*deg;
    int iQuarter = iBlade % nQuarter;
    double zShift = - zShiftMax + iQuarter * effBladeZShift;
    
    // compute rotation matrix from mother to blade frame :
    
    HepRotation* rotMatrix = new HepRotation(Hep3Vector(0.,0.,1.), phi);
    (*rotMatrix) *= bladeRotMatrix;
    
    // convert translation vector from blade frame to mother frame, and add Z shift :
    
    Hep3Vector translation = (*rotMatrix)(childTranslation + Hep3Vector(0., ancorRadius, 0.));
    translation += Hep3Vector(0., 0., zShift + zPlane);
    
    // create DDRotation for placing the child if not already existent :

    DDRotation rotation;   
    std::string rotstr = DDSplit(childName).first + int_to_string(copy);
    rotation = DDRotation(DDName(rotstr, idNameSpace));

    if (!rotation) {
      (*rotMatrix) *= childRotMatrix;
      // due to conversion to ROOT::Math::Rotation3D; 
      // also, maybe I'm dumb, but the ownership of these pointers is very confusing,
      // so it seemed safer to make DD stuff "as needed" -- Michael Case
      DDRotationMatrix* temp = new DDRotationMatrix(rotMatrix->xx(), rotMatrix->xy(), rotMatrix->xz(),
						    rotMatrix->yx(), rotMatrix->yy(), rotMatrix->yz(),
						    rotMatrix->zx(), rotMatrix->zy(), rotMatrix->zz() );
      rotation = DDrot(DDName(rotstr, idNameSpace), temp);
    } else {
      delete rotMatrix;
    }

    // position the child :

    // due to conversion to ROOT::Math::Rotation3D; 
    // also, maybe I'm dumb, but the ownership of these pointers is very confusing,
    // so it seemed safer to make DD stuff "as needed" -- Michael Case
    DDTranslation ddtran(translation.x(), translation.y(), translation.z());
    DDpos(child, mother, copy, ddtran, rotation);
    // LogDebug("PixelGeom") << "DDPixFwdBlades: " << child << " Copy " << copy << " positioned in " << mother << " at " << translation << " with rotation " << rotation;
  }

  // End of cycle over Phi positions

}

// -- Helpers :  -------------------------------------------------------------------------

int DDPixFwdBlades::issueCopyNumber() {
  if (copyNumbers.count(childName) == 0) copyNumbers[childName] = 0;
  return ++copyNumbers[childName];
}


// -- Calculating Nipple parameters :  ---------------------------------------------------

void DDPixFwdBlades::computeNippleParameters(double endcap) {
	
  double effBladeAngle = endcap * bladeAngle;
  
  Hep3Vector jC; // Point J in the "cover" blade frame
  Hep3Vector kB; // Point K in the "body" blade frame
  std::string rotNameNippleToCover;
  std::string rotNameCoverToNipple;
  std::string rotNameNippleToBody;

  if (endcap > 0.) {
  	jC = Hep3Vector(jX, jY + ancorRadius, jZ);
  	kB = Hep3Vector(kX, kY + ancorRadius, kZ);
	rotNameNippleToCover = "NippleToCoverZPlus";
	rotNameCoverToNipple = "CoverToNippleZPlus";
	rotNameNippleToBody = "NippleToBodyZPlus";
  } else {
  	jC = Hep3Vector(-jX, jY + ancorRadius, jZ);
  	kB = Hep3Vector(-kX, kY + ancorRadius, kZ);
	rotNameNippleToCover = "NippleToCoverZMinus";
	rotNameCoverToNipple = "CoverToNippleZMinus";
	rotNameNippleToBody = "NippleToBodyZMinus";
  }
		
  // Z-shift from "cover" to "body" blade frame:
  
  Hep3Vector tCB(bladeZShift*sin(effBladeAngle), 0., bladeZShift*cos(effBladeAngle));
  
  // Rotation from "cover" blade frame into "body" blade frame :
  
  double deltaPhi = endcap*(360./nBlades)*deg;
  HepRotation rCB(Hep3Vector(1.*sin(effBladeAngle), 0., 1.*cos(effBladeAngle)), deltaPhi);
  
  // Transform vector k into "cover" blade frame :
  
  Hep3Vector kC = rCB * (kB + tCB);
  
  // Vector JK in the "cover" blade frame:
  
  Hep3Vector jkC = kC - jC;
  double* jkLength = new double(jkC.mag());
  DDConstant JK(DDName("JK", "pixfwdNipple"), jkLength);
  LogDebug("PixelGeom") << "+++++++++++++++ DDPixFwdBlades: " << "JK Length " <<  *jkLength * mm;
  
  // Position of the center of a nipple in "cover" blade frame :
  
  Hep3Vector nippleTranslation((kC+jC)/2. - Hep3Vector(0., ancorRadius, 0.));
  if (endcap > 0) {
  	nippleTranslationX = nippleTranslation.x();
  	nippleTranslationY = nippleTranslation.y();
  	nippleTranslationZ = nippleTranslation.z();
  }
  LogDebug("PixelGeom") << "Child translation : " << nippleTranslation;
  
  // Rotations from nipple frame to "cover" blade frame and back :
  
  Hep3Vector vZ(0.,0.,1.);
  Hep3Vector axis = vZ.cross(jkC);
  double angleCover = vZ.angle(jkC);
  LogDebug("PixelGeom") << " Angle to Cover: " << angleCover;
  HepRotation* rpCN = new HepRotation(axis, angleCover);
  if (endcap > 0.) {
  	nippleRotationZPlus = rpCN;
  } else {
  	nippleRotationZMinus = rpCN;
  }
  //( endcap > 0. ? nippleRotationZPlus : nippleRotationZMinus ) = rpCN;

  // due to conversion to ROOT::Math::Rotation3D;
  // also, maybe I'm dumb, but the ownership of these pointers is very confusing,
  // so it seemed safer to make DD stuff "as needed" -- Michael Case
  DDRotationMatrix* ddrpCN = new DDRotationMatrix(rpCN->xx(), rpCN->xy(), rpCN->xz(),
						  rpCN->yx(), rpCN->yy(), rpCN->yz(),
						  rpCN->zx(), rpCN->zy(), rpCN->zz() );

  DDrot(DDName(rotNameCoverToNipple, "pixfwdNipple"), ddrpCN);
  HepRotation* rpNC = new HepRotation(axis, -angleCover);
  DDRotationMatrix* ddrpNC = new DDRotationMatrix(rpNC->xx(), rpNC->xy(), rpNC->xz(),
						  rpNC->yx(), rpNC->yy(), rpNC->yz(),
						  rpNC->zx(), rpNC->zy(), rpNC->zz() );

  DDrot(DDName(rotNameNippleToCover, "pixfwdNipple"), ddrpNC);
  
  // Rotation from nipple frame to "body" blade frame :
  
  HepRotation* rpNB = new HepRotation( (*rpNC) * rCB );
  DDRotationMatrix* ddrpNB = new DDRotationMatrix(rpNB->xx(), rpNB->xy(), rpNB->xz(),
						  rpNB->yx(), rpNB->yy(), rpNB->yz(),
						  rpNB->zx(), rpNB->zy(), rpNB->zz() );

  DDrot(DDName(rotNameNippleToBody, "pixfwdNipple"), ddrpNB);
  double angleBody = vZ.angle(*rpNB * vZ);
  LogDebug("PixelGeom") << " Angle to body : " << angleBody;
  
}

// ---------------------------------------------------------------------------------------

