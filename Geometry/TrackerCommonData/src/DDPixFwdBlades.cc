/* 
== CMS Forward Pixels Geometry ==

 @version 2.01.01 Dec 06, 2005
 @created Dmitry Onoprienko

  Algorithm for placing one-per-blade components.
  See header file (DDPixFwdBlades.h) for a detailed description.
*/

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDPixFwdBlades.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

// Constructors & Destructor :  ----------------------------------------------------------

DDPixFwdBlades::DDPixFwdBlades() {}

DDPixFwdBlades::~DDPixFwdBlades() {}

std::map<std::string, int> DDPixFwdBlades::copyNumbers;
  
// Initialization :  ---------------------------------------------------------------------

void DDPixFwdBlades::initialize(const DDNumericArguments & nArgs,
				  const DDVectorArguments & vArgs,
				  const DDMapArguments & ,
				  const DDStringArguments & sArgs,
				  const DDStringVectorArguments & ) {

  nBlades      = 24;
  
  bladeAngle   = nArgs["BladeAngle"];
  zPlane       = nArgs["ZPlane"];
  bladeZShift  = nArgs["BladeZShift"];

  DCOUT('A', "DDPixFwdBlades: nBlades " << nBlades << " bladeAngle " << bladeAngle/deg << " zPlane " << zPlane << " bladeZShift " << bladeZShift);
  try {
    flagString = sArgs["FlagString"];
    flagSelector = sArgs["FlagSelector"];
  } catch (...) {
    flagString = "YYYYYYYYYYYYYYYYYYYYYYYY";
    flagSelector = "Y";
  }
  DCOUT('A', "DDPixFwdBlades: flagString " << flagString << " flagSelector " << flagSelector);
 
  childName   = sArgs["Child"]; 

  childX = nArgs["ChildX"];
  childY = nArgs["ChildY"];
  childZ = nArgs["ChildZ"];
  try {
    childRotationName = sArgs["ChildRotation"];
  } catch (...) {
    childRotationName = "";
  }

  idNameSpace = DDCurrentNamespace::ns();
  DCOUT('A', "DDPixFwdBlades: childName " << childName << " Position ("	<< childX << ", " << childY << ", " << childZ << ") Rotation " << childRotationName << " NameSpace " << idNameSpace);
}
  
// Execution :  --------------------------------------------------------------------------

void DDPixFwdBlades::execute() {
  
  // -- Names of mother and child volumes :

  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  
  // -- Get final rotation (with respect to "blade frame") matrix, if any :
  
  HepRotation* childRotMatrix = 0;
  if (childRotationName != "") {
    DDRotation childRotation = DDRotation(DDName(DDSplit(childRotationName).first, DDSplit(childRotationName).second));
    childRotMatrix = childRotation.rotation();
  }
  
  // Create a matrix for rotation around blade axis (to "blade frame") :
  
  HepRotation bladeRotMatrix(Hep3Vector(0.,1.,0.), - bladeAngle);
  
  // Cycle over Phi positions, placing copies of the child volume :

  double deltaPhi = (360./nBlades)*deg;
  int nQuarter = nBlades/4;
  double zShiftMax = bladeZShift*((nQuarter-1)/2.);

  for (int iBlade=0; iBlade < nBlades; iBlade++) {
    
    // check if this blade position should be skipped :
  	
  	if (flagString[iBlade] != flagSelector[0]) continue;
    int copy = issueCopyNumber();
    
    // calculate Phi and Z shift for this blade :

    double phi = (iBlade + 0.5) * deltaPhi - 90.*deg;
    int iQuarter = iBlade % nQuarter;
    double zShift = - zShiftMax + iQuarter * bladeZShift;
    
    // compute rotation matrix from mother to blade frame :
    
    HepRotation* rotMatrix = new HepRotation(Hep3Vector(0.,0.,1.), phi);
    (*rotMatrix) *= bladeRotMatrix;
    
    // convert translation vector from blade frame to mother frame, and add Z shift :
    
    Hep3Vector translation = (*rotMatrix)(Hep3Vector(childX,childY,childZ));
    translation += Hep3Vector(0., 0., zShift + zPlane);
    
    // create DDRotation for the child if not already existent :

    DDRotation rotation;   
    string rotstr = DDSplit(childName).first + int_to_string(copy);
    rotation = DDRotation(DDName(rotstr, idNameSpace));

    if (!rotation) {
      if (childRotMatrix) (*rotMatrix) *= (*childRotMatrix);
      rotation = DDrot(DDName(rotstr, idNameSpace), rotMatrix);
    } else {
      delete rotMatrix;
    }

    // position the child :

    DDpos(child, mother, copy, translation, rotation);
    DCOUT('a', "DDPixFwdBlades: " << child << " Copy " << copy << " positioned in " << mother << " at " << translation << " with rotation " << rotation);
  }

  // End of cycle over Phi positions

}

// -- Helpers :  -------------------------------------------------------------------------

int DDPixFwdBlades::issueCopyNumber() {
  if (copyNumbers.count(childName) == 0) copyNumbers[childName] = 0;
  return ++copyNumbers[childName];
}

// ---------------------------------------------------------------------------------------

