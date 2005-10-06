#define DEBUG 0
#define COUT if (DEBUG) cout
///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerLinear.cc
// Description: Position n copies at given intervals along an axis
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "Geometry/TrackerSimData/interface/DDTrackerLinear.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTrackerLinear::DDTrackerLinear() {
  COUT << "DDTrackerLinear info: Creating an instance" << endl;
}

DDTrackerLinear::~DDTrackerLinear() {}

void DDTrackerLinear::initialize(const DDNumericArguments & nArgs,
				 const DDVectorArguments & vArgs,
				 const DDMapArguments & ,
				 const DDStringArguments & sArgs,
				 const DDStringVectorArguments &) {

  number    = int(nArgs["Number"]);
  theta     = nArgs["Theta"];
  phi       = nArgs["Phi"];
  offset    = nArgs["Offset"];
  delta     = nArgs["Delta"];
  centre    = vArgs["Center"];
  rotMat    = sArgs["Rotation"];
  
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  COUT << "DDTrackerLinear debug: Parent " << parentName 
		<< "\tChild " << childName << " NameSpace " << idNameSpace 
		<< "\tNumber " << number << "\tAxis (theta/phi) " << theta/deg
		<< ", " << phi/deg << "\t(Offset/Delta) " << offset << ", " 
		<< delta << "\tCentre " << centre[0] << ", " << centre[1] 
		<< ", "	<< centre[2] << "\tRotation " << rotMat << endl;
}

void DDTrackerLinear::execute() {

  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  DDTranslation direction(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));
  DDTranslation base(centre[0],centre[1],centre[2]);
  string rotstr = DDSplit(rotMat).first;
  DDRotation rot;
  if (rotstr != "NULL") {
    string rotns  = DDSplit(rotMat).second;
    rot = DDRotation(DDName(rotstr, rotns));
  }

  for (int i=0; i<number; i++) {
	
    DDTranslation tran = base + (offset + double(i)*delta)*direction;
    DDpos (child, mother, i+1, tran, rot);
    COUT << "DDTrackerLinear test: " << child << " number " << i+1
		 << " positioned in " << mother << " at " << tran << " with "
		 << rot << endl;
  }
}
