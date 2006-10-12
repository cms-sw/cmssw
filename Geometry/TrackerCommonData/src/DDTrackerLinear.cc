///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerLinear.cc
// Description: Position n copies at given intervals along an axis
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerLinear.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTrackerLinear::DDTrackerLinear() {
  edm::LogInfo("TrackerGeom") << "DDTrackerLinear info: Creating an instance";
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
  LogDebug("TrackerGeom") << "DDTrackerLinear debug: Parent " << parentName 
			  << "\tChild " << childName << " NameSpace " 
			  << idNameSpace << "\tNumber " << number 
			  << "\tAxis (theta/phi) " << theta/deg << ", "
			  << phi/deg << "\t(Offset/Delta) " << offset << ", " 
			  << delta << "\tCentre " << centre[0] << ", " 
			  << centre[1] << ", " << centre[2] << "\tRotation "
			  << rotMat;
}

void DDTrackerLinear::execute() {

  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  DDTranslation direction(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));
  DDTranslation base(centre[0],centre[1],centre[2]);
  std::string rotstr = DDSplit(rotMat).first;
  DDRotation rot;
  if (rotstr != "NULL") {
    std::string rotns  = DDSplit(rotMat).second;
    rot = DDRotation(DDName(rotstr, rotns));
  }

  for (int i=0; i<number; i++) {
	
    DDTranslation tran = base + (offset + double(i)*delta)*direction;
    DDpos (child, mother, i+1, tran, rot);
    LogDebug("TrackerGeom") << "DDTrackerLinear test: " << child << " number "
			    << i+1 << " positioned in " << mother << " at "
			    << tran << " with " << rot;
  }
}
