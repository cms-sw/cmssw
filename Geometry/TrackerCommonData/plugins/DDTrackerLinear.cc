///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerLinear.cc
// Description: Position n copies at given intervals along an axis
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerLinear.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTrackerLinear::DDTrackerLinear() : startcn(1), incrcn(1) {
  LogDebug("TrackerGeom") << "DDTrackerLinear info: Creating an instance";
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
  if ( nArgs.find("StartCopyNo") != nArgs.end() ) {
    startcn = size_t(nArgs["StartCopyNo"]);
  } else {
    startcn = 1;
  }
  if ( nArgs.find("IncrCopyNo") != nArgs.end() ) {
    incrcn = int(nArgs["IncrCopyNo"]);
  } else {
    incrcn = 1;
  }
  
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerLinear debug: Parent " << parentName 
			  << "\tChild " << childName << " NameSpace " 
			  << DDCurrentNamespace() << "\tNumber " << number 
			  << "\tAxis (theta/phi) " << theta/CLHEP::deg << ", "
			  << phi/CLHEP::deg << "\t(Offset/Delta) " << offset 
			  << ", "  << delta << "\tCentre " << centre[0] << ", "
			  << centre[1] << ", " << centre[2] << "\tRotation "
			  << rotMat;
}

void DDTrackerLinear::execute(DDCompactView& cpv) {

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
  int ci = startcn;
  for (int i=0; i<number; i++) {
	
    DDTranslation tran = base + (offset + double(i)*delta)*direction;
   cpv.position(child, mother, ci, tran, rot);
    LogDebug("TrackerGeom") << "DDTrackerLinear test: " << child << " number "
			    << ci << " positioned in " << mother << " at "
			    << tran << " with " << rot;
    ++ci;
  }
}
