///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerZPosAlgo.cc
// Description: Position n copies at given z-values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerZPosAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTrackerZPosAlgo::DDTrackerZPosAlgo() {
  LogDebug("TrackerGeom") <<"DDTrackerZPosAlgo info: Creating an instance";
}

DDTrackerZPosAlgo::~DDTrackerZPosAlgo() {}

void DDTrackerZPosAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & vsArgs) {

  startCopyNo = int(nArgs["StartCopyNo"]);
  incrCopyNo  = int(nArgs["IncrCopyNo"]);
  zvec        = vArgs["ZPositions"];
  rotMat      = vsArgs["Rotations"];
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerZPosAlgo debug: Parent " << parentName 
			  << "\tChild " << childName << " NameSpace " 
			  << DDCurrentNamespace() << "\tCopyNo (Start/Increment) " 
			  << startCopyNo << ", " << incrCopyNo << "\tNumber " 
			  << zvec.size();
  for (int i = 0; i < (int)(zvec.size()); i++) {
    LogDebug("TrackerGeom") << "\t[" << i << "]\tZ = " << zvec[i] 
			    << ", Rot.Matrix = " << rotMat[i];
  }
}

void DDTrackerZPosAlgo::execute(DDCompactView& cpv) {

  int    copy   = startCopyNo;
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);

  for (int i=0; i<(int)(zvec.size()); i++) {
	
    DDTranslation tran(0, 0, zvec[i]);
    std::string rotstr = DDSplit(rotMat[i]).first;
    DDRotation rot;
    if (rotstr != "NULL") {
      std::string rotns  = DDSplit(rotMat[i]).second;
      rot = DDRotation(DDName(rotstr, rotns));
    }
   cpv.position(child, mother, copy, tran, rot);
    LogDebug("TrackerGeom") << "DDTrackerZPosAlgo test: " << child <<" number "
			    << copy << " positioned in " << mother << " at "
			    << tran << " with " << rot;
    copy += incrCopyNo;
  }
}
