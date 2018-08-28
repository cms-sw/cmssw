///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerXYZPosAlgo.cc
// Description: Position n copies at given x-values, y-values and z-values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerXYZPosAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTrackerXYZPosAlgo::DDTrackerXYZPosAlgo() {
  LogDebug("TrackerGeom") <<"DDTrackerXYZPosAlgo info: Creating an instance";
}

DDTrackerXYZPosAlgo::~DDTrackerXYZPosAlgo() {}

void DDTrackerXYZPosAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & vsArgs) {

  startCopyNo = int(nArgs["StartCopyNo"]);
  incrCopyNo  = int(nArgs["IncrCopyNo"]);
  xvec        = vArgs["XPositions"];
  yvec        = vArgs["YPositions"];
  zvec        = vArgs["ZPositions"];
  rotMat      = vsArgs["Rotations"];
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerXYZPosAlgo debug: Parent " << parentName 
			  << "\tChild " << childName << " NameSpace " 
			  << DDCurrentNamespace() << "\tCopyNo (Start/Increment) " 
			  << startCopyNo << ", " << incrCopyNo << "\tNumber " 
			  << xvec.size() << ", " << yvec.size() << ", " << zvec.size();
  for (int i = 0; i < (int)(zvec.size()); i++) {
    LogDebug("TrackerGeom") << "\t[" << i << "]\tX = " << xvec[i]
                            << "\t[" << i << "]\tY = " << yvec[i] 
                            << "\t[" << i << "]\tZ = " << zvec[i] 
			    << ", Rot.Matrix = " << rotMat[i];
  }
}

void DDTrackerXYZPosAlgo::execute(DDCompactView& cpv) {

  int    copy   = startCopyNo;
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);

  for (int i=0; i<(int)(zvec.size()); i++) {
	
    DDTranslation tran(xvec[i], yvec[i], zvec[i]);
    std::string rotstr = DDSplit(rotMat[i]).first;
    DDRotation rot;
    if (rotstr != "NULL") {
      std::string rotns  = DDSplit(rotMat[i]).second;
      rot = DDRotation(DDName(rotstr, rotns));
    }
   cpv.position(child, mother, copy, tran, rot);
    LogDebug("TrackerGeom") << "DDTrackerXYZPosAlgo test: " << child 
			    <<" number " << copy << " positioned in " 
			    << mother << " at " << tran << " with " << rot;
    copy += incrCopyNo;
  }
}
