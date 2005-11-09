///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerZPosAlgo.cc
// Description: Position n copies at given z-values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerZPosAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTrackerZPosAlgo::DDTrackerZPosAlgo() {
  DCOUT('a', "DDTrackerZPosAlgo info: Creating an instance");
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
  
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  DCOUT('A', "DDTrackerZPosAlgo debug: Parent " << parentName << "\tChild " << childName << " NameSpace " << idNameSpace << "\tCopyNo (Start/Increment) " << startCopyNo << ", " << incrCopyNo << "\tNumber " << zvec.size());
  for (unsigned int i = 0; i < zvec.size(); i++) {
    DCOUT('A', "\tZ = " << zvec[i] << ", Rot.Matrix = " << rotMat[i] << "; ");
  }
}

void DDTrackerZPosAlgo::execute() {

  int    copy   = startCopyNo;
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);

  for (unsigned int i=0; i<zvec.size(); i++) {
	
    DDTranslation tran(0, 0, zvec[i]);
    string rotstr = DDSplit(rotMat[i]).first;
    DDRotation rot;
    if (rotstr != "NULL") {
      string rotns  = DDSplit(rotMat[i]).second;
      rot = DDRotation(DDName(rotstr, rotns));
    }
    DDpos (child, mother, copy, tran, rot);
    DCOUT('a', "DDTrackerZPosAlgo test: " << child << " number " << copy << " positioned in " << mother << " at " << tran << " with " << rot);
    copy += incrCopyNo;
  }
}
