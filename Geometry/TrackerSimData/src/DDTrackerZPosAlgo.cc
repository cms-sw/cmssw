#define DEBUG 0
#define COUT if (DEBUG) cout
///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerZPosAlgo.cc
// Description: Position n copies at given z-values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "Geometry/TrackerSimData/interface/DDTrackerZPosAlgo.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTrackerZPosAlgo::DDTrackerZPosAlgo() {
  COUT << "DDTrackerZPosAlgo info: Creating an instance" << endl;
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
  COUT << "DDTrackerZPosAlgo debug: Parent " << parentName 
		<< "\tChild " << childName << " NameSpace " << idNameSpace 
		<< "\tCopyNo (Start/Increment) " << startCopyNo << ", " 
		<< incrCopyNo << "\tNumber " << zvec.size() << endl;
  for (unsigned int i = 0; i < zvec.size(); i++) {
    COUT << " Z = " << zvec[i] << ", Rot.Matrix = " << rotMat[i]
		  << "; ";
    if (i%3 == 2) COUT << endl;
  }
  if ((zvec.size())%3 != 0) COUT << endl;
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
    COUT << "DDTrackerZPosAlgo test: " << child << " number " << copy
		 << " positioned in " << mother << " at " << tran << " with "
		 << rot << endl;
    copy += incrCopyNo;
  }
}
