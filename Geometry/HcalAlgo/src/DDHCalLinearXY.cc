///////////////////////////////////////////////////////////////////////////////
// File: DDHCalLinearXY.cc
// Description: Position nxXny copies at given intervals along x and y axis
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/interface/DDHCalLinearXY.h"


DDHCalLinearXY::DDHCalLinearXY() {
  LogDebug("HCalGeom") <<"DDHCalLinearXY info: Creating an instance";
}

DDHCalLinearXY::~DDHCalLinearXY() {}

void DDHCalLinearXY::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments &) {

  numberX   = int(nArgs["NumberX"]);
  deltaX    = nArgs["DeltaX"];
  numberY   = int(nArgs["NumberY"]);
  deltaY    = nArgs["DeltaY"];
  centre    = vArgs["Center"];
  
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  LogDebug("HCalGeom") << "DDHCalLinearXY debug: Parent " << parentName
		       << "\tChild " << childName << " NameSpace " 
		       << idNameSpace << "\tNumber along X/Y " << numberX
		       << "/" << numberY << "\tDelta along X/Y " << deltaX
		       << "/" << deltaY << "\tCentre " << centre[0] << ", " 
		       << centre[1] << ", "	<< centre[2];
}

void DDHCalLinearXY::execute(DDCompactView& cpv) {

  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  DDRotation rot;
  double xoff = centre[0] - (numberX-1)*deltaX/2.;
  double yoff = centre[1] - (numberY-1)*deltaY/2.;
  int    copy = 0;

  for (int i=0; i<numberX; i++) {
    for (int j=0; j<numberY; j++) {
	
      DDTranslation tran(xoff+i*deltaX,yoff+j*deltaY,centre[2]);
      copy++;
     cpv.position(child, mother, copy, tran, rot);
      LogDebug("HCalGeom") << "DDHCalLinearXY test: " << child 
			   << " number " << copy << " positioned in "
			   << mother << " at " << tran << " with " << rot;
    }
  }
}
