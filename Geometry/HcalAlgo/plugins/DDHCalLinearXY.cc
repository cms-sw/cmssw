///////////////////////////////////////////////////////////////////////////////
// File: DDHCalLinearXY.cc
// Description: Position nxXny copies at given intervals along x and y axis
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/plugins/DDHCalLinearXY.h"


DDHCalLinearXY::DDHCalLinearXY() {
  LogDebug("HCalGeom") <<"DDHCalLinearXY info: Creating an instance";
}

DDHCalLinearXY::~DDHCalLinearXY() {}

void DDHCalLinearXY::initialize(const DDNumericArguments & nArgs,
				const DDVectorArguments & vArgs,
				const DDMapArguments & ,
				const DDStringArguments & sArgs,
				const DDStringVectorArguments & vsArgs) {

  numberX   = int(nArgs["NumberX"]);
  deltaX    = nArgs["DeltaX"];
  numberY   = int(nArgs["NumberY"]);
  deltaY    = nArgs["DeltaY"];
  centre    = vArgs["Center"];
  
  idNameSpace = DDCurrentNamespace::ns();
  childName   = vsArgs["Child"]; 
  DDName parentName = parent().name();
  LogDebug("HCalGeom") << "DDHCalLinearXY debug: Parent " << parentName
		       << "\twith " << childName.size() << " children";
  for (unsigned int i=0; i<childName.size(); ++i) 
    LogDebug("HCalGeom") << "DDHCalLinearXY debug: Child[" << i << "] = "
			 << childName[i];
  LogDebug("HCalGeom") << "DDHCalLinearXY debug: NameSpace " 
		       << idNameSpace << "\tNumber along X/Y " << numberX
		       << "/" << numberY << "\tDelta along X/Y " << deltaX
		       << "/" << deltaY << "\tCentre " << centre[0] << ", " 
		       << centre[1] << ", "	<< centre[2];
}

void DDHCalLinearXY::execute(DDCompactView& cpv) {

  DDName mother = parent().name();
  DDName child;
  DDRotation rot;
  double xoff = centre[0] - (numberX-1)*deltaX/2.;
  double yoff = centre[1] - (numberY-1)*deltaY/2.;
  int    copy = 0;

  for (int i=0; i<numberX; i++) {
    for (int j=0; j<numberY; j++) {
	
      DDTranslation tran(xoff+i*deltaX,yoff+j*deltaY,centre[2]);
      bool     place = true;
      unsigned int k = copy;
      if (childName.size() == 1) k = 0;
      if (k < childName.size() && (childName[k] != " " && childName[k] != "Null")) {
	child = DDName(DDSplit(childName[k]).first, DDSplit(childName[k]).second);
      } else {
	place = false;
      }
      copy++;
      if (place) {
	cpv.position(child, mother, copy, tran, rot);
	LogDebug("HCalGeom") << "DDHCalLinearXY test: " << child 
			     << " number " << copy << " positioned in "
			     << mother << " at " << tran << " with " << rot;
      } else {
	LogDebug("HCalGeom") << "DDHCalLinearXY test: No child placed for ["
			     << copy << "]";
      }
    }
  }
}
