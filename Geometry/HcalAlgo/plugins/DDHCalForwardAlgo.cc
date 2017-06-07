///////////////////////////////////////////////////////////////////////////////
// File: DDHCalForwardAlgo.cc
// Description: Cable mockup between barrel and endcap gap
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/plugins/DDHCalForwardAlgo.h"

DDHCalForwardAlgo::DDHCalForwardAlgo(): number(0),size(0),type(0) {
  LogDebug("HCalGeom") << "DDHCalForwardAlgo info: Creating an instance";
}

DDHCalForwardAlgo::~DDHCalForwardAlgo() {}


void DDHCalForwardAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & vsArgs) {

  cellMat     = sArgs["CellMaterial"];
  cellDx      = nArgs["CellDx"];
  cellDy      = nArgs["CellDy"];
  cellDz      = nArgs["CellDz"];
  startY      = nArgs["StartY"];

  childName   = vsArgs["Child"];
  number      = dbl_to_int(vArgs["Number"]);
  size        = dbl_to_int(vArgs["Size"]);
  type        = dbl_to_int(vArgs["Type"]);

  LogDebug("HCalGeom") << "DDHCalForwardAlgo debug: Cell material " << cellMat
		       << "\tCell Size "  << cellDx << ", " << cellDy << ", "
		       << cellDz << "\tStarting Y " << startY << "\tChildren "
		       << childName[0] << ", " << childName[1] << "\n"
		       << "                         Cell positioning done for "
		       << number.size() << " times";
  for (unsigned int i = 0; i < number.size(); i++)
    LogDebug("HCalGeom") << "\t" << i << " Number of children " << size[i] 
			 << " occurence " << number[i] << " first child index "
			 << type[i];

  idNameSpace = DDCurrentNamespace::ns();
  DDName parentName = parent().name(); 
  LogDebug("HCalGeom") << "DDHCalForwardAlgo debug: Parent " << parentName
		       << " NameSpace " << idNameSpace;
}

void DDHCalForwardAlgo::execute(DDCompactView& cpv) {
  
  LogDebug("HCalGeom") << "==>> Constructing DDHCalForwardAlgo...";

  DDName parentName = parent().name(); 
  double ypos       = startY;
  int    box        = 0;

  for (unsigned int i=0; i<number.size(); i++) {
    double dx   = cellDx*size[i];
    int    indx = type[i];
    for (int j=0; j<number[i]; j++) {
      box++;
      string name = parentName.name() + std::to_string(box);
      DDSolid solid = DDSolidFactory::box(DDName(name, idNameSpace),
					  dx, cellDy, cellDz);
      LogDebug("HCalGeom") << "DDHCalForwardAlgo test: " 
			   << DDName(name, idNameSpace) << " Box made of " 
			   << cellMat << " of Size " << dx << ", " << cellDy
			   << ", " << cellDz;
  
      DDName matname(DDSplit(cellMat).first, DDSplit(cellMat).second); 
      DDMaterial matter(matname);
      DDLogicalPart genlogic(solid.ddname(), matter, solid);

      DDTranslation r0(0.0, ypos, 0.0);
      DDRotation rot;
      cpv.position(solid.ddname(), parentName, box, r0, rot);
      LogDebug("HCalGeom") << "DDHCalForwardAlgo test: " << solid.ddname() 
			   << " number " << box << " positioned in " 
			   << parentName << " at " << r0 << " with " << rot;
  
      DDName child(DDSplit(childName[indx]).first, 
		   DDSplit(childName[indx]).second); 
      double xpos = -dx + cellDx;
      ypos       += 2*cellDy;
      indx        = 1 - indx;

      for (int k=0; k<size[i]; k++) {
	DDTranslation r1(xpos, 0.0, 0.0);
	cpv.position(child, solid.ddname(), k+1, r1, rot);
	LogDebug("HCalGeom") << "DDHCalForwardAlgo test: " << child 
			     << " number " << k+1 << " positioned in " 
			     << solid.ddname() << " at " << r1 << " with "
			     << rot;
	xpos += 2*cellDx;
      }
    }  
  }
  LogDebug("HCalGeom") << "<<== End of DDHCalForwardAlgo construction ...";
}
