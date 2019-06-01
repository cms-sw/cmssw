///////////////////////////////////////////////////////////////////////////////
// File: DDHCalForwardAlgo.cc
// Description: Cable mockup between barrel and endcap gap
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/plugins/DDHCalForwardAlgo.h"

//#define EDM_ML_DEBUG

DDHCalForwardAlgo::DDHCalForwardAlgo() : number(0), size(0), type(0) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalForwardAlgo: Creating an instance";
#endif
}

DDHCalForwardAlgo::~DDHCalForwardAlgo() {}

void DDHCalForwardAlgo::initialize(const DDNumericArguments& nArgs,
                                   const DDVectorArguments& vArgs,
                                   const DDMapArguments&,
                                   const DDStringArguments& sArgs,
                                   const DDStringVectorArguments& vsArgs) {
  cellMat = sArgs["CellMaterial"];
  cellDx = nArgs["CellDx"];
  cellDy = nArgs["CellDy"];
  cellDz = nArgs["CellDz"];
  startY = nArgs["StartY"];

  childName = vsArgs["Child"];
  number = dbl_to_int(vArgs["Number"]);
  size = dbl_to_int(vArgs["Size"]);
  type = dbl_to_int(vArgs["Type"]);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalForwardAlgo: Cell material " << cellMat << "\tCell Size " << cellDx << ", "
                               << cellDy << ", " << cellDz << "\tStarting Y " << startY << "\tChildren " << childName[0]
                               << ", " << childName[1] << "\n               "
                               << "          Cell positioning done for " << number.size() << " times";
  for (unsigned int i = 0; i < number.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\t" << i << " Number of children " << size[i] << " occurence " << number[i]
                                 << " first child index " << type[i];
#endif
  idNameSpace = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalForwardAlgo debug: Parent " << parent().name() << " NameSpace " << idNameSpace;
#endif
}

void DDHCalForwardAlgo::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "==>> Constructing DDHCalForwardAlgo...";
#endif
  DDName parentName = parent().name();
  double ypos = startY;
  int box = 0;

  for (unsigned int i = 0; i < number.size(); i++) {
    double dx = cellDx * size[i];
    int indx = type[i];
    for (int j = 0; j < number[i]; j++) {
      box++;
      std::string name = parentName.name() + std::to_string(box);
      DDSolid solid = DDSolidFactory::box(DDName(name, idNameSpace), dx, cellDy, cellDz);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalForwardAlgo: " << DDName(name, idNameSpace) << " Box made of " << cellMat
                                   << " of Size " << dx << ", " << cellDy << ", " << cellDz;
#endif
      DDName matname(DDSplit(cellMat).first, DDSplit(cellMat).second);
      DDMaterial matter(matname);
      DDLogicalPart genlogic(solid.ddname(), matter, solid);

      DDTranslation r0(0.0, ypos, 0.0);
      DDRotation rot;
      cpv.position(solid.ddname(), parentName, box, r0, rot);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalForwardAlgo: " << solid.ddname() << " number " << box << " positioned in "
                                   << parentName << " at " << r0 << " with " << rot;
#endif
      DDName child(DDSplit(childName[indx]).first, DDSplit(childName[indx]).second);
      double xpos = -dx + cellDx;
      ypos += 2 * cellDy;
      indx = 1 - indx;

      for (int k = 0; k < size[i]; k++) {
        DDTranslation r1(xpos, 0.0, 0.0);
        cpv.position(child, solid.ddname(), k + 1, r1, rot);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalForwardAlgo: " << child << " number " << k + 1 << " positioned in "
                                     << solid.ddname() << " at " << r1 << " with " << rot;
#endif
        xpos += 2 * cellDx;
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "<<== End of DDHCalForwardAlgo construction";
#endif
}
