///////////////////////////////////////////////////////////////////////////////
// File: DDHCalLinearXY.cc
// Description: Position nxXny copies at given intervals along x and y axis
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"

//#define EDM_ML_DEBUG

class DDHCalLinearXY : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDHCalLinearXY();
  ~DDHCalLinearXY() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  std::string idNameSpace;             //Namespace of this and ALL sub-parts
  std::vector<std::string> childName;  //Child name
  int numberX;                         //Number of positioning along X-axis
  double deltaX;                       //Increment               .........
  int numberY;                         //Number of positioning along Y-axis
  double deltaY;                       //Increment               .........
  std::vector<double> centre;          //Centre
};

DDHCalLinearXY::DDHCalLinearXY() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalLinearXY: Creating an instance";
#endif
}

DDHCalLinearXY::~DDHCalLinearXY() {}

void DDHCalLinearXY::initialize(const DDNumericArguments& nArgs,
                                const DDVectorArguments& vArgs,
                                const DDMapArguments&,
                                const DDStringArguments& sArgs,
                                const DDStringVectorArguments& vsArgs) {
  numberX = int(nArgs["NumberX"]);
  deltaX = nArgs["DeltaX"];
  numberY = int(nArgs["NumberY"]);
  deltaY = nArgs["DeltaY"];
  centre = vArgs["Center"];

  idNameSpace = DDCurrentNamespace::ns();
  childName = vsArgs["Child"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalLinearXY: Parent " << parent().name() << "\twith " << childName.size()
                               << " children";
  for (unsigned int i = 0; i < childName.size(); ++i)
    edm::LogVerbatim("HCalGeom") << "DDHCalLinearXY: Child[" << i << "] = " << childName[i];
  edm::LogVerbatim("HCalGeom") << "DDHCalLinearXY: NameSpace " << idNameSpace << "\tNumber along X/Y " << numberX << "/"
                               << numberY << "\tDelta along X/Y " << deltaX << "/" << deltaY << "\tCentre " << centre[0]
                               << ", " << centre[1] << ", " << centre[2];
#endif
}

void DDHCalLinearXY::execute(DDCompactView& cpv) {
  DDName mother = parent().name();
  DDName child;
  DDRotation rot;
  double xoff = centre[0] - (numberX - 1) * deltaX / 2.;
  double yoff = centre[1] - (numberY - 1) * deltaY / 2.;
  int copy = 0;

  for (int i = 0; i < numberX; i++) {
    for (int j = 0; j < numberY; j++) {
      DDTranslation tran(xoff + i * deltaX, yoff + j * deltaY, centre[2]);
      bool place = true;
      unsigned int k = copy;
      if (childName.size() == 1)
        k = 0;
      if (k < childName.size() && (childName[k] != " " && childName[k] != "Null")) {
        child = DDName(DDSplit(childName[k]).first, DDSplit(childName[k]).second);
      } else {
        place = false;
      }
      copy++;
      if (place) {
        cpv.position(child, mother, copy, tran, rot);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalLinearXY: " << child << " number " << copy << " positioned in " << mother
                                     << " at " << tran << " with " << rot;
#endif
      } else {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalLinearXY: No child placed for [" << copy << "]";
#endif
      }
    }
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHCalLinearXY, "hcal:DDHCalLinearXY");
