///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalWaferAlgo.cc
// Description: Position inside the mother according to (eta,phi) 
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalWaferAlgo.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

DDHGCalWaferAlgo::DDHGCalWaferAlgo() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HGCalGeom") << "DDHGCalWaferAlgo test: Creating an instance";
#endif
}

DDHGCalWaferAlgo::~DDHGCalWaferAlgo() {}

void DDHGCalWaferAlgo::initialize(const DDNumericArguments & nArgs,
				  const DDVectorArguments & vArgs,
				  const DDMapArguments & ,
				  const DDStringArguments & sArgs,
				  const DDStringVectorArguments & vsArgs) {

  cellSize    = nArgs["CellSize"];
  cellType    = (int)(nArgs["CellType"]);
  childNames  = vsArgs["ChildNames"];
  positionX   = dbl_to_int(vArgs["PositionX"]);
  positionY   = dbl_to_int(vArgs["PositionY"]);
  angles      = vArgs["Angles"];
  detectorType= dbl_to_int(vArgs["DetectorType"]);
#ifdef EDM_ML_DEBUG
  std::cout << childNames.size() << " children: " << childNames[0] << "; "
	    << childNames[1] << " positioned " << positionX.size() 
	    << " times with cell size " << cellSize << std::endl;
  for (unsigned int k=0; k<positionX.size(); ++k)
    std::cout << "[" << k << "] x " << positionX[k] << " y "  << positionY[k]
	      << " angle " << angles[k] << " detector " << detectorType[k]
	      << std::endl;
#endif
  rotns       = sArgs["RotNameSpace"];
  DDCurrentNamespace ns;
  idNameSpace = *ns;
  parentName  = parent().name(); 
#ifdef EDM_ML_DEBUG
  std::cout << "DDHGCalWaferAlgo debug: Parent " << parentName << " NameSpace "
	    << idNameSpace << " for Rotation " << rotns << std::endl;
#endif
}

void DDHGCalWaferAlgo::execute(DDCompactView& cpv) {
  
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HGCalGeom") << "==>> Constructing DDHGCalWaferAlgo...";
#endif
  double dx = 0.5*cellSize;
  double dy = 0.5*dx*tan(30.0*CLHEP::deg);

  for (unsigned int k=0; k<positionX.size(); ++k) {
    std::string name(childNames[detectorType[k]]);
    DDRotation rotation;
    std::string rotstr("NULL");
    if (angles[k] != 0) {
      if (angles[k] >=0 && angles[k] < 100) rotstr = "R0"; 
      else                                  rotstr = "R"; 
      rotstr = rotstr + std::to_string(angles[k]);
      rotation = DDRotation(DDName(rotstr, rotns)); 
      if (!rotation) {
#ifdef EDM_ML_DEBUG
	std::cout << "DDHGCalWaferAlgo: Creating new rotation "
		  << DDName(rotstr, rotns) << "\t90, " << angles[k] << ", 90, "
		  << (angles[k]+90) << ", 0, 0" << std::endl;
#endif
        rotation = DDrot(DDName(rotstr, rotns), 90*CLHEP::deg, 
                         angles[k]*CLHEP::deg, 90*CLHEP::deg, 
                         (90+angles[k])*CLHEP::deg, 0*CLHEP::deg,0*CLHEP::deg);
      }
    }
    double xpos = dx*positionX[k];
    double ypos = dy*positionY[k];
    DDTranslation tran(xpos, ypos, 0);
    int copy = cellType*1000+k;
    cpv.position(DDName(name,idNameSpace), parentName, copy, tran, rotation);
#ifdef EDM_ML_DEBUG
    std::cout << "DDHGCalWaferAlgo: " << DDName(name,idNameSpace) << " number "
	      << copy << " positioned in " << parentName << " at " << tran 
	      << " with " << rotation << std::endl;
#endif
  }
}
