///////////////////////////////////////////////////////////////////////////////
// File: DDShashlikNoTaperModule.cc
// Description: Create a Shashlik module (using box)
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "Geometry/HGCalCommonData/plugins/DDShashlikNoTaperModule.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDShashlikNoTaperModule::DDShashlikNoTaperModule() {
  edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule test: Creating an instance";
}

DDShashlikNoTaperModule::~DDShashlikNoTaperModule() {}

void DDShashlikNoTaperModule::initialize(const DDNumericArguments & nArgs,
					 const DDVectorArguments & vArgs,
					 const DDMapArguments & ,
					 const DDStringArguments & sArgs,
					 const DDStringVectorArguments & ) {

  activeMat      = sArgs["ActiveMaterial"];
  activeName     = sArgs["ActiveName"];
  activeLayers   = int (nArgs["ActiveLayers"]);
  activeThick    = nArgs["ActiveThickness"];
  absorbMat      = sArgs["AbsorberMaterial"];
  absorbName     = sArgs["AbsorberName"];
  absorbThick    = nArgs["AbsorberThickness"];
  widthFront     = nArgs["WidthFront"];
  widthBack      = nArgs["WidthBack"];
  moduleThick    = nArgs["ModuleThickness"];
  holeR          = nArgs["HoleRadius"];
  fibreMat       = sArgs["FibreMaterial"];
  fibreName      = sArgs["FibreName"];
  holeX          = vArgs["HoleX"];
  holeY          = vArgs["HoleY"];
  calibFibreName = sArgs["CalibFibreName"];
  calibFibrePars = vArgs["CalibFibreParameters"];
  edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:: Active: " 
			    << activeLayers << " of " << activeName << " with "
			    << activeMat << " thickness " << activeThick 
			    << " Absorber: " << absorbName << " with " 
			    << absorbMat << " thickness " << absorbThick 
			    << " width " << widthFront << "|" << widthBack 
			    << " module size " << moduleThick << holeX.size()
			    << " holes of radius " << holeR << " for fibres "
			    << fibreName << " with " << fibreMat 
			    << " Calibration fibre " << calibFibreName 
			    << " with parameters " << calibFibrePars[0] << ":" 
			    << calibFibrePars[1] << ":" << calibFibrePars[2];
  for (unsigned int i=0; i<holeX.size(); ++i) {
    edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule: Hole[" << i 
			      << "] at (" << holeX[i] << ", " << holeY[i] 
			      << ")";
  }
  idNameSpace = DDCurrentNamespace::ns();
  edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule: NameSpace " 
			    << idNameSpace << "\tParent " << parent().name();
}

void DDShashlikNoTaperModule::execute(DDCompactView& cpv) {

  std::string  baseName1  = DDSplit(activeName).first;
  std::string  baseName2  = DDSplit(absorbName).first;
  double       dx1        = 0.5*widthFront;
  DDName       name       = DDName(baseName1, DDSplit(activeName).second);
  DDSolid      solid      = DDSolidFactory::box(name, dx1,dx1, 0.5*activeThick);
  DDMaterial   matter     = DDMaterial(DDName(DDSplit(activeMat).first, 
					      DDSplit(activeMat).second));
  DDLogicalPart active(solid.ddname(), matter, solid);
  edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:: "<< name
			    << " box made of " << matter.name() << " dim: " 
			    << dx1 << ":" << dx1 << ":" << 0.5*activeThick;
  name                    = DDName(baseName2, DDSplit(absorbName).second);
  solid                   = DDSolidFactory::box(name, dx1,dx1, 0.5*absorbThick);
  matter                  = DDMaterial(DDName(DDSplit(absorbMat).first, 
					      DDSplit(absorbMat).second));
  DDLogicalPart absorb(solid.ddname(), matter, solid);
  edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:: "<< name
			    << " box made of " << matter.name() << " dim: " 
			    << dx1 << ":" << dx1 << ":" << 0.5*absorbThick;
  matter                  = DDMaterial(DDName(DDSplit(fibreMat).first, 
					      DDSplit(fibreMat).second));
  DDRotation   rot;
  if (holeX.size() > 0) {
    name      = DDName(DDSplit(fibreName).first+"inActive", 
		       DDSplit(fibreName).second);
    solid     = DDSolidFactory::tubs(name, 0.5*activeThick, 0, holeR,
				     0, CLHEP::twopi);
    DDLogicalPart fibre1(solid.ddname(), matter, solid);
    edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:: " << name
			      << " tube made of " << matter.name() << " dim: "
			      << 0.5*activeThick << ":0:" << holeR << ":0:"
			      << CLHEP::twopi;
    name      = DDName(DDSplit(fibreName).first+"inAbsorber", 
		       DDSplit(fibreName).second);
    solid     = DDSolidFactory::tubs(name, 0.5*absorbThick, 0, holeR,
				     0, CLHEP::twopi);
    DDLogicalPart fibre2(solid.ddname(), matter, solid);
    edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:: " << name
			      << " tube made of " << matter.name() << " dim: "
			      << 0.5*absorbThick << ":0:" << holeR << ":0:"
			      << CLHEP::twopi;
    for (unsigned int k=0; k<holeX.size(); ++k) {
      DDTranslation tran(holeX[k],holeY[k],0);
      cpv.position(fibre1, active, k+1, tran, rot);   
      edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:" << fibre1.name()
				<< " no " << k+1 << " positioned in " 
				<< active.name() << " at " << tran 
				<< " with no rotation";
      cpv.position(fibre2, absorb, k+1, tran, rot);   
      edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:" << fibre2.name()
				<< " no " << k+1 << " positioned in " 
				<< absorb.name() << " at " << tran 
				<< " with no rotation";
    }
  }
  name    = DDName(DDSplit(calibFibreName).first+"inActive", 
		   DDSplit(calibFibreName).second);
  solid   = DDSolidFactory::tubs(name, 0.5*activeThick, 0,
				 calibFibrePars[0], 0, CLHEP::twopi);
  DDLogicalPart fibre1(solid.ddname(), matter, solid);
  edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:: " << name
			    << " tube made of " << matter.name() << " dim: "
			    << 0.5*activeThick << ":0:" << calibFibrePars[0]
			    << ":0:" << CLHEP::twopi;
  name    = DDName(DDSplit(calibFibreName).first+"inAbsorber", 
		   DDSplit(calibFibreName).second);
  solid   = DDSolidFactory::tubs(name, 0.5*absorbThick, 0,
				 calibFibrePars[0], 0, CLHEP::twopi);
  DDLogicalPart fibre2(solid.ddname(), matter, solid);
  edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:: " << name
			    << " tube made of " << matter.name() << " dim: "
			    << 0.5*absorbThick << ":0:" << calibFibrePars[0]
			    << ":0:" << CLHEP::twopi;
  DDTranslation tranc(calibFibrePars[1],calibFibrePars[2],0);
  cpv.position(fibre1, active, 1, tranc, rot);   
  edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:" << fibre1.name() 
			    << " no 1 positioned in " << active.name() 
			    << " at " << tranc << " with no rotation";
  cpv.position(fibre2, absorb, 1, tranc, rot);   
  edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:" << fibre2.name() 
			    << " no 1 positioned in " << absorb.name() 
			    << " at " << tranc << " with no rotation";

  for (int ii=0; ii<activeLayers; ii++) {
    double z = -0.5*(moduleThick-activeThick)+ii*(activeThick+absorbThick);
    DDTranslation tran(0,0,z);
    cpv.position(active, parent(), ii+1, tran, rot);   
    edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:" << active.name() 
			      << " no " << ii+1 << " positioned in " 
			      << parent().name() << " at " << tran 
			      << " with no rotation";
    if (ii < activeLayers-1) {
      z       += 0.5*(activeThick+absorbThick);
      DDTranslation tran(0,0,z);
      cpv.position(absorb, parent(), ii+1, tran, rot);   
      edm::LogInfo("HGCalGeom") << "DDShashlikNoTaperModule:" << absorb.name()
				<< " no " << ii+1 << " positioned in " 
				<< parent().name() << " at " << tran 
				<< " with no rotation";
    }
  }
}

