///////////////////////////////////////////////////////////////////////////////
// File: DDShashlikTaperModule.cc
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
#include "Geometry/HGCalCommonData/plugins/DDShashlikTaperModule.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDShashlikTaperModule::DDShashlikTaperModule() {
  edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule test: Creating an instance";
}

DDShashlikTaperModule::~DDShashlikTaperModule() {}

void DDShashlikTaperModule::initialize(const DDNumericArguments & nArgs,
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
  edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:: Active: " 
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
    edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule: Hole[" << i 
			      << "] at (" << holeX[i] << ", " << holeY[i] 
			      << ")";
  }
  idNameSpace = DDCurrentNamespace::ns();
  edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule: NameSpace " 
			    << idNameSpace << "\tParent " << parent().name();
}

void DDShashlikTaperModule::execute(DDCompactView& cpv) {

  std::string  baseName1  = DDSplit(activeName).first;
  std::string  baseName2  = DDSplit(absorbName).first;
  DDMaterial   matter     = DDMaterial(DDName(DDSplit(fibreMat).first, 
					      DDSplit(fibreMat).second));
  DDName       name       = DDName(DDSplit(fibreName).first+"inActive", 
				   DDSplit(fibreName).second);
  DDSolid      solid      = DDSolidFactory::tubs(name, 0.5*activeThick, 0, 
						 holeR, 0, CLHEP::twopi);
  DDLogicalPart fibre1(solid.ddname(), matter, solid);
  edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:: " << name
			    << " tube made of " << matter.name() << " dim: "
			    << 0.5*activeThick << ":0:" << holeR << ":0:"
			    << CLHEP::twopi;
  name                    = DDName(DDSplit(fibreName).first+"inAbsorber", 
				   DDSplit(fibreName).second);
  solid                   = DDSolidFactory::tubs(name, 0.5*absorbThick, 0,
						 holeR, 0, CLHEP::twopi);
  DDLogicalPart fibre2(solid.ddname(), matter, solid);
  edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:: " << name
			    << " tube made of " << matter.name() << " dim: "
			    << 0.5*absorbThick << ":0:" << holeR << ":0:"
			    << CLHEP::twopi;
  name                    = DDName(DDSplit(calibFibreName).first+"inActive", 
				   DDSplit(calibFibreName).second);
  solid                   = DDSolidFactory::tubs(name, 0.5*activeThick, 0,
						 calibFibrePars[0], 0, 
						 CLHEP::twopi);
  DDLogicalPart fibre3(solid.ddname(), matter, solid);
  edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:: " << name
			    << " tube made of " << matter.name() << " dim: "
			    << 0.5*activeThick << ":0:" << calibFibrePars[0]
			    << ":0:" << CLHEP::twopi;
  name                    = DDName(DDSplit(calibFibreName).first+"inAbsorber", 
				   DDSplit(calibFibreName).second);
  solid                   = DDSolidFactory::tubs(name, 0.5*absorbThick, 0,
						 calibFibrePars[0], 0, 
						 CLHEP::twopi);
  DDLogicalPart fibre4(solid.ddname(), matter, solid);
  edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:: " << name
			    << " tube made of " << matter.name() << " dim: "
			    << 0.5*absorbThick << ":0:" << calibFibrePars[0]
			    << ":0:" << CLHEP::twopi;

  DDRotation   rot;
  double dWidth = (widthBack-widthFront)/moduleThick;
  for (int ii=0; ii<activeLayers; ii++) {
    name = DDName(baseName1+int_to_string(ii), DDSplit(activeName).second);
    double dx1        = 0.5*(widthFront+dWidth*ii*(activeThick+absorbThick));
    double dx2        = dx1+0.5*dWidth*activeThick;
    solid             = DDSolidFactory::trap(name,0.5*activeThick, 0, 0, dx1,
					     dx1, dx1, 0, dx2, dx2, dx2, 0);
    matter            = DDMaterial(DDName(DDSplit(activeMat).first, 
					  DDSplit(activeMat).second));
    DDLogicalPart active(solid.ddname(), matter, solid);
    edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:: " << name 
			      << " trap made of " << matter.name() <<" of dim "
			      << 0.5*activeThick << ":0:0:" << dx1 << ":"
			      << dx1 << ":" << dx1 << ":0:" << dx2 << ":" 
			      << dx2 << ":" << dx2 <<":0";
    for (unsigned int k=0; k<holeX.size(); ++k) {
      DDTranslation tran(holeX[k],holeY[k],0);
      cpv.position(fibre1, active, k+1, tran, rot);   
      edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:" << fibre1.name()
				<< " no " << k+1 << " positioned in " 
				<< active.name() << " at " << tran 
				<< " with no rotation";
    }
    DDTranslation tranc(calibFibrePars[1],calibFibrePars[2],0);
    cpv.position(fibre3, active, 1, tranc, rot);   
    edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:" << fibre3.name() 
			      << " no 1 positioned in " << active.name() 
			      << " at " << tranc << " with no rotation";
    double z = -0.5*(moduleThick-activeThick)+ii*(activeThick+absorbThick);
    DDTranslation tran(0,0,z);
    cpv.position(active, parent(), ii+1, tran, rot);   
    edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:" << active.name() 
			      << " no " << ii+1 << " positioned in " 
			      << parent().name() << " at " << tran 
			      << " with no rotation";
    if (ii < activeLayers-1) {
      name              = DDName(baseName2+int_to_string(ii), 
				 DDSplit(absorbName).second);
      dx1               = dx2;
      dx2               = dx1+0.5*dWidth*absorbThick;
      solid             = DDSolidFactory::trap(name,0.5*absorbThick, 0, 0, dx1,
					       dx1, dx1, 0, dx2, dx2, dx2, 0);
      matter            = DDMaterial(DDName(DDSplit(absorbMat).first, 
					    DDSplit(absorbMat).second));
      DDLogicalPart absorb(solid.ddname(), matter, solid);
      edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:: " << name 
				<< " trap made of " << matter.name()<<" of dim "
				<< 0.5*absorbThick << ":0:0:" << dx1 << ":"
				<< dx1 << ":" << dx1 << ":0:" << dx2 << ":" 
				<< dx2 << ":" << dx2 <<":0";
      for (unsigned int k=0; k<holeX.size(); ++k) {
	DDTranslation tran(holeX[k],holeY[k],0);
	cpv.position(fibre2, absorb, k+1, tran, rot);   
	edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:" << fibre2.name()
				  << " no " << k+1 << " positioned in " 
				  << absorb.name() << " at " << tran 
				  << " with no rotation";
      }
      DDTranslation tranc(calibFibrePars[1],calibFibrePars[2],0);
      cpv.position(fibre4, absorb, 1, tranc, rot);   
      edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:" << fibre4.name() 
				<< " no 1 positioned in " << absorb.name() 
				<< " at " << tranc << " with no rotation";
      z       += 0.5*(activeThick+absorbThick);
      DDTranslation tran(0,0,z);
      cpv.position(absorb, parent(), ii+1, tran, rot);   
      edm::LogInfo("HGCalGeom") << "DDShashlikTaperModule:" << absorb.name()
				<< " no " << ii+1 << " positioned in " 
				<< parent().name() << " at " << tran 
				<< " with no rotation";
    }
  }
}

