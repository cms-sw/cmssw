///////////////////////////////////////////////////////////////////////////////
// File: DDShashlikModule.cc
// Description: Create a Shashlik module
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
#include "Geometry/HGCalCommonData/plugins/DDShashlikModule.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDShashlikModule::DDShashlikModule() {
  edm::LogInfo("HGCalGeom") << "DDShashlikModule test: Creating an instance";
}

DDShashlikModule::~DDShashlikModule() {}

void DDShashlikModule::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & ) {

  activeMat      = sArgs["ActiveMaterial"];
  activeName     = sArgs["ActiveName"];
  activeLayers   = int (nArgs["ActiveLayers"]);
  activeThick    = nArgs["ActiveThickness"];
  absorbThick    = nArgs["AbsorberThickness"];
  widthFront     = nArgs["WidthFront"];
  widthBack      = nArgs["WidthBack"];
  moduleThick    = nArgs["ModuleThickness"];
  moduleTaperAngle = nArgs["ModuleTaperAngle"];
  holeR          = nArgs["HoleRadius"];
  fibreMat       = sArgs["FibreMaterial"];
  fibreName      = sArgs["FibreName"];
  holeX          = vArgs["HoleX"];
  holeY          = vArgs["HoleY"];
  calibFibreName = sArgs["CalibFibreName"];
  calibFibrePars = vArgs["CalibFibreParameters"];
  edm::LogInfo("HGCalGeom") << "DDShashlikModule:: Active: " << activeLayers 
			    << " of " << activeName << " with " << activeMat 
			    << " thickness " << activeThick <<"|"<< absorbThick
			    << " width " << widthFront << "|" << widthBack 
			    << " module size " << moduleThick 
			    << " module taper angle " << moduleTaperAngle/CLHEP::deg << "deg  "
			    << holeX.size() << " holes of radius " << holeR 
			    << " for fibres "<<fibreName <<" with "<< fibreMat
			    << " Calibration fibre " << calibFibreName 
			    << " with parameters " << calibFibrePars[0] << ":"
			    << calibFibrePars[1] << ":" << calibFibrePars[2];
  for (unsigned int i=0; i<holeX.size(); ++i) {
    edm::LogInfo("HGCalGeom") << "DDShashlikModule: Hole[" << i << "] at ("
			      << holeX[i] << ", " << holeY[i] << ")";
  }
  idNameSpace = DDCurrentNamespace::ns();
  edm::LogInfo("HGCalGeom") << "DDShashlikModule: NameSpace " << idNameSpace 
			    << "\tParent " << parent().name();
}

void DDShashlikModule::execute(DDCompactView& cpv) {

  std::string  baseName   = DDSplit(activeName).first;
  DDName       name       = DDName(baseName+"Hole", DDSplit(activeName).second);
  DDSolid      solidHole  = DDSolidFactory::tubs(name,0.5*activeThick,0,holeR,
						 0,CLHEP::twopi);
  edm::LogInfo("HGCalGeom") << "DDShashlikModule:: " << name <<" tube of dim "
			    << 0.5*activeThick << ":0:" << holeR << ":0:"
			    << CLHEP::twopi;
  name                    = DDName(baseName+"CalibHole", DDSplit(activeName).second);
  DDSolid      solidCHole = DDSolidFactory::tubs(name,0.5*activeThick,0,
						 calibFibrePars[0],0,CLHEP::twopi);
  DDRotation rot;
  edm::LogInfo("HGCalGeom") << "DDShashlikModule:: " << name <<" tube of dim "
			    << 0.5*activeThick << ":0:" << calibFibrePars[0]
			    << ":0:" << CLHEP::twopi;

  double dWidth = (widthBack-widthFront)/moduleThick;
  for (int ii=0; ii<activeLayers; ii++) {
    std::string namex = baseName+int_to_string(ii);
    name = DDName(namex+"_A", DDSplit(activeName).second);
    double dx1        = 0.5*(widthFront+dWidth*ii*(activeThick+absorbThick));
    double dx2        = dx1+0.5*dWidth*activeThick;
    DDSolid solid     = DDSolidFactory::trap(name,0.5*activeThick, 0, 0, dx1,
					     dx1, dx1, 0, dx2, dx2, dx2, 0);
    edm::LogInfo("HGCalGeom") << "DDShashlikModule:: "<< name<<" trap of dim "
			      << 0.5*activeThick << ":0:0:" << dx1 << ":"
			      << dx1 << ":" << dx1 << ":0:" << dx2 << ":" << dx2
			      << ":" << dx2 <<":0";
    for (unsigned int k=0; k<holeX.size(); ++k) {
      name = DDName(namex+"_"+int_to_string(k), DDSplit(activeName).second);
      DDTranslation tran(holeX[k],holeY[k],0);
      DDSolid solx =DDSolidFactory::subtraction(name,solid,solidHole,tran,rot);
      edm::LogInfo("HGCalGeom") << "DDShashlikModule:: Construct " << solx.name() << " by putting the hole in " << solid.name() << " at " << tran;
      solid = solx;
    }
    name = DDName(namex, DDSplit(activeName).second);
    DDTranslation tranc(calibFibrePars[1],calibFibrePars[2],0);
    DDSolid solx =DDSolidFactory::subtraction(name,solid,solidCHole,tranc,rot);
    edm::LogInfo("HGCalGeom") << "DDShashlikModule:: Construct " << solx.name() << " by putting the hole in " << solid.name() << " at " << tranc;
    solid = solx;
    DDMaterial matter = DDMaterial(DDName(DDSplit(activeMat).first, 
					  DDSplit(activeMat).second));
    DDLogicalPart active(solid.ddname(), matter, solid);
    DDTranslation tran(0,0,(-0.5*(moduleThick-activeThick)+ii*(activeThick+absorbThick)));
    cpv.position(active, parent(), ii+1, tran, rot);   
    edm::LogInfo("HGCalGeom") << "DDShashlikModule:" << active.name() <<" no "
			      << ii+1 << " positioned in " << parent().name()
			      << " at " << tran << " with no rotation";
  }

  // Now the fibres
  DDMaterial matter = DDMaterial(DDName(DDSplit(fibreMat).first, 
					DDSplit(fibreMat).second));
  if (holeX.size() > 0) {
    name       = DDName(DDSplit(fibreName).first, DDSplit(fibreName).second);
    solidHole  = DDSolidFactory::tubs(name,0.5*moduleThick,0,holeR,
				      0,CLHEP::twopi);
    DDLogicalPart fibre(solidHole.ddname(), matter, solidHole);
    edm::LogInfo("HGCalGeom") << "DDShashlikModule:: " << fibre.name() 
			      << " tube made of " << matter.name() << " dim: "
			      << 0.5*moduleThick << ":0:" << holeR << ":0:"
			      << CLHEP::twopi;
    for (unsigned int k=0; k<holeX.size(); ++k) {
      DDTranslation tran(holeX[k],holeY[k],0);
      cpv.position(fibre, parent(), k+1, tran, rot);   
      edm::LogInfo("HGCalGeom") << "DDShashlikModule:" << fibre.name()<<" no "
				<< k+1 << " positioned in " << parent().name()
				<< " at " << tran << " with no rotation";
    }
  }
  name       = DDName(DDSplit(calibFibreName).first, DDSplit(calibFibreName).second);
  solidCHole = DDSolidFactory::tubs(name,0.5*moduleThick,0,calibFibrePars[0],
				    0,CLHEP::twopi);
  DDLogicalPart fibre(solidCHole.ddname(), matter, solidCHole);
  edm::LogInfo("HGCalGeom") << "DDShashlikModule:: " << fibre.name() 
			    << " tube made of " << matter.name() << " dim: "
			    << 0.5*moduleThick << ":0:" << calibFibrePars[0]
			    << ":0:" << CLHEP::twopi;
  DDTranslation tranc(calibFibrePars[1],calibFibrePars[2],0);
  cpv.position(fibre, parent(), 1, tranc, rot);   
  edm::LogInfo("HGCalGeom") << "DDShashlikModule:" << fibre.name() <<" no 1"
			    << " positioned in " << parent().name() << " at " 
			    << tranc << " with no rotation";
}

