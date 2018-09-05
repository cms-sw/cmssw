///////////////////////////////////////////////////////////////////////////////
// File: DDTIBLayerAlgo_MTCC.cc
// Description: Makes a TIB layer and position the strings with a tilt angle
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/MTCCTrackerCommonData/plugins/DDTIBLayerAlgo_MTCC.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDTIBLayerAlgo_MTCC::DDTIBLayerAlgo_MTCC(): ribW(0),ribPhi(0) {
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC info: Creating an instance";
}

DDTIBLayerAlgo_MTCC::~DDTIBLayerAlgo_MTCC() {}

void DDTIBLayerAlgo_MTCC::initialize(const DDNumericArguments & nArgs,
				     const DDVectorArguments & vArgs,
				     const DDMapArguments & ,
				     const DDStringArguments & sArgs,
				     const DDStringVectorArguments & ) {
  DDCurrentNamespace ns;
  idNameSpace  = *ns;
  genMat       = sArgs["GeneralMaterial"];
  DDName parentName = parent().name(); 
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC debug: Parent " << parentName 
		      << " NameSpace " << idNameSpace << " General Material " 
		      << genMat;
  
  detectorTilt = nArgs["DetectorTilt"];
  layerL       = nArgs["LayerL"];
  detectorTol  = nArgs["LayerTolerance"];
  detectorW    = nArgs["DetectorWidth"];
  detectorT    = nArgs["DetectorThickness"];
  coolTubeW    = nArgs["CoolTubeWidth"];
  coolTubeT    = nArgs["CoolTubeThickness"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC debug: Tilt Angle " 
		      << detectorTilt/CLHEP::deg << " Layer Length/tolerance " 
		      << layerL << " " << detectorTol
		      << " Detector layer Width/Thick " << detectorW << ", " 
		      << detectorT << " Cooling Tube/Cable layer Width/Thick " 
		      << coolTubeW << ", " << coolTubeT;
  
  radiusLo     = nArgs["RadiusLo"];
  phioffLo     = nArgs["PhiOffsetLo"];
  phiMinLo     = nArgs["PhiMinimumLo"];
  phiMaxLo     = nArgs["PhiMaximumLo"];
  stringsLo    = int(nArgs["StringsLo"]);
  stringLoList = vArgs["StringLoList"];
  detectorLo   = sArgs["StringDetLoName"];
  emptyDetectorLo = sArgs["EmptyStringDetLoName"];
  roffDetLo    = nArgs["ROffsetDetLo"];
  coolCableLo  = sArgs["StringCabLoName"];
  emptyCoolCableLo = sArgs["EmptyStringCabLoName"];
  roffCableLo  = nArgs["ROffsetCabLo"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC debug: Lower layer Radius " 
		      << radiusLo << " Phi offset " << phioffLo/CLHEP::deg 
		      << " min " << phiMinLo/CLHEP::deg << " max " 
		      << phiMaxLo/CLHEP::deg << " Number " << stringsLo 
		      << " String " << detectorLo << " at offset " 
		      << roffDetLo << " String " << coolCableLo <<" at offset "
		      << roffCableLo << " Strings filled: ";
  for(unsigned int i=0; i<stringLoList.size(); i++) {
    LogDebug("TIBGeom") << "String " << i << " " << (int)stringLoList[i];
  }
  LogDebug("TIBGeom") << " Empty String " << emptyDetectorLo  << " at offset "
		      << roffDetLo << " Empty String " << emptyCoolCableLo 
		      << " at offset " << roffCableLo;
  
  radiusUp     = nArgs["RadiusUp"];
  phioffUp     = nArgs["PhiOffsetUp"];
  phiMinUp     = nArgs["PhiMinimumUp"];
  phiMaxUp     = nArgs["PhiMaximumUp"];
  stringsUp    = int(nArgs["StringsUp"]);
  stringUpList = vArgs["StringUpList"];
  detectorUp   = sArgs["StringDetUpName"];
  emptyDetectorUp = sArgs["EmptyStringDetUpName"];
  roffDetUp    = nArgs["ROffsetDetUp"];
  coolCableUp  = sArgs["StringCabUpName"];
  emptyCoolCableUp = sArgs["EmptyStringCabUpName"];
  roffCableUp  = nArgs["ROffsetCabUp"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC debug: Upper layer Radius " 
		      << radiusUp << " Phi offset " << phioffUp/CLHEP::deg 
		      << " min " << phiMinUp/CLHEP::deg << " max " 
		      << phiMaxUp/CLHEP::deg << " Number " << stringsUp 
		      << " String " << detectorUp << " at offset " << roffDetUp
		      << " String " << coolCableUp << " at offset " 
		      << roffCableUp << " Strings filled: ";
  for(unsigned int i=0; i<stringUpList.size(); i++) {
    LogDebug("TIBGeom") << "String " << i << " " << (int)stringUpList[i];
  }
  LogDebug("TIBGeom") << " Empty String " << emptyDetectorUp  << " at offset "
		      << roffDetUp << " Empty String " << emptyCoolCableUp 
		      << " at offset " << roffCableUp;
  
  cylinderT    = nArgs["CylinderThickness"];
  cylinderMat  = sArgs["CylinderMaterial"];
  supportW     = nArgs["SupportWidth"];
  supportT     = nArgs["SupportThickness"];
  supportMat   = sArgs["SupportMaterial"];
  ribMat       = sArgs["RibMaterial"];
  ribW         = vArgs["RibWidth"];
  ribPhi       = vArgs["RibPhi"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC debug: Cylinder Material/"
		      << "thickness " << cylinderMat << " " << cylinderT 
		      << " Support Wall Material/Width/Thickness " 
		      << supportMat << " " << supportW << " " << supportT 
		      << " Rib Material " << ribMat << " at "
		      << ribW.size() << " positions with width/phi";
  for (unsigned int i = 0; i < ribW.size(); i++)
    LogDebug("TIBGeom") << "Rib " <<  i << " " <<  ribW[i] << " " 
			<< ribPhi[i]/CLHEP::deg;
  
  dohmN               = int(nArgs["DOHMPhiNumber"]);
  dohmCarrierW        = nArgs["DOHMCarrierWidth"];
  dohmCarrierT        = nArgs["DOHMCarrierThickness"];
  dohmCarrierR        = nArgs["DOHMCarrierRadialHeight"];
  dohmCarrierMaterial = sArgs["DOHMCarrierMaterial"];
  dohmCableMaterial   = sArgs["DOHMCableMaterial"];
  dohmPrimW           = nArgs["DOHMPRIMWidth"];
  dohmPrimL           = nArgs["DOHMPRIMLength"];
  dohmPrimT           = nArgs["DOHMPRIMThickness"];
  dohmPrimMaterial    = sArgs["DOHMPRIMMaterial"];
  dohmAuxW            = nArgs["DOHMAUXWidth"];
  dohmAuxL            = nArgs["DOHMAUXLength"];
  dohmAuxT            = nArgs["DOHMAUXThickness"];
  dohmAuxMaterial     = sArgs["DOHMAUXMaterial"];
  dohmList            = vArgs["DOHMList"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC debug: DOHM PRIMary " << dohmN
		      << " Width/Length/Thickness " << " Material " 
		      << dohmPrimMaterial << " " << dohmPrimW << " " 
		      << dohmPrimL << " " << dohmPrimT
		      << " at positions:";
  for(unsigned int i=0; i<dohmList.size(); i++) {
    if((int)dohmList[i]>0) LogDebug("TIBGeom") <<  i+1 << ",";
  }
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC debug: DOHM AUXiliary "
		      << " Material " << dohmAuxMaterial << " " 
		      << dohmAuxW << " " << dohmAuxL << " " << dohmAuxT
		      << " at positions:";
  for(unsigned int i=0; i<dohmList.size(); i++) {
    if((int)dohmList[i]==2) LogDebug("TIBGeom") << i+1 << ",";
  }
  LogDebug("TIBGeom") << " in Carrier Width/Thickness/Radius " 
		      << dohmCarrierW << " " << dohmCarrierT << " " 
		      << dohmCarrierR << " Carrier Material " 
		      << dohmCarrierMaterial
		      << "\n with cables and connectors Material " 
		      << dohmCableMaterial << "\n"
		      << "DDTIBLayerAlgo_MTCC debug: no DOHM "
		      << " at positions: ";
  for(unsigned int i=0; i<dohmList.size(); i++) {
    if((int)dohmList[i]==0) LogDebug("TIBGeom") << i+1 << ",";
  }
  
}


void DDTIBLayerAlgo_MTCC::execute(DDCompactView& cpv) {
  
  LogDebug("TIBGeom") << "==>> Constructing DDTIBLayerAlgo_MTCC...";
  
  //Parameters for the tilt of the layer
  double rotsi  = std::abs(detectorTilt);
  double redgd1 = 0.5*(detectorW*sin(rotsi)+detectorT*cos(rotsi));
  double redgd2 = 0.5*(detectorW*cos(rotsi)-detectorT*sin(rotsi));
  double redgc1 = 0.5*(coolTubeW*sin(rotsi)+coolTubeT*cos(rotsi));
  double redgc2 = 0.5*(coolTubeW*cos(rotsi)-coolTubeT*sin(rotsi));
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test DeltaR (Detector Tilt) " 
		      << redgd1 << ", " << redgd2 << " DeltaR (Cable+Cool) "
		      << redgc1	<< ", " << redgc2;
  
  DDName parentName = parent().name(); 
  const std::string &idName = parentName.name();
  double rmin = radiusLo + roffDetLo - redgd1 - detectorTol;
  double rmax = sqrt((radiusUp+roffDetUp+redgd1)*(radiusUp+roffDetUp+redgd1)+
		     redgd2*redgd2) + detectorTol;
  DDSolid solid = DDSolidFactory::tubs(DDName(idName, idNameSpace), 0.5*layerL,
				       rmin, rmax, 0, CLHEP::twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " 
		      << DDName(idName,idNameSpace) << " Tubs made of " 
		      << genMat << " from 0 to " << CLHEP::twopi/CLHEP::deg 
		      << " with Rin " << rmin << " Rout " << rmax << " ZHalf "
		      << 0.5*layerL;
  DDMaterial matter( DDName( DDSplit(genMat).first, DDSplit(genMat).second ));
  DDLogicalPart layer(solid.ddname(), matter, solid);
  
  //Lower part first
  double rin  = rmin;
  double rout = 0.5*(radiusLo+radiusUp-cylinderT);
  std::string name = idName + "Down";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, CLHEP::twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: "
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << genMat << " from 0 to " << CLHEP::twopi/CLHEP::deg 
		      << " with Rin " << rin << " Rout " << rout << " ZHalf " 
		      << 0.5*layerL;
  DDLogicalPart layerIn(solid.ddname(), matter, solid);
 cpv.position(layerIn, layer, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " << layerIn.name() 
		      << " number 1 positioned in " << layer.name()
		      << " at (0,0,0) with no rotation";
  
  double rposdet = radiusLo + roffDetLo;
  double rposcab = rposdet + roffCableLo;
  double dphi    = CLHEP::twopi/stringsLo;
  DDName detIn(DDSplit(detectorLo).first, DDSplit(detectorLo).second);
  DDName cabIn(DDSplit(coolCableLo).first, DDSplit(coolCableLo).second);
  for (int n = 0; n < stringsLo; n++) {
    double phi    = phioffLo + n*dphi;
    if( phi>=phiMinLo && phi<phiMaxLo ) { // phi range
      double phix   = phi - detectorTilt + 90*CLHEP::deg;
      double phideg = phix/CLHEP::deg;
      DDRotation rotation;
      if (phideg != 0) {
	double theta  = 90*CLHEP::deg;
	double phiy   = phix + 90.*CLHEP::deg;
	std::string rotstr = idName + std::to_string(phideg*10.);
	rotation = DDRotation(DDName(rotstr, idNameSpace));
	if (!rotation) {
	  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test: Creating a new "
			      << "rotation: " << rotstr << "\t90., " 
			      << phix/CLHEP::deg << ", 90.," 
			      << phiy/CLHEP::deg << ", 0, 0";
	  rotation = DDrot(DDName(rotstr, idNameSpace), theta,phix, theta,phiy,
			   0., 0.);
	}
      }
      
      // fill strings in the stringLoList with modules, the others with only structure
      bool empty=true;
      for(double i : stringLoList) {
	if(n+1==(int)i) {
	  empty=false;
	}
      }
      if(empty) {
	if(emptyDetectorLo!="nothing") {
	  DDName emptyDetIn(DDSplit(emptyDetectorLo).first, DDSplit(emptyDetectorLo).second);
	  DDTranslation trdet(rposdet*cos(phi), rposdet*sin(phi), 0);
	 cpv.position(emptyDetIn, layerIn, n+1, trdet, rotation);
	  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << emptyDetIn.name()
			      << " number " << n+1 << " positioned in " 
			      << layerIn.name() << " at " << trdet 
			      << " with " << rotation;      
	}
	if(emptyCoolCableLo!="nothing") {
	  DDName emptyCabIn(DDSplit(emptyCoolCableLo).first, DDSplit(emptyCoolCableLo).second);
	  DDTranslation trcab(rposcab*cos(phi), rposcab*sin(phi), 0);
	 cpv.position(emptyCabIn, layerIn, n+1, trcab, rotation);
	  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << emptyCabIn.name() 
			      << " number " << n+1 << " positioned in " 
			      << layerIn.name() << " at " << trcab 
			      << " with " << rotation;
	}
      } else {
	DDTranslation trdet(rposdet*cos(phi), rposdet*sin(phi), 0);
	cpv.position(detIn, layerIn, n+1, trdet, rotation);
	LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << detIn.name() 
			    << " number " << n+1 << " positioned in " 
			    << layerIn.name() << " at " << trdet 
			    << " with " << rotation;
	DDTranslation trcab(rposcab*cos(phi), rposcab*sin(phi), 0);
	cpv.position(cabIn, layerIn, n+1, trcab, rotation);
	LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << cabIn.name() 
			    << " number " << n+1 << " positioned in " 
			    << layerIn.name() << " at " << trcab 
			    << " with " << rotation;
      }
      //
      
    } // phi range
    
  }
  
  //Now the upper part
  rin  = 0.5*(radiusLo+radiusUp+cylinderT);
  rout = rmax;
  name = idName + "Up";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, CLHEP::twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << genMat << " from 0 to " << CLHEP::twopi/CLHEP::deg 
		      << " with Rin " << rin << " Rout " << rout << " ZHalf " 
		      << 0.5*layerL;
  DDLogicalPart layerOut(solid.ddname(), matter, solid);
 cpv.position(layerOut, layer, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " << layerOut.name() 
		      << " number 1 positioned in " << layer.name()
		      << " at (0,0,0) with no rotation";
  
  rposdet = radiusUp + roffDetUp;
  rposcab = rposdet + roffCableUp;
  dphi    = CLHEP::twopi/stringsUp;
  DDName detOut(DDSplit(detectorUp).first, DDSplit(detectorUp).second);
  DDName cabOut(DDSplit(coolCableUp).first, DDSplit(coolCableUp).second);
  for (int n = 0; n < stringsUp; n++) {
    double phi    = phioffUp + n*dphi;
    if( phi>=phiMinUp && phi<phiMaxUp ) { // phi range
      double phix   = phi - detectorTilt - 90*CLHEP::deg;
      double phideg = phix/CLHEP::deg;
      DDRotation rotation;
      if (phideg != 0) {
	double theta  = 90*CLHEP::deg;
	double phiy   = phix + 90.*CLHEP::deg;
	std::string rotstr = idName + std::to_string(phideg*10.);
	rotation = DDRotation(DDName(rotstr, idNameSpace));
	if (!rotation) {
	  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test: Creating a new "
			      << "rotation: " << rotstr << "\t90., " 
			      << phix/CLHEP::deg << ", 90.," 
			      << phiy/CLHEP::deg << ", 0, 0";
	  rotation = DDrot(DDName(rotstr, idNameSpace), theta,phix, theta,phiy,
			   0., 0.);
	}
      }
      
      // fill strings in the stringUpList with modules, the others with only structure
      bool empty=true;
      for(double i : stringUpList) {
	if(n+1==(int)i) {
	  empty=false;
	}
      }
      if(empty) {
	if(emptyDetectorUp!="nothing") {
	  DDName emptyDetOut(DDSplit(emptyDetectorUp).first, DDSplit(emptyDetectorUp).second);
	  DDTranslation trdet(rposdet*cos(phi), rposdet*sin(phi), 0);
	 cpv.position(emptyDetOut, layerOut, n+1, trdet, rotation);
	  LogDebug("TIBGeom") << "DDTIBLayer test " << emptyDetOut.name()
			      << " number " << n+1 << " positioned in " 
			      << layerOut.name() << " at " << trdet 
			      << " with " << rotation;
	  
	}
	if(emptyCoolCableUp!="nothing") {
	  DDName emptyCabOut(DDSplit(emptyCoolCableUp).first, DDSplit(emptyCoolCableUp).second);
	  DDTranslation trcab(rposcab*cos(phi), rposcab*sin(phi), 0);
	 cpv.position(emptyCabOut, layerOut, n+1, trcab, rotation);
	  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << emptyCabOut.name()
			      << " number " << n+1 << " positioned in " 
			      << layerOut.name() << " at " << trcab 
			      << " with " << rotation;
	}
      } else {
	DDTranslation trdet(rposdet*cos(phi), rposdet*sin(phi), 0);
	cpv.position(detOut, layerOut, n+1, trdet, rotation);
	LogDebug("TIBGeom") << "DDTIBLayer test " << detOut.name() 
			    << " number " << n+1 << " positioned in " 
			    << layerOut.name() << " at " << trdet 
			    << " with " << rotation;
	DDTranslation trcab(rposcab*cos(phi), rposcab*sin(phi), 0);
	cpv.position(cabOut, layerOut, n+1, trcab, rotation);
	LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << cabOut.name() 
			    << " number " << n+1 << " positioned in " 
			    << layerOut.name() << " at " << trcab 
			    << " with " << rotation;
      }
      //
      
    } // phi range
    
  }
  
  double phiMin  = phiMinUp-phioffUp;   // lower phi for cylinders
  double phiMax  = phiMaxUp-phioffUp;   // upper phi for cylinders
  double phidiff = fabs(phiMax-phiMin); // cylinders will not be twopi but phidiff
  //Finally the inner cylinder, support wall and ribs
  rin  = 0.5*(radiusLo+radiusUp-cylinderT);
  rout = 0.5*(radiusLo+radiusUp+cylinderT);
  name = idName + "Cylinder";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.25*layerL,
			       rin, rout, phiMin, phidiff);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << cylinderMat << " from " << phiMin/CLHEP::deg << " to "
		      << (phiMin+phidiff)/CLHEP::deg << " with Rin " << rin 
		      << " Rout " << rout << " ZHalf " << 0.25*layerL;
  DDMaterial matcyl( DDName( DDSplit(cylinderMat).first, DDSplit(cylinderMat).second ));
  DDLogicalPart cylinder(solid.ddname(), matcyl, solid);
  cpv.position(cylinder, layer, 1, DDTranslation(0.0,0.0,0.25*layerL), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " << cylinder.name() 
		      << " number 1 positioned in " << layer.name()
		      << " at (0,0," << 0.25*layerL << ") with no rotation";
  rin  += supportT;
  rout -= supportT;
  name  = idName + "CylinderIn";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, phiMin, phidiff);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << genMat << " from " << phiMin/CLHEP::deg << " to " 
		      << (phiMin+phidiff)/CLHEP::deg << phidiff/CLHEP::deg 
		      << " with Rin " << rin << " Rout " << rout << " ZHalf " 
		      << 0.5*layerL;
  DDLogicalPart cylinderIn(solid.ddname(), matter, solid);
  cpv.position(cylinderIn, cylinder, 1, DDTranslation(0.0, 0.0, -0.25*layerL), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " << cylinderIn.name() 
		      << " number 1 positioned in " << cylinder.name()
		      << " at (0,0," << -0.25*layerL << ") with no rotation";
  name  = idName + "CylinderInSup";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*supportW,
			       rin, rout, phiMin, phidiff);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << genMat << " from " << phiMin/CLHEP::deg << " to " 
		      << (phiMin+phidiff)/CLHEP::deg << " with Rin " << rin 
		      << " Rout " << rout << " ZHalf " << 0.5*supportW;
  DDMaterial matsup( DDName( DDSplit(supportMat).first, DDSplit(supportMat).second ));
  DDLogicalPart cylinderSup(solid.ddname(), matsup, solid);
  cpv.position(cylinderSup, cylinderIn, 1, DDTranslation(0., 0., 0.), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " << cylinderSup.name() 
		      << " number 1 positioned in " << cylinderIn.name()
		      << " at (0,0,0) with no rotation";
  DDMaterial matrib( DDName( DDSplit(ribMat).first, DDSplit(ribMat).second ));
  for (unsigned int i = 0; i < ribW.size(); i++) {
    name = idName + "Rib" + std::to_string(i);
    double width = 2.*ribW[i]/(rin+rout);
    double dz    = 0.25*(layerL - supportW);
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, rout, 
				 -0.5*width, width);
    LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " 
			<< DDName(name, idNameSpace) << " Tubs made of " 
			<< ribMat << " from " << -0.5*width/CLHEP::deg
			<< " to " << 0.5*width/CLHEP::deg << " with Rin " 
			<< rin << " Rout " << rout << " ZHalf "  << dz;
    DDLogicalPart cylinderRib(solid.ddname(), matrib, solid);
    double phix   = ribPhi[i];
    double phideg = phix/CLHEP::deg;
    if( phideg>=phiMin/CLHEP::deg && phideg<phiMax/CLHEP::deg ) { // phi range
      DDRotation rotation;
      if (phideg != 0) {
	double theta  = 90*CLHEP::deg;
	double phiy   = phix + 90.*CLHEP::deg;
	std::string rotstr = idName + std::to_string(phideg*10.);
	rotation = DDRotation(DDName(rotstr, idNameSpace));
	if (!rotation) {
	  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test: Creating a new "
			      << "rotation: " << rotstr << "\t90., " 
			      << phix/CLHEP::deg << ", 90.," 
			      << phiy/CLHEP::deg << ", 0, 0";
	  rotation = DDrot(DDName(rotstr, idNameSpace), theta,phix, theta,phiy,
			   0., 0.);
	}
      }
      DDTranslation tran(0, 0, +0.25*(layerL+supportW));
     cpv.position(cylinderRib, cylinderIn, 1, tran, rotation);
      LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << cylinderRib.name() 
			  << " number 1 positioned in " << cylinderIn.name() 
			  << " at " << tran << " with " << rotation;
    } // phi range
  }

  
  // DOHM + carrier (portadohm)
  double dz_dohm    = 0.5*dohmCarrierW;
  double dphi_dohm  = CLHEP::twopi/((double)dohmN);
  double rout_dohm  = 0.5*(radiusLo+radiusUp+cylinderT)+dohmCarrierR;
  
  // DOHM Carrier TIB+ & TIB-
  // lower
  name = idName + "DOHMCarrier_lo";
  double rin_lo  = rout_dohm;
  double rout_lo = rin_lo + dohmCarrierT;
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz_dohm, 
			       rin_lo, rout_lo, 
			       -0.5*dphi_dohm, dphi_dohm);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << dohmCarrierMaterial << " from " 
		      << -0.5*(dphi_dohm)/CLHEP::deg << " to " 
		      << +0.5*(dphi_dohm)/CLHEP::deg << " with Rin " 
		      << rin_lo << " Rout " << rout_lo << " ZHalf "  
		      << dz_dohm;
  // create different name objects for only PRIMary DOHMs and PRIMary+AUXiliary DOHM Carriers
  std::string name_lo_r = name + "_PRIM_AUX" + "_lo" + "_r";
  std::string name_lo_l = name + "_PRIM_AUX" + "_lo" + "_l";
  DDLogicalPart dohmCarrierPrimAux_lo_r(name_lo_r, DDMaterial(dohmCarrierMaterial), solid);
  DDLogicalPart dohmCarrierPrimAux_lo_l(name_lo_l, DDMaterial(dohmCarrierMaterial), solid);
  name_lo_r = name + "_PRIM" + "_lo" + "_r";
  name_lo_l = name + "_PRIM" + "_lo" + "_l";
  DDLogicalPart dohmCarrierPrim_lo_r(name_lo_r, DDMaterial(dohmCarrierMaterial), solid);
  DDLogicalPart dohmCarrierPrim_lo_l(name_lo_l, DDMaterial(dohmCarrierMaterial), solid);
  // upper
  name = idName + "DOHMCarrier_up";
  double rin_up  = rout_lo + 2.*dohmAuxT;
  double rout_up = rin_up + dohmCarrierT;
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz_dohm, 
			       rin_up, rout_up, 
			       -0.5*dphi_dohm, dphi_dohm);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << dohmCarrierMaterial << " from " 
		      << -0.5*(dphi_dohm)/CLHEP::deg << " to " 
		      << +0.5*(dphi_dohm)/CLHEP::deg << " with Rin " 
		      << rin_up << " Rout " << rout_up << " ZHalf "
		      << dz_dohm;
  // create different name objects for only PRIMary DOHMs and PRIMary+AUXiliary DOHM Carriers
  std::string name_up_r = name + "_PRIM_AUX" + "_up" + "_r";
  std::string name_up_l = name + "_PRIM_AUX" + "_up" + "_l";
  DDLogicalPart dohmCarrierPrimAux_up_r(name_up_r, DDMaterial(dohmCarrierMaterial), solid);
  DDLogicalPart dohmCarrierPrimAux_up_l(name_up_l, DDMaterial(dohmCarrierMaterial), solid);
  name_up_r = name + "_PRIM" + "_up" + "_r";
  name_up_l = name + "_PRIM" + "_up" + "_l";
  DDLogicalPart dohmCarrierPrim_up_r(name_up_r, DDMaterial(dohmCarrierMaterial), solid);
  DDLogicalPart dohmCarrierPrim_up_l(name_up_l, DDMaterial(dohmCarrierMaterial), solid);
  //
  for (unsigned int i = 0; i < (unsigned int)dohmN; i++) {
    DDLogicalPart dohmCarrier_lo_r;
    DDLogicalPart dohmCarrier_lo_l;
    DDLogicalPart dohmCarrier_up_r;
    DDLogicalPart dohmCarrier_up_l;
    // create different name objects for only PRIMary DOHMs and PRIMary+AUXiliary DOHMs
    bool prim = false;
    bool aux  = false;
    if((unsigned int)dohmList[i]==2) {
      prim = true;
      aux  = true;
    } else if((unsigned int)dohmList[i]==1) {
      prim = true;
      aux  = false;
    } else {
      prim = false;
      aux  = false;      
    }
    
    if(prim) {
      dohmCarrier_lo_r = dohmCarrierPrim_lo_r;
      dohmCarrier_lo_l = dohmCarrierPrim_lo_l;
      dohmCarrier_up_r = dohmCarrierPrim_up_r;
      dohmCarrier_up_l = dohmCarrierPrim_up_l;
    }
    if(prim && aux) {
      dohmCarrier_lo_r = dohmCarrierPrimAux_lo_r;
      dohmCarrier_lo_l = dohmCarrierPrimAux_lo_l;
      dohmCarrier_up_r = dohmCarrierPrimAux_up_r;
      dohmCarrier_up_l = dohmCarrierPrimAux_up_l;
    }
    //
    
    if(prim) {
      double phix   = ((double)i+0.5)*dphi_dohm;
      double phideg = phix/CLHEP::deg;
      //    if( phideg>=phiMin/CLHEP::deg && phideg<phiMax/CLHEP::deg ) { // phi range
      DDRotation rotation;
      if (phideg != 0) {
	double theta  = 90*CLHEP::deg;
	double phiy   = phix + 90.*CLHEP::deg;
	std::string rotstr = idName + std::to_string(phideg*10.);
	rotation = DDRotation(DDName(rotstr, idNameSpace));
	if (!rotation) {
	  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test: Creating a new "
			      << "rotation: " << rotstr << "\t90., " 
			      << phix/CLHEP::deg << ", 90.," 
			      << phiy/CLHEP::deg << ", 0, 0";
	  rotation = DDrot(DDName(rotstr, idNameSpace), theta,phix, theta,phiy,
			   0., 0.);
	}
      }
      // TIB+ DOHM Carrier - lower
      DDTranslation tran(0, 0, 0.5*layerL-dz_dohm);
     cpv.position(dohmCarrier_lo_r, parent(), i+1, tran, rotation );
      LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << dohmCarrier_lo_r.name()
			  << " z+ number " << i+1 << " positioned in " 
			  << parent().name() << " at " << tran
			  << " with " << rotation;
      // TIB+ DOHM Carrier - upper
     cpv.position(dohmCarrier_up_r, parent(), i+1+(unsigned int)dohmN, tran, rotation );
      LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << dohmCarrier_up_r.name()
			  << " z+ number " << i+1 << " positioned in " 
			  << parent().name() << " at " << tran
			  << " with " << rotation;
    }
    
    //    } // phi range
  }
  
  
  // DOHM only PRIMary
  double dx = 0.5*dohmPrimT;
  double dy = 0.5*dohmPrimW;
  double dz = 0.5*dohmPrimL;
  name = idName + "DOHM_PRIM";
  solid = DDSolidFactory::box(DDName(name, idNameSpace), dx, dy, dz);
  DDLogicalPart dohmPrim(solid.ddname(), DDMaterial(dohmPrimMaterial), solid);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: "
		      << DDName(name, idNameSpace) << " Box made of " 
		      << dohmPrimMaterial << " of dimensions " << dx << ", " 
		      << dy << ", " << dz;
  name = idName + "DOHM_PRIM_Cable";
  double dx_cable = 0.25*dohmPrimT;
  double dy_cable = 0.40*dohmPrimW;
  double dz_cable = 0.5*dohmPrimL;
  solid = DDSolidFactory::box(DDName(name, idNameSpace), dx_cable, dy_cable, dz_cable);
  DDLogicalPart dohmCablePrim(solid.ddname(), DDMaterial(dohmCableMaterial), solid);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: "
		      << DDName(name, idNameSpace) << " Box made of "
		      << dohmCableMaterial << " of dimensions " << dx_cable
		      << ", " << dy_cable << ", " << dz_cable;
  // TIB+ DOHM
  DDTranslation tran(rout_dohm+0.5*dohmPrimT, 0. , 0.);
 cpv.position(dohmPrim, dohmCarrierPrim_lo_r, 1, tran, DDRotation() );
  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << dohmPrim.name() 
		      << " z+ number " << 1 << " positioned in " 
		      << dohmCarrierPrim_lo_r.name() << " at " << tran 
		      << " with no rotation";
  tran = DDTranslation(rout_dohm+dx_cable, 0.5*dohmPrimW , 0.);
 cpv.position(dohmCablePrim, dohmCarrierPrim_lo_r, 1, tran, DDRotation() );
  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << dohmCablePrim.name() 
		      << " copy number " << 1 << " positioned in "
		      << dohmCarrierPrim_lo_r.name()
		      << " at " << tran << " with no rotation";
  tran = DDTranslation(rout_dohm+dx_cable, -0.5*dohmPrimW , 0.);
 cpv.position(dohmCablePrim, dohmCarrierPrim_lo_r, 2, tran, DDRotation() );
  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << dohmCablePrim.name()
		      << " copy number " << 2 << " positioned in " 
		      << dohmCarrierPrim_lo_r.name()
		      << " at " << tran << " with no rotation";
  
  // DOHM PRIMary + AUXiliary
  dx = 0.5*dohmPrimT;
  dy = 0.5*dohmPrimW;
  dz = 0.5*dohmPrimL;
  name = idName + "DOHM_PRIM";
  solid = DDSolidFactory::box(DDName(name, idNameSpace), dx, dy, dz);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " 
		      << DDName(name, idNameSpace) << " Box made of "
		      << dohmPrimMaterial << " of dimensions " << dx << ", " 
		      << dy << ", " << dz;
  dohmPrim = DDLogicalPart(solid.ddname(), DDMaterial(dohmPrimMaterial), solid);
  name = idName + "DOHM_PRIM_Cable";
  dx_cable = 0.25*dohmPrimT;
  dy_cable = 0.40*dohmPrimW;
  dz_cable = 0.5*dohmPrimL;
  solid = DDSolidFactory::box(DDName(name, idNameSpace), dx_cable, dy_cable, dz_cable);
  dohmCablePrim = DDLogicalPart(solid.ddname(), DDMaterial(dohmCableMaterial), solid);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: "
		      << DDName(name, idNameSpace) << " Box made of " 
		      << dohmCableMaterial << " of dimensions " << dx_cable 
		      << ", " << dy_cable << ", " << dz_cable;
  dx = 0.5*dohmAuxT;
  dy = 0.5*dohmAuxW;
  dz = 0.5*dohmAuxL;
  name = idName + "DOHM_AUX";
  solid = DDSolidFactory::box(DDName(name, idNameSpace), dx, dy, dz);
  DDLogicalPart dohmAux(solid.ddname(), DDMaterial(dohmAuxMaterial), solid);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " 
		      << DDName(name, idNameSpace) << " Box made of " 
		      << dohmAuxMaterial << " of dimensions " << dx << ", " 
		      << dy << ", " << dz;
  name = idName + "DOHM_AUX_Cable";
  solid = DDSolidFactory::box(DDName(name, idNameSpace), dx_cable, dy_cable, dz_cable);
  DDLogicalPart dohmCableAux(solid.ddname(), DDMaterial(dohmCableMaterial), solid);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo_MTCC test: " 
		      << DDName(name, idNameSpace) << " Box made of " 
		      << dohmCableMaterial << " of dimensions " << dx_cable 
		      << ", " << dy_cable << ", " << dz_cable;
  // TIB+ DOHM
  tran = DDTranslation(rout_dohm+0.5*dohmPrimT, -0.75*dohmPrimW , 0.);
 cpv.position(dohmPrim, dohmCarrierPrimAux_lo_r, 1, tran, DDRotation() );
  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << dohmAux.name() 
		      << " z+ number " << 1 << " positioned in " 
		      << dohmCarrierPrimAux_lo_r.name()	<< " at " << tran 
		      << " with no rotation";
  tran = DDTranslation(rout_dohm+dx_cable, -0.75*dohmPrimW+0.5*dohmPrimW , 0.);
 cpv.position(dohmCablePrim, dohmCarrierPrimAux_lo_r, 1, tran, DDRotation() );
  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << dohmCablePrim.name() 
		      << " copy number " << 1 << " positioned in " 
		      << dohmCarrierPrimAux_lo_r.name()	<< " at " << tran 
		      << " with no rotation";
  tran = DDTranslation(rout_dohm+dx_cable, -0.75*dohmPrimW-0.5*dohmPrimW , 0.);
 cpv.position(dohmCablePrim, dohmCarrierPrimAux_lo_r, 2, tran, DDRotation() );
  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << dohmCablePrim.name() 
		      << " copy number " << 2 << " positioned in "
		      << dohmCarrierPrimAux_lo_r.name()	<< " at " << tran 
		      << " with no rotation";
  tran = DDTranslation(rout_dohm+0.5*dohmAuxT, 0.75*dohmAuxW , 0.);
 cpv.position(dohmAux, dohmCarrierPrimAux_lo_r, 1, tran, DDRotation() );
  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << dohmAux.name() 
		      << " z+ number " << 1 << " positioned in " 
		      << dohmCarrierPrimAux_lo_r.name()
		      << " at (0,0,0) with no rotation";
  tran = DDTranslation(rout_dohm+dx_cable, 0.75*dohmAuxW+0.5*dohmPrimW , 0.);
 cpv.position(dohmCableAux, dohmCarrierPrimAux_lo_r, 1, tran, DDRotation() );
  LogDebug("TIBGeom") << "DDTIBLayer_MTCC test " << dohmCableAux.name() 
		      << " copy number " << 1 << " positioned in " 
		      << dohmCarrierPrimAux_lo_r.name()
		      << " at " << tran << " with no rotation";
}
