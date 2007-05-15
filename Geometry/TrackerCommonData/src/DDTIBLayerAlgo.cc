///////////////////////////////////////////////////////////////////////////////
// File: DDTIBLayerAlgo.cc
// Description: Makes a TIB layer and position the strings with a tilt angle
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDTIBLayerAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTIBLayerAlgo::DDTIBLayerAlgo(): ribW(0),ribPhi(0) {
  LogDebug("TIBGeom") << "DDTIBLayerAlgo info: Creating an instance";
}

DDTIBLayerAlgo::~DDTIBLayerAlgo() {}

void DDTIBLayerAlgo::initialize(const DDNumericArguments & nArgs,
				const DDVectorArguments & vArgs,
				const DDMapArguments & ,
				const DDStringArguments & sArgs,
				const DDStringVectorArguments & ) {

  idNameSpace  = DDCurrentNamespace::ns();
  genMat       = sArgs["GeneralMaterial"];
  DDName parentName = parent().name(); 
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: Parent " << parentName 
		      << " NameSpace " << idNameSpace 
		      << " General Material " << genMat;

  detectorTilt = nArgs["DetectorTilt"];
  layerL       = nArgs["LayerL"];
  detectorTol  = nArgs["LayerTolerance"];
  detectorW    = nArgs["DetectorWidth"];
  detectorT    = nArgs["DetectorThickness"];
  coolTubeW    = nArgs["CoolTubeWidth"];
  coolTubeT    = nArgs["CoolTubeThickness"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: Tilt Angle " 
		      << detectorTilt/deg << " Layer Length/tolerance " 
		      << layerL << " " << detectorTol 
		      << " Detector layer Width/Thick " << detectorW 
		      << ", " << detectorT << " Cooling Tube/Cable layer "
		      << "Width/Thick " << coolTubeW << ", " << coolTubeT;

  radiusLo     = nArgs["RadiusLo"];
  phioffLo     = nArgs["PhiOffsetLo"];
  stringsLo    = int(nArgs["StringsLo"]);
  detectorLo   = sArgs["StringDetLoName"];
  roffDetLo    = nArgs["ROffsetDetLo"];
  coolCableLo  = sArgs["StringCabLoName"];
  // roffCableLo no more used!!!
  roffCableLo  = nArgs["ROffsetCabLo"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: Lower layer Radius " 
		      << radiusLo << " Phi offset " << phioffLo/deg 
		      << " Number " << stringsLo << " String "
		      << detectorLo << " at offset " << roffDetLo
		      << " String " << coolCableLo << " at offset "
		      << roffCableLo;

  radiusUp     = nArgs["RadiusUp"];
  phioffUp     = nArgs["PhiOffsetUp"];
  stringsUp    = int(nArgs["StringsUp"]);
  detectorUp   = sArgs["StringDetUpName"];
  roffDetUp    = nArgs["ROffsetDetUp"];
  coolCableUp  = sArgs["StringCabUpName"];
  // roffCableUp no more used!!!
  roffCableUp  = nArgs["ROffsetCabUp"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: Upper layer Radius "
		      << radiusUp << " Phi offset " << phioffUp/deg 
		      << " Number " << stringsUp << " String "
		      << detectorUp << " at offset " << roffDetUp 
		      << " String " << coolCableUp << " at offset " 
		      << roffCableUp;

  cylinderT    = nArgs["CylinderThickness"];
  cylinderMat  = sArgs["CylinderMaterial"];
  supportW     = nArgs["SupportWidth"];
  supportT     = nArgs["SupportThickness"];
  supportMat   = sArgs["SupportMaterial"];
  ribMat       = sArgs["RibMaterial"];
  ribW         = vArgs["RibWidth"];
  ribPhi       = vArgs["RibPhi"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: Cylinder Material/"
		      << "thickness " << cylinderMat << " " << cylinderT 
		      << " Support Wall Material/" << "Width/Thickness " 
		      << supportMat << " " << supportW << " " 
		      << supportT << " Rib Material " << ribMat << " at "
		      << ribW.size() << " positions with width/phi";
  for (int i = 0; i < (int)(ribW.size()); i++)
    LogDebug("TIBGeom") << "\tribW[" << i << "] = " <<  ribW[i] 
			<< "\tribPhi[" << i << "] = " << ribPhi[i]/deg;
  
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
  dohmRotPlus         = sArgs["DOHMRotstrPlus"];
  dohmRotMinus        = sArgs["DOHMRotstrMinus"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: DOHM PRIMary " << dohmN
		      << " Material " << dohmPrimMaterial << " Width "
		      << dohmPrimW << " Length " << dohmPrimL 
		      << " Thickness " << dohmPrimT << " at:";
  for (int i=0; i<(int)(dohmList.size()); i++) {
    if ((int)dohmList[i]>0) LogDebug("TIBGeom") << "Position " << i+1;
  }
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: DOHM AUXiliary "
		      << " Material " << dohmAuxMaterial << " Width "
		      << dohmAuxW << " Length " << dohmAuxL 
		      << " Thickness " << dohmAuxT << " at:";
  for (int i=0; i<(int)(dohmList.size()); i++) {
    if ((int)dohmList[i]==2) LogDebug("TIBGeom") << "Position " << i+1;
  }
  LogDebug("TIBGeom") << " in Carrier Width/Thickness/Radius " 
		      << dohmCarrierW << " " << dohmCarrierT << " "
		      << dohmCarrierR << " Carrier Material " 
		      << dohmCarrierMaterial <<"\n"
		      << " with cables and connectors Material "
		      << dohmCableMaterial << "\n"
		      << "DDTIBLayer debug: no DOHM at:";
  for (int i=0; i<(int)(dohmList.size()); i++) {
    if ((int)dohmList[i]==0) LogDebug("TIBGeom") << "Position " << i+1;
  }
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: DOHM placed in TIB+ with "
		      << "rotation " << dohmRotPlus << " and in TIB- with"
		      << " rotation " << dohmRotMinus;
  
}

void DDTIBLayerAlgo::execute() {

  LogDebug("TIBGeom") << "==>> Constructing DDTIBLayerAlgo...";

  //Parameters for the tilt of the layer
  double rotsi  = detectorTilt;
  if (rotsi < 0.) rotsi = -rotsi;
  double redgd1 = 0.5*(detectorW*sin(rotsi)+detectorT*cos(rotsi));
  double redgd2 = 0.5*(detectorW*cos(rotsi)-detectorT*sin(rotsi));
  double redgc1 = 0.5*(coolTubeW*sin(rotsi)+coolTubeT*cos(rotsi));
  double redgc2 = 0.5*(coolTubeW*cos(rotsi)-coolTubeT*sin(rotsi));
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test DeltaR (Detector Tilt) "
			  << redgd1 << ", " << redgd2 <<" DeltaR (Cable+Cool) "
			  << redgc1 << ", " << redgc2;

  DDName  parentName = parent().name(); 
  std::string idName = DDSplit(parentName).first;
  double rmin = radiusLo + roffDetLo - redgd1 - detectorTol;
  double rmax = sqrt((radiusUp+roffDetUp+redgd1)*(radiusUp+roffDetUp+redgd1)+
		     redgd2*redgd2) + detectorTol;
  double rmaxTube = rmax + 2*(dohmCarrierT+dohmAuxT);
  DDSolid solid = DDSolidFactory::tubs(DDName(idName, idNameSpace), 0.5*layerL,
				       rmin, rmaxTube, 0, twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: "  
		      << DDName(idName,idNameSpace) << " Tubs made of " 
		      << genMat << " from 0 to " << twopi/deg 
		      << " with Rin " << rmin << " Rout " << rmax 
		      << " ZHalf " << 0.5*layerL;
  DDName matname(DDSplit(genMat).first, DDSplit(genMat).second);
  DDMaterial matter(matname);
  DDLogicalPart layer(solid.ddname(), matter, solid);

  //Lower part first
  double rin  = rmin;
  double rout = 0.5*(radiusLo+radiusUp-cylinderT);
  std::string name = idName + "Down";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << genMat << " from 0 to " << twopi/deg 
		      << " with Rin " << rin << " Rout " << rout 
		      << " ZHalf " << 0.5*layerL;
  DDLogicalPart layerIn(solid.ddname(), matter, solid);
  DDpos (layerIn, layer, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << layerIn.name()
		      << " number 1 positioned in " << layer.name()
		      << " at (0,0,0) with no rotation";

  double rposdet = radiusLo + roffDetLo;
  roffCableLo = detectorT/2. + coolTubeT/2.; 
  double rposcab = rposdet + roffCableLo;
  double dphi    = twopi/stringsLo;
  DDName detIn(DDSplit(detectorLo).first, DDSplit(detectorLo).second);
  DDName cabIn(DDSplit(coolCableLo).first, DDSplit(coolCableLo).second);
  for (int n = 0; n < stringsLo; n++) {
//    double phi    = phioffLo + n*dphi;
    double phi    = (n+0.5)*dphi;
    double phix   = phi - detectorTilt + 90*deg;
    double phideg = phix/deg;
    DDRotation rotation;
    if (phideg != 0) {
      double theta  = 90*deg;
      double phiy   = phix + 90.*deg;
      std::string rotstr = idName + dbl_to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
        LogDebug("TIBGeom") << "DDTIBLayerAlgo test: Creating a new "
			    << "rotation: "	<< rotstr << "\t90., " 
			    << phix/deg << ", 90.,"	<< phiy/deg <<", 0, 0";
        rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
                         0., 0.);
      }
    }
    DDTranslation trdet(rposdet*cos(phi), rposdet*sin(phi), 0);
    DDpos (detIn, layerIn, n+1, trdet, rotation);
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << detIn.name() 
			<< " number " << n+1 << " positioned in " 
			<< layerIn.name() << " at " << trdet << " with "
			<< rotation;
    DDTranslation trcab(rposcab*cos(phi), rposcab*sin(phi), 0);
    DDpos (cabIn, layerIn, n+1, trcab, rotation);
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << cabIn.name() 
			<< " number " << n+1 << " positioned in " 
			<< layerIn.name() << " at " << trcab << " with "
			<< rotation;
  }

  //Now the upper part
  rin  = 0.5*(radiusLo+radiusUp+cylinderT);
  rout = rmax;
  name = idName + "Up";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << genMat << " from 0 to " << twopi/deg 
		      << " with Rin " << rin << " Rout " << rout
		      << " ZHalf " << 0.5*layerL;
  DDLogicalPart layerOut(solid.ddname(), matter, solid);
  DDpos (layerOut, layer, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << layerOut.name() 
		      << " number 1 positioned in " << layer.name() 
		      << " at (0,0,0) with no rotation";

  rposdet = radiusUp + roffDetUp;
  roffCableUp = - detectorT/2. - coolTubeT/2.; 
  rposcab = rposdet + roffCableUp;
  dphi    = twopi/stringsUp;
  DDName detOut(DDSplit(detectorUp).first, DDSplit(detectorUp).second);
  DDName cabOut(DDSplit(coolCableUp).first, DDSplit(coolCableUp).second);
  for (int n = 0; n < stringsUp; n++) {
//    double phi    = phioffUp + n*dphi;
    double phi    = (n+0.5)*dphi;
    double phix   = phi - detectorTilt - 90*deg;
    double phideg = phix/deg;
    DDRotation rotation;
    if (phideg != 0) {
      double theta  = 90*deg;
      double phiy   = phix + 90.*deg;
      std::string rotstr = idName + dbl_to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
        LogDebug("TIBGeom") << "DDTIBLayerAlgo test: Creating a new "
			    << "rotation: " << rotstr << "\t90., " 
			    << phix/deg << ", 90.,"	<< phiy/deg <<", 0, 0";
        rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
                         0., 0.);
      }
    }
    DDTranslation trdet(rposdet*cos(phi), rposdet*sin(phi), 0);
    DDpos (detOut, layerOut, n+1, trdet, rotation);
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << detOut.name() 
			<< " number " << n+1 << " positioned in " 
			<< layerOut.name() << " at " << trdet << " with "
			<< rotation;
    DDTranslation trcab(rposcab*cos(phi), rposcab*sin(phi), 0);
    DDpos (cabOut, layerOut, n+1, trcab, rotation);
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << cabOut.name() 
			<< " number " << n+1 << " positioned in " 
			<< layerOut.name() << " at " << trcab << " with "
			<< rotation;
  }

  //Finally the inner cylinder, support wall and ribs
  rin  = 0.5*(radiusLo+radiusUp-cylinderT);
  rout = 0.5*(radiusLo+radiusUp+cylinderT);
  name = idName + "Cylinder";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << cylinderMat << " from 0 to " << twopi/deg 
		      << " with Rin " << rin << " Rout " << rout 
		      << " ZHalf " << 0.5*layerL;
  matname = DDName(DDSplit(cylinderMat).first, DDSplit(cylinderMat).second);
  DDMaterial matcyl(matname);
  DDLogicalPart cylinder(solid.ddname(), matcyl, solid);
  DDpos (cylinder, layer, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << cylinder.name() 
		      << " number 1 positioned in " << layer.name()
		      << " at (0,0,0) with no rotation";
  rin  += supportT;
  rout -= supportT;
  name  = idName + "CylinderIn";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: "
		      << DDName(name, idNameSpace) << " Tubs made of "
		      << genMat << " from 0 to " << twopi/deg 
		      << " with Rin " << rin << " Rout " << rout 
		      << " ZHalf " << 0.5*layerL;
  DDLogicalPart cylinderIn(solid.ddname(), matter, solid);
  DDpos (cylinderIn, cylinder, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << cylinderIn.name() 
		      << " number 1 positioned in " << cylinder.name() 
		      << " at (0,0,0) with no rotation";
  name  = idName + "CylinderInSup";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*supportW,
			       rin, rout, 0, twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << genMat << " from 0 to " << twopi/deg 
		      << " with Rin " << rin << " Rout " << rout 
		      << " ZHalf " << 0.5*supportW;
  matname = DDName(DDSplit(supportMat).first, DDSplit(supportMat).second);
  DDMaterial matsup(matname);
  DDLogicalPart cylinderSup(solid.ddname(), matsup, solid);
  DDpos (cylinderSup, cylinderIn, 1, DDTranslation(0., 0., 0.), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << cylinderSup.name()
		      << " number 1 positioned in " << cylinderIn.name()
		      << " at (0,0,0) with no rotation";
  matname = DDName(DDSplit(ribMat).first, DDSplit(ribMat).second);
  DDMaterial matrib(matname);
  for (int i = 0; i < (int)(ribW.size()); i++) {
    name = idName + "Rib" + dbl_to_string(i);
    double width = 2.*ribW[i]/(rin+rout);
    double dz    = 0.25*(layerL - supportW);
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, rout, 
				 -0.5*width, width);
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
			<< DDName(name, idNameSpace) << " Tubs made of " 
			<< ribMat << " from " << -0.5*width/deg << " to "
			<< 0.5*width/deg << " with Rin " << rin << " Rout "
			<< rout << " ZHalf "  << dz;
    DDLogicalPart cylinderRib(solid.ddname(), matrib, solid);
    double phix   = ribPhi[i];
    double phideg = phix/deg;
    DDRotation rotation;
    if (phideg != 0) {
      double theta  = 90*deg;
      double phiy   = phix + 90.*deg;
      std::string rotstr = idName + dbl_to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
        LogDebug("TIBGeom") << "DDTIBLayerAlgo test: Creating a new "
			    << "rotation: "	<< rotstr << "\t90., " 
			    << phix/deg << ", 90.," << phiy/deg <<", 0, 0";
        rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
                         0., 0.);
      }
    }
    DDTranslation tran(0, 0, -0.25*(layerL+supportW));
    DDpos (cylinderRib, cylinderIn, 1, tran, rotation);
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << cylinderRib.name()
			<< " number 1" << " positioned in " 
			<< cylinderIn.name() << " at " << tran << " with " 
			<< rotation;
    tran = DDTranslation(0, 0, 0.25*(layerL+supportW));
    DDpos (cylinderRib, cylinderIn, 2, tran, rotation);
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << cylinderRib.name()
			<< " number 2" << " positioned in "
			<< cylinderIn.name() << " at " << tran << " with "
			<< rotation;
  }

  // DOHM + carrier (portadohm)
  double dz_dohm    = 0.5*dohmCarrierW;
  double dphi_dohm  = twopi/((double)dohmN);
  //  double rout_dohm  = 0.5*(radiusLo+radiusUp+cylinderT)+dohmCarrierR;
  double rout_dohm = rmax;
  
  // DOHM Carrier TIB+ & TIB-
  // lower
  name = idName + "DOHMCarrier_lo";
  double rin_lo  = rout_dohm;
  double rout_lo = rin_lo + dohmCarrierT + dohmAuxT;
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz_dohm, 
			       rin_lo, rout_lo, 
			       -0.5*dphi_dohm, dphi_dohm);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of "
		      << dohmCarrierMaterial << " from "
		      << -0.5*(dphi_dohm)/deg << " to " 
		      << -0.5*(dphi_dohm)/deg+dphi_dohm/deg << " with Rin "
		      << rin_lo << " Rout " << rout_lo << " ZHalf "  
		      << dz_dohm;
  // create different name objects for only PRIMary DOHMs and PRIMary+AUXiliary DOHM Carriers
  std::string name_lo_r = name + "_PRIM_AUX" + "_lo" + "_r";
  std::string name_lo_l = name + "_PRIM_AUX" + "_lo" + "_l";
  DDLogicalPart dohmCarrierPrimAux_lo_r(name_lo_r,
					DDMaterial(dohmCarrierMaterial),solid);
  DDLogicalPart dohmCarrierPrimAux_lo_l(name_lo_l,
					DDMaterial(dohmCarrierMaterial),solid);
  name_lo_r = name + "_PRIM" + "_lo" + "_r";
  name_lo_l = name + "_PRIM" + "_lo" + "_l";
  DDLogicalPart dohmCarrierPrim_lo_r(name_lo_r,DDMaterial(dohmCarrierMaterial),
				     solid);
  DDLogicalPart dohmCarrierPrim_lo_l(name_lo_l,DDMaterial(dohmCarrierMaterial),
				     solid);
  // upper
  name = idName + "DOHMCarrier_up";
  double rin_up  = rout_lo;
  double rout_up = rin_up + dohmCarrierT + dohmAuxT;
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz_dohm, 
			       rin_up, rout_up, 
			       -0.5*dphi_dohm, dphi_dohm);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of "
		      << dohmCarrierMaterial << " from " 
		      << -0.5*(dphi_dohm)/deg << " to " 
		      << -0.5*(dphi_dohm)/deg+dphi_dohm/deg 
		      << " with Rin " << rin_up << " Rout " << rout_up
		      << " ZHalf "  << dz_dohm;
  // create different name objects for only PRIMary DOHMs and PRIMary+AUXiliary DOHM Carriers
  std::string name_up_r = name + "_PRIM_AUX" + "_up" + "_r";
  std::string name_up_l = name + "_PRIM_AUX" + "_up" + "_l";
  DDLogicalPart dohmCarrierPrimAux_up_r(name_up_r,
					DDMaterial(dohmCarrierMaterial),solid);
  DDLogicalPart dohmCarrierPrimAux_up_l(name_up_l, 
					DDMaterial(dohmCarrierMaterial),solid);
  name_up_r = name + "_PRIM" + "_up" + "_r";
  name_up_l = name + "_PRIM" + "_up" + "_l";
  DDLogicalPart dohmCarrierPrim_up_r(name_up_r, 
				     DDMaterial(dohmCarrierMaterial), solid);
  DDLogicalPart dohmCarrierPrim_up_l(name_up_l,
				     DDMaterial(dohmCarrierMaterial), solid);
  //
  for (int i = 0; i < dohmN; i++) {
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
      double phideg = phix/deg;
      //    if( phideg>=phiMin/deg && phideg<phiMax/deg ) { // phi range
      DDRotation rotation;
      if (phideg != 0) {
	double theta  = 90*deg;
	double phiy   = phix + 90.*deg;
	std::string rotstr = idName + dbl_to_string(phideg*10.);
	rotation = DDRotation(DDName(rotstr, idNameSpace));
	if (!rotation) {
	  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: Creating a new "
			      << "rotation: " << rotstr << "\t90., " 
			      << phix/deg << ", 90.," << phiy/deg 
			      << ", 0, 0";
	  rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, 
			   phiy, 0., 0.);
	}
      }
      // TIB+ DOHM Carrier - lower
      DDTranslation tran(0, 0, 0.5*layerL-dz_dohm);
      DDpos (dohmCarrier_lo_r, parent(), i+1, tran, rotation );
      LogDebug("TIBGeom") << "DDTIBLayerAlgo test "
			  << dohmCarrier_lo_r.name() << " z+ number " <<i+1
			  << " positioned in " << parent().name() << " at "
			  << tran << " with " << rotation;
      // TIB+ DOHM Carrier - upper
      DDpos (dohmCarrier_up_r, parent(), i+1+dohmN, tran,
	     rotation );
      LogDebug("TIBGeom") << "DDTIBLayerAlgo test " 
			  << dohmCarrier_up_r.name() << " z+ number " <<i+1
			  << " positioned in " << parent().name() << " at "
			  << tran << " with " << rotation;
      // TIB- DOHM Carrier - lower
      tran = DDTranslation(0, 0, -0.5*layerL+dz_dohm);
      DDpos (dohmCarrier_lo_l, parent(), i+1, tran, rotation);
      LogDebug("TIBGeom") << "DDTIBLayerAlgo test "
			  << dohmCarrier_lo_l.name() << " z- number " <<i+1
			  << " positioned in " << parent().name() << " at "
			  << tran << " with " << rotation;
      // TIB- DOHM Carrier - upper
      DDpos (dohmCarrier_up_l, parent(), i+1, tran, rotation);
      LogDebug("TIBGeom") << "DDTIBLayerAlgo test "
			  << dohmCarrier_up_l.name() << " z- number " <<i+1
			  << " positioned in " << parent().name() << " at "
			  << tran << " with " << rotation;
    }
    
    //    } // phi range
  }
  
  // DOHM PRIMary + Auxiliary
  dphi = std::atan2(dohmPrimW, rout_dohm+0.5*dohmPrimT);
  name = idName + "DOHM_PRIM";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*dohmPrimL, 
			       rout_dohm, rout_dohm+dohmPrimT,
			       -0.5*dphi, dphi);
  DDLogicalPart dohmPrim(solid.ddname(), DDMaterial(dohmPrimMaterial), solid);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << solid.ddname()
		      << " Tubs made of " << dohmPrimMaterial 
		      << " of dimensions " << 0.5*dohmPrimL << ", "
		      << rout_dohm << ", " << rout_dohm+dohmPrimT << ", "
		      << -0.5*dphi/deg << ", " << dphi/deg;

  double dphi_cable = std::atan2(0.8*dohmPrimW, rout_dohm+0.5*dohmPrimT);
  name = idName + "DOHM_PRIM_Cable";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*dohmPrimL,
			       rout_dohm, rout_dohm+0.5*dohmPrimT,
			       -0.5*dphi_cable, dphi_cable);
  DDLogicalPart dohmCablePrim(solid.ddname(), DDMaterial(dohmCableMaterial), 
			      solid);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << solid.ddname()
		      << " Tubs made of " << dohmCableMaterial 
		      << " of dimensions " << 0.5*dohmPrimL << ", "
		      << rout_dohm << ", " << rout_dohm+0.5*dohmPrimT << ", "
		      << -0.5*dphi_cable/deg << ", " << dphi_cable/deg;

  dphi = std::atan2(dohmAuxW, rout_dohm+dohmCarrierT+0.5*dohmAuxT);
  name = idName + "DOHM_AUX";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*dohmAuxL, 
			       rout_dohm+dohmCarrierT, rout_dohm+dohmCarrierT+dohmAuxT,
			       -0.5*dphi, dphi);
  DDLogicalPart dohmAux(solid.ddname(), DDMaterial(dohmAuxMaterial), solid);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << solid.ddname()
		      << " Tubs made of " << dohmAuxMaterial 
		      << " of dimensions " << 0.5*dohmAuxL << ", "
		      << rout_dohm+dohmCarrierT << ", " 
		      << rout_dohm+dohmCarrierT+dohmAuxT << ", "
		      << -0.5*dphi/deg << ", " << dphi/deg;

  dphi_cable = std::atan2(0.8*dohmPrimW, rout_dohm+dohmCarrierT+0.5*dohmPrimT);
  name = idName + "DOHM_AUX_Cable";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*dohmPrimL,
			       rout_dohm+dohmCarrierT, rout_dohm+dohmCarrierT+0.5*dohmPrimT,
			       -0.5*dphi_cable, dphi_cable);
  DDLogicalPart dohmCableAux(solid.ddname(), DDMaterial(dohmCableMaterial),
			     solid);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: "  << solid.ddname()
		      << " Tubs made of " << dohmCableMaterial 
		      << " of dimensions " << 0.5*dohmPrimL << ", "
		      << rout_dohm+dohmCarrierT << ", " 
		      << rout_dohm+dohmCarrierT+0.5*dohmPrimT << ", "
		      << -0.5*dphi_cable/deg << ", " << dphi_cable/deg;

  DDRotation rotation_r, rotation_l;
  double phix   =  0.5*(std::atan2(dohmPrimW, rout_dohm+0.5*dohmPrimT)+std::atan2(0.8*dohmPrimW, rout_dohm+0.5*dohmPrimT));
  double phideg = phix/deg;
  if (phideg != 0) {
    double theta  = 90*deg;
    double phiy   = phix + 90.*deg;
    std::string rotstr = idName + dbl_to_string(phideg*10.);
    rotation_r = DDRotation(DDName(rotstr, idNameSpace));
    if (!rotation_r) {
      LogDebug("TIBGeom") << "DDTIBLayerAlgo test: Creating a new "
			  << "rotation: " << rotstr << "\t90., " 
			  << phix/deg << ", 90.," << phiy/deg 
			  << ", 0, 0";
      rotation_r = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, 
			 phiy, 0., 0.);
    }
  }
  phix   = -phix;
  phideg = phix/deg;
  if (phideg != 0) {
    double theta  = 90*deg;
    double phiy   = phix + 90.*deg;
    std::string rotstr = idName + dbl_to_string(phideg*10.);
    rotation_l = DDRotation(DDName(rotstr, idNameSpace));
    if (!rotation_l) {
      LogDebug("TIBGeom") << "DDTIBLayerAlgo test: Creating a new "
			  << "rotation: " << rotstr << "\t90., " 
			  << phix/deg << ", 90.," << phiy/deg 
			  << ", 0, 0";
      rotation_l = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, 
			 phiy, 0., 0.);
    }
  }
  
  // TIB+ DOHM (Primary Only)
  DDTranslation tran;
  DDpos (dohmPrim, dohmCarrierPrim_lo_r, 1, tran, DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmPrim.name()
		      << " z+ number " << 1	<< " positioned in " 
		      << dohmCarrierPrim_lo_r.name() << " at " << tran
		      << " with no rotation";
  DDpos (dohmCablePrim, dohmCarrierPrim_lo_r, 1, tran, rotation_r);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmCablePrim.name()
		      << " copy number 1 positioned in " 
		      << dohmCarrierPrim_lo_r.name() << " at " << tran 
		      << " with " << rotation_r;
  DDpos (dohmCablePrim, dohmCarrierPrim_lo_r, 2, tran, rotation_l);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmCablePrim.name()
		      << " copy number 2 positioned in " 
		      << dohmCarrierPrim_lo_r.name() << " at " << tran 
		      << " with " << rotation_l;
  // TIB- DOHM (Primary Only)
  DDpos (dohmPrim, dohmCarrierPrim_lo_l, 1, tran, DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmPrim.name() 
		      << " z+ number 1 positioned in " 
		      << dohmCarrierPrim_lo_l.name() << " at " << tran 
		      << " with no rotation";
  DDpos (dohmCablePrim, dohmCarrierPrim_lo_l, 1, tran, rotation_r);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmCablePrim.name()
		      << " copy number 1 positioned in " 
		      << dohmCarrierPrim_lo_l.name() << " at " << tran 
		      << " with " << rotation_r;
  DDpos (dohmCablePrim, dohmCarrierPrim_lo_l, 2, tran, rotation_l);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmCablePrim.name()
		      << " copy number 2 positioned in " 
		      << dohmCarrierPrim_lo_l.name() << " at " << tran
		      << " with " << rotation_l;
  
  // DOHM PRIMary + AUXiliary
  // TIB+ DOHM
  DDpos (dohmPrim, dohmCarrierPrimAux_lo_r, 1, tran, DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmAux.name() 
		      << " z+ number 1 positioned in " 
		      << dohmCarrierPrimAux_lo_r.name() << " at " << tran 
		      << " with no rotation";
  DDpos (dohmCablePrim, dohmCarrierPrimAux_lo_r, 1, tran, rotation_r);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmCablePrim.name()
		      << " copy number 1 positioned in " 
		      << dohmCarrierPrimAux_lo_r.name() << " at " << tran 
		      << " with " << rotation_r;
  DDpos (dohmCablePrim, dohmCarrierPrimAux_lo_r, 2, tran, rotation_l);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmCablePrim.name()
		      << " copy number 2 positioned in " 
		      << dohmCarrierPrimAux_lo_r.name() << " at " << tran 
		      << " with " << rotation_l;
  DDpos (dohmAux, dohmCarrierPrimAux_lo_r, 1, tran, DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmAux.name() 
		      << " z+ number 1 positioned in " 
		      << dohmCarrierPrimAux_lo_r.name()
		      << " at " << tran << " with no rotation";
  phix   = 0.5*(std::atan2(dohmAuxW, rout_dohm+dohmCarrierT+0.5*dohmAuxT)+
		std::atan2(0.8*dohmPrimW, rout_dohm+dohmCarrierT+0.5*dohmPrimT));
  phideg = phix/deg;
  DDRotation rotation_1;
  if (phideg != 0) {
    double theta  = 90*deg;
    double phiy   = phix + 90.*deg;
    std::string rotstr = idName + dbl_to_string(phideg*10.);
    rotation_1 = DDRotation(DDName(rotstr, idNameSpace));
    if (!rotation_1) {
      LogDebug("TIBGeom") << "DDTIBLayerAlgo test: Creating a new "
			  << "rotation: " << rotstr << "\t90., " 
			  << phix/deg << ", 90.," << phiy/deg 
			  << ", 0, 0";
      rotation_1 = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, 
			 phiy, 0., 0.);
    }
  }
  DDpos (dohmCableAux, dohmCarrierPrimAux_lo_r, 1, tran, rotation_1);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmCableAux.name()
		      << " copy number 1 positioned in " 
		      << dohmCarrierPrimAux_lo_r.name() << " at " << tran
		      << " with " << rotation_1;

  // TIB- DOHM
  DDpos (dohmPrim, dohmCarrierPrimAux_lo_l, 1, tran, DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmPrim.name() 
		      << " z+ number 1 positioned in " 
		      << dohmCarrierPrimAux_lo_l.name() << " at "<< tran
		      << " with no rotation";
  DDpos (dohmCablePrim, dohmCarrierPrimAux_lo_l, 1, tran, rotation_r);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmCablePrim.name()
		      << " copy number 1 positioned in " 
		      << dohmCarrierPrimAux_lo_l.name() << " at " << tran 
		      << " with " << rotation_r;
  DDpos (dohmCablePrim, dohmCarrierPrimAux_lo_l, 2, tran, rotation_l);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmCablePrim.name()
		      << " copy number 2 positioned in " 
		      << dohmCarrierPrimAux_lo_l.name() << " at " << tran 
		      << " with " << rotation_l;
  DDpos (dohmAux, dohmCarrierPrimAux_lo_l, 1, tran, DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmAux.name() 
		      << " z+ number " << 1	<< " positioned in "
		      << dohmCarrierPrimAux_lo_l.name()
		      << " at " << tran << " with no rotation";
  phix   =-0.5*(std::atan2(dohmAuxW, rout_dohm+dohmCarrierT+0.5*dohmAuxT)+
		std::atan2(0.8*dohmPrimW, rout_dohm+dohmCarrierT+0.5*dohmPrimT));
  phideg = phix/deg;
  DDRotation rotation_2;
  if (phideg != 0) {
    double theta  = 90*deg;
    double phiy   = phix + 90.*deg;
    std::string rotstr = idName + dbl_to_string(phideg*10.);
    rotation_2 = DDRotation(DDName(rotstr, idNameSpace));
    if (!rotation_2) {
      LogDebug("TIBGeom") << "DDTIBLayerAlgo test: Creating a new "
			  << "rotation: " << rotstr << "\t90., " 
			  << phix/deg << ", 90.," << phiy/deg 
			  << ", 0, 0";
      rotation_2 = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, 
			 phiy, 0., 0.);
    }
  }
  DDpos (dohmCableAux, dohmCarrierPrimAux_lo_l, 1, tran, rotation_2);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohmCableAux.name()
		      << " copy number 1 positioned in " 
		      << dohmCarrierPrimAux_lo_l.name() << " at " << tran
		      << " with " << rotation_2;
  
}
