///////////////////////////////////////////////////////////////////////////////
// File: DDTIBLayerAlgo.cc
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
#include "Geometry/TrackerCommonData/plugins/DDTIBLayerAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTIBLayerAlgo::DDTIBLayerAlgo(): ribW(0),ribPhi(0) {
  LogDebug("TIBGeom") << "DDTIBLayerAlgo info: Creating an instance";
}

DDTIBLayerAlgo::~DDTIBLayerAlgo() {}

void DDTIBLayerAlgo::initialize(const DDNumericArguments & nArgs,
				const DDVectorArguments & vArgs,
				const DDMapArguments & ,
				const DDStringArguments & sArgs,
				const DDStringVectorArguments & ) {
  DDCurrentNamespace ns;
  idNameSpace  = *ns;
  genMat       = sArgs["GeneralMaterial"];
  DDName parentName = parent().name(); 
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: Parent " << parentName 
		      << " NameSpace " << idNameSpace 
		      << " General Material " << genMat;

  detectorTilt = nArgs["DetectorTilt"];
  layerL       = nArgs["LayerL"];

  radiusLo     = nArgs["RadiusLo"];
  stringsLo    = int(nArgs["StringsLo"]);
  detectorLo   = sArgs["StringDetLoName"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: Lower layer Radius " 
		      << radiusLo << " Number " << stringsLo << " String "
		      << detectorLo;

  radiusUp     = nArgs["RadiusUp"];
  stringsUp    = int(nArgs["StringsUp"]);
  detectorUp   = sArgs["StringDetUpName"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: Upper layer Radius "
		      << radiusUp << " Number " << stringsUp << " String "
		      << detectorUp;

  cylinderT    = nArgs["CylinderThickness"];
  cylinderInR  = nArgs["CylinderInnerRadius"];
  cylinderMat  = sArgs["CylinderMaterial"];
  MFRingInR    = nArgs["MFRingInnerRadius"]; 
  MFRingOutR   = nArgs["MFRingOuterRadius"]; 
  MFRingT      = nArgs["MFRingThickness"];   
  MFRingDz     = nArgs["MFRingDeltaz"];      
  MFIntRingMat = sArgs["MFIntRingMaterial"];      
  MFExtRingMat = sArgs["MFExtRingMaterial"];      

  supportT     = nArgs["SupportThickness"];

  centMat      = sArgs["CentRingMaterial"];
  centRing1par = vArgs["CentRing1"];
  centRing2par = vArgs["CentRing2"];

  fillerMat    = sArgs["FillerMaterial"];
  fillerDz     = nArgs["FillerDeltaz"];

  ribMat       = sArgs["RibMaterial"];
  ribW         = vArgs["RibWidth"];
  ribPhi       = vArgs["RibPhi"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: Cylinder Material/"
		      << "thickness " << cylinderMat << " " << cylinderT 
		      << " Rib Material " << ribMat << " at "
		      << ribW.size() << " positions with width/phi";

  for (unsigned int i = 0; i < ribW.size(); i++)
    LogDebug("TIBGeom") << "\tribW[" << i << "] = " <<  ribW[i] 
			<< "\tribPhi[" << i << "] = " << ribPhi[i]/CLHEP::deg;
  
  dohmCarrierPhiOff   = nArgs["DOHMCarrierPhiOffset"];

  dohmtoMF            = nArgs["DOHMtoMFDist"];

  dohmPrimName         = sArgs["StringDOHMPrimName"];
  dohmAuxName          = sArgs["StringDOHMAuxName"];

  dohmCarrierMaterial = sArgs["DOHMCarrierMaterial"];
  dohmCableMaterial   = sArgs["DOHMCableMaterial"];
  dohmPrimL           = nArgs["DOHMPRIMLength"];
  dohmPrimMaterial    = sArgs["DOHMPRIMMaterial"];
  dohmAuxL            = nArgs["DOHMAUXLength"];
  dohmAuxMaterial     = sArgs["DOHMAUXMaterial"];
  dohmListFW          = vArgs["DOHMListFW"];
  dohmListBW          = vArgs["DOHMListBW"];
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: DOHM Primary "
		      << " Material " << dohmPrimMaterial << " Length " << dohmPrimL;
  LogDebug("TIBGeom") << "DDTIBLayerAlgo debug: DOHM Aux     "
		      << " Material " << dohmAuxMaterial << " Length " << dohmAuxL;
  for (double i : dohmListFW) {
    if (i>0.) LogDebug("TIBGeom") << "DOHM Primary at FW Position " << i;
    if (i<0.) LogDebug("TIBGeom") << "DOHM Aux     at FW Position " << -i;
  }
  for (double i : dohmListBW) {
    if (i>0.) LogDebug("TIBGeom") << "DOHM Primary at BW Position " << i;
    if (i<0.) LogDebug("TIBGeom") << "DOHM Aux     at BW Position " << -i;
  }

  //Pillar Material
  pillarMaterial        = sArgs["PillarMaterial"];

  // Internal Pillar Parameters
  fwIntPillarDz         = nArgs["FWIntPillarDz"];
  fwIntPillarDPhi       = nArgs["FWIntPillarDPhi"];
  fwIntPillarZ          = vArgs["FWIntPillarZ"];
  fwIntPillarPhi        = vArgs["FWIntPillarPhi"];
  bwIntPillarDz         = nArgs["BWIntPillarDz"];
  bwIntPillarDPhi       = nArgs["BWIntPillarDPhi"];
  bwIntPillarZ          = vArgs["BWIntPillarZ"];
  bwIntPillarPhi        = vArgs["BWIntPillarPhi"];
  LogDebug("TIBGeom") << "FW Internal Pillar [Dz, DPhi] " 
		      << fwIntPillarDz << ", " 
		      << fwIntPillarDPhi; 
  for (unsigned int i=0; i<fwIntPillarZ.size(); i++) {
    if( fwIntPillarPhi[i]>0. ) { 
      LogDebug("TIBGeom") << " at positions [z, phi] " 
			  << fwIntPillarZ[i] << " " << fwIntPillarPhi[i];
    }
  }
  LogDebug("TIBGeom") << "BW Internal Pillar [Dz, DPhi] " 
		      << bwIntPillarDz << ", " 
		      << bwIntPillarDPhi; 
  for (unsigned int i=0; i<bwIntPillarZ.size(); i++) {
    if( bwIntPillarPhi[i]>0. ) { 
      LogDebug("TIBGeom") << " at positions [z, phi] " 
			  << bwIntPillarZ[i] << " " << bwIntPillarPhi[i];
    }
  }

  // External Pillar Parameters
  fwExtPillarDz         = nArgs["FWExtPillarDz"];
  fwExtPillarDPhi       = nArgs["FWExtPillarDPhi"];
  fwExtPillarZ          = vArgs["FWExtPillarZ"];
  fwExtPillarPhi        = vArgs["FWExtPillarPhi"];
  bwExtPillarDz         = nArgs["BWExtPillarDz"];
  bwExtPillarDPhi       = nArgs["BWExtPillarDPhi"];
  bwExtPillarZ          = vArgs["BWExtPillarZ"];
  bwExtPillarPhi        = vArgs["BWExtPillarPhi"];
  LogDebug("TIBGeom") << "FW External Pillar [Dz, DPhi] " 
		      << fwExtPillarDz << ", " 
		      << fwExtPillarDPhi; 
  for (unsigned int i=0; i<fwExtPillarZ.size(); i++) {
    if( fwExtPillarPhi[i]>0. ) { 
      LogDebug("TIBGeom") << " at positions [z, phi] " 
			  << fwExtPillarZ[i] << " " << fwExtPillarPhi[i];
    }
  }
  LogDebug("TIBGeom") << "BW External Pillar [Dz, DPhi] " 
		      << bwExtPillarDz << ", " 
		      << bwExtPillarDPhi; 
  for (unsigned int i=0; i<bwExtPillarZ.size(); i++) {
    if( bwExtPillarPhi[i]>0. ) { 
      LogDebug("TIBGeom") << " at positions [z, phi] " 
			  << bwExtPillarZ[i] << " " << bwExtPillarPhi[i];
    }
  }
}

void DDTIBLayerAlgo::execute(DDCompactView& cpv) {

  LogDebug("TIBGeom") << "==>> Constructing DDTIBLayerAlgo...";

  DDName  parentName = parent().name(); 
  const std::string &idName = parentName.name();

  double rmin = MFRingInR;
  double rmax = MFRingOutR;

  DDSolid solid = DDSolidFactory::tubs(DDName(idName, idNameSpace), 0.5*layerL,
				       rmin, rmax, 0, CLHEP::twopi);

  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: "  
		      << DDName(idName,idNameSpace) << " Tubs made of " 
		      << genMat << " from 0 to " << CLHEP::twopi/CLHEP::deg 
		      << " with Rin " << rmin << " Rout " << rmax 
		      << " ZHalf " << 0.5*layerL;

  DDName matname(DDSplit(genMat).first, DDSplit(genMat).second);
  DDMaterial matter(matname);
  DDLogicalPart layer(solid.ddname(), matter, solid);

  //Internal layer first
  double rin  = rmin+MFRingT;
  //  double rout = 0.5*(radiusLo+radiusUp-cylinderT);
  double rout = cylinderInR;
  std::string name = idName + "Down";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, CLHEP::twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << genMat << " from 0 to " << CLHEP::twopi/CLHEP::deg 
		      << " with Rin " << rin << " Rout " << rout 
		      << " ZHalf " << 0.5*layerL;
  DDLogicalPart layerIn(solid.ddname(), matter, solid);
 cpv.position(layerIn, layer, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << layerIn.name()
		      << " number 1 positioned in " << layer.name()
		      << " at (0,0,0) with no rotation";

  double rposdet = radiusLo;
  double dphi    = CLHEP::twopi/stringsLo;
  DDName detIn(DDSplit(detectorLo).first, DDSplit(detectorLo).second);
  for (int n = 0; n < stringsLo; n++) {
    double phi    = (n+0.5)*dphi;
    double phix   = phi - detectorTilt + 90*CLHEP::deg;
    double phideg = phix/CLHEP::deg;
    DDRotation rotation;
    if (phideg != 0) {
      double theta  = 90*CLHEP::deg;
      double phiy   = phix + 90.*CLHEP::deg;
      std::string rotstr = idName + std::to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
        LogDebug("TIBGeom") << "DDTIBLayerAlgo test: Creating a new "
			    << "rotation: "	<< rotstr << "\t90., " 
			    << phix/CLHEP::deg << ", 90.,"
			    << phiy/CLHEP::deg << ", 0, 0";
        rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
                         0., 0.);
      }
    }
    DDTranslation trdet(rposdet*cos(phi), rposdet*sin(phi), 0);
   cpv.position(detIn, layerIn, n+1, trdet, rotation);
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << detIn.name() 
			<< " number " << n+1 << " positioned in " 
			<< layerIn.name() << " at " << trdet << " with "
			<< rotation;
  }

  //Now the external layer
  rin  = cylinderInR + cylinderT;
  rout = rmax-MFRingT;
  name = idName + "Up";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, CLHEP::twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << genMat << " from 0 to " << CLHEP::twopi/CLHEP::deg 
		      << " with Rin " << rin << " Rout " << rout
		      << " ZHalf " << 0.5*layerL;
  DDLogicalPart layerOut(solid.ddname(), matter, solid);
 cpv.position(layerOut, layer, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << layerOut.name() 
		      << " number 1 positioned in " << layer.name() 
		      << " at (0,0,0) with no rotation";

  rposdet = radiusUp;
  dphi    = CLHEP::twopi/stringsUp;
  DDName detOut(DDSplit(detectorUp).first, DDSplit(detectorUp).second);
  for (int n = 0; n < stringsUp; n++) {
    double phi    = (n+0.5)*dphi;
    double phix   = phi - detectorTilt - 90*CLHEP::deg;
    double phideg = phix/CLHEP::deg;
    DDRotation rotation;
    if (phideg != 0) {
      double theta  = 90*CLHEP::deg;
      double phiy   = phix + 90.*CLHEP::deg;
      std::string rotstr = idName + std::to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
        LogDebug("TIBGeom") << "DDTIBLayerAlgo test: Creating a new "
			    << "rotation: " << rotstr << "\t90., " 
			    << phix/CLHEP::deg << ", 90.,"
			    << phiy/CLHEP::deg << ", 0, 0";
        rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
                         0., 0.);
      }
    }
    DDTranslation trdet(rposdet*cos(phi), rposdet*sin(phi), 0);
   cpv.position(detOut, layerOut, n+1, trdet, rotation);
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << detOut.name() 
			<< " number " << n+1 << " positioned in " 
			<< layerOut.name() << " at " << trdet << " with "
			<< rotation;
  }

  //
  // Inner cylinder, support wall and ribs
  //
  // External skins
  rin  = cylinderInR;
  rout = cylinderInR+cylinderT;
  name = idName + "Cylinder";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, CLHEP::twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << cylinderMat << " from 0 to " 
		      << CLHEP::twopi/CLHEP::deg << " with Rin " << rin 
		      << " Rout " << rout << " ZHalf " << 0.5*layerL;
  matname = DDName(DDSplit(cylinderMat).first, DDSplit(cylinderMat).second);
  DDMaterial matcyl(matname);
  DDLogicalPart cylinder(solid.ddname(), matcyl, solid);
 cpv.position(cylinder, layer, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << cylinder.name() 
		      << " number 1 positioned in " << layer.name()
		      << " at (0,0,0) with no rotation";
  //
  // inner part of the cylinder
  //
  rin  += supportT;
  rout -= supportT;
  name  = idName + "CylinderIn";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, CLHEP::twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: "
		      << DDName(name, idNameSpace) << " Tubs made of "
		      << genMat << " from 0 to " << CLHEP::twopi/CLHEP::deg 
		      << " with Rin " << rin << " Rout " << rout 
		      << " ZHalf " << 0.5*layerL;
  DDLogicalPart cylinderIn(solid.ddname(), matter, solid);
 cpv.position(cylinderIn, cylinder, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << cylinderIn.name() 
		      << " number 1 positioned in " << cylinder.name() 
		      << " at (0,0,0) with no rotation";
  //
  // Filler Rings
  //
  matname = DDName(DDSplit(fillerMat).first, DDSplit(fillerMat).second);
  DDMaterial matfiller(matname);
  name = idName + "Filler";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), fillerDz, rin, rout, 
			       0., CLHEP::twopi);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << fillerMat << " from " << 0. << " to "
		      << CLHEP::twopi/CLHEP::deg << " with Rin " << rin 
		      << " Rout " << rout << " ZHalf "  << fillerDz;
  DDLogicalPart cylinderFiller(solid.ddname(), matfiller, solid);
 cpv.position(cylinderFiller, cylinderIn, 1, DDTranslation(0.0, 0.0, 0.5*layerL-fillerDz), DDRotation());
 cpv.position(cylinderFiller, cylinderIn, 2, DDTranslation(0.0, 0.0,-0.5*layerL+fillerDz), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << cylinderFiller.name()
		      << " number 1" << " positioned in " 
		      << cylinderIn.name() << " at " << DDTranslation(0.0, 0.0, 0.5*layerL-fillerDz)
		      << " number 2" << " positioned in " 
		      << cylinderIn.name() << " at " << DDTranslation(0.0, 0.0,-0.5*layerL+fillerDz);

  //
  // Ribs
  //
  matname = DDName(DDSplit(ribMat).first, DDSplit(ribMat).second);
  DDMaterial matrib(matname);
  for (int i = 0; i < (int)(ribW.size()); i++) {
    name = idName + "Rib" + std::to_string(i);
    double width = 2.*ribW[i]/(rin+rout);
    double dz    = 0.5*layerL-2.*fillerDz;
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, 
				 rin+0.5*CLHEP::mm, rout-0.5*CLHEP::mm, 
				 -0.5*width, width);
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
			<< DDName(name, idNameSpace) << " Tubs made of " 
			<< ribMat << " from " << -0.5*width/CLHEP::deg <<" to "
			<< 0.5*width/CLHEP::deg << " with Rin " 
			<< rin+0.5*CLHEP::mm << " Rout " 
			<< rout-0.5*CLHEP::mm << " ZHalf "  << dz;
    DDLogicalPart cylinderRib(solid.ddname(), matrib, solid);
    double phix   = ribPhi[i];
    double phideg = phix/CLHEP::deg;
    DDRotation rotation;
    if (phideg != 0) {
      double theta  = 90*CLHEP::deg;
      double phiy   = phix + 90.*CLHEP::deg;
      std::string rotstr = idName + std::to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
        LogDebug("TIBGeom") << "DDTIBLayerAlgo test: Creating a new "
			    << "rotation: "	<< rotstr << "\t90., " 
			    << phix/CLHEP::deg << ", 90.," << phiy/CLHEP::deg 
			    << ", 0, 0";
        rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
                         0., 0.);
      }
    }
    DDTranslation tran(0, 0, 0);
   cpv.position(cylinderRib, cylinderIn, 1, tran, rotation);
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << cylinderRib.name()
			<< " number 1" << " positioned in " 
			<< cylinderIn.name() << " at " << tran << " with " 
			<< rotation;
  }

  //Manifold rings
  //
  // Inner ones first
  matname = DDName(DDSplit(MFIntRingMat).first, DDSplit(MFIntRingMat).second);
  DDMaterial matintmfr(matname);
  rin  = MFRingInR;
  rout = rin + MFRingT;
  name = idName + "InnerMFRing";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), MFRingDz,
			       rin, rout, 0, CLHEP::twopi);

  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << MFIntRingMat << " from 0 to " 
		      << CLHEP::twopi/CLHEP::deg << " with Rin " << rin 
		      << " Rout " << rout << " ZHalf " << MFRingDz;

  DDLogicalPart inmfr(solid.ddname(), matintmfr, solid);
 cpv.position(inmfr, layer, 1, DDTranslation(0.0, 0.0, -0.5*layerL+MFRingDz), DDRotation());
 cpv.position(inmfr, layer, 2, DDTranslation(0.0, 0.0, +0.5*layerL-MFRingDz), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << inmfr.name() 
		      << " number 1 and 2 positioned in " << layer.name()
		      << " at (0,0,+-" << 0.5*layerL-MFRingDz << ") with no rotation";
  // Outer ones
  matname = DDName(DDSplit(MFExtRingMat).first, DDSplit(MFExtRingMat).second);
  DDMaterial matextmfr(matname);
  rout  = MFRingOutR;
  rin   = rout - MFRingT;
  name = idName + "OuterMFRing";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), MFRingDz,
			       rin, rout, 0, CLHEP::twopi);

  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << MFExtRingMat << " from 0 to " 
		      << CLHEP::twopi/CLHEP::deg << " with Rin " << rin 
		      << " Rout " << rout << " ZHalf " << MFRingDz;

  DDLogicalPart outmfr(solid.ddname(), matextmfr, solid);
 cpv.position(outmfr, layer, 1, DDTranslation(0.0, 0.0, -0.5*layerL+MFRingDz), DDRotation());
 cpv.position(outmfr, layer, 2, DDTranslation(0.0, 0.0, +0.5*layerL-MFRingDz), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << outmfr.name() 
		      << " number 1 and 2 positioned in " << layer.name()
		      << " at (0,0,+-" << 0.5*layerL-MFRingDz 
		      << ") with no rotation";

  //Central Support rings
  //
  matname = DDName(DDSplit(centMat).first, DDSplit(centMat).second);
  DDMaterial matcent(matname);
  // Ring 1
  double centZ  = centRing1par[0];
  double centDz = 0.5*centRing1par[1];
  rin  = centRing1par[2];
  rout = centRing1par[3];
  name = idName + "CentRing1";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), centDz,
			       rin, rout, 0, CLHEP::twopi);

  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << centMat << " from 0 to " << CLHEP::twopi/CLHEP::deg 
		      << " with Rin " << rin << " Rout " << rout 
		      << " ZHalf " << centDz;

  DDLogicalPart cent1(solid.ddname(), matcent, solid);
 cpv.position(cent1, layer, 1, DDTranslation(0.0, 0.0, centZ), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << cent1.name() 
		      << " positioned in " << layer.name()
		      << " at (0,0," << centZ << ") with no rotation";
  // Ring 2
  centZ  = centRing2par[0];
  centDz = 0.5*centRing2par[1];
  rin  = centRing2par[2];
  rout = centRing2par[3];
  name = idName + "CentRing2";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), centDz,
			       rin, rout, 0, CLHEP::twopi);

  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << centMat << " from 0 to " << CLHEP::twopi/CLHEP::deg 
		      << " with Rin " << rin << " Rout " << rout 
		      << " ZHalf " << centDz;

  DDLogicalPart cent2(solid.ddname(), matcent, solid);
 cpv.position(cent2, layer, 1, DDTranslation(0.0, 0.0, centZ), DDRotation());
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " << cent2.name() 
		      << " positioned in " << layer.name()
		      << " at (0,0," << centZ << ") with no rotation";

  ////// DOHM
  //
  // Preparing DOHM Carrier solid

  name = idName + "DOHMCarrier";

  double dohmCarrierRin   = MFRingOutR - MFRingT;
  double dohmCarrierRout  = MFRingOutR;
  double dohmCarrierDz    = 0.5*(dohmPrimL+dohmtoMF);
  double dohmCarrierZ     = 0.5*layerL-2.*MFRingDz-dohmCarrierDz;

  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dohmCarrierDz, 
			       dohmCarrierRin, dohmCarrierRout, 
			       dohmCarrierPhiOff, 
			       180.*CLHEP::deg-2.*dohmCarrierPhiOff);
  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of "
		      << dohmCarrierMaterial << " from "
		      << dohmCarrierPhiOff << " to " 
		      << 180.*CLHEP::deg-dohmCarrierPhiOff << " with Rin "
		      << dohmCarrierRin << " Rout " << MFRingOutR << " ZHalf " 
		      << dohmCarrierDz;


  // Define FW and BW carrier logical volume and
  // place DOHM Primary and auxiliary modules inside it

  dphi = CLHEP::twopi/stringsUp;

  DDRotation dohmRotation;

  double dohmR = 0.5*(dohmCarrierRin+dohmCarrierRout);


  for (int j = 0; j<4; j++) {

    std::vector<double> dohmList;
    DDTranslation tran;
    std::string rotstr;
    DDRotation rotation;
    int dohmCarrierReplica=0;
    int placeDohm = 0;

    switch (j){
    case 0:
      name = idName + "DOHMCarrierFW";
      dohmList = dohmListFW;
      tran = DDTranslation(0., 0., dohmCarrierZ);
      rotstr = idName + "FwUp";
      rotation = DDRotation();
      dohmCarrierReplica = 1;
      placeDohm=1;
      break;
    case 1:
      name = idName + "DOHMCarrierFW";
      dohmList = dohmListFW;
      tran = DDTranslation(0., 0., dohmCarrierZ);
      rotstr = idName + "FwDown";
      rotation = DDrot(DDName(rotstr, idNameSpace), 90.*CLHEP::deg, 
		       180.*CLHEP::deg, 90.*CLHEP::deg,270.*CLHEP::deg, 0.,0.);
      dohmCarrierReplica = 2;
      placeDohm=0;
      break;
    case 2:
      name = idName + "DOHMCarrierBW";
      dohmList = dohmListBW;
      tran = DDTranslation(0., 0., -dohmCarrierZ);
      rotstr = idName + "BwUp";
      rotation = DDrot(DDName(rotstr, idNameSpace), 90.*CLHEP::deg, 
		       180.*CLHEP::deg, 90.*CLHEP::deg, 90.*CLHEP::deg, 
		       180.*CLHEP::deg, 0.);
      dohmCarrierReplica = 1;
      placeDohm=1;
      break;
    case 3:
      name = idName + "DOHMCarrierBW";
      dohmList = dohmListBW;
      tran = DDTranslation(0., 0., -dohmCarrierZ);
      rotstr = idName + "BwDown";
      rotation = DDrot(DDName(rotstr, idNameSpace), 90.*CLHEP::deg, 0., 
		       90.*CLHEP::deg, 270.*CLHEP::deg, 180.*CLHEP::deg, 0.);
      dohmCarrierReplica = 2;
      placeDohm=0;
      break;
    }

    DDLogicalPart dohmCarrier(name,DDMaterial(dohmCarrierMaterial),solid);

    int primReplica = 0;
    int auxReplica = 0;

    for (int i = 0; i < placeDohm*((int)(dohmList.size())); i++) {

      double phi    = (std::abs(dohmList[i])+0.5-1.)*dphi;
      double phix   = phi + 90*CLHEP::deg;
      double phideg = phix/CLHEP::deg;
      if (phideg != 0) {
	double theta  = 90*CLHEP::deg;
	double phiy   = phix + 90.*CLHEP::deg;
	std::string   rotstr = idName + std::to_string(std::abs(dohmList[i])-1.);
	dohmRotation = DDRotation(DDName(rotstr, idNameSpace));
	if (!dohmRotation) {
	  LogDebug("TIBGeom") << "DDTIBLayerAlgo test: Creating a new "
			      << "rotation: "	<< rotstr << "\t" << theta 
			      << ", " << phix/CLHEP::deg << ", " << theta 
			      << ", " << phiy/CLHEP::deg <<", 0, 0";
	  dohmRotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta,
			       phiy, 0., 0.);
	}
      }
      
      std::string dohmName;
      int dohmReplica = 0;
      double dohmZ = 0.;
      
      if(dohmList[i]<0.) {
	// Place a Auxiliary DOHM
	dohmName = dohmAuxName;
	dohmZ = dohmCarrierDz - 0.5*dohmAuxL - dohmtoMF;
	primReplica++;
	dohmReplica = primReplica;
	
      } else {
	// Place a Primary DOHM
	dohmName = dohmPrimName;
	dohmZ = dohmCarrierDz - 0.5*dohmPrimL - dohmtoMF;
	auxReplica++;
	dohmReplica = auxReplica;
      }
      
      DDName dohm(DDSplit(dohmName).first, DDSplit(dohmName).second);
      DDTranslation dohmTrasl(dohmR*cos(phi), dohmR*sin(phi), dohmZ);
     cpv.position(dohm, dohmCarrier, dohmReplica, dohmTrasl, dohmRotation);
      LogDebug("TIBGeom") << "DDTIBLayerAlgo test " << dohm.name() 
			  << " replica " << dohmReplica << " positioned in " 
			  << dohmCarrier.name() << " at " << dohmTrasl << " with "
			  << dohmRotation;
      
    }
    
    
   cpv.position(dohmCarrier, parent(), dohmCarrierReplica, tran, rotation );
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test "
			<< dohmCarrier.name() << " positioned in " << parent().name() << " at "
			<< tran << " with " << rotation;
    
  }

  ////// PILLARS

  for (int j = 0; j<4; j++) {
    
    matname = DDName(DDSplit(pillarMaterial).first, DDSplit(pillarMaterial).second);
    DDMaterial pillarMat(matname);
    std::vector<double> pillarZ;
    std::vector<double> pillarPhi;
    double pillarDz=0, pillarDPhi=0, pillarRin=0, pillarRout=0;
    
    switch (j){
    case 0:
      name = idName + "FWIntPillar";
      pillarZ    = fwIntPillarZ;
      pillarPhi  = fwIntPillarPhi;
      pillarRin  = MFRingInR;
      pillarRout = MFRingInR + MFRingT;
      pillarDz   = fwIntPillarDz;
      pillarDPhi = fwIntPillarDPhi;
      break;
    case 1:
      name = idName + "BWIntPillar";
      pillarZ    = bwIntPillarZ;
      pillarPhi  = bwIntPillarPhi;
      pillarRin  = MFRingInR;
      pillarRout = MFRingInR + MFRingT;
      pillarDz   = bwIntPillarDz;
      pillarDPhi = bwIntPillarDPhi;
      break;
    case 2:
      name = idName + "FWExtPillar";
      pillarZ    = fwExtPillarZ;
      pillarPhi  = fwExtPillarPhi;
      pillarRin  = MFRingOutR - MFRingT;
      pillarRout = MFRingOutR;
      pillarDz   = fwExtPillarDz;
      pillarDPhi = fwExtPillarDPhi;
      break;
    case 3:
      name = idName + "BWExtPillar";
      pillarZ    = bwExtPillarZ;
      pillarPhi  = bwExtPillarPhi;
      pillarRin  = MFRingOutR - MFRingT;
      pillarRout = MFRingOutR;
      pillarDz   = bwExtPillarDz;
      pillarDPhi = bwExtPillarDPhi;
      break;
    }
    
    
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), pillarDz, 
				 pillarRin, pillarRout, 
				 -pillarDPhi, 2.*pillarDPhi);
    
    DDLogicalPart Pillar(name,DDMaterial(pillarMat),solid);
    
    LogDebug("TIBGeom") << "DDTIBLayerAlgo test: " 
			<< DDName(name, idNameSpace) << " Tubs made of "
			<< pillarMat << " from "
			<< -pillarDPhi << " to " 
			<< pillarDPhi << " with Rin "
			<< pillarRin << " Rout " << pillarRout << " ZHalf "  
			<< pillarDz;
    
    DDTranslation pillarTran;
    DDRotation pillarRota;
    int pillarReplica = 0;
    for (unsigned int i=0; i<pillarZ.size(); i++) {
      if( pillarPhi[i]>0. ) {
	
	pillarTran = DDTranslation(0., 0., pillarZ[i]);
	pillarRota = DDanonymousRot(std::unique_ptr<DDRotationMatrix>(DDcreateRotationMatrix(90.*CLHEP::deg, pillarPhi[i], 90.*CLHEP::deg,
											     90.*CLHEP::deg+pillarPhi[i], 0., 0.)));
	
	cpv.position(Pillar, parent(), i, pillarTran, pillarRota);
	LogDebug("TIBGeom") << "DDTIBLayerAlgo test "
			    << Pillar.name() << " positioned in " 
			    << parent().name() << " at "
			    << pillarTran << " with " << pillarRota 
			    << " copy number " << pillarReplica;
	
	pillarReplica++;
      }

    }
    
  }

}
