#define DEBUG 0
#define COUT if (DEBUG) cout
///////////////////////////////////////////////////////////////////////////////
// File: DDTIBLayerAlgo.cc
// Description: Makes a TIB layer and position the strings with a tilt angle
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "Geometry/TrackerSimData/interface/DDTIBLayerAlgo.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTIBLayerAlgo::DDTIBLayerAlgo(): ribW(0),ribPhi(0) {
  COUT << "DDTIBLayerAlgo info: Creating an instance" << endl;
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
  COUT << "DDTIBLayerAlgo debug: Parent " << parentName 
                << " NameSpace " << idNameSpace << " General Material " 
                << genMat << endl;

  detectorTilt = nArgs["DetectorTilt"];
  layerL       = nArgs["LayerL"];
  detectorTol  = nArgs["LayerTolerance"];
  detectorW    = nArgs["DetectorWidth"];
  detectorT    = nArgs["DetectorThickness"];
  coolTubeW    = nArgs["CoolTubeWidth"];
  coolTubeT    = nArgs["CoolTubeThickness"];
  COUT << "DDTIBLayerAlgo debug: Tilt Angle " << detectorTilt/deg
		<< " Layer Length/tolerance " << layerL << " " << detectorTol
		<< " Detector layer Width/Thick " << detectorW << ", " 
		<< detectorT << " Cooling Tube/Cable layer Width/Thick " 
		<< coolTubeW << ", " << coolTubeT << endl;

  radiusLo     = nArgs["RadiusLo"];
  phioffLo     = nArgs["PhiOffsetLo"];
  stringsLo    = int(nArgs["StringsLo"]);
  detectorLo   = sArgs["StringDetLoName"];
  roffDetLo    = nArgs["ROffsetDetLo"];
  coolCableLo  = sArgs["StringCabLoName"];
  roffCableLo  = nArgs["ROffsetCabLo"];
  COUT << "DDTIBLayerAlgo debug: Lower layer Radius " << radiusLo
		<< " Phi offset " << phioffLo/deg << " Number " << stringsLo
		<< " String " << detectorLo << " at offset " << roffDetLo
		<< " String " << coolCableLo << " at offset " << roffCableLo
		<< endl;

  radiusUp     = nArgs["RadiusUp"];
  phioffUp     = nArgs["PhiOffsetUp"];
  stringsUp    = int(nArgs["StringsUp"]);
  detectorUp   = sArgs["StringDetUpName"];
  roffDetUp    = nArgs["ROffsetDetUp"];
  coolCableUp  = sArgs["StringCabUpName"];
  roffCableUp  = nArgs["ROffsetCabUp"];
  COUT << "DDTIBLayerAlgo debug: Upper layer Radius " << radiusUp
		<< " Phi offset " << phioffUp/deg << " Number " << stringsUp
		<< " String " << detectorUp << " at offset " << roffDetUp
		<< " String " << coolCableUp << " at offset " << roffCableUp
		<< endl;

  cylinderT    = nArgs["CylinderThickness"];
  cylinderMat  = sArgs["CylinderMaterial"];
  supportW     = nArgs["SupportWidth"];
  supportT     = nArgs["SupportThickness"];
  supportMat   = sArgs["SupportMaterial"];
  ribMat       = sArgs["RibMaterial"];
  ribW         = vArgs["RibWidth"];
  ribPhi       = vArgs["RibPhi"];
  COUT << "DDTIBLayerAlgo debug: Cylinder Material/thickness " 
		<< cylinderMat << " " << cylinderT << " Support Wall Material/"
		<< "Width/Thickness " << supportMat << " " << supportW << " "
		<< supportT << " Rib Material " << ribMat << " at "
		<< ribW.size() << " positions with width/phi " << endl;
  for (unsigned int i = 0; i < ribW.size(); i++)
    COUT << " " << i << " " <<  ribW[i] << " " << ribPhi[i]/deg;
  COUT << endl;
}

void DDTIBLayerAlgo::execute() {

  COUT << "==>> Constructing DDTIBLayerAlgo..." << endl;

  //Parameters for the tilt of the layer
  double rotsi  = abs(detectorTilt);
  double redgd1 = 0.5*(detectorW*sin(rotsi)+detectorT*cos(rotsi));
  double redgd2 = 0.5*(detectorW*cos(rotsi)-detectorT*sin(rotsi));
  double redgc1 = 0.5*(coolTubeW*sin(rotsi)+coolTubeT*cos(rotsi));
  double redgc2 = 0.5*(coolTubeW*cos(rotsi)-coolTubeT*sin(rotsi));
  COUT << "DDTIBLayerAlgo test DeltaR (Detector Tilt) " << redgd1 
	       << ", " << redgd2 << " DeltaR (Cable+Cool) " << redgc1
	       << ", " << redgc2 << endl;

  DDName parentName = parent().name(); 
  string idName = DDSplit(parentName).first;
  double rmin = radiusLo + roffDetLo - redgd1 - detectorTol;
  double rmax = sqrt((radiusUp+roffDetUp+redgd1)*(radiusUp+roffDetUp+redgd1)+
		     redgd2*redgd2) + detectorTol;
  DDSolid solid = DDSolidFactory::tubs(DDName(idName, idNameSpace), 0.5*layerL,
				       rmin, rmax, 0, twopi);
  COUT << "DDTIBLayerAlgo test: " << DDName(idName, idNameSpace) 
	       << " Tubs made of " << genMat << " from 0 to " << twopi/deg 
	       << " with Rin " << rmin << " Rout " << rmax << " ZHalf " 
	       << 0.5*layerL << endl;
  DDName matname(DDSplit(genMat).first, DDSplit(genMat).second);
  DDMaterial matter(matname);
  DDLogicalPart layer(solid.ddname(), matter, solid);

  //Lower part first
  double rin  = rmin;
  double rout = 0.5*(radiusLo+radiusUp-cylinderT);
  string name = idName + "Down";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, twopi);
  COUT << "DDTIBLayerAlgo test: " << DDName(name, idNameSpace) 
	       << " Tubs made of " << genMat << " from 0 to " << twopi/deg 
	       << " with Rin " << rin << " Rout " << rout << " ZHalf " 
	       << 0.5*layerL << endl;
  DDLogicalPart layerIn(solid.ddname(), matter, solid);
  DDpos (layerIn, layer, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  COUT << "DDTIBLayerAlgo test: " << layerIn.name() 
               << " number 1 positioned in " << layer.name()
               << " at (0,0,0) with no rotation" << endl;

  double rposdet = radiusLo + roffDetLo;
  double rposcab = rposdet + roffCableLo;
  double dphi    = twopi/stringsLo;
  DDName detIn(DDSplit(detectorLo).first, DDSplit(detectorLo).second);
  DDName cabIn(DDSplit(coolCableLo).first, DDSplit(coolCableLo).second);
  for (int n = 0; n < stringsLo; n++) {
    double phi    = phioffLo + n*dphi;
    double phix   = phi - detectorTilt + 90*deg;
    double phideg = phix/deg;
    DDRotation rotation;
    if (phideg != 0) {
      double theta  = 90*deg;
      double phiy   = phix + 90.*deg;
      string rotstr = idName + dbl_to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
        COUT << "DDTIBLayer test: Creating a new rotation: " 
                     << rotstr << "\t90., " << phix/deg << ", 90.," << phiy/deg
                     << ", 0, 0" << endl;
        rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
                         0., 0.);
      }
    }
    DDTranslation trdet(rposdet*cos(phi), rposdet*sin(phi), 0);
    DDpos (detIn, layerIn, n+1, trdet, rotation);
    COUT << "DDTIBLayer test " << detIn.name() << " number " << n+1
                 << " positioned in " << layerIn.name() << " at " << trdet 
		 << " with " << rotation << endl;
    DDTranslation trcab(rposcab*cos(phi), rposcab*sin(phi), 0);
    DDpos (cabIn, layerIn, n+1, trcab, rotation);
    COUT << "DDTIBLayer test " << cabIn.name() << " number " << n+1
                 << " positioned in " << layerIn.name() << " at " << trcab 
		 << " with " << rotation << endl;
  }

  //Now the upper part
  rin  = 0.5*(radiusLo+radiusUp+cylinderT);
  rout = rmax;
  name = idName + "Up";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, twopi);
  COUT << "DDTIBLayerAlgo test: " << DDName(name, idNameSpace) 
	       << " Tubs made of " << genMat << " from 0 to " << twopi/deg 
	       << " with Rin " << rin << " Rout " << rout << " ZHalf " 
	       << 0.5*layerL << endl;
  DDLogicalPart layerOut(solid.ddname(), matter, solid);
  DDpos (layerOut, layer, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  COUT << "DDTIBLayerAlgo test: " << layerOut.name() 
               << " number 1 positioned in " << layer.name()
               << " at (0,0,0) with no rotation" << endl;

  rposdet = radiusUp + roffDetUp;
  rposcab = rposdet + roffCableUp;
  dphi    = twopi/stringsUp;
  DDName detOut(DDSplit(detectorUp).first, DDSplit(detectorUp).second);
  DDName cabOut(DDSplit(coolCableUp).first, DDSplit(coolCableUp).second);
  for (int n = 0; n < stringsUp; n++) {
    double phi    = phioffUp + n*dphi;
    double phix   = phi - detectorTilt - 90*deg;
    double phideg = phix/deg;
    DDRotation rotation;
    if (phideg != 0) {
      double theta  = 90*deg;
      double phiy   = phix + 90.*deg;
      string rotstr = idName + dbl_to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
        COUT << "DDTIBLayer test: Creating a new rotation: " 
                     << rotstr << "\t90., " << phix/deg << ", 90.," << phiy/deg
                     << ", 0, 0" << endl;
        rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
                         0., 0.);
      }
    }
    DDTranslation trdet(rposdet*cos(phi), rposdet*sin(phi), 0);
    DDpos (detOut, layerOut, n+1, trdet, rotation);
    COUT << "DDTIBLayer test " << detOut.name() << " number " << n+1
                 << " positioned in " << layerOut.name() << " at " << trdet 
		 << " with " << rotation << endl;
    DDTranslation trcab(rposcab*cos(phi), rposcab*sin(phi), 0);
    DDpos (cabOut, layerOut, n+1, trcab, rotation);
    COUT << "DDTIBLayer test " << cabOut.name() << " number " << n+1
                 << " positioned in " << layerOut.name() << " at " << trcab 
		 << " with " << rotation << endl;
  }

  //Finally the inner cylinder, support wall and ribs
  rin  = 0.5*(radiusLo+radiusUp-cylinderT);
  rout = 0.5*(radiusLo+radiusUp+cylinderT);
  name = idName + "Cylinder";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, twopi);
  COUT << "DDTIBLayerAlgo test: " << DDName(name, idNameSpace) 
	       << " Tubs made of " << cylinderMat << " from 0 to " << twopi/deg
	       << " with Rin " << rin << " Rout " << rout << " ZHalf " 
	       << 0.5*layerL << endl;
  matname = DDName(DDSplit(cylinderMat).first, DDSplit(cylinderMat).second);
  DDMaterial matcyl(matname);
  DDLogicalPart cylinder(solid.ddname(), matcyl, solid);
  DDpos (cylinder, layer, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  COUT << "DDTIBLayerAlgo test: " << cylinder.name() 
               << " number 1 positioned in " << layer.name()
               << " at (0,0,0) with no rotation" << endl;
  rin  += supportT;
  rout -= supportT;
  name  = idName + "CylinderIn";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*layerL,
			       rin, rout, 0, twopi);
  COUT << "DDTIBLayerAlgo test: " << DDName(name, idNameSpace) 
	       << " Tubs made of " << genMat << " from 0 to " << twopi/deg
	       << " with Rin " << rin << " Rout " << rout << " ZHalf " 
	       << 0.5*layerL << endl;
  DDLogicalPart cylinderIn(solid.ddname(), matter, solid);
  DDpos (cylinderIn, cylinder, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  COUT << "DDTIBLayerAlgo test: " << cylinderIn.name() 
               << " number 1 positioned in " << cylinder.name()
               << " at (0,0,0) with no rotation" << endl;
  name  = idName + "CylinderInSup";
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*supportW,
			       rin, rout, 0, twopi);
  COUT << "DDTIBLayerAlgo test: " << DDName(name, idNameSpace) 
	       << " Tubs made of " << genMat << " from 0 to " << twopi/deg
	       << " with Rin " << rin << " Rout " << rout << " ZHalf " 
	       << 0.5*supportW << endl;
  matname = DDName(DDSplit(supportMat).first, DDSplit(supportMat).second);
  DDMaterial matsup(matname);
  DDLogicalPart cylinderSup(solid.ddname(), matsup, solid);
  DDpos (cylinderSup, cylinderIn, 1, DDTranslation(0., 0., 0.), DDRotation());
  COUT << "DDTIBLayerAlgo test: " << cylinderSup.name() 
               << " number 1 positioned in " << cylinderIn.name()
               << " at (0,0,0) with no rotation" << endl;
  matname = DDName(DDSplit(ribMat).first, DDSplit(ribMat).second);
  DDMaterial matrib(matname);
  for (unsigned int i = 0; i < ribW.size(); i++) {
    name = idName + "Rib" + dbl_to_string(i);
    double width = 2.*ribW[i]/(rin+rout);
    double dz    = 0.25*(layerL - supportW);
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, rout, 
				 -0.5*width, width);
    COUT << "DDTIBLayerAlgo test: " << DDName(name, idNameSpace) 
		 << " Tubs made of " << ribMat << " from " << -0.5*width/deg
		 << " to " << 0.5*width/deg << " with Rin " << rin << " Rout " 
		 << rout << " ZHalf "  << dz << endl;
    DDLogicalPart cylinderRib(solid.ddname(), matrib, solid);
    double phix   = ribPhi[i];
    double phideg = phix/deg;
    DDRotation rotation;
    if (phideg != 0) {
      double theta  = 90*deg;
      double phiy   = phix + 90.*deg;
      string rotstr = idName + dbl_to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
        COUT << "DDTIBLayer test: Creating a new rotation: " 
                     << rotstr << "\t90., " << phix/deg << ", 90.," << phiy/deg
                     << ", 0, 0" << endl;
        rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
                         0., 0.);
      }
    }
    DDTranslation tran(0, 0, -0.25*(layerL+supportW));
    DDpos (cylinderRib, cylinderIn, 1, tran, rotation);
    COUT << "DDTIBLayer test " << cylinderRib.name() << " number 1"
                 << " positioned in " << cylinderIn.name() << " at " << tran
		 << " with " << rotation << endl;
    tran = DDTranslation(0, 0, 0.25*(layerL+supportW));
    DDpos (cylinderRib, cylinderIn, 2, tran, rotation);
    COUT << "DDTIBLayer test " << cylinderRib.name() << " number 2"
                 << " positioned in " << cylinderIn.name() << " at " << tran
		 << " with " << rotation << endl;
  }
}
