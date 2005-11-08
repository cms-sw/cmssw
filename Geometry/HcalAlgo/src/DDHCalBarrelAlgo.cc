///////////////////////////////////////////////////////////////////////////////
// File: DDHCalBarrelAlgo.cc
//   adapted from CCal(G4)HcalBarrel.cc
// Description: Geometry factory class for Hcal Barrel
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/interface/DDHCalBarrelAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

DDHCalBarrelAlgo::DDHCalBarrelAlgo():
  theta(0),rmax(0),zoff(0),ttheta(0),layerId(0),layerLabel(0),layerMat(0),
  layerWidth(0),layerD1(0),layerD2(0),layerAlpha(0),layerT(0),layerAbsorb(0),
  layerGap(0),absorbName(0),absorbMat(0),absorbD(0),absorbAlpha(0),absorbT(0),
  detType(0),detT1(0),detTsc(0),detT2(0),detWidth1(0),detWidth2(0),detPosY(0) {
  DCOUT('a', "DDHCalBarrelAlgo info: Creating an instance");
}

DDHCalBarrelAlgo::~DDHCalBarrelAlgo() {}


void DDHCalBarrelAlgo::initialize(const DDNumericArguments & nArgs,
				  const DDVectorArguments & vArgs,
				  const DDMapArguments & ,
				  const DDStringArguments & sArgs,
				  const DDStringVectorArguments & vsArgs) {

  genMaterial = sArgs["MaterialName"];
  nsectors    = int (nArgs["NSector"]);
  nsectortot  = int (nArgs["NSectorTot"]);
  nhalf       = int (nArgs["NHalf"]);
  rin         = nArgs["RIn"];
  rout        = nArgs["ROut"];
  rzones      = int (nArgs["RZones"]);
  rotHalf     = sArgs["RotHalf"];
  rotns       = sArgs["RotNameSpace"];

  theta       = vArgs["Theta"];
  rmax        = vArgs["RMax"];
  zoff        = vArgs["ZOff"];
  int i = 0;
  for (i = 0; i < rzones; i++) {
    ttheta.push_back(tan(theta[i])); //*deg already done in XML
  }
  if (rzones > 3)
    rmax[2] = (zoff[3] - zoff[2]) / ttheta[2];

  DCOUT('A', "DDHCalBarrelAlgo debug: General material " << genMaterial << "\tSectors " << nsectors << ", " << nsectortot <<"\tHalves "	<< nhalf << "\tRotation matrix " << rotns << ":" << rotHalf); 
  DCOUT('A', "\t\t" << rin << "\t" << rout << "\t" << rzones);
  for (i = 0; i < rzones; i++)
    DCOUT('A', "\t" << i << " Theta " << theta[i] << " rmax " << rmax[i] << " zoff " << zoff[i]);

  ///////////////////////////////////////////////////////////////
  //Layers
  nLayers = int(nArgs["NLayers"]);
  DCOUT('A', "DDHCalBarrelAlgo debug: Layer\t" << nLayers);
  layerId     = dbl_to_int (vArgs["Id"]);
  layerLabel  = vsArgs["LayerLabel"];
  layerMat    = vsArgs["LayerMat"];
  layerWidth  = vArgs["LayerWidth"];
  layerD1     = vArgs["D1"];
  layerD2     = vArgs["D2"];
  layerAlpha  = vArgs["Alpha2"]; 
  layerT      = vArgs["T"];
  layerAbsorb = dbl_to_int(vArgs["AbsL"]);
  layerGap    = vArgs["Gap"];
  for (i = 0; i < nLayers; i++) {
    DCOUT('A', layerLabel[i] << "\t" << layerId[i] << "\t" << layerMat[i] << "\t" << layerWidth[i] << "\t" << layerD1[i] << "\t" << layerD2[i]  << "\t" << layerAlpha[i] << "\t" << layerT[i] << "\t" << layerAbsorb[i] << "\t" << layerGap[i]);
  }
  
  ///////////////////////////////////////////////////////////////
  //Absorber Layer
  absorbName  = vsArgs["AbsorbName"];
  absorbMat   = vsArgs["AbsorbMat"];
  absorbD     = vArgs["AbsorbD"];
  absorbAlpha = vArgs["AbsorbAlpha"];
  absorbT     = vArgs["AbsorbT"];
  int nAbs    = absorbName.size();
  for (i = 0; i < nAbs; i++) {
    DCOUT('A', "DDHCalBarrelAlgo debug: " << absorbName[i] <<" Material " <<  absorbMat[i] << " d " << absorbD[i] << " Alpha "  << absorbAlpha[i] << " t " << absorbT[i]);
  }

  ///////////////////////////////////////////////////////////////
  // Detectors

  detMat   = sArgs["DetMat"];
  detRot   = sArgs["DetRot"];
  detMatPl = sArgs["DetMatPl"];
  detMatSc = sArgs["DetMatSc"];
  DCOUT('A', "DDHCalBarrelAlgo debug: Detector (" <<  nLayers << ") Rotation matrix " << rotns << ":" << detRot);
  DCOUT('A', "\t\t" << detMat << "\t" << detMatPl  << "\t" << detMatSc);

  detType   = dbl_to_int(vArgs["DetType"]);
  detdP1    = vArgs["DetdP1"];
  detT1     = vArgs["DetT1"];
  detTsc    = vArgs["DetTsc"];
  detT2     = vArgs["DetT2"];
  detWidth1 = vArgs["DetWidth1"];
  detWidth2 = vArgs["DetWidth2"];
  detPosY   = dbl_to_int(vArgs["DetPosY"]);
  for (i = 0; i < nLayers; i ++) {
    DCOUT('A', i+1 << "\t" << detType[i] << "\t" << detdP1[i] << "\t"  << detT1[i] << "\t" << detTsc[i] << "\t" << detT2[i] << "\t" << detWidth1[i] << "\t" << detWidth2[i] << "\t" << detPosY[i]);
  }

  //  idName = parentName.name();
  idName      = sArgs["MotherName"];
  idNameSpace = DDCurrentNamespace::ns();
  idOffset = int (nArgs["IdOffset"]); 
  DDName parentName = parent().name(); 
  DCOUT('A', "DDHCalBarrelAlgo debug: Parent " << parentName <<" idName " << idName << " NameSpace " << idNameSpace << " Offset " << idOffset);
}

////////////////////////////////////////////////////////////////////
// DDHCalBarrelAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHCalBarrelAlgo::execute() {

  DCOUT('a', "==>> Constructing DDHCalBarrelAlgo...");
  constructGeneralVolume();
  DCOUT('a', "<<== End of DDHCalBarrelAlgo construction ...");
}

//----------------------start here for DDD work!!! ---------------

void DDHCalBarrelAlgo::constructGeneralVolume() {
  
  DCOUT('a', "DDHCalBarrelAlgo test: General volume...");
  unsigned int i=0;

  DDRotation rot = DDRotation();

  double alpha = pi/getNsectors();
  double dphi  = getNsectortot()*twopi/getNsectors();
  int nsec, ntot=15;
  if (getNhalf() == 1)
    nsec = 8;
  else
    nsec = 15;
  int nf = ntot - nsec;

  //Calculate zmin... see HCalBarrel.hh picture. For polyhedra
  //Rmin and Rmax are distances to vertex
  double zmax   = getZoff(3);
  double zstep5 = getZoff(4);
  double zstep4 =(getZoff(1) + getRmax(1)*getTanTheta(1));
  double zstep3 =(getZoff(1) + getRmax(0)*getTanTheta(1));
  double zstep2 =(getZoff(0) + getRmax(0)*getTanTheta(0));
  double zstep1 =(getZoff(0) + getRin()  *getTanTheta(0));
  double rout   = getRout();
  double rout1  = getRmax(3);
  double rin    = getRin();
  double rmid1  = getRmax(0);
  double rmid2  = getRmax(1);
  double rmid3  =(getZoff(4) - getZoff(2))/getTanTheta(2);
  double rmid4  = getRmax(2);

  vector<double> pgonZ;
  pgonZ.push_back( -zmax); 
  pgonZ.push_back( -zstep5); 
  pgonZ.push_back( -zstep5); 
  pgonZ.push_back( -zstep4); 
  pgonZ.push_back( -zstep3); 
  pgonZ.push_back( -zstep2); 
  pgonZ.push_back( -zstep1); 
  pgonZ.push_back(       0); 
  pgonZ.push_back(  zstep1); 
  pgonZ.push_back(  zstep2); 
  pgonZ.push_back(  zstep3); 
  pgonZ.push_back(  zstep4); 
  pgonZ.push_back(  zstep5); 
  pgonZ.push_back(  zstep5); 
  pgonZ.push_back(    zmax);

  vector<double> pgonRmin;
  pgonRmin.push_back(   rmid4); 
  pgonRmin.push_back(   rmid3); 
  pgonRmin.push_back(   rmid3); 
  pgonRmin.push_back(   rmid2); 
  pgonRmin.push_back(   rmid1); 
  pgonRmin.push_back(   rmid1); 
  pgonRmin.push_back(     rin); 
  pgonRmin.push_back(     rin); 
  pgonRmin.push_back(     rin); 
  pgonRmin.push_back(   rmid1); 
  pgonRmin.push_back(   rmid1); 
  pgonRmin.push_back(   rmid2); 
  pgonRmin.push_back(   rmid3); 
  pgonRmin.push_back(   rmid3); 
  pgonRmin.push_back(   rmid4);

  vector<double> pgonRmax;
  pgonRmax.push_back(   rout1); 
  pgonRmax.push_back(   rout1); 
  pgonRmax.push_back(    rout); 
  pgonRmax.push_back(    rout); 
  pgonRmax.push_back(    rout); 
  pgonRmax.push_back(    rout); 
  pgonRmax.push_back(    rout); 
  pgonRmax.push_back(    rout); 
  pgonRmax.push_back(    rout); 
  pgonRmax.push_back(    rout); 
  pgonRmax.push_back(    rout); 
  pgonRmax.push_back(    rout); 
  pgonRmax.push_back(    rout); 
  pgonRmax.push_back(   rout1);
  pgonRmax.push_back(   rout1);

  vector<double> pgonZHalf;
  pgonZHalf.push_back(       0); 
  pgonZHalf.push_back(  zstep1); 
  pgonZHalf.push_back(  zstep2); 
  pgonZHalf.push_back(  zstep3); 
  pgonZHalf.push_back(  zstep4); 
  pgonZHalf.push_back(  zstep5); 
  pgonZHalf.push_back(  zstep5); 
  pgonZHalf.push_back(    zmax);

  vector<double> pgonRminHalf;
  pgonRminHalf.push_back(     rin); 
  pgonRminHalf.push_back(     rin); 
  pgonRminHalf.push_back(   rmid1); 
  pgonRminHalf.push_back(   rmid1); 
  pgonRminHalf.push_back(   rmid2); 
  pgonRminHalf.push_back(   rmid3); 
  pgonRminHalf.push_back(   rmid3); 
  pgonRminHalf.push_back(   rmid4);

  vector<double> pgonRmaxHalf;
  pgonRmaxHalf.push_back(    rout); 
  pgonRmaxHalf.push_back(    rout); 
  pgonRmaxHalf.push_back(    rout); 
  pgonRmaxHalf.push_back(    rout); 
  pgonRmaxHalf.push_back(    rout); 
  pgonRmaxHalf.push_back(    rout); 
  pgonRmaxHalf.push_back(   rout1);
  pgonRmaxHalf.push_back(   rout1);

  string name("Null");
  DDSolid solid;
  if (nf == 0) { 
    solid = DDSolidFactory::polyhedra(DDName(idName, idNameSpace),
				      getNsectortot(), -alpha, dphi, pgonZ, 
				      pgonRmin, pgonRmax);
    DCOUT('a', "DDHCalBarrelAlgo test: " << DDName(idName, idNameSpace) << " Polyhedra made of " << getGenMaterial() << " with " << getNsectortot() << " sectors from " << -alpha/deg <<" to " << (-alpha+dphi)/deg << " and with " << nsec << " sections ");
    for (i = 0; i <pgonZ.size(); i++) 
      DCOUT('a', "\t" << "\tZ = " << pgonZ[i] << "\tRmin = " <<pgonRmin[i] << "\tRmax = " << pgonRmax[i]);
  } else {
    solid = DDSolidFactory::polyhedra(DDName(idName, idNameSpace),
				      getNsectortot(), -alpha, dphi, pgonZHalf,
				      pgonRminHalf, pgonRmaxHalf);
    DCOUT('a', "DDHCalBarrelAlgo test: " << DDName(idName, idNameSpace) << " Polyhedra made of " << getGenMaterial() << " with " << getNsectortot() << " sectors from " << -alpha/deg << " to " << (-alpha+dphi)/deg << " and with " << nsec << " sections ");
    for (i = 0; i < pgonZHalf.size(); i++) 
      DCOUT('a', "\t" << "\tZ = " << pgonZHalf[i] << "\tRmin = " << pgonRminHalf[i] << "\tRmax = " << pgonRmaxHalf[i]);
  }  
  

  DDName matname(DDSplit(getGenMaterial()).first, DDSplit(getGenMaterial()).second);
  DDMaterial matter(matname);
  DDLogicalPart genlogic(DDName(idName, idNameSpace), matter, solid);

  DDName parentName = parent().name(); 
  DDTranslation r0(0,0,0);
  DDpos(DDName(idName, idNameSpace), parentName, 1, r0, rot);
  DCOUT('a', "DDHCalBarrelAlgo test: " << DDName(idName, idNameSpace) << " number 1 positioned in " << parentName << " at " << r0 << " with " << rot);

  //Forward and backwards halfs
  name = idName + "Half";
  nf   = (ntot+1)/2;
  DCOUT('a', "DDHCalBarrelAlgo test: " << DDName(name, idNameSpace) << " Polyhedra made of " << getGenMaterial() << " with " << getNsectortot() << " sectors from " << -alpha/deg << " to " << (-alpha+dphi)/deg << " and with " << nf << " sections "); 
  for (i = 0; i < pgonZHalf.size(); i++) 
    DCOUT('a', "\t" << "\tZ = " << pgonZHalf[i] << "\tRmin = " << pgonRminHalf[i] << "\tRmax = " << pgonRmaxHalf[i]);

  solid =   DDSolidFactory::polyhedra(DDName(name, idNameSpace),
				      getNsectortot(), -alpha, dphi, pgonZHalf,
				      pgonRminHalf, pgonRmaxHalf);
  DDLogicalPart genlogich(DDName(name, idNameSpace), matter, solid);

  DDpos(genlogich, genlogic, 1, r0, rot);
  DCOUT('a', "DDHCalBarrelAlgo test: "  << genlogich.name() << " number 1 positioned in " << genlogic.name() << " at " << r0 << " with " << rot);

  if (getNhalf() != 1) {
    rot = DDRotation(DDName(rotHalf, rotns));
    DDpos (genlogich, genlogic, 2, r0, rot);
    DCOUT('a', "DDHCalBarrelAlgo test:  " << genlogich.name() << " number 2 positioned in " << genlogic.name() << " at " << r0 << " with " << rot);
  } //end if (getNhalf...
  
  //Construct sector (from -alpha to +alpha)
  name = idName + "Module";
  DCOUT('a', "DDHCalBarrelAlgo test: " << DDName(name, idNameSpace) << " Polyhedra made of " << getGenMaterial() << " with 1 sector from " << -alpha/deg << " to "  << alpha/deg << " and with " << nf << " sections");
  for (i = 0; i < pgonZHalf.size(); i++) 
    DCOUT('a', "\t" << "\tZ = " << pgonZHalf[i] << "\tRmin = " << pgonRminHalf[i] << "\tRmax = " << pgonRmaxHalf[i]);

  solid =   DDSolidFactory::polyhedra(DDName(name, idNameSpace),
				      1, -alpha, 2*alpha, pgonZHalf,
				      pgonRminHalf, pgonRmaxHalf);
  DDLogicalPart seclogic(DDName(name, idNameSpace), matter, solid);
  
  for (int ii=0; ii<getNsectortot(); ii++) {
    double phi    = ii*2*alpha;
    double phideg = phi/deg;
    
    DDRotation rotation;
    string rotstr("NULL");
    if (phideg != 0) {
      rotstr = "R"; 
      if (phideg < 100)	rotstr = "R0"; 
      rotstr = rotstr + dbl_to_string(phideg);
      rotation = DDRotation(DDName(rotstr, rotns)); 
      if (!rotation) {
	DCOUT('a', "DDHCalBarrelAlgo test: Creating a new rotation " << rotstr << "\t" << 90 << "," << phideg << ","  << 90 << "," << (phideg+90) << ","    << 0 << "," << 0);
	rotation = DDrot(DDName(rotstr, idNameSpace), 90*deg, phideg*deg, 
			 90*deg, (90+phideg)*deg, 0*deg,  0*deg);
      } //if !rotation
    } //if phideg!=0
  
    DDpos (seclogic, genlogich, ii+1, r0, rotation);
    DCOUT('a', "DDHCalBarrelAlgo test: " << seclogic.name() << " number " << ii+1 << " positioned in " << genlogich.name() << " at " << r0 << " with " << rotation);
  }
  
  //Construct the things inside the sector
  constructInsideSector(seclogic);
}


void DDHCalBarrelAlgo::constructInsideSector(DDLogicalPart sector) {
  
  DCOUT('a', "DDHCalBarrelAlgo test: Layers (" << getNLayers() << ") ...");
  
  double alpha = pi/getNsectors();
  double rin   = getRin();
  for (int i = 0; i < getNLayers(); i++) {
    string  name   = idName + getLayerLabel(i);
    DDName matname(DDSplit(getLayerMaterial(i)).first, 
		   DDSplit(getLayerMaterial(i)).second); //idNameSpace);
    DDMaterial matter(matname);

    double width = getLayerWidth(i);
    double rout  = rin + width;

    int    in = 0, out = 0;
    for (int j = 0; j < getRzones()-1; j++) {
      if (rin >= getRmax(j)) in = j+1;
      if (rout>  getRmax(j)) out= j+1;
    }
    double zout  = getZoff(in) + rin*getTanTheta(in);

    //!!!!!!!!!!!!!!!!!Should be zero. And removed as soon as
    //vertical walls are allowed in SolidPolyhedra
    double deltaz = 0;
    
    int    nsec=2;
    vector<double> pgonZ, pgonRmin, pgonRmax;
    // index 0
    pgonZ.push_back(0);
    pgonRmin.push_back(rin); 
    pgonRmax.push_back(rout);
    // index 1
    pgonZ.push_back(zout);  
    pgonRmin.push_back(rin); 
    pgonRmax.push_back(rout);
    if (in == out) {
      if (in <= 3) {
	//index 2
	pgonZ.push_back(getZoff(in) + rout*getTanTheta(in));
	pgonRmin.push_back(pgonRmax[1]);
	pgonRmax.push_back(pgonRmax[1]);
	nsec++;
      }
    } else {
      if (in == 3) {
	//redo index 1, add index 2
	pgonZ[1]    =(getZoff(out) + getRmax(out)*getTanTheta(out));
	pgonZ.push_back(pgonZ[1] + deltaz);
	pgonRmin.push_back(pgonRmin[1]); 
	pgonRmax.push_back(getRmax(in));
	//index 3 
	pgonZ.push_back(getZoff(in) + getRmax(in)*getTanTheta(in));
	pgonRmin.push_back(pgonRmin[2]); 
	pgonRmax.push_back(pgonRmax[2]);
        nsec       += 2;
      } else {
	//index 2
	pgonZ.push_back(getZoff(in) + getRmax(in)*getTanTheta(in));
	pgonRmin.push_back(getRmax(in)); 
	pgonRmax.push_back(pgonRmax[1]); 
	nsec++;
	if (in == 0) {
	  pgonZ.push_back(getZoff(out) + getRmax(in)*getTanTheta(out));
          pgonRmin.push_back(pgonRmin[2]); 
	  pgonRmax.push_back(pgonRmax[2]);
	  nsec++;
	}
	if (in <= 1) {
	  pgonZ.push_back(getZoff(out) + rout*getTanTheta(out));
	  pgonRmin.push_back(rout);
	  pgonRmax.push_back(rout);
	  nsec++;
	}
      }
    }
    //Solid & volume
    DDSolid solid;
    double  alpha1 = alpha;
    if (getLayerGap(i)>1.e-6) {
      double rmid  = 0.5*(rin+rout);
      double width = rmid*tan(alpha) - getLayerGap(i);
      alpha1 = atan(width/rmid);
      DCOUT('a', "\t" << "Alpha_1 modified from " << alpha/deg << " to " << alpha1/deg << " Rmid " << rmid << " Reduced width " << width);
    }
    DCOUT('a', "DDHCalBarrelAlgo test: " << name << " (Layer " << i << ") Polyhedra made of " << getLayerMaterial(i) << " with 1 sector from " << -alpha1/deg << " to " << alpha1/deg << " and with " << nsec << " sections");
    for (unsigned int k=0; k<pgonZ.size(); k++)
      DCOUT('a', "\t" << "\t" << pgonZ[k] << "\t" << pgonRmin[k] << "\t" << pgonRmax[k]);
    
    solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace), 
				      1, -alpha1, 2*alpha1,
				      pgonZ, pgonRmin, pgonRmax);
    DDLogicalPart glog(DDName(name, idNameSpace), matter, solid);

    DDpos (glog, sector, getLayerId(i), DDTranslation(0.0, 0.0, 0.0), 
	   DDRotation());
    DCOUT('a', "DDHCalBarrelAlgo test: " << glog.name() << " number " << getLayerId(i) << " positioned in " << sector.name() << " at (0,0,0) with no rotation");

    constructInsideLayers(glog, getLayerLabel(i), getLayerId(i), 
			  getLayerAbsorb(i), rin,  getLayerD1(i), alpha1, 
			  getLayerD2(i), getLayerAlpha(i), getLayerT(i));
    rin = rout;
  }
  
}

void DDHCalBarrelAlgo::constructInsideLayers(DDLogicalPart laylog,
					     string nm, int id, int nAbs, 
					     double rin, double d1, 
					     double alpha1, double d2, 
					     double alpha2, double t) {
  
  DCOUT('a', "DDHCalBarrelAlgo test: \t\tInside layer " << id << "...");

  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  DDRotation rot(DDName(detRot, rotns));

  string nam0 = nm + "In";
  string name = idName + nam0;
  DDName matName(DDSplit(getDetMat()).first, DDSplit(getDetMat()).second);
  DDMaterial matter (matName);

  DDSolid solid;
  DDLogicalPart glog;
  double         rsi, dx, dy, dz, x, y;
  int            i, in;
  //Two lower volumes
  if (alpha1 > 0) {
    rsi = rin + d1;
    in  = 0;
    for (i = 0; i < getRzones()-1; i++) {
      if (rsi >= getRmax(i)) in = i+1;
    }
    dx = 0.5*t;
    dy = 0.5*rsi*(tan(alpha1)-tan(alpha2));
    dz = 0.5*(getZoff(in) + rsi*getTanTheta(in));
    x  = rsi + dx;
    y  = 0.5*rsi*(tan(alpha1)+tan(alpha2));
    DDTranslation r11(x, y, dz);
    DDTranslation r12(x, -y, dz);

    solid = DDSolidFactory::box(DDName(name+"1", idNameSpace), dx, dy, dz);
    DCOUT('a', "DDHCalBarrelAlgo test: " << solid.name() <<" Box made of " << getDetMat() << " of dimensions " << dx << ", " << dy << ", " << dz);
    glog = DDLogicalPart(solid.ddname(), matter, solid);

    DDpos(glog, laylog, idOffset+1, r11, DDRotation());
    DDpos(glog, laylog, idOffset+2, r12, rot);
    DCOUT('a', "DDHCalBarrelAlgo test: " << glog.name() << " Number " << idOffset+1 << " positioned in " << laylog.name() << " at " << r11 << " with no rotation");
    DCOUT('a', "DDHCalBarrelAlgo test: " << glog.name() << " Number " << idOffset+2 << " positioned in " << laylog.name() << " at " << r12 << " with " << rot);

    //Constructin the plastics and scintillators inside
    constructInsideDetectors(glog, nam0+"1", id, dx, dy, dz, 1);
  }

  //Upper volume
  rsi = rin + d2;
  in  = 0;
  for (i = 0; i < getRzones()-1; i++) {
    if (rsi >= getRmax(i)) in = i+1;
  }
  dx  = 0.5*t;
  dy  = 0.5*rsi*tan(alpha2);
  dz  = 0.5*(getZoff(in) + rsi*getTanTheta(in));
  x   = rsi + dx;
  DDTranslation r21(x, dy, dz);
  DDTranslation r22(x, -dy, dz);
  
  solid = DDSolidFactory::box(DDName(name+"2", idNameSpace), dx, dy, dz);
  DCOUT('a', "DDHCalBarrelAlgo test: " << solid.name() << " Box made of " << getDetMat() << " of dimensions " << dx << ", " << dy << ", " << dz);
  glog = DDLogicalPart(solid.ddname(), matter, solid);

  DDpos (glog, laylog, idOffset+3, r21, DDRotation());
  DDpos (glog, laylog, idOffset+4, r22, rot);
  DCOUT('a', "DDHCalBarrelAlgo test: " << glog.name() << " Number " << idOffset+3 << " positioned in " << laylog.name() << " at " << r21 << " with no rotation");
  DCOUT('a', "DDHCalBarrelAlgo test: " << glog.name() << " Number " << idOffset+4 << " positioned in " << laylog.name() << " at " << r22 << " with " << rot);

  //Constructin the plastics and scintillators inside
  constructInsideDetectors(glog, nam0+"2", id, dx, dy, dz, 2);

  //Extra absorber layer
  for (int k = 0; k < nAbs; k++) {
    string namek = name + getAbsorbName(k);
    rsi = rin + getAbsorbD(k);
    in  = 0;
    for (i = 0; i < getRzones()-1; i++) {
      if (rsi >= getRmax(i)) in = i+1;
    }
    vector<double> pgonZ, pgonRmin, pgonRmax;
    double zout = getZoff(in) + rin*getTanTheta(in);
    // index 0
    pgonZ.push_back(0.0);     
    pgonRmin.push_back(rsi); 
    pgonRmax.push_back(rsi+getAbsorbT(k));
    // index 1
    pgonZ.push_back(zout);  
    pgonRmin.push_back(rsi); 
    pgonRmax.push_back(pgonRmax[0]);
    // index 2
    pgonZ.push_back(getZoff(in) + pgonRmax[0]*getTanTheta(in));
    pgonRmin.push_back(pgonRmax[1]);
    pgonRmax.push_back(pgonRmax[1]);
    solid = DDSolidFactory::polyhedra(DDName(namek, idNameSpace), 1,
				      -getAbsorbAlpha(k), 2*getAbsorbAlpha(k),
				      pgonZ, pgonRmin, pgonRmax);
    DCOUT('a', "DDHCalBarrelAlgo test: " << solid.name() << " Polyhedra made of " << getAbsorbMat(k) << " with 1 sector from " << -getAbsorbAlpha(k)/deg << " to " << getAbsorbAlpha(k)/deg << " and with " << pgonZ.size() << " sections");
    for (unsigned int ii=0; ii<pgonZ.size(); ii++)
      DCOUT('a', "\t\tZ = " << pgonZ[ii] << "\tRmin = " << pgonRmin[ii] << "\tRmax = " << pgonRmax[ii]);

    DDName matName(DDSplit(getAbsorbMat(k)).first, DDSplit(getAbsorbMat(k)).second);
    DDMaterial matter(matName);
    glog = DDLogicalPart(solid.ddname(), matter, solid);

    DDpos(glog, laylog, 1, DDTranslation(), DDRotation());
    DCOUT('a', "DDHCalBarrelAlgo test: " << glog.name() << " Number 1 positioned in " << laylog.name() << " at (0,0,0) with no rotation");
  }
}

 
void DDHCalBarrelAlgo::constructInsideDetectors(DDLogicalPart detector,
						string name, int id, double dx,
						double dy, double dz,
						int type) {

  DCOUT('a', "DDHCalBarrelAlgo test: \t\tInside detector " << id << "...");
  
  DDName plmatname(DDSplit(getDetMatPl()).first, DDSplit(getDetMatPl()).second);
  DDMaterial plmatter(plmatname);
  DDName scmatname(DDSplit(getDetMatSc()).first, DDSplit(getDetMatSc()).second);
  DDMaterial scmatter(scmatname);
  
  string plname = detector.name().name()+"Plastic_";
  string scname = idName+"Scintillator"+name;
  
  id--;
  DDSolid solid;
  DDLogicalPart glog;
  double  wid, y=0;

  if (type == 1) {
    wid = 0.5*getDetWidth1(id);
    if (getDetPosY(id)>0) y =-dy+wid;
  } else {
    wid = 0.5*getDetWidth2(id);
  }

  solid = DDSolidFactory::box(DDName(plname+"1", idNameSpace), 
			      0.5*getDetT1(id) , wid, dz);
  DCOUT('a', "DDHCalBarrelAlgo test: " << solid.name() << " Box made of " << getDetMatPl() << " of dimensions " << 0.5*getDetT1(id) <<", " << wid << ", "  << dz);
  glog = DDLogicalPart(solid.ddname(), plmatter, solid); 

  double x = getDetdP1(id) + 0.5*getDetT1(id) - dx;
  DDpos(glog, detector, 1, DDTranslation(x,y,0), DDRotation());
  DCOUT('a', "DDHCalBarrelAlgo test: " << glog.name() << " Number 1 positioned in " << detector.name() << " at (" << x << "," << y << ",0) with no rotation");

  solid = DDSolidFactory::box(DDName(scname, idNameSpace), 
			      0.5*getDetTsc(id), wid, dz);
  DCOUT('a', "DDHCalBarrelAlgo test: " << solid.name() << " Box made of " << getDetMatSc() << " of dimensions " << 0.5*getDetTsc(id) << ", " << wid << ", " << dz);
  glog = DDLogicalPart(solid.ddname(), scmatter, solid);

  x += 0.5*(getDetT1(id) + getDetTsc(id));
  int copyNo = id*10 + getDetType(id);
  DDpos(glog, detector, copyNo, DDTranslation(x, y, 0), DDRotation());
  DCOUT('a', "DDHCalBarrelAlgo test: " << glog.name() << " Number " << copyNo << " positioned in " << detector.name() << " at (" << x << "," << y  << ",0) with no rotation");

  solid = DDSolidFactory::box(DDName(plname+"2", idNameSpace), 
			      0.5*getDetT2(id) , wid, dz);
  DCOUT('a', "DDHCalBarrelAlgo test: " << solid.name() << " Box made of " << getDetMatPl() << " of dimensions " << 0.5*getDetT2(id) <<", " << wid << ", "  << dz);
  glog = DDLogicalPart(solid.ddname(), plmatter, solid);

  x+=0.5*(getDetTsc(id) + getDetT2(id));
  DDpos (glog, detector, 1, DDTranslation(x, y, 0), DDRotation());
  DCOUT('a', "DDHCalBarrelAlgo test: " << glog.name() << " Number 1 positioned in " << detector.name() << " at (" << x << "," << y << ",0) with no rotation");

}
