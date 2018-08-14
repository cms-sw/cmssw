/////////////////////////////////////////////////////////////////////////////
// File: DDHCalBarrelAlgo.cc
//   adapted from CCal(G4)HcalBarrel.cc
// Description: Geometry factory class for Hcal Barrel
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/plugins/DDHCalBarrelAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDHCalBarrelAlgo::DDHCalBarrelAlgo():
  theta(0),rmax(0),zoff(0),ttheta(0),layerId(0),layerLabel(0),layerMat(0),
  layerWidth(0),layerD1(0),layerD2(0),layerAlpha(0),layerT1(0),layerT2(0),
  layerAbsorb(0),layerGap(0),absorbName(0),absorbMat(0),absorbD(0),absorbT(0),
  midName(0),midMat(0),midW(0),midT(0),sideMat(0),sideD(0),sideT(0),
  sideAbsName(0),sideAbsMat(0),sideAbsW(0),detType(0),detdP1(0),detdP2(0),
  detT11(0),detT12(0),detTsc(0),detT21(0),detT22(0),detWidth1(0),detWidth2(0),
  detPosY(0) {
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo info: Creating an instance";
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
    ttheta.emplace_back(tan(theta[i])); //*deg already done in XML
  }
  if (rzones > 3)
    rmax[2] = (zoff[3] - zoff[2]) / ttheta[2];

  LogDebug("HCalGeom") << "DDHCalBarrelAlgo debug: General material " 
		       << genMaterial << "\tSectors " << nsectors << ", " 
		       << nsectortot <<"\tHalves "	<< nhalf 
		       << "\tRotation matrix " << rotns << ":" << rotHalf 
		       << "\n\t\t" << rin << "\t" << rout << "\t" << rzones;
  for (i = 0; i < rzones; i++) {
    LogDebug("HCalGeom") << "\tTheta[" << i << "] = " << theta[i] << "\trmax["
			 << i << "] = " << rmax[i] << "\tzoff[" << i << "] = "
			 << zoff[i];
  }
  ///////////////////////////////////////////////////////////////
  //Layers
  nLayers = int(nArgs["NLayers"]);
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo debug: Layer\t" << nLayers;
  layerId     = dbl_to_int (vArgs["Id"]);
  layerLabel  = vsArgs["LayerLabel"];
  layerMat    = vsArgs["LayerMat"];
  layerWidth  = vArgs["LayerWidth"];
  layerD1     = vArgs["D1"];
  layerD2     = vArgs["D2"];
  layerAlpha  = vArgs["Alpha2"]; 
  layerT1     = vArgs["T1"];
  layerT2     = vArgs["T2"];
  layerAbsorb = dbl_to_int(vArgs["AbsL"]);
  layerGap    = vArgs["Gap"];
  for (i = 0; i < nLayers; i++) {
    LogDebug("HCalGeom") << layerLabel[i] << "\t" << layerId[i] << "\t" 
			 << layerMat[i] << "\t" << layerWidth[i] << "\t" 
			 << layerD1[i] << "\t" << layerD2[i]  << "\t" 
			 << layerAlpha[i] << "\t" << layerT1[i] << "\t"
			 << layerT2[i] << "\t" << layerAbsorb[i] << "\t" 
			 << layerGap[i];
  }
  
  ///////////////////////////////////////////////////////////////
  //Absorber Layers and middle part
  absorbName  = vsArgs["AbsorbName"];
  absorbMat   = vsArgs["AbsorbMat"];
  absorbD     = vArgs["AbsorbD"];
  absorbT     = vArgs["AbsorbT"];
  nAbsorber   = absorbName.size();
  for (i = 0; i < nAbsorber; i++) {
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo debug: " << absorbName[i]
			 <<" Material " <<  absorbMat[i] << " d " << absorbD[i]
			 << " t " <<absorbT[i];
  }
  middleMat   = sArgs["MiddleMat"];
  middleD     = nArgs["MiddleD"];
  middleW     = nArgs["MiddleW"];
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo debug: Middle material " 
		       << middleMat << " d " << middleD << " w " << middleW;
  midName     = vsArgs["MidAbsName"];
  midMat      = vsArgs["MidAbsMat"];
  midW        = vArgs["MidAbsW"];
  midT        = vArgs["MidAbsT"];
  nMidAbs     = midName.size();
  for (i = 0; i < nMidAbs; i++) {
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo debug: " << midName[i]
			 << " Material " <<  midMat[i] << " W " << midW[i]
			 << " T " << midT[i];
  }

  //Absorber layers in the side part
  sideMat     = vsArgs["SideMat"];
  sideD       = vArgs["SideD"];
  sideT       = vArgs["SideT"];
  int nSide   = sideMat.size();
  for (int i = 0; i < nSide; i++) {
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo debug: Side material " 
			 << sideMat[i] << " d " << sideD[i] << " t "
			 << sideT[i];
  }
  sideAbsName = vsArgs["SideAbsName"];
  sideAbsMat  = vsArgs["SideAbsMat"];
  sideAbsW    = vArgs["SideAbsW"];
  nSideAbs    = sideAbsName.size();
  for (i = 0; i < nSideAbs; i++) {
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo debug: " << sideAbsName[i]
			 <<" Material " <<  sideAbsMat[i] << " W "
			 << sideAbsW[i];
  }

  ///////////////////////////////////////////////////////////////
  // Detectors

  detMat   = sArgs["DetMat"];
  detRot   = sArgs["DetRot"];
  detMatPl = sArgs["DetMatPl"];
  detMatSc = sArgs["DetMatSc"];
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo debug: Detector (" <<  nLayers 
		       << ") Rotation matrix " << rotns << ":" << detRot
		       << "\n\t\t" << detMat << "\t" << detMatPl  << "\t"
		       << detMatSc;

  detType   = dbl_to_int(vArgs["DetType"]);
  detdP1    = vArgs["DetdP1"];
  detdP2    = vArgs["DetdP2"];
  detT11    = vArgs["DetT11"];
  detT12    = vArgs["DetT12"];
  detTsc    = vArgs["DetTsc"];
  detT21    = vArgs["DetT21"];
  detT22    = vArgs["DetT22"];
  detWidth1 = vArgs["DetWidth1"];
  detWidth2 = vArgs["DetWidth2"];
  detPosY   = dbl_to_int(vArgs["DetPosY"]);
  for (i = 0; i < nLayers; i ++) {
    LogDebug("HCalGeom") << i+1 << "\t" << detType[i] << "\t" << detdP1[i]
			 << ", "  << detdP2[i] << "\t" << detT11[i] << ", " 
			 << detT12[i] << "\t" << detTsc[i] << "\t" << detT21[i]
			 <<", " << detT22[i] << "\t" << detWidth1[i] << "\t" 
			 << detWidth2[i] << "\t" << detPosY[i];
  }

  //  idName = parentName.name();
  idName      = sArgs["MotherName"];
  idNameSpace = DDCurrentNamespace::ns();
  idOffset = int (nArgs["IdOffset"]); 
  DDName parentName = parent().name(); 
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo debug: Parent " << parentName
		       <<" idName " << idName << " NameSpace " << idNameSpace
		       << " Offset " << idOffset;
}

////////////////////////////////////////////////////////////////////
// DDHCalBarrelAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHCalBarrelAlgo::execute(DDCompactView& cpv) {

  LogDebug("HCalGeom") << "==>> Constructing DDHCalBarrelAlgo...";
  constructGeneralVolume(cpv);
  LogDebug("HCalGeom") << "<<== End of DDHCalBarrelAlgo construction ...";
}

//----------------------start here for DDD work!!! ---------------

void DDHCalBarrelAlgo::constructGeneralVolume(DDCompactView& cpv) {
  
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: General volume...";
  unsigned int i=0;

  DDRotation rot = DDRotation();

  double alpha = CLHEP::pi/getNsectors();
  double dphi  = getNsectortot()*CLHEP::twopi/getNsectors();
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
  if ((getZoff(2)+getRmax(1)*getTanTheta(2)) > zstep4)
    zstep4 = (getZoff(2)+getRmax(1)*getTanTheta(2));
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
  pgonZ.emplace_back( -zmax); 
  pgonZ.emplace_back( -zstep5); 
  pgonZ.emplace_back( -zstep5); 
  pgonZ.emplace_back( -zstep4); 
  pgonZ.emplace_back( -zstep3); 
  pgonZ.emplace_back( -zstep2); 
  pgonZ.emplace_back( -zstep1); 
  pgonZ.emplace_back(       0); 
  pgonZ.emplace_back(  zstep1); 
  pgonZ.emplace_back(  zstep2); 
  pgonZ.emplace_back(  zstep3); 
  pgonZ.emplace_back(  zstep4); 
  pgonZ.emplace_back(  zstep5); 
  pgonZ.emplace_back(  zstep5); 
  pgonZ.emplace_back(    zmax);

  vector<double> pgonRmin;
  pgonRmin.emplace_back(   rmid4); 
  pgonRmin.emplace_back(   rmid3); 
  pgonRmin.emplace_back(   rmid3); 
  pgonRmin.emplace_back(   rmid2); 
  pgonRmin.emplace_back(   rmid1); 
  pgonRmin.emplace_back(   rmid1); 
  pgonRmin.emplace_back(     rin); 
  pgonRmin.emplace_back(     rin); 
  pgonRmin.emplace_back(     rin); 
  pgonRmin.emplace_back(   rmid1); 
  pgonRmin.emplace_back(   rmid1); 
  pgonRmin.emplace_back(   rmid2); 
  pgonRmin.emplace_back(   rmid3); 
  pgonRmin.emplace_back(   rmid3); 
  pgonRmin.emplace_back(   rmid4);

  vector<double> pgonRmax;
  pgonRmax.emplace_back(   rout1); 
  pgonRmax.emplace_back(   rout1); 
  pgonRmax.emplace_back(    rout); 
  pgonRmax.emplace_back(    rout); 
  pgonRmax.emplace_back(    rout); 
  pgonRmax.emplace_back(    rout); 
  pgonRmax.emplace_back(    rout); 
  pgonRmax.emplace_back(    rout); 
  pgonRmax.emplace_back(    rout); 
  pgonRmax.emplace_back(    rout); 
  pgonRmax.emplace_back(    rout); 
  pgonRmax.emplace_back(    rout); 
  pgonRmax.emplace_back(    rout); 
  pgonRmax.emplace_back(   rout1);
  pgonRmax.emplace_back(   rout1);

  vector<double> pgonZHalf;
  pgonZHalf.emplace_back(       0); 
  pgonZHalf.emplace_back(  zstep1); 
  pgonZHalf.emplace_back(  zstep2); 
  pgonZHalf.emplace_back(  zstep3); 
  pgonZHalf.emplace_back(  zstep4); 
  pgonZHalf.emplace_back(  zstep5); 
  pgonZHalf.emplace_back(  zstep5); 
  pgonZHalf.emplace_back(    zmax);

  vector<double> pgonRminHalf;
  pgonRminHalf.emplace_back(     rin); 
  pgonRminHalf.emplace_back(     rin); 
  pgonRminHalf.emplace_back(   rmid1); 
  pgonRminHalf.emplace_back(   rmid1); 
  pgonRminHalf.emplace_back(   rmid2); 
  pgonRminHalf.emplace_back(   rmid3); 
  pgonRminHalf.emplace_back(   rmid3); 
  pgonRminHalf.emplace_back(   rmid4);

  vector<double> pgonRmaxHalf;
  pgonRmaxHalf.emplace_back(    rout); 
  pgonRmaxHalf.emplace_back(    rout); 
  pgonRmaxHalf.emplace_back(    rout); 
  pgonRmaxHalf.emplace_back(    rout); 
  pgonRmaxHalf.emplace_back(    rout); 
  pgonRmaxHalf.emplace_back(    rout); 
  pgonRmaxHalf.emplace_back(   rout1);
  pgonRmaxHalf.emplace_back(   rout1);

  string name("Null");
  DDSolid solid;
  if (nf == 0) { 
    solid = DDSolidFactory::polyhedra(DDName(idName, idNameSpace),
				      getNsectortot(), -alpha, dphi, pgonZ, 
				      pgonRmin, pgonRmax);
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: "
			 << DDName(idName, idNameSpace) <<" Polyhedra made of "
			 << getGenMaterial() << " with " << getNsectortot()
			 << " sectors from " << -alpha/CLHEP::deg <<" to "
			 << (-alpha+dphi)/CLHEP::deg << " and with " << nsec
			 << " sections ";
    for (i = 0; i <pgonZ.size(); i++) {
      LogDebug("HCalGeom") << "\t" << "\tZ = " << pgonZ[i] << "\tRmin = " 
			   << pgonRmin[i] << "\tRmax = " << pgonRmax[i];
    }
  } else {
    solid = DDSolidFactory::polyhedra(DDName(idName, idNameSpace),
				      getNsectortot(), -alpha, dphi, pgonZHalf,
				      pgonRminHalf, pgonRmaxHalf);
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " 
			 << DDName(idName, idNameSpace) <<" Polyhedra made of "
			 << getGenMaterial() << " with " << getNsectortot()
			 << " sectors from " << -alpha/CLHEP::deg << " to " 
			 << (-alpha+dphi)/CLHEP::deg << " and with " << nsec 
			 << " sections ";
    for (i = 0; i < pgonZHalf.size(); i++) {
      LogDebug("HCalGeom") << "\t" << "\tZ = " << pgonZHalf[i] << "\tRmin = "
			   << pgonRminHalf[i] << "\tRmax = " <<pgonRmaxHalf[i];
    }
  }  
  

  DDName matname(DDSplit(getGenMaterial()).first, DDSplit(getGenMaterial()).second);
  DDMaterial matter(matname);
  DDLogicalPart genlogic(DDName(idName, idNameSpace), matter, solid);

  DDName parentName = parent().name(); 
  DDTranslation r0(0,0,0);
  cpv.position(DDName(idName, idNameSpace), parentName, 1, r0, rot);
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " 
		       << DDName(idName, idNameSpace) << " number 1 positioned"
		       << " in " << parentName << " at " << r0 <<" with "<<rot;

  //Forward and backwards halfs
  name = idName + "Half";
  nf   = (ntot+1)/2;
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << DDName(name,idNameSpace)
		       << " Polyhedra made of " << getGenMaterial() << " with "
		       << getNsectortot() << " sectors from " 
		       << -alpha/CLHEP::deg << " to " 
		       << (-alpha+dphi)/CLHEP::deg << " and with " << nf
		       << " sections "; 
  for (i = 0; i < pgonZHalf.size(); i++) {
    LogDebug("HCalGeom") << "\t" << "\tZ = " << pgonZHalf[i] << "\tRmin = "
			 << pgonRminHalf[i] << "\tRmax = " << pgonRmaxHalf[i];
  }

  solid =   DDSolidFactory::polyhedra(DDName(name, idNameSpace),
				      getNsectortot(), -alpha, dphi, pgonZHalf,
				      pgonRminHalf, pgonRmaxHalf);
  DDLogicalPart genlogich(DDName(name, idNameSpace), matter, solid);

  cpv.position(genlogich, genlogic, 1, r0, rot);
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: "  << genlogich.name() 
		       << " number 1 positioned in " << genlogic.name() 
		       << " at " << r0 << " with " << rot;

  if (getNhalf() != 1) {
    rot = DDRotation(DDName(rotHalf, rotns));
   cpv.position(genlogich, genlogic, 2, r0, rot);
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo test:  " << genlogich.name()
			 << " number 2 positioned in " << genlogic.name()
			 << " at " << r0 << " with " << rot;
  } //end if (getNhalf...
  
  //Construct sector (from -alpha to +alpha)
  name = idName + "Module";
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << DDName(name,idNameSpace)
		       << " Polyhedra made of " << getGenMaterial() 
		       << " with 1 sector from " << -alpha/CLHEP::deg << " to "
		       << alpha/CLHEP::deg << " and with " << nf <<" sections";
  for (i = 0; i < pgonZHalf.size(); i++) {
    LogDebug("HCalGeom") << "\t" << "\tZ = " << pgonZHalf[i] << "\tRmin = " 
			 << pgonRminHalf[i] << "\tRmax = " << pgonRmaxHalf[i];
  }

  solid =   DDSolidFactory::polyhedra(DDName(name, idNameSpace),
				      1, -alpha, 2*alpha, pgonZHalf,
				      pgonRminHalf, pgonRmaxHalf);
  DDLogicalPart seclogic(DDName(name, idNameSpace), matter, solid);
  
  for (int ii=0; ii<getNsectortot(); ii++) {
    double phi    = ii*2*alpha;
    double phideg = phi/CLHEP::deg;
    
    DDRotation rotation;
    string rotstr("NULL");
    if (phideg != 0) {
      rotstr = "R"; 
      if (phideg < 100)	rotstr = "R0"; 
      rotstr = rotstr + std::to_string(phideg);
      rotation = DDRotation(DDName(rotstr, rotns)); 
      if (!rotation) {
	LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: Creating a new rotat"
			     << "ion " << rotstr << "\t" << 90 << "," << phideg
			     << ","  << 90 << "," << (phideg+90) << ", 0, 0";
	rotation = DDrot(DDName(rotstr, rotns), 90*CLHEP::deg, 
			 phideg*CLHEP::deg, 90*CLHEP::deg, 
			 (90+phideg)*CLHEP::deg, 0*CLHEP::deg,  0*CLHEP::deg);
      } //if !rotation
    } //if phideg!=0
  
   cpv.position(seclogic, genlogich, ii+1, r0, rotation);
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << seclogic.name() 
			 << " number " << ii+1 << " positioned in " 
			 << genlogich.name() << " at " << r0 << " with "
			 << rotation;
  }
  
  //Construct the things inside the sector
  constructInsideSector(seclogic, cpv);
}


void DDHCalBarrelAlgo::constructInsideSector(const DDLogicalPart& sector, DDCompactView& cpv) {
  
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: Layers (" << getNLayers()
		       << ") ...";

  double alpha = CLHEP::pi/getNsectors();
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
    pgonZ.emplace_back(0);
    pgonRmin.emplace_back(rin); 
    pgonRmax.emplace_back(rout);
    // index 1
    pgonZ.emplace_back(zout);  
    pgonRmin.emplace_back(rin); 
    pgonRmax.emplace_back(rout);
    if (in == out) {
      if (in <= 3) {
	//index 2
	pgonZ.emplace_back(getZoff(in) + rout*getTanTheta(in));
	pgonRmin.emplace_back(pgonRmax[1]);
	pgonRmax.emplace_back(pgonRmax[1]);
	nsec++;
      }
    } else {
      if (in == 3) {
	//redo index 1, add index 2
	pgonZ[1]    =(getZoff(out) + getRmax(out)*getTanTheta(out));
	pgonZ.emplace_back(pgonZ[1] + deltaz);
	pgonRmin.emplace_back(pgonRmin[1]); 
	pgonRmax.emplace_back(getRmax(in));
	//index 3 
	pgonZ.emplace_back(getZoff(in) + getRmax(in)*getTanTheta(in));
	pgonRmin.emplace_back(pgonRmin[2]); 
	pgonRmax.emplace_back(pgonRmax[2]);
        nsec       += 2;
      } else {
	//index 2
	pgonZ.emplace_back(getZoff(in) + getRmax(in)*getTanTheta(in));
	pgonRmin.emplace_back(getRmax(in)); 
	pgonRmax.emplace_back(pgonRmax[1]); 
	nsec++;
	if (in == 0) {
	  pgonZ.emplace_back(getZoff(out) + getRmax(in)*getTanTheta(out));
          pgonRmin.emplace_back(pgonRmin[2]); 
	  pgonRmax.emplace_back(pgonRmax[2]);
	  nsec++;
	}
	if (in <= 1) {
	  pgonZ.emplace_back(getZoff(out) + rout*getTanTheta(out));
	  pgonRmin.emplace_back(rout);
	  pgonRmax.emplace_back(rout);
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
      LogDebug("HCalGeom") << "\t" << "Alpha_1 modified from " 
			   << alpha/CLHEP::deg << " to " << alpha1/CLHEP::deg 
			   << " Rmid " << rmid << " Reduced width " << width;
    }
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << name << " (Layer " 
			 << i << ") Polyhedra made of " << getLayerMaterial(i)
			 << " with 1 sector from " << -alpha1/CLHEP::deg 
			 << " to " << alpha1/CLHEP::deg << " and with " 
			 << nsec << " sections";
    for (unsigned int k=0; k<pgonZ.size(); k++) {
      LogDebug("HCalGeom") << "\t" << "\t" << pgonZ[k] << "\t" << pgonRmin[k]
			   << "\t" << pgonRmax[k];
    }    
    solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace), 
				      1, -alpha1, 2*alpha1,
				      pgonZ, pgonRmin, pgonRmax);
    DDLogicalPart glog(DDName(name, idNameSpace), matter, solid);

   cpv.position(glog, sector, getLayerId(i), DDTranslation(0.0, 0.0, 0.0), 
	   DDRotation());
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << glog.name() 
			 << " number " << getLayerId(i) << " positioned in " 
			 << sector.name() << " at (0,0,0) with no rotation";

    constructInsideLayers(glog, getLayerLabel(i), getLayerId(i), 
			  getLayerAbsorb(i), rin,  getLayerD1(i), alpha1, 
			  getLayerD2(i), getLayerAlpha(i), getLayerT1(i),
			  getLayerT2(i), cpv);
    rin = rout;
  }
  
}

void DDHCalBarrelAlgo::constructInsideLayers(const DDLogicalPart& laylog,
					     const string& nm, int id, int nAbs, 
					     double rin, double d1, 
					     double alpha1, double d2, 
					     double alpha2, double t1,
					     double t2, DDCompactView& cpv) {
  
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: \t\tInside layer " << id 
		       << "...";

  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  DDRotation rot(DDName(detRot, rotns));

  string nam0 = nm + "In";
  string name = idName + nam0;
  DDName matName(DDSplit(getDetMat()).first, DDSplit(getDetMat()).second);
  DDMaterial matter (matName);

  DDSolid        solid;
  DDLogicalPart  glog, mother;
  double         rsi, dx, dy, dz, x, y;
  int            i, in;
  //Two lower volumes
  if (alpha1 > 0) {
    rsi = rin + d1;
    in  = 0;
    for (i = 0; i < getRzones()-1; i++) {
      if (rsi >= getRmax(i)) in = i+1;
    }
    dx = 0.5*t1;
    dy = 0.5*rsi*(tan(alpha1)-tan(alpha2));
    dz = 0.5*(getZoff(in) + rsi*getTanTheta(in));
    x  = rsi + dx;
    y  = 0.5*rsi*(tan(alpha1)+tan(alpha2));
    DDTranslation r11(x, y, dz);
    DDTranslation r12(x, -y, dz);

    solid = DDSolidFactory::box(DDName(name+"1", idNameSpace), dx, dy, dz);
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << solid.name() 
			 <<" Box made of " << getDetMat() << " of dimensions "
			 << dx << ", " << dy << ", " << dz;
    glog = DDLogicalPart(solid.ddname(), matter, solid);

    if (nAbs != 0) {
      mother = constructSideLayer(laylog, name, nAbs, rin, alpha1, cpv);
    } else {
      mother = laylog;
    }
    cpv.position(glog, mother, idOffset+1, r11, DDRotation());
    cpv.position(glog, mother, idOffset+2, r12, rot);
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << glog.name() 
			 << " Number " << idOffset+1 << " positioned in " 
			 << mother.name() << " at " << r11 
			 << " with no rotation\n"
			 << "DDHCalBarrelAlgo test: " << glog.name() 
			 << " Number " << idOffset+2 << " positioned in " 
			 << mother.name() << " at " << r12 << " with " << rot;

    //Constructin the plastics and scintillators inside
    constructInsideDetectors(glog, nam0+"1", id, dx, dy, dz, 1, cpv);
  }

  //Upper volume
  rsi = rin + d2;
  in  = 0;
  for (i = 0; i < getRzones()-1; i++) {
    if (rsi >= getRmax(i)) in = i+1;
  }
  dx  = 0.5*t2;
  dy  = 0.5*rsi*tan(alpha2);
  dz  = 0.5*(getZoff(in) + rsi*getTanTheta(in));
  x   = rsi + dx;
  DDTranslation r21(x, dy, dz);
  DDTranslation r22(x, -dy, dz);
  
  solid = DDSolidFactory::box(DDName(name+"2", idNameSpace), dx, dy, dz);
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << solid.name() 
		       << " Box made of " << getDetMat() << " of dimensions "
		       << dx << ", " << dy << ", " << dz;
  glog = DDLogicalPart(solid.ddname(), matter, solid);

  if (nAbs < 0) {
    mother = constructMidLayer(laylog, name, rin, alpha1, cpv);
  } else {
    mother = laylog;
  }
 cpv.position(glog, mother, idOffset+3, r21, DDRotation());
 cpv.position(glog, mother, idOffset+4, r22, rot);
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << glog.name() <<" Number "
		       << idOffset+3 << " positioned in " << mother.name() 
		       << " at " << r21 << " with no rotation\n"
		       << "DDHCalBarrelAlgo test: " << glog.name() <<" Number "
		       << idOffset+4 << " positioned in " << mother.name()
		       << " at " << r22 << " with " << rot;

  //Constructin the plastics and scintillators inside
  constructInsideDetectors(glog, nam0+"2", id, dx, dy, dz, 2, cpv);
}

DDLogicalPart DDHCalBarrelAlgo::constructSideLayer(const DDLogicalPart& laylog,
						   const string& nm, int nAbs, 
						   double rin, double alpha,
						   DDCompactView& cpv) {

  //Extra absorber layer
  int k = abs(nAbs) - 1;
  string namek = nm + "Side";
  double rsi   = rin + getSideD(k);
  int    in  = 0;
  for (int i = 0; i < getRzones()-1; i++) {
    if (rsi >= getRmax(i)) in = i+1;
  }
  vector<double> pgonZ, pgonRmin, pgonRmax;
  // index 0
  pgonZ.emplace_back(0.0);     
  pgonRmin.emplace_back(rsi); 
  pgonRmax.emplace_back(rsi+getSideT(k));
  // index 1
  pgonZ.emplace_back(getZoff(in) + rsi*getTanTheta(in));  
  pgonRmin.emplace_back(rsi); 
  pgonRmax.emplace_back(pgonRmax[0]);
  // index 2
  pgonZ.emplace_back(getZoff(in) + pgonRmax[0]*getTanTheta(in));
  pgonRmin.emplace_back(pgonRmax[1]);
  pgonRmax.emplace_back(pgonRmax[1]);
  DDSolid solid = DDSolidFactory::polyhedra(DDName(namek, idNameSpace), 1, 
					    -alpha, 2*alpha, pgonZ, pgonRmin, 
					    pgonRmax);
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << solid.name() 
		       << " Polyhedra made of " << getSideMat(k) 
		       << " with 1 sector from " << -alpha/CLHEP::deg
		       << " to " << alpha/CLHEP::deg << " and with "
		       << pgonZ.size() << " sections";
  for (unsigned int ii=0; ii<pgonZ.size(); ii++) {
    LogDebug("HCalGeom") << "\t\tZ = " << pgonZ[ii] << "\tRmin = " 
			 << pgonRmin[ii] << "\tRmax = " << pgonRmax[ii];
  }

  DDName matName(DDSplit(getSideMat(k)).first, DDSplit(getSideMat(k)).second);
  DDMaterial matter(matName);
  DDLogicalPart glog = DDLogicalPart(solid.ddname(), matter, solid);

  cpv.position(glog, laylog, 1, DDTranslation(), DDRotation());
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << glog.name() 
		       << " Number 1 positioned in " << laylog.name()
		       << " at (0,0,0) with no rotation";

  if (nAbs < 0) {
    DDLogicalPart mother = glog;
    double rmid  = pgonRmax[0];
    for (int i = 0; i < getSideAbsorber(); i++) {
      double alpha1 = atan(getSideAbsW(i)/rmid);  
      if (alpha1 > 0) {
	string name   = namek + getSideAbsName(i);
	solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace), 1, 
					  -alpha1, 2*alpha1, pgonZ, pgonRmin, 
					  pgonRmax);
	LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << solid.name() 
			     << " Polyhedra made of " << getSideAbsMat(i) 
			     << " with 1 sector from " << -alpha1/CLHEP::deg
			     << " to " << alpha1/CLHEP::deg << " and with "
			     << pgonZ.size() << " sections";
	for (unsigned int ii=0; ii<pgonZ.size(); ii++) {
	  LogDebug("HCalGeom") << "\t\tZ = " << pgonZ[ii] << "\tRmin = " 
			       << pgonRmin[ii] << "\tRmax = " << pgonRmax[ii];
	}

	DDName matName(DDSplit(getSideAbsMat(i)).first, 
		       DDSplit(getSideAbsMat(i)).second);
	DDMaterial matter(matName);
	DDLogicalPart log = DDLogicalPart(solid.ddname(), matter, solid);

	cpv.position(log, mother, 1, DDTranslation(), DDRotation());
	LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << log.name() 
			     << " Number 1 positioned in " << mother.name()
			     << " at (0,0,0) with no rotation";
	mother = log;
      }
    }
  }
  return glog;
}

DDLogicalPart DDHCalBarrelAlgo::constructMidLayer(const DDLogicalPart& laylog,
						  const string& nm, double rin, 
						  double alpha, DDCompactView& cpv) {

  DDSolid       solid;
  DDLogicalPart log, glog;
  string name = nm + "Mid";
  for (int k=0; k < getAbsorberN(); k++) {
    string namek = name + getAbsorbName(k);
    double rsi   = rin + getAbsorbD(k);
    int    in  = 0;
    for (int i = 0; i < getRzones()-1; i++) {
      if (rsi >= getRmax(i)) in = i+1;
    }
    vector<double> pgonZ, pgonRmin, pgonRmax;
    // index 0
    pgonZ.emplace_back(0.0);     
    pgonRmin.emplace_back(rsi); 
    pgonRmax.emplace_back(rsi+getAbsorbT(k));
    // index 1
    pgonZ.emplace_back(getZoff(in) + rsi*getTanTheta(in));  
    pgonRmin.emplace_back(rsi); 
    pgonRmax.emplace_back(pgonRmax[0]);
    // index 2
    pgonZ.emplace_back(getZoff(in) + pgonRmax[0]*getTanTheta(in));
    pgonRmin.emplace_back(pgonRmax[1]);
    pgonRmax.emplace_back(pgonRmax[1]);
    solid = DDSolidFactory::polyhedra(DDName(namek, idNameSpace), 1, -alpha, 
				      2*alpha, pgonZ, pgonRmin, pgonRmax);
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << solid.name() 
			 << " Polyhedra made of " << getAbsorbMat(k) 
			 << " with 1 sector from " << -alpha/CLHEP::deg
			 << " to " << alpha/CLHEP::deg << " and with "
			 << pgonZ.size() << " sections";
    for (unsigned int ii=0; ii<pgonZ.size(); ii++) {
      LogDebug("HCalGeom") << "\t\tZ = " << pgonZ[ii] << "\tRmin = " 
			   << pgonRmin[ii] << "\tRmax = " << pgonRmax[ii];
    }

    DDName matName(DDSplit(getAbsorbMat(k)).first, DDSplit(getAbsorbMat(k)).second);
    DDMaterial matter(matName);
    log = DDLogicalPart(solid.ddname(), matter, solid);

    cpv.position(log, laylog, 1, DDTranslation(), DDRotation());
    LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << log.name() 
			 << " Number 1 positioned in " << laylog.name()
			 << " at (0,0,0) with no rotation";
    
    if (k==0) {
      double rmin   = pgonRmin[0];
      double rmax   = pgonRmax[0];
      DDLogicalPart mother = log;
      for (int i=0; i<1; i++) {
	double alpha1 = atan(getMidAbsW(i)/rmin);
	string namek  = name + getMidAbsName(i);
	solid = DDSolidFactory::polyhedra(DDName(namek, idNameSpace), 1, 
					  -alpha1, 2*alpha1, pgonZ, pgonRmin, 
					  pgonRmax);
	LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << solid.name() 
			     << " Polyhedra made of " << getMidAbsMat(i) 
			     << " with 1 sector from " << -alpha1/CLHEP::deg
			   << " to " << alpha1/CLHEP::deg << " and with "
			     << pgonZ.size() << " sections";
	for (unsigned int ii=0; ii<pgonZ.size(); ii++) {
	  LogDebug("HCalGeom") << "\t\tZ = " << pgonZ[ii] << "\tRmin = " 
			       << pgonRmin[ii] << "\tRmax = " << pgonRmax[ii];
	}

	DDName matNam1(DDSplit(getMidAbsMat(i)).first, 
		       DDSplit(getMidAbsMat(i)).second);
	DDMaterial matter1(matNam1);
	log = DDLogicalPart(solid.ddname(), matter1, solid);

	cpv.position(log, mother, 1, DDTranslation(), DDRotation());
	LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << log.name() 
			     << " Number 1 positioned in " << mother.name()
			     << " at (0,0,0) with no rotation";
	mother = log;
      }

      // Now the layer with detectors
      double rmid = rmin + getMiddleD();
      pgonRmin[0] = rmid; pgonRmax[0] = rmax;
      pgonRmin[1] = rmid; pgonRmax[1] = rmax; pgonZ[1] = getZoff(in) + rmid*getTanTheta(in);
      pgonRmin[2] = rmax; pgonRmax[2] = rmax; pgonZ[2] = getZoff(in) + rmax*getTanTheta(in);
      double alpha1 = atan(getMiddleW()/rmin);
      solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace), 1, 
					-alpha1, 2*alpha1, pgonZ, pgonRmin, 
					pgonRmax);
      LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << solid.name() 
			   << " Polyhedra made of " << getMiddleMat() 
			   << " with 1 sector from " << -alpha1/CLHEP::deg
			   << " to " << alpha1/CLHEP::deg << " and with "
			   << pgonZ.size() << " sections";
      for (unsigned int ii=0; ii<pgonZ.size(); ii++) {
	LogDebug("HCalGeom") << "\t\tZ = " << pgonZ[ii] << "\tRmin = " 
			     << pgonRmin[ii] << "\tRmax = " << pgonRmax[ii];
      }

      DDName matNam1(DDSplit(getMiddleMat()).first, 
		     DDSplit(getMiddleMat()).second);
      DDMaterial matter1(matNam1);
      glog = DDLogicalPart(solid.ddname(), matter1, solid);

      cpv.position(glog, mother, 1, DDTranslation(), DDRotation());
      LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << glog.name() 
			   << " Number 1 positioned in " << mother.name()
			   << " at (0,0,0) with no rotation";

      // Now the remaining absorber layers
      for (int i = 1; i < getMidAbsorber(); i++) {
	namek  = name + getMidAbsName(i);
	rmid   = rmin + getMidAbsT(i);
	pgonRmin[0] = rmin; pgonRmax[0] = rmid;
	pgonRmin[1] = rmin; pgonRmax[1] = rmid; pgonZ[1] = getZoff(in) + rmin*getTanTheta(in);
	pgonRmin[2] = rmid; pgonRmax[2] = rmid; pgonZ[2] = getZoff(in) + rmid*getTanTheta(in);
	alpha1 = atan(getMidAbsW(i)/rmin);
	solid = DDSolidFactory::polyhedra(DDName(namek, idNameSpace), 1, 
					  -alpha1, 2*alpha1, pgonZ, pgonRmin, 
					  pgonRmax);
	LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << solid.name() 
			     << " Polyhedra made of " << getMidAbsMat(i) 
			     << " with 1 sector from " << -alpha1/CLHEP::deg
			     << " to " << alpha1/CLHEP::deg << " and with "
			     << pgonZ.size() << " sections";
	for (unsigned int ii=0; ii<pgonZ.size(); ii++) {
	  LogDebug("HCalGeom") << "\t\tZ = " << pgonZ[ii] << "\tRmin = " 
			       << pgonRmin[ii] << "\tRmax = " << pgonRmax[ii];
	}

	DDName matName2(DDSplit(getMidAbsMat(i)).first, 
			DDSplit(getMidAbsMat(i)).second);
	DDMaterial matter2(matName2);
	log = DDLogicalPart(solid.ddname(), matter2, solid);

	cpv.position(log, mother, i, DDTranslation(), DDRotation());
	LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << log.name() 
			     << " Number " << i << " positioned in " 
			     << mother.name() << " at (0,0,0) with no "
			     << "rotation";
	mother = log;
      }
    }
  }
  return glog;
}
 
void DDHCalBarrelAlgo::constructInsideDetectors(const DDLogicalPart& detector,
						const string& name, int id, double dx,
						double dy, double dz,
						int type, DDCompactView& cpv) {

  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: \t\tInside detector " << id 
		       << "...";
  
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
  double  dx1, dx2, shiftX;

  if (type == 1) {
    wid = 0.5*getDetWidth1(id);
    if (getDetPosY(id)>0) y =-dy+wid;
    dx1    = 0.5*getDetT11(id);
    dx2    = 0.5*getDetT21(id);
    shiftX = getDetdP1(id);
  } else {
    wid = 0.5*getDetWidth2(id);
    dx1    = 0.5*getDetT12(id);
    dx2    = 0.5*getDetT22(id);
    shiftX = getDetdP2(id);
  }

  solid = DDSolidFactory::box(DDName(plname+"1", idNameSpace), dx1, wid, dz);
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << solid.name() 
		       << " Box made of " << getDetMatPl() << " of dimensions "
		       << dx1 <<", " << wid << ", "  << dz;
  glog = DDLogicalPart(solid.ddname(), plmatter, solid); 

  double x = shiftX + dx1 - dx;
  cpv.position(glog, detector, 1, DDTranslation(x,y,0), DDRotation());
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << glog.name() 
		       << " Number 1 positioned in " << detector.name() 
		       << " at (" << x << "," << y << ",0) with no rotation";

  solid = DDSolidFactory::box(DDName(scname, idNameSpace), 
			      0.5*getDetTsc(id), wid, dz);
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << solid.name() 
		       << " Box made of " << getDetMatSc() << " of dimensions "
		       << 0.5*getDetTsc(id) << ", " << wid << ", " << dz;
  glog = DDLogicalPart(solid.ddname(), scmatter, solid);

  x += dx1 + 0.5*getDetTsc(id);
  int copyNo = id*10 + getDetType(id);
  cpv.position(glog, detector, copyNo, DDTranslation(x, y, 0), DDRotation());
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << glog.name() <<" Number "
		       << copyNo << " positioned in " << detector.name() 
		       << " at (" << x << "," << y  << ",0) with no rotation";

  solid = DDSolidFactory::box(DDName(plname+"2", idNameSpace), dx2, wid, dz);
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << solid.name() 
		       << " Box made of " << getDetMatPl() << " of dimensions "
		       << dx2 <<", " << wid << ", "  << dz;
  glog = DDLogicalPart(solid.ddname(), plmatter, solid);

  x+=0.5*getDetTsc(id) + dx2;
 cpv.position(glog, detector, 1, DDTranslation(x, y, 0), DDRotation());
  LogDebug("HCalGeom") << "DDHCalBarrelAlgo test: " << glog.name() 
		       << " Number 1 positioned in " << detector.name() 
		       << " at (" << x << "," << y << ",0) with no rotation";

}
