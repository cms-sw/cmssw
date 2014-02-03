///////////////////////////////////////////////////////////////////////////////
// File: DDHCalEndcapAlgo.cc
//   adapted from CCal(G4)HcalEndcap.cc
// Description: Geometry factory class for Hcal Endcap
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/plugins/DDHCalEndcapAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDHCalEndcapAlgo::DDHCalEndcapAlgo():
  modMat(0),modType(0),sectionModule(0),layerN(0),layerN0(0),layerN1(0),
  layerN2(0),layerN3(0),layerN4(0),layerN5(0),thick(0),trimLeft(0),
  trimRight(0),zminBlock(0),zmaxBlock(0),rinBlock1(0),routBlock1(0),
  rinBlock2(0),routBlock2(0),layerType(0),layerT(0),scintT(0) {
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo info: Creating an instance";
}

DDHCalEndcapAlgo::~DDHCalEndcapAlgo() {}

int DDHCalEndcapAlgo::getLayer(unsigned int i, unsigned int j) const {

  switch (i) {
  case 0: 
    return layerN0[j];
    break;

  case 1: 
    return layerN1[j];
    break;

  case 2: 
    return layerN2[j];
    break;

  case 3: 
    return layerN3[j];
    break;

  case 4: 
    return layerN4[j];
    break;

  case 5: 
    return layerN5[j];
    break;

  default:
    return 0;
  }
}

double DDHCalEndcapAlgo::getTrim(unsigned int i, unsigned int j) const {
 
 if (j == 0)
    return trimLeft[i];
  else
    return trimRight[j];
}

void DDHCalEndcapAlgo::initialize(const DDNumericArguments & nArgs,
				  const DDVectorArguments & vArgs,
				  const DDMapArguments & ,
				  const DDStringArguments & sArgs,
				  const DDStringVectorArguments & vsArgs) {

  int i,j;
  genMaterial   = sArgs["MaterialName"];
  rotation      = sArgs["Rotation"];
  nsectors      = int (nArgs["Sector"]);
  nsectortot    = int (nArgs["SectorTot"]);
  nEndcap       = int (nArgs["Endcap"]);
  rotHalf       = sArgs["RotHalf"];
  rotns         = sArgs["RotNameSpace"];
  zShift        = nArgs["ZShift"];

  zFront        = nArgs["ZFront"];
  zEnd          = nArgs["ZEnd"];
  ziNose        = nArgs["ZiNose"];
  ziL0Nose      = nArgs["ZiL0Nose"];
  ziBody        = nArgs["ZiBody"];
  ziL0Body      = nArgs["ZiL0Body"];
  z0Beam        = nArgs["Z0Beam"];
  ziDip         = nArgs["ZiDip"];
  dzStep        = nArgs["DzStep"];
  zShiftHac2    = nArgs["ZShiftHac2"];
  double gap    = nArgs["Gap"];
  double z1     = nArgs["Z1"];
  double r1     = nArgs["R1"];
  rout          = nArgs["Rout"];
  heboxDepth    = nArgs["HEboxDepth"];
  drEnd         = nArgs["DrEnd"];
  double etamin = nArgs["Etamin"];
  angBot        = nArgs["AngBot"];
  angGap        = nArgs["AngGap"];

  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo debug: General material " 
		       << genMaterial << "\tSectors "  << nsectors << ",  " 
		       << nsectortot << "\tEndcaps " << nEndcap  
		       << "\tRotation matrix for half " << rotns 
		       << ":" << rotHalf << "\n\tzFront " << zFront << " zEnd "
		       << zEnd << " ziNose " << ziNose << " ziL0Nose " 
		       << ziL0Nose << " ziBody " << ziBody  << " ziL0Body " 
		       << ziL0Body << " z0Beam " << z0Beam << " ziDip " 
		       << ziDip << " dzStep " << dzStep << " Gap " << gap 
		       << " z1 " << z1 << "\n\tr1 " << r1 << " rout " << rout
		       << " HeboxDepth " << heboxDepth << " drEnd " << drEnd 
		       << "\tetamin " << etamin << " Bottom angle " << angBot
		       << " Gap angle " << angGap << " Z-Shift " << zShift
		       << " " << zShiftHac2;

  //Derived quantities
  angTop   = 2.0 * atan (exp(-etamin));
  slope    = tan(angGap);
  z1Beam   = z1 - r1/slope;
  ziKink   = z1Beam + rout/slope;
  riKink   = ziKink*tan(angBot);
  riDip    = ziDip*tan(angBot);
  roDip    = rout - heboxDepth;
  dzShift  = (z1Beam - z0Beam) - gap/sin(angGap);
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo debug: angTop " 
		       << angTop/CLHEP::deg  <<"\tSlope " << slope 
		       << "\tDzShift " << dzShift << "\n\tz1Beam " << z1Beam 
		       << "\tziKink" << ziKink << "\triKink " << riKink 
		       << "\triDip " << riDip << "\n\troDip " << roDip 
		       << "\tRotation " << rotation;

  ///////////////////////////////////////////////////////////////
  //Modules
  absMat        = sArgs["AbsMat"];
  modules       = int(nArgs["Modules"]);
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo debug: Number of modules " 
		       << modules << " and absorber material " << absMat;

  modName       = vsArgs["ModuleName"];
  modMat        = vsArgs["ModuleMat"];
  modType       = dbl_to_int(vArgs["ModuleType"]);
  sectionModule = dbl_to_int(vArgs["SectionModule"]);
  thick         = vArgs["ModuleThick"];
  trimLeft      = vArgs["TrimLeft"]; 
  trimRight     = vArgs["TrimRight"]; 
  eModule       = dbl_to_int(vArgs["EquipModule"]);
  layerN        = dbl_to_int(vArgs["LayerN"]);
  layerN0       = dbl_to_int(vArgs["LayerN0"]);
  layerN1       = dbl_to_int(vArgs["LayerN1"]);
  layerN2       = dbl_to_int(vArgs["LayerN2"]);
  layerN3       = dbl_to_int(vArgs["LayerN3"]);
  layerN4       = dbl_to_int(vArgs["LayerN4"]);
  layerN5       = dbl_to_int(vArgs["LayerN5"]);
  for (i = 0; i < modules; i++) {
    edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo debug: " << modName[i] <<" type "
			 << modType[i] << " Sections " << sectionModule[i] 
			 << " thickness of absorber/air " << thick[i] 
			 << " trim " << trimLeft[i] << ", " << trimRight[i] 
			 << " equip module " << eModule[i] << " with "
			 << layerN[i] << " layers";
    if (i == 0) {
      for (j = 0; j < layerN[i]; j++) {
	edm::LogInfo("HCalGeom") << "\t " << layerN0[j] << "/" << layerN0[j+1];
      }
    } else if (i == 1) {
      for (j = 0; j < layerN[i]; j++) {
	edm::LogInfo("HCalGeom") << "\t " << layerN1[j] << "/" << layerN1[j+1];
      }
    } else if (i == 2) {
      for (j = 0; j < layerN[i]; j++) {
	edm::LogInfo("HCalGeom") << "\t " << layerN2[j];
      }
    } else if (i == 3) {
      for (j = 0; j < layerN[i]; j++) {
	edm::LogInfo("HCalGeom") << "\t " << layerN3[j];
      }
    } else if (i == 4) {
      for (j = 0; j < layerN[i]; j++) {
	edm::LogInfo("HCalGeom") << "\t " << layerN4[j];
      }
    } else if (i == 5) {
      for (j = 0; j < layerN[i]; j++) {
	edm::LogInfo("HCalGeom") << "\t " << layerN5[j];
      }
    }
  }
  
  ///////////////////////////////////////////////////////////////
  //Layers
  phiSections = int(nArgs["PhiSections"]);
  phiName     = vsArgs["PhiName"];
  layers      = int(nArgs["Layers"]);
  layerName   = vsArgs["LayerName"];
  layerType   = dbl_to_int(vArgs["LayerType"]);
  layerT      = vArgs["LayerT"];
  scintT      = vArgs["ScintT"];
  scintMat    = sArgs["ScintMat"];
  plastMat    = sArgs["PlastMat"];
  rotmat      = sArgs["RotMat"];
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo debug: Phi Sections " 
		       << phiSections;
  for (i = 0; i < phiSections; i++) 
    edm::LogInfo("HCalGeom") << "\tName[" << i << "] : " << phiName[i];
  edm::LogInfo("HCalGeom") << "\tPlastic: " << plastMat << "\tScintillator: "
		       << scintMat << "\tRotation matrix " << rotns << ":" 
		       << rotmat << "\n\tNumber of layers " << layers;
  for (i = 0; i < layers; i++) {
    edm::LogInfo("HCalGeom") << "\t" << layerName[i] << "\tType " << layerType[i]
			 << "\tThickness " << layerT[i] << "\tScint.Thick " 
			 << scintT[i];
  }

  ///////////////////////////////////////////////////////////////
  // Derive bounding of the modules
  int module = 0;
  // Layer 0 (Nose)
  if (modules > 0) {
    zminBlock.push_back(ziL0Nose);
    zmaxBlock.push_back(zminBlock[module] + layerT[0] + 0.5*dzStep);
    rinBlock1.push_back(zminBlock[module] * tan(angTop));
    rinBlock2.push_back(zmaxBlock[module] * tan(angTop));
    routBlock1.push_back((zminBlock[module] - z1Beam) * slope);
    routBlock2.push_back((zmaxBlock[module] - z1Beam) * slope);
    module++;
  }

  // Layer 0 (Body)
  if (modules > 1) {
    zminBlock.push_back(ziL0Body);
    zmaxBlock.push_back(zminBlock[module] + layerT[0] + 0.5*dzStep);
    rinBlock1.push_back(zminBlock[module] * tan(angBot));
    rinBlock2.push_back(zmaxBlock[module] * tan(angBot));
    routBlock1.push_back(zminBlock[module] * tan(angTop));
    routBlock2.push_back(zmaxBlock[module] * tan(angTop));
    module++;
  }

  // Hac1
  if (modules > 2) {
    zminBlock.push_back(ziNose);
    zmaxBlock.push_back(ziBody);
    rinBlock1.push_back(zminBlock[module] * tan(angTop));
    rinBlock2.push_back(zmaxBlock[module] * tan(angTop));
    routBlock1.push_back((zminBlock[module] - z1Beam) * slope);
    routBlock2.push_back((zmaxBlock[module] - z1Beam) * slope);
    module++;
  }

  // Hac2
  if (modules > 3) {
    zminBlock.push_back(ziBody);
    zmaxBlock.push_back(zminBlock[module] + layerN[3]*dzStep);
    rinBlock1.push_back(zminBlock[module] * tan(angBot));
    rinBlock2.push_back(zmaxBlock[module] * tan(angBot));
    routBlock1.push_back((zmaxBlock[module-1] - z1Beam) * slope);
    routBlock2.push_back(rout);
    module++;
  }

  // Hac3
  if (modules > 4) {
    zminBlock.push_back(zmaxBlock[module-1]);
    zmaxBlock.push_back(zminBlock[module] + layerN[4]*dzStep);
    rinBlock1.push_back(zminBlock[module] * tan(angBot));
    rinBlock2.push_back(zmaxBlock[module] * tan(angBot));
    routBlock1.push_back(rout);
    routBlock2.push_back(rout);
    module++;
  }

  // Hac4
  if (modules > 5) {
    zminBlock.push_back(zmaxBlock[module-1]);
    zmaxBlock.push_back(zminBlock[module] + layerN[5]*dzStep);
    rinBlock1.push_back(zminBlock[module] * tan(angBot));
    rinBlock2.push_back(zmaxBlock[module] * tan(angBot));
    routBlock1.push_back(rout);
    routBlock2.push_back(roDip);
    module++;
  }

  for (i = 0; i < module; i++)
    edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo debug: Module " << i 
			 << "\tZ/Rin/Rout " << zminBlock[i] << ", " 
			 << zmaxBlock[i] << "/ " << rinBlock1[i] << ", " 
			 << rinBlock2[i] << "/ " << routBlock1[i] << ", " 
			 << routBlock2[i];

  idName      = sArgs["MotherName"];
  idNameSpace = DDCurrentNamespace::ns();
  idOffset = int (nArgs["IdOffset"]); 
  DDName parentName = parent().name(); 
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo debug: Parent " << parentName 
		       << " idName " << idName << " NameSpace " << idNameSpace
		       << " Offset " << idOffset;

  tolPos      = nArgs["TolPos"];
  tolAbs      = nArgs["TolAbs"];
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo debug: Tolerances - Positioning "
		       << tolPos << " Absorber " << tolAbs;
}

////////////////////////////////////////////////////////////////////
// DDHCalEndcapAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHCalEndcapAlgo::execute(DDCompactView& cpv) {
  
  edm::LogInfo("HCalGeom") << "==>> Constructing DDHCalEndcapAlgo...";
  constructGeneralVolume(cpv);
  edm::LogInfo("HCalGeom") << "<<== End of DDHCalEndcapAlgo construction ...";
}

//----------------------start here for DDD work!!! ---------------

void DDHCalEndcapAlgo::constructGeneralVolume(DDCompactView& cpv) {
  
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: General volume...";
  bool proto = true;
  for (int i=0; i<3; i++) 
    if (equipModule(i) > 0) proto = false;

  DDRotation    rot;
  if (DDSplit(getRotation()).first == "NULL") rot = DDRotation();
  else rot = DDRotation(DDName(DDSplit(getRotation()).first,DDSplit(getRotation()).second));
  edm::LogInfo("HCalGeom") << " First " << DDSplit(getRotation()).first
		       << " Second " << DDSplit(getRotation()).second 
		       << " Rotation " << rot;
  DDTranslation r0(0,0,getZShift());
  double alpha = CLHEP::pi/getNsectors();
  double dphi  = getNsectortot()*CLHEP::twopi/getNsectors();

  //!!!!!!!!!!!!!!!!!Should be zero. And removed as soon as
  //vertical walls are allowed in SolidPolyhedra
  double delz = 0;

  vector<double> pgonZ, pgonRmin, pgonRmax;
  if (proto) {
    double zf = getZiBody() + getZShiftHac2();
    pgonZ.push_back(zf - getDzShift()); 
    pgonRmin.push_back(zf * tan(getAngBot())); 
    pgonRmax.push_back((zf - getZ1Beam())*getSlope()); 
  } else {
    pgonZ.push_back(getZFront()   - getDzShift()); 
    pgonRmin.push_back(getZFront()   * tan(getAngTop())); 
    pgonRmax.push_back((getZFront()   - getZ1Beam())*getSlope()); 
    pgonZ.push_back(getZiL0Body() - getDzShift()); 
    pgonRmin.push_back(getZiL0Body() * tan(getAngTop())); 
    pgonRmax.push_back((getZiL0Body() - getZ1Beam())*getSlope()); 
    pgonZ.push_back(getZiL0Body() - getDzShift()); 
    pgonRmin.push_back(getZiL0Body() * tan(getAngBot())); 
    pgonRmax.push_back((getZiL0Body() - getZ1Beam())*getSlope()); 
  }
  pgonZ.push_back(getZiKink()   - getDzShift()); 
  pgonRmin.push_back(getRinKink()); 
  pgonRmax.push_back(getRout()); 
  pgonZ.push_back(getZiDip()    - getDzShift()); 
  pgonRmin.push_back(getRinDip()); 
  pgonRmax.push_back(getRout()); 
  pgonZ.push_back(getZiDip()    - getDzShift() + delz); 
  pgonRmin.push_back(getRinDip()); 
  pgonRmax.push_back(getRoutDip()); 
  pgonZ.push_back(getZEnd()     - getDzShift()); 
  pgonRmin.push_back(getZEnd() * tan(getAngBot())); 
  pgonRmax.push_back(getRoutDip()); 
  pgonZ.push_back(getZEnd()); 
  pgonRmin.push_back(getZEnd() * tan(getAngBot())); 
  pgonRmax.push_back(getRoutDip()); 

  string name("Null");
  unsigned int i=0;
  DDSolid solid;
  solid = DDSolidFactory::polyhedra(DDName(idName, idNameSpace),
				    getNsectortot(), -alpha, dphi, pgonZ, 
				    pgonRmin, pgonRmax);
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " 
		       << DDName(idName, idNameSpace) << " Polyhedra made of "
		       << getGenMat() << " with " << getNsectortot() 
		       << " sectors from " << -alpha/CLHEP::deg << " to " 
		       << (-alpha+dphi)/CLHEP::deg << " and with " 
		       << pgonZ.size() << " sections";
  for (i = 0; i <pgonZ.size(); i++) 
    edm::LogInfo("HCalGeom") << "\t\tZ = " << pgonZ[i] << "\tRmin = " <<pgonRmin[i]
			 << "\tRmax = " << pgonRmax[i];

  DDName matname(DDSplit(getGenMat()).first, DDSplit(getGenMat()).second); 
  DDMaterial matter(matname);
  DDLogicalPart genlogic(DDName(idName, idNameSpace), matter, solid);

  DDName parentName = parent().name(); 
  cpv.position(DDName(idName, idNameSpace), parentName, 1, r0, rot);
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " 
		       << DDName(idName, idNameSpace) << " number 1 positioned"
		       << " in " << parentName << " at " << r0 << " with " 
		       << rot;
  if (getEndcaps() != 1) {
    rot = DDRotation(DDName(rotHalf,rotns));
   cpv.position(DDName(idName, idNameSpace), parentName, 2, r0, rot);
    edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " 
			 << DDName(idName, idNameSpace) << " number 2 "
			 << "positioned in " << parentName  << " at " << r0
			 << " with " << rot;
  }

  //Forward half
  name  = idName + "Front";
  vector<double> pgonZMod, pgonRminMod, pgonRmaxMod;
  for (i=0; i < (pgonZ.size()-1); i++) {
    pgonZMod.push_back(pgonZ[i] + getDzShift()); 
    pgonRminMod.push_back(pgonRmin[i]); 
    pgonRmaxMod.push_back(pgonRmax[i]); 
  }
  solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace),
				    getNsectortot(), -alpha, dphi, pgonZMod,
				    pgonRminMod, pgonRmaxMod);
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << DDName(name,idNameSpace)
		       << " Polyhedra made of " << getGenMat() << " with "
		       << getNsectortot() << " sectors from " 
		       << -alpha/CLHEP::deg << " to " 
		       << (-alpha+dphi)/CLHEP::deg << " and with "
		       << pgonZMod.size() << " sections ";
  for (i = 0; i < pgonZMod.size(); i++) 
    edm::LogInfo("HCalGeom") << "\t\tZ = " << pgonZMod[i] << "\tRmin = " 
			 << pgonRminMod[i] << "\tRmax = " << pgonRmaxMod[i];
  DDLogicalPart genlogich(DDName(name, idNameSpace), matter, solid);

  cpv.position(genlogich, genlogic, 1, DDTranslation(0.0, 0.0, -getDzShift()),
	DDRotation());
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << genlogich.name() 
		       << " number 1 positioned in " << genlogic.name() 
		       << " at (0,0," << -getDzShift() << ") with no rotation";
  
  //Construct sector (from -alpha to +alpha)
  name  = idName + "Module";
  solid =   DDSolidFactory::polyhedra(DDName(name, idNameSpace),
				      1, -alpha, 2*alpha, pgonZMod,
				      pgonRminMod, pgonRmaxMod);
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << DDName(name,idNameSpace)
		       << " Polyhedra made of " << getGenMat() 
		       <<" with 1 sector from " << -alpha/CLHEP::deg << " to " 
		       << alpha/CLHEP::deg << " and with " << pgonZMod.size() 
		       << " sections";
  for (i = 0; i < pgonZMod.size(); i++) 
    edm::LogInfo("HCalGeom") << "\t\tZ = " << pgonZMod[i] << "\tRmin = " 
			 << pgonRminMod[i] << "\tRmax = " << pgonRmaxMod[i];

  DDLogicalPart seclogic(DDName(name, idNameSpace), matter, solid);
  
  for (int ii=0; ii<getNsectortot(); ii++) {
    double phi    = ii*2*alpha;
    double phideg = phi/CLHEP::deg;
    
    DDRotation rotation;
    string rotstr("NULL");
    if (phideg != 0) {
      rotstr = "R"; 
      if (phideg < 100)	rotstr = "R0"; 
      rotstr = rotstr + dbl_to_string(phideg);
      rotation = DDRotation(DDName(rotstr, rotns)); 
      if (!rotation) {
	edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: Creating a new "
			     << "rotation " << rotstr << "\t" << 90 << "," 
			     << phideg << ","  << 90 << "," << (phideg+90)
			     << ", 0, 0";
	rotation = DDrot(DDName(rotstr, rotns), 90*CLHEP::deg, 
			 phideg*CLHEP::deg, 90*CLHEP::deg, 
			 (90+phideg)*CLHEP::deg, 0*CLHEP::deg,  0*CLHEP::deg);
      } //if !rotation
    } //if phideg!=0
  
   cpv.position(seclogic, genlogich, ii+1, DDTranslation(0.0, 0.0, 0.0), rotation);
    edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << seclogic.name() 
			 << " number " << ii+1 << " positioned in " 
			 << genlogich.name() << " at (0,0,0) with " <<rotation;
  }
  
  //Construct the things inside the sector
  constructInsideSector(seclogic, cpv);

  //Backward half
  name  = idName + "Back";
  vector<double> pgonZBack, pgonRminBack, pgonRmaxBack;
  pgonZBack.push_back(getZEnd() - getDzShift()); 
  pgonRminBack.push_back(pgonZBack[0]*tan(getAngBot()) + getDrEnd()); 
  pgonRmaxBack.push_back(getRoutDip()); 
  pgonZBack.push_back(getZEnd()); 
  pgonRminBack.push_back(pgonZBack[1]*tan(getAngBot()) + getDrEnd()); 
  pgonRmaxBack.push_back(getRoutDip()); 
  solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace),
				    getNsectortot(), -alpha, dphi, pgonZBack,
				    pgonRminBack, pgonRmaxBack);
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << DDName(name,idNameSpace)
		       << " Polyhedra made of " << getAbsMat() << " with " 
		       << getNsectortot() << " sectors from " 
		       << -alpha/CLHEP::deg << " to " 
		       << (-alpha+dphi)/CLHEP::deg << " and with " 
		       << pgonZBack.size()	<< " sections";
  for (i = 0; i < pgonZBack.size(); i++) 
    edm::LogInfo("HCalGeom") << "\t\tZ = " << pgonZBack[i] << "\tRmin = " 
			 << pgonRminBack[i] << "\tRmax = " << pgonRmaxBack[i];
  DDName absMatname(DDSplit(getAbsMat()).first, DDSplit(getAbsMat()).second); 
  DDMaterial absMatter(absMatname);
  DDLogicalPart glog(DDName(name, idNameSpace), absMatter, solid);

  cpv.position(glog, genlogic, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << glog.name() 
		       << " number 1 positioned in "  << genlogic.name() 
		       << " at (0,0,0) with no rotation";
}


void DDHCalEndcapAlgo::constructInsideSector(DDLogicalPart sector, DDCompactView& cpv) {
  
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: Modules (" << getModules()
		       << ") ...";
  double alpha = CLHEP::pi/getNsectors();

  for (int i = 0; i < getModules(); i++) {
    string  name   = idName + getModName(i);
    DDName matname(DDSplit(getModMat(i)).first, DDSplit(getModMat(i)).second); 
    DDMaterial matter(matname);
    
    if (equipModule(i)>0) {
      int nsec = getSectionModule(i);

      //!!!!!!!!!!!!!!!!!Should be zero. And removed as soon as
      //vertical walls are allowed in SolidPolyhedra
      double deltaz = 0;
    
      vector<double> pgonZ, pgonRmin, pgonRmax;
      if (nsec == 3) {
	double zf = getZminBlock(i) + getZShiftHac2();
	pgonZ.push_back(zf);
	pgonRmin.push_back(zf*tan(getAngBot())); 
	pgonRmax.push_back((zf-getZ1Beam())*getSlope());
	pgonZ.push_back(getZiKink());  
	pgonRmin.push_back(getRinKink()); 
	pgonRmax.push_back(getRout());
      } else {
	pgonZ.push_back(getZminBlock(i));
	pgonRmin.push_back(getRinBlock1(i)); 
	pgonRmax.push_back(getRoutBlock1(i));
      }
      if (nsec == 4) {
	pgonZ.push_back(getZiDip());
	pgonRmin.push_back(getRinDip());
	pgonRmax.push_back(getRout());
	pgonZ.push_back(pgonZ[1] + deltaz);
	pgonRmin.push_back(pgonRmin[1]); 
	pgonRmax.push_back(getRoutDip());
      }
      pgonZ.push_back(getZmaxBlock(i));
      pgonRmin.push_back(getRinBlock2(i)); 
      pgonRmax.push_back(getRoutBlock2(i));

      //Solid & volume
      DDSolid solid;
      solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace), 
					1, -alpha, 2*alpha,
					pgonZ, pgonRmin, pgonRmax);
      edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " 
			   << DDName(name,idNameSpace) << " Polyhedra made of "
			   << getModMat(i) << " with 1 sector from "
			   << -alpha/CLHEP::deg << " to " << alpha/CLHEP::deg 
			   << " and with " << nsec << " sections";
      for (unsigned int k=0; k<pgonZ.size(); k++)
	edm::LogInfo("HCalGeom") << "\t\tZ = " << pgonZ[k] << "\tRmin = "
			     << pgonRmin[k] << "\tRmax = " << pgonRmax[k];
    
      DDLogicalPart glog(DDName(name, idNameSpace), matter, solid);

     cpv.position(glog, sector, i+1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
      edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << glog.name() 
			   << " number " << i+1 << " positioned in " 
			   << sector.name() << " at (0,0,0) with no rotation";
      
      if (getModType(i) == 0) 
	constructInsideModule0 (glog, i, cpv);
      else
	constructInsideModule  (glog, i, cpv);
    }
  }
  
}

void DDHCalEndcapAlgo::parameterLayer0(int mod, int layer, int iphi, 
				       double& yh, double& bl, double& tl, 
				       double& alp, double& xpos, double& ypos,
                                       double& zpos) {

  //Given module and layer number compute parameters of trapezoid
  //and positioning parameters
  double alpha = CLHEP::pi/getNsectors();
  edm::LogInfo("HCalGeom") << "Input " << iphi << " " << layer << " " << iphi
		       << " Alpha " << alpha/CLHEP::deg;

  double zi, zo;
  if (iphi == 0) {
    zi = getZminBlock(mod);
    zo = zi + getLayerT(layer);
  } else {
    zo = getZmaxBlock(mod);
    zi = zo - getLayerT(layer);
  }
  double rin, rout;
  if (mod == 0) {
    rin  = zo * tan(getAngTop());
    rout = (zi - getZ1Beam()) * getSlope();
  } else {
    rin  = zo * tan(getAngBot());
    rout = zi * tan(getAngTop());
  }
  edm::LogInfo("HCalGeom") << "ModNumber " << mod << " " << zi << " " << zo << " " << tan(getAngTop()) << " " << getSlope() << " " << tan(getAngBot()) << " " << rin << " " << rout << " " << getTrim(mod,iphi);
  yh   = 0.5 * (rout - rin);
  bl   = 0.5 * rin * tan (alpha);
  tl   = 0.5 * rout * tan(alpha);
  xpos = 0.5 * (rin + rout);
  ypos = 0.5 * (bl + tl);
  zpos = 0.5 * (zi + zo);
  yh  -= getTrim(mod,iphi);
  bl  -= getTrim(mod,iphi);
  tl  -= getTrim(mod,iphi);
  alp  = atan(0.5 * tan(alpha));
  if (iphi == 0) {
    ypos  = -ypos;
  } else {
    alp  = -alp;
  }
  edm::LogInfo("HCalGeom") << "Output Dimensions " << yh << " " << bl << " "
		       << tl << " " << alp/CLHEP::deg << " Position " << xpos 
		       << " " << ypos << " " << zpos;
}


void DDHCalEndcapAlgo::parameterLayer(int iphi, double rinF, double routF,
				      double rinB, double routB, double zi,
				      double zo, double& yh1, double& bl1,
				      double& tl1, double& yh2, double& bl2,
				      double& tl2, double& alp, double& theta,
				      double& phi, double& xpos, double& ypos,
				      double& zpos) {

  //Given rin, rout compute parameters of the trapezoid and 
  //position of the trapezoid for a standrd layer
  double alpha = CLHEP::pi/getNsectors();
  edm::LogInfo("HCalGeom") << "Input " << iphi << " Front " << rinF << " " << routF
		       << " " << zi << " Back " << rinB << " " << routB << " "
		       << zo << " Alpha " << alpha/CLHEP::deg;

  yh1 = 0.5 * (routF - rinB);
  bl1 = 0.5 * rinB  * tan(alpha);
  tl1 = 0.5 * routF * tan(alpha);
  yh2 = 0.5 * (routF - rinB);
  bl2 = 0.5 * rinB  * tan(alpha);
  tl2 = 0.5 * routF * tan(alpha);
  double dx  = 0.25* (bl2+tl2-bl1-tl1);
  double dy  = 0.5 * (rinB+routF-rinB-routF);
  xpos = 0.25*(rinB+routF+rinB+routF);
  ypos = 0.25*(bl2+tl2+bl1+tl1);
  zpos = 0.5*(zi+zo);
  alp  = atan(0.5 * tan(alpha));
  //  ypos-= getTolPos();
  if (iphi == 0) {
    ypos  = -ypos;
  } else {
    alp  = -alp;
    dx   = -dx;
  }
  double r   = sqrt (dx*dx + dy*dy);
  theta= atan (r/(zo-zi));
  phi  = atan2 (dy, dx);
  edm::LogInfo("HCalGeom") << "Output Dimensions " << yh1 << " " << bl1 << " "
		       << tl1 << " " << yh2 << " " << bl2 << " " << tl2
		       << " " << alp/CLHEP::deg << " " << theta/CLHEP::deg
		       << " " << phi/CLHEP::deg << " Position " << xpos << " "
		       << ypos << " " << zpos;
}


void DDHCalEndcapAlgo::constructInsideModule0(DDLogicalPart module, int mod, DDCompactView& cpv) {
  
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: \t\tInside module0 ..."<<mod;

  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  string rotstr = getRotMat();
  DDRotation rot(DDName(rotstr, rotns));
  DDName matName(DDSplit(getAbsMat()).first, DDSplit(getAbsMat()).second);
  DDMaterial matabsorbr(matName);
  DDName plasName(DDSplit(getPlastMat()).first, DDSplit(getPlastMat()).second);
  DDMaterial matplastic(plasName);

  int     layer  = getLayer(mod,0);
  int     layer0 = getLayer(mod,1);
  string  name;
  double  xpos, ypos, zpos;
  DDSolid solid;
  DDLogicalPart glog, plog;
  for (int iphi = 0; iphi < getPhi(); iphi++) {
    double yh, bl, tl, alp;
    parameterLayer0(mod, layer, iphi, yh, bl, tl, alp, xpos, ypos, zpos);
    name = module.name().name()+getLayerName(layer)+getPhiName(iphi);
    solid = DDSolidFactory::trap(DDName(name, idNameSpace), 
				 0.5*getLayerT(layer), 0, 0, yh,
				 bl, tl, alp, yh, bl, tl, alp);
    edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << solid.name() 
			 << " Trap made of " << getPlastMat() 
			 << " of dimensions " << 0.5*getLayerT(layer) 
			 << ", 0, 0, " << yh << ", " << bl << ", " << tl 
			 << ", " << alp/CLHEP::deg << ", " << yh << ", " << bl 
			 << ", " << tl << ", " << alp/CLHEP::deg;
    glog = DDLogicalPart(solid.ddname(), matplastic, solid);

    DDTranslation r1(xpos, ypos, zpos);
    cpv.position(glog, module, idOffset+layer+1, r1, rot);
    edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << glog.name() 
			 << " number " << idOffset+layer+1 << " positioned in "
			 << module.name() << " at " << r1 << " with " << rot;
    //Now construct the layer of scintillator inside this
    int copyNo = layer0*10 + getLayerType(layer);
    name = getModName(mod)+getLayerName(layer)+getPhiName(iphi);
    constructScintLayer (glog, getScintT(layer), yh, bl, tl, alp, name, copyNo, cpv);
  }

  //Now the absorber layer
  double zi = getZminBlock(mod) + getLayerT(layer);
  double zo = zi + 0.5*getDzStep();
  double rinF, routF, rinB, routB;
  if (mod == 0) {
    rinF  = zi * tan(getAngTop());
    routF =(zi - getZ1Beam()) * getSlope();
    rinB  = zo * tan(getAngTop());
    routB =(zo - getZ1Beam()) * getSlope();
  } else {
    rinF  = zi * tan(getAngBot());
    routF = zi * tan(getAngTop());
    rinB  = zo * tan(getAngBot());
    routB = zo * tan(getAngTop());
  }
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: Module " << mod << " Front "
		       << zi << ", " << rinF << ", " << routF << " Back "
		       << zo << ", " << rinB << ", " << routB;
  double yh1, bl1, tl1, yh2, bl2, tl2, theta, phi, alp;
  parameterLayer(0, rinF, routF, rinB, routB, zi, zo, yh1, bl1, tl1, yh2, bl2, 
                 tl2, alp, theta, phi, xpos, ypos, zpos);
  double fact = getTolAbs();
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: Trim " << fact << " Param "
		       << yh1 << ", " << bl1 << ", " << tl1 << ", " << yh2
		       << ", " << bl2 << ", " << tl2;
  bl1 -= fact;
  tl1 -= fact;
  bl2 -= fact;
  tl2 -= fact;

  name = module.name().name()+"Absorber";
  solid = DDSolidFactory::trap(DDName(name, idNameSpace), 
			       0.5*getThick(mod), theta, phi, yh1,
			       bl1, tl1, alp, yh2, bl2, tl2, alp);
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << solid.name() 
		       << " Trap made of " << getAbsMat() << " of dimensions " 
		       << 0.5*getThick(mod) << ", " << theta/CLHEP::deg << ", "
		       << phi/CLHEP::deg << ", " << yh1 << ", " << bl1 << ", " 
		       << tl1 << ", " << alp/CLHEP::deg << ", " << yh2 << ", "
		       << bl2 << ", " << tl2 << ", " << alp/CLHEP::deg;
  glog = DDLogicalPart(solid.ddname(), matabsorbr, solid);

  DDTranslation r2(xpos, ypos, zpos);
  cpv.position(glog, module, 1, r2, rot);
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << glog.name() 
		       << " number 1 positioned in " << module.name() << " at "
		       << r2 << " with " << rot;
}


void DDHCalEndcapAlgo::constructInsideModule(DDLogicalPart module, int mod, DDCompactView& cpv) {
  
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: \t\tInside module ..." <<mod;

  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  string rotstr = getRotMat();
  DDRotation rot(DDName(rotstr, rotns));
  DDName matName(DDSplit(getGenMat()).first, DDSplit(getGenMat()).second);
  DDMaterial matter(matName);
  DDName plasName(DDSplit(getPlastMat()).first, DDSplit(getPlastMat()).second);
  DDMaterial matplastic(plasName);

  double  alpha = CLHEP::pi/getNsectors();
  double  zi    = getZminBlock(mod);

  for (int i = 0; i < getLayerN(mod); i++) {
    string name;
    DDSolid solid;
    DDLogicalPart glog, plog;
    int     layer  = getLayer(mod,i);
    double  zo     = zi + 0.5*getDzStep();

    for (int iphi = 0; iphi < getPhi(); iphi++) {
      double  ziAir = zo - getThick(mod);
      double  rinF, rinB;
      if (layer == 1) {
        rinF  = ziAir * tan(getAngTop());
        rinB  = zo    * tan(getAngTop());
      } else {
        rinF  = ziAir * tan(getAngBot());
        rinB  = zo    * tan(getAngBot());
      }
      double routF = (ziAir - getZ1Beam()) * getSlope();
      double routB = (zo    - getZ1Beam()) * getSlope();
      if (routF > getRoutBlock2(mod)) routF =  getRoutBlock2(mod);
      if (routB > getRoutBlock2(mod)) routB =  getRoutBlock2(mod);
      edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: Layer " << i << " Phi "
			   << iphi << " Front " << ziAir << ", " << rinF
			   << ", " << routF << " Back " << zo << ", " << rinB
			   << ", " << routB;
      double yh1, bl1, tl1, yh2, bl2, tl2, theta, phi, alp;
      double xpos, ypos, zpos;
      parameterLayer(iphi, rinF, routF, rinB, routB, ziAir, zo, yh1, bl1, tl1, 
                     yh2, bl2, tl2, alp, theta, phi, xpos, ypos, zpos);
      
      name = module.name().name()+getLayerName(layer)+getPhiName(iphi)+"Air";
      solid = DDSolidFactory::trap(DDName(name, idNameSpace), 
				   0.5*getThick(mod), theta, phi, yh1,
				   bl1, tl1, alp, yh2, bl2, tl2, alp);
      edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << solid.name() 
			   << " Trap made of " << getGenMat() 
			   << " of dimensions " << 0.5*getThick(mod) << ", " 
			   << theta/CLHEP::deg << ", " << phi/CLHEP::deg
			   << ", " << yh1 << ", " << bl1 << ", " << tl1 << ", "
			   << alp/CLHEP::deg << ", " << yh2 << ", " << bl2 
			   << ", " << tl2 << ", " << alp/CLHEP::deg;
      glog = DDLogicalPart(solid.ddname(), matter, solid);

      DDTranslation r1(xpos, ypos, zpos);
      cpv.position(glog, module, layer+1, r1, rot);
      edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << glog.name() 
			   << " number " << layer+1 << " positioned in " 
			   << module.name() << " at " << r1 << " with " << rot;

      //Now the plastic with scintillators
      double yh = 0.5 * (routF - rinB) - getTrim(mod,iphi);
      double bl = 0.5 * rinB  * tan(alpha) - getTrim(mod,iphi);
      double tl = 0.5 * routF * tan(alpha) - getTrim(mod,iphi);
      name = module.name().name()+getLayerName(layer)+getPhiName(iphi);
      solid = DDSolidFactory::trap(DDName(name, idNameSpace), 
				   0.5*getLayerT(layer), 0, 0, yh,
				   bl, tl, alp, yh, bl, tl, alp);
      edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << solid.name() 
			   << " Trap made of " << getPlastMat() 
			   << " of dimensions " << 0.5*getLayerT(layer) 
			   << ", 0, 0, " << yh << ", " << bl << ", " << tl
			   << ", " << alp/CLHEP::deg << ", " << yh << ", " 
			   << bl << ", " << tl << ", " << alp/CLHEP::deg;
      plog = DDLogicalPart(solid.ddname(), matplastic, solid);

      ypos = 0.5*(routF+rinB) - xpos;
      DDTranslation r2(0., ypos, 0.);
      cpv.position(plog, glog, idOffset+layer+1, r2, DDRotation());
      edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << plog.name() 
			   << " number " << idOffset+layer+1 
			   << " positioned in " << glog.name() << " at " << r2
			   << " with no rotation";

      //Constructin the scintillators inside
      int copyNo = layer*10 + getLayerType(layer);
      name = getModName(mod)+getLayerName(layer)+getPhiName(iphi);
      constructScintLayer (plog, getScintT(layer), yh,bl,tl, alp,name,copyNo, cpv);
      zo += 0.5*getDzStep();
    } // End of loop over phi indices
    zi = zo - 0.5*getDzStep();
  }   // End of loop on layers
}

 
void DDHCalEndcapAlgo::constructScintLayer(DDLogicalPart detector, double dz,
                                           double yh, double bl, double tl, 
					   double alp, string nm, int id, DDCompactView& cpv) {

  DDName matname(DDSplit(getScintMat()).first, DDSplit(getScintMat()).second);
  DDMaterial matter(matname);
  string name = idName+"Scintillator"+nm;

  DDSolid solid = DDSolidFactory::trap(DDName(name, idNameSpace), 0.5*dz, 0, 0,
				       yh, bl, tl, alp, yh, bl, tl, alp);
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << DDName(name,idNameSpace)
		       << " Trap made of " << getScintMat() <<" of dimensions "
		       << 0.5*dz << ", 0, 0, " << yh << ", "  << bl << ", " 
		       << tl << ", " << alp/CLHEP::deg << ", " << yh << ", " 
		       << bl << ", " << tl << ", " << alp/CLHEP::deg;

  DDLogicalPart glog(solid.ddname(), matter, solid); 

  cpv.position(glog, detector, id, DDTranslation(0,0,0), DDRotation());
  edm::LogInfo("HCalGeom") << "DDHCalEndcapAlgo test: " << glog.name() 
		       << " number " << id << " positioned in " 
		       << detector.name() << " at (0,0,0) with no rotation";

}
