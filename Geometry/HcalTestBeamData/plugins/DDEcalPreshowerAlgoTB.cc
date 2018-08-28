#include "Geometry/HcalTestBeamData/plugins/DDEcalPreshowerAlgoTB.h"

#include <cmath>
#include <algorithm>
#include <ostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

using namespace CLHEP;

DDEcalPreshowerAlgoTB::DDEcalPreshowerAlgoTB() : DDAlgorithm() {
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB info: Creating an instance" ;

}

void DDEcalPreshowerAlgoTB::initialize(const DDNumericArguments & nArgs,
				       const DDVectorArguments & vArgs,
				       const DDMapArguments & mArgs,
				       const DDStringArguments & sArgs,
				       const DDStringVectorArguments & vsArgs){

  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB info: Initialize" ;
  quadMin_ = vArgs["IQUAD_MIN"];
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB IQUAD_MIN";
  quadMax_ = vArgs["IQUAD_MAX"];  
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB IQUAD_MAX";
  thickLayers_ = vArgs["Layers"];
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB Layers";
  thickness_ = double(nArgs["PRESH_Z_TOTAL"]);
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB PRESH_Z_TOTAL";
  materials_ = vsArgs["LayMat"];
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB material";
  rmaxVec = vArgs["R_MAX"]; // inner radii
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB R_MAX";
  rminVec = vArgs["R_MIN"]; // outer radii
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB R_MIN";
  waf_intra_col_sep = double(nArgs["waf_intra_col_sep"]);
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB waf_intra_col_sep";
  waf_inter_col_sep = double(nArgs["waf_inter_col_sep"]);
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB waf_intra_col_sep = "<<waf_inter_col_sep;
  waf_active = double(nArgs["waf_active"]);
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB waf_active = " << waf_active;
  wedge_length = double(nArgs["wedge_length"]);
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB wedge_length = " 
		       << wedge_length;
  wedge_offset = double(nArgs["wedge_offset"]);
  zwedge_ceramic_diff = double(nArgs["zwedge_ceramic_diff"]);
  ywedge_ceramic_diff = double(nArgs["ywedge_ceramic_diff"]);
  micromodulesx = vArgs["MicromodulesX"]; // micromodules in X plane
  micromodulesy = vArgs["MicromodulesY"]; // micromodules in Y plane
  absorbx = double(nArgs["absorbx"]);
  absorby = double(nArgs["absorby"]);
  trabsorbx = double(nArgs["trabsorbx"]);
  trabsorby = double(nArgs["trabsorby"]);
  ScndplaneXshift = double(nArgs["2ndPlaneXshift"]);
  ScndplaneYshift = double(nArgs["2ndPlaneYshift"]);
  TotSFXshift = double(nArgs["SF07vsSF_Xshift"]);
  TotSFYshift = double(nArgs["SF07vsSF_Yshift"]);
  dummyMaterial = sArgs["DummyMaterial"];
  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB Dummy Material = "
		       << dummyMaterial;

  DDCurrentNamespace ns;
  idNameSpace = *ns;

  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB info: end initialize" ;
}

void DDEcalPreshowerAlgoTB::execute(DDCompactView& cpv) {

  LogDebug("HCalGeom") << "******** DDEcalPreshowerAlgoTB execute!";

  // creates all the tube-like layers of the preshower
  doLayers(cpv);
  // places the wedges and silicon strip detectors in their x and y layer
  doWedges(cpv);
  // places the slicon strips in active silicon wafers
  doSens(cpv);
  
}

void DDEcalPreshowerAlgoTB::doLayers(DDCompactView& cpv) {
  //double sum_z=0;
  double zpos = -thickness_/2.;
  for(size_t i = 0; i<rminVec.size(); ++i) {
    int I = int(i)+1; // FOTRAN I (offset +1)
    
    double zHalf(0);
      
    // create the name
    std::ostringstream name;
    name << "SF" << int((50+I)/10) << I-int(I/10)*10;
    DDName ddname(name.str(), idNameSpace); // namespace:name
      
    // tube dimensions
    //     rIn = rmaxVec[i];
    //     rOut = rminVec[i];
    zHalf = thickLayers_[i]/2.;

    // position the logical part w.r.t. the parent volume
    zpos += zHalf;
    //sum_z += thickLayers_[i];
    if (I==7 || I==15) { // skip layers with detectors
      zpos += zHalf;
      continue;
    }
    if ( I==2 ) {
      zfoam1_ = zpos;
    }
    if ( I==6 ) {
      zlead1_ = zpos + zHalf;
    }
    if ( I==14 ) {
      zlead2_ = zpos + zHalf;
    }
      
    if (getMaterial(i) != dummyMaterial ) {
      // create a logical part representing a single layer in the preshower
      //DDSolid solid = DDSolidFactory::tubs(ddname,zHalf,rIn,rOut,0.,360.*deg);

      DDSolid solid = DDSolidFactory::box(ddname,absorbx,absorby,zHalf);
    
      DDName        matname(getMaterial(i),"materials"); 
      DDMaterial    material(matname);
      DDLogicalPart layer = DDLogicalPart(ddname,material,solid);

      DDTranslation tran=DDTranslation(trabsorbx+TotSFXshift,trabsorby+TotSFYshift, zpos);
      cpv.position(layer, parent(), 1, tran, DDRotation());
      LogDebug("HCalGeom") <<"DDEcalPreshowerAlgoTB debug : Child "
			   << layer << " Copy 1 in " << parent().name()
			   << " with translation " << tran
			   << " and rotation " << DDRotation();
    }
    zpos += zHalf;
  }
  
}

void DDEcalPreshowerAlgoTB::doWedges(DDCompactView& cpv) {

  LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : doWedges()";
  int nx(0), ny(0), icopy(0), icopyt(0);
  double xpos(0), ypos(0), zpos(0);// zposY(0);
  int sz = int(quadMax_.size());

  DDTranslation tran;
  DDName name1("SWED", idNameSpace);
  DDName name2("SFBX", idNameSpace);
  DDRotation rot1("rotations:RM1299");
  DDRotation rot2("rotations:RM1298");
  // Do Plane X
  for(int I=1; I<=sz;++I) {
    for(int J=int(quadMax_[I-1]); J>=int(quadMin_[I-1]); --J) {
      //LogDebug("HCalGeom") <<"DDEcalPreshowerAlgoTB::I=" << I << " J="  << J;
      nx += 1;
      icopy += 1;
      go=0; 
      for (double m : micromodulesx) 
	if (m==icopy) {go=1; icopyt +=1; }
      xpos = -1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep 
		  - waf_intra_col_sep/2.);
      ypos = (sz-int(I))*waf_active + wedge_length/2. + 0.05*cm;
      xpos = xpos + TotSFXshift; ypos = ypos + TotSFYshift;
      zpos = zlead1_ + wedge_offset;
      // place the wedge
      if (go==1) {
	tran = DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name1), parent(), icopyt, tran, rot1);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name1 << " copy = " << icopy << " (" 
			     << icopyt << ") in Mother " << parent().name()
			     << " translation " <<tran  << " rotation " <<rot1;
      }
      zpos = zlead1_ + zwedge_ceramic_diff;
      ypos = ypos + ywedge_ceramic_diff;
      if (go==1) {
	tran = DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name2), parent(), icopyt, tran, rot2);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name2 << " copy = " << icopy << " (" 
			     << icopyt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot2;
      }
    }
    
    for(int J=int(quadMin_[I-1]); J<=int(quadMax_[I-1]); ++J) {
      nx += 1;
      icopy += 1;
      go=0; 
      for (double m : micromodulesx) 
	if (m==icopy) {go=1; icopyt +=1;}
      xpos = 1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep 
		 - waf_intra_col_sep/2.);
      ypos = (sz-int(I))*waf_active + wedge_length/2. + 0.05*cm;
      xpos = xpos + TotSFXshift; ypos = ypos + TotSFYshift;
      zpos = zlead1_ + wedge_offset;      
      if(go==1) {
	tran = DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name1), parent(), icopyt, tran, rot1);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name1 << " copy = " << icopy << " (" 
			     << icopyt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot1;
      }
      zpos = zlead1_ + zwedge_ceramic_diff;
      ypos = ypos + ywedge_ceramic_diff;
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name2), parent(), icopyt, tran, rot2);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name2 << " copy = " << icopy << " (" 
			     << icopyt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot2;
      }
    }
    
  }
  
  // mirror image system
  rot1 = DDRotation("rotations:RM1303");
  rot2 = DDRotation("rotations:RM1302");
  for(int I=sz; I>=1;--I) {
    for(int J=int(quadMax_[I-1]); J>=int(quadMin_[I-1]); --J) {
      nx += 1;
      icopy += 1;
      go=0; 
      for (double m : micromodulesx) 
	if (m==icopy) {go=1; icopyt +=1;}
      //LogDebug("HCalGeom") <<"DDEcalPreshowerAlgoTB::I=" << I << " J="  << J;
      xpos = -1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep 
		  - waf_intra_col_sep/2.);
      ypos = -1.*(sz-int(I))*waf_active - wedge_length/2. - 0.05*cm;
      xpos = xpos + TotSFXshift; ypos = ypos + TotSFYshift;
      zpos = zlead1_ + wedge_offset;
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name1), parent(), icopyt, tran, rot1);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name1 << " copy = " << icopy << " (" 
			     << icopyt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " << rot1;
      }
      zpos = zlead1_ + zwedge_ceramic_diff;
      ypos = ypos - ywedge_ceramic_diff;
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name2), parent(), icopyt, tran, rot2);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name2 << " copy = " << icopy << " (" 
			     << icopyt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation "<<rot2;
      }
    }

    for(int J=int(quadMin_[I-1]); J<=int(quadMax_[I-1]); ++J) {
      nx += 1;
      icopy += 1;
      go=0; 
      for (double m : micromodulesx) 
	if (m==icopy) {go=1; icopyt +=1;}
      xpos = 1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep 
		 - waf_intra_col_sep/2.);
      ypos = -1.*(sz-int(I))*waf_active - wedge_length/2. - 0.05*cm;
      xpos = xpos + TotSFXshift; ypos = ypos + TotSFYshift;
      zpos = zlead1_ + wedge_offset;      
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name1), parent(), icopyt, tran, rot1);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name1 << " copy = " << icopy << " (" 
			     << icopyt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot1;
      }
      zpos = zlead1_ + zwedge_ceramic_diff;
      ypos = ypos - ywedge_ceramic_diff;
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name2), parent(), icopyt, tran, rot2);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name2 << " copy = " << icopy << " (" 
			     << icopyt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot2;
      }
    }
  }

  // Do Plane Y
  icopy =  0; 
  int nxt = micromodulesy.size();
  name1 = DDName("SWED", idNameSpace);
  name2 = DDName("SFBY", idNameSpace);
  rot1 = DDRotation(DDName("RM1301B",idNameSpace));
  rot2 = DDRotation(DDName("RM1300B",idNameSpace));
  for(int I=1; I<=sz;++I) {
    for(int J=int(quadMax_[I-1]); J>=int(quadMin_[I-1]); --J) {
      ny += 1;
      icopy += 1;
      go=0; 
      for (double m : micromodulesy) 
	if (m==icopy) {go=1; icopyt +=1;}
      //LogDebug("HCalGeom") <<"DDEcalPreshowerAlgoTB::I=" << I << " J="  << J;
      ypos = -1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep 
		  - waf_intra_col_sep/2.);
      xpos = (sz-int(I))*waf_active + wedge_length/2. + 0.05*cm + ScndplaneXshift;
      xpos = xpos + TotSFXshift; ypos = ypos + TotSFYshift;
      zpos = zlead2_ + wedge_offset;
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name1), parent(), icopy+nxt, tran, rot1);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name1 << " copy = " << icopy << " (" 
			     << icopyt+nxt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot1;
      }
      zpos = zlead2_ + zwedge_ceramic_diff;
      xpos = xpos - ywedge_ceramic_diff;
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name2), parent(), icopyt, tran, rot2);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name2 << " copy = " << icopy << " (" 
			     << icopyt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot2;
      }
    }

    for(int J=int(quadMin_[I-1]); J<=int(quadMax_[I-1]); ++J) {
      ny += 1;
      icopy += 1;
      go=0; 
      for (double m : micromodulesy) 
	if (m==icopy) {go=1; icopyt +=1;}
      //LogDebug("HCalGeom") <<"DDEcalPreshowerAlgoTB::I=" << I << " J="  << J;
      ypos = 1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep 
		 - waf_intra_col_sep/2.);
      xpos = (sz-int(I))*waf_active + wedge_length/2. + 0.05*cm + ScndplaneXshift;
      xpos = xpos + TotSFXshift; ypos = ypos + TotSFYshift;
      zpos = zlead2_ + wedge_offset;
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name1), parent(), icopyt+nxt, tran, rot1);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name1 << " copy = " << icopy << " (" 
			     << icopyt+nxt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot1;
      }
      zpos = zlead2_ + zwedge_ceramic_diff;
      xpos = xpos - ywedge_ceramic_diff;
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name2), parent(), icopyt, tran, rot2);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name2 << " copy = " << icopy << " (" 
			     << icopyt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot2;
      }
    }
  }

  // mirror image system
  rot1 = DDRotation(DDName("RM1305B",idNameSpace));
  rot2 = DDRotation(DDName("RM1304B",idNameSpace));
  for(int I=sz; I>=1;--I) {
    for(int J=int(quadMax_[I-1]); J>=int(quadMin_[I-1]); --J) {
      ny += 1;
      icopy += 1;
      go=0; 
      for (double m : micromodulesy) 
	if (m==icopy) {go=1; icopyt +=1;}
      //LogDebug("HCalGeom") <<"DDEcalPreshowerAlgoTB::I=" << I << " J="  << J;
      ypos = -1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep 
		  - waf_intra_col_sep/2.);
      xpos = -1.*(sz-int(I))*waf_active - wedge_length/2. - 0.05*cm + ScndplaneXshift;
      xpos = xpos + TotSFXshift; ypos = ypos + TotSFYshift;
      zpos = zlead2_ + wedge_offset;
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name1), parent(), icopyt+nxt, tran, rot1);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name1 << " copy = " << icopy << " (" 
			     << icopyt+nxt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot1;
      }      
      zpos = zlead2_ + zwedge_ceramic_diff;
      xpos = xpos + ywedge_ceramic_diff;
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name2), parent(), icopyt, tran, rot2);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name2 << " copy = " << icopy << " (" 
			     << icopyt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot2;
      }
    }

    for(int J=int(quadMin_[I-1]); J<=int(quadMax_[I-1]); ++J) {
      ny += 1;
      icopy += 1;
      go=0; 
      for (double m : micromodulesy) 
	if (m==icopy) {go=1; icopyt +=1;}
      //LogDebug("HCalGeom") <<"DDEcalPreshowerAlgoTB::I=" << I << " J="  << J;
      ypos = 1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep 
		 - waf_intra_col_sep/2.);
      xpos = -1.*(sz-int(I))*waf_active - wedge_length/2. - 0.05*cm + ScndplaneXshift;
      xpos = xpos + TotSFXshift; ypos = ypos + TotSFYshift;
      zpos = zlead2_ + wedge_offset;
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name1), parent(), icopyt+nxt, tran, rot1);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name1 << " copy = " << icopy << " (" 
			     << icopyt+nxt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot1;
      }
      zpos = zlead2_ + zwedge_ceramic_diff;
      xpos = xpos + ywedge_ceramic_diff;
      if (go==1) {
	tran =  DDTranslation(xpos,ypos,zpos);
	cpv.position(DDLogicalPart(name2), parent(), icopyt, tran, rot2);
	LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " 
			     << name2 << " copy = " << icopy << " (" 
			     << icopyt << ") in Mother " << parent().name()
			     << " translation " <<tran << " rotation " <<rot2;
      }
    }
  }

}

void DDEcalPreshowerAlgoTB::doSens(DDCompactView& cpv) {

  double xpos(0), ypos(0);
  DDTranslation tran;
  DDName        child1  = DDName("SFSX", idNameSpace);
  DDName        child2  = DDName("SFSY", idNameSpace);
  DDName        mother1 = DDName("SFAX", idNameSpace);
  DDName        mother2 = DDName("SFAY", idNameSpace);
  DDRotation    rot;
  for(size_t i = 0; i<32; ++i) {
    xpos = -waf_active/2. + i*waf_active/32. + waf_active/64.;
    tran = DDTranslation(xpos,0., 0.);
    cpv.position(DDLogicalPart(child1), mother1, i+1, tran, rot);
    LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " << child1
			 << "\ncopy number " << i+1 << " in " << mother1
			 << " translation "<< tran << " rotation " << rot;
      
    ypos = -waf_active/2. + i*waf_active/32. + waf_active/64.;
    tran = DDTranslation(0.,ypos, 0.);
    cpv.position(DDLogicalPart(child2), mother2, i+1, tran, rot);
    LogDebug("HCalGeom") << "DDEcalPreshowerAlgoTB::debug : Child " << child2
			 << "\ncopy number " << i+1 << " in " << mother2
			 << " translation "<< tran << " rotation " << rot;
    
  }
}
