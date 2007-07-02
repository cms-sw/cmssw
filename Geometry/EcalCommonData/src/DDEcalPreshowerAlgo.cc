#include "Geometry/EcalCommonData/interface/DDEcalPreshowerAlgo.h"


#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


//#include "DDD/DDCore/interface/DDD.h"
//#include <iostream>
//#include <sstream>
//using namespace DD;
//using namespace std;

DDEcalPreshowerAlgo::DDEcalPreshowerAlgo() : DDAlgorithm() {
   edm::LogInfo("EcalGeom") << "DDEcalPreshowerAlgo info: Creating an instance" ;
}

void DDEcalPreshowerAlgo::initialize(const DDNumericArguments & nArgs,
                           const DDVectorArguments & vArgs,
                           const DDMapArguments & mArgs,
			   const DDStringArguments & sArgs,
			   const DDStringVectorArguments & vsArgs)
{

  edm::LogInfo("EcalGeom") << "DDEcalPreshowerAlgo info: Initialize" ;

  quadMin_ = vArgs["IQUAD_MIN"];
  LogDebug("SFGeom")<< "DDEcalPreshowerAlgo IQUAD_MIN";
  quadMax_ = vArgs["IQUAD_MAX"];  
  LogDebug("SFGeom")<< "DDEcalPreshowerAlgo IQUAD_MAX";
  thickLayers_ = vArgs["Layers"];
  LogDebug("SFGeom")<< "DDEcalPreshowerAlgo Layers";
  thickness_ = double(nArgs["PRESH_Z_TOTAL"]);
  LogDebug("SFGeom")<< "DDEcalPreshowerAlgo PRESH_Z_TOTAL";
  materials_ = vsArgs["LayMat"];
  LogDebug("SFGeom")<< "DDEcalPreshowerAlgo material";
   rmaxVec = vArgs["R_MAX"]; // inner radii
  LogDebug("SFGeom")<< "DDEcalPreshowerAlgo R_MAX";
  rminVec = vArgs["R_MIN"]; // outer radii
  LogDebug("SFGeom")<< "DDEcalPreshowerAlgo R_MIN";
   waf_intra_col_sep = double(nArgs["waf_intra_col_sep"]);
  LogDebug("SFGeom")<< "DDEcalPreshowerAlgo waf_intra_col_sep";
   waf_inter_col_sep = double(nArgs["waf_inter_col_sep"]);
  LogDebug("SFGeom")<< "DDEcalPreshowerAlgo waf_intra_col_sep = "<<waf_inter_col_sep<<endl;
   waf_active = double(nArgs["waf_active"]);
  LogDebug("SFGeom")<< "DDEcalPreshowerAlgo waf_active = "<<waf_active<<endl;
   wedge_length = double(nArgs["wedge_length"]);
  LogDebug("SFGeom")<< "DDEcalPreshowerAlgo wedge_length = "<<wedge_length<<endl;
   wedge_offset = double(nArgs["wedge_offset"]);
   zwedge_ceramic_diff = double(nArgs["zwedge_ceramic_diff"]);
   ywedge_ceramic_diff = double(nArgs["ywedge_ceramic_diff"]);

   edm::LogInfo("EcalGeom") << "DDEcalPreshowerAlgo info: end initialize" ;
}

void DDEcalPreshowerAlgo::execute()
{
   edm::LogInfo("EcalGeom") << "******** DDEcalPreshowerAlgo execute!";

  // creates all the tube-like layers of the preshower
   doLayers();
  // places the wedges and silicon strip detectors in their x and y layer
     doWedges();
  // places the slicon strips in active silicon wafers
     doSens();

}

void DDEcalPreshowerAlgo::doLayers()
{
  //double sum_z=0;
  double zpos = -thickness_/2.;
  for(size_t i = 0; i<rminVec.size(); ++i) 
    {
    int I = int(i)+1; // FOTRAN I (offset +1)
    
    double rIn(0),rOut(0),zHalf(0);
    
    // create the name
    ostringstream name;
    name << "esalgo:SF";
    name << int((50+I)/10) << I-int(I/10)*10;
    DDName ddname(name.str()); // namespace:name
    
    // tube dimensions
    rIn = rmaxVec[i];
    rOut = rminVec[i];
    zHalf = thickLayers_[i]/2.;

    // create a logical part representing a single layer in the preshower
    DDSolid solid = DDSolidFactory::tubs(ddname,zHalf,rIn,rOut,0.,360.*deg);
    DDName matname(getMaterial(i),"materials"); 
    DDMaterial material(matname);
    DDLogicalPart layer = DDLogicalPart(ddname,material,solid);

    // position the logical part w.r.t. the parent volume
    zpos += zHalf;
    //sum_z += thickLayers_[i];
    if (I==10 || I==20) { // skip layers with detectors
      zpos += zHalf;
      continue;
    }
    if ( I==2 ) {
      zfoam1_ = zpos;
    }
    if ( I==9 ) {
      zlead1_ = zpos + zHalf;
    }
    if ( I==19 ) {
      zlead2_ = zpos + zHalf;
    }
    if ( I==23 ) {
      zfoam2_ = zpos;
    }

    DDpos(layer, 
          parent(),
          1,
          DDTranslation(0.,0., zpos),
          DDRotation());

    LogDebug("SFGeom")<<" debug : tubs, Logical part: "<<DDLogicalPart(ddname,material,solid)<<endl<<" translation "<<DDTranslation(0.,0.,zpos)<<" rotation "<<DDRotation()<< endl;
    zpos += zHalf; 
  }

}
  void DDEcalPreshowerAlgo::doWedges()
{
  LogDebug("SFGeom")<< "debug : doWedges()" << endl;
  int nx(0), ny(0), icopy(0);
  double xpos(0), ypos(0), zpos(0);// zposY(0);
 int sz = int(quadMax_.size());
  for(int I=1; I<=sz;++I) {
    for(int J=int(quadMax_[I-1]); J>=int(quadMin_[I-1]); --J) {
      LogDebug("SFGeom")<< "I=" << I << " J="  << J << endl;
      nx += 1;
      icopy += 1;
      xpos = -1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      ypos = (sz-int(I))*waf_active + wedge_length/2. + 0.05*cm;
      zpos = zlead1_ + wedge_offset;
      // place the wedge
      DDpos(DDLogicalPart("esalgo:SWED"), parent(), icopy, DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1299"));
      
    LogDebug("SFGeom")<<" debug : SWED, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SWED")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1299")<< endl;

      zpos = zlead1_ + zwedge_ceramic_diff;
      ypos = ypos + ywedge_ceramic_diff;
      DDpos(DDLogicalPart("esalgo:SFBX"), parent(), icopy, DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1298"));


    LogDebug("SFGeom")<<" debug : SFBX, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SFBX")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1298")<< endl;

    }
    for(int J=int(quadMin_[I-1]); J<=int(quadMax_[I-1]); ++J) {
      nx += 1;
      icopy += 1;
      xpos = 1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      ypos = (sz-int(I))*waf_active + wedge_length/2. + 0.05*cm;
      zpos = zlead1_ + wedge_offset;      
      DDpos(DDLogicalPart("esalgo:SWED"), parent(), icopy, DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1299"));

    LogDebug("SFGeom")<<" debug : SWED, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SWED")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1299")<< endl;

      zpos = zlead1_ + zwedge_ceramic_diff;
      ypos = ypos + ywedge_ceramic_diff;
      DDpos(DDLogicalPart("esalgo:SFBX"), parent(),icopy,DDTranslation(xpos,ypos,zpos),
           DDRotation("rotations:RM1298"));

    LogDebug("SFGeom")<<" debug : SFBX, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SFBX")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1298")<< endl;
    }

  }

  // mirror image system
  for(int I=sz; I>=1;--I) {
    for(int J=int(quadMax_[I-1]); J>=int(quadMin_[I-1]); --J) {
      nx += 1;
      icopy += 1;
      LogDebug("SFGeom")<< "I=" << I << " J="  << J << endl;
      xpos = -1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      ypos = -1.*(sz-int(I))*waf_active - wedge_length/2. - 0.05*cm;
      zpos = zlead1_ + wedge_offset;
      // place the wedge
      DDpos(DDLogicalPart("esalgo:SWED"), 
            parent(),
            icopy,
            DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1303"));

    LogDebug("SFGeom")<<" debug : SWED, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SWED")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1303")<< endl;

      zpos = zlead1_ + zwedge_ceramic_diff;
      ypos = ypos - ywedge_ceramic_diff;
      DDpos(DDLogicalPart("esalgo:SFBX"), 
            parent(),
            icopy,
            DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1302"));

    LogDebug("SFGeom")<<" debug : SFBX, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SFBX")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1302")<< endl;

    }
    for(int J=int(quadMin_[I-1]); J<=int(quadMax_[I-1]); ++J) {
      nx += 1;
      icopy += 1;
      xpos = 1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      ypos = -1.*(sz-int(I))*waf_active - wedge_length/2. - 0.05*cm;
      zpos = zlead1_ + wedge_offset;      
      DDpos(DDLogicalPart("esalgo:SWED"), 
            parent(),
            icopy,
            DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1303"));

    LogDebug("SFGeom")<<" debug : SWED, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SWED")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1303")<< endl;

      zpos = zlead1_ + zwedge_ceramic_diff;
      ypos = ypos - ywedge_ceramic_diff;
      DDpos(DDLogicalPart("esalgo:SFBX"), 
            parent(),
            icopy,
            DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1302"));
    LogDebug("SFGeom")<<" debug : SFBX, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SFBX")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1302")<< endl;
    }
  }

  // Second Plane (Y Plane)
  // Top half first
  icopy =  0;
  for(int I=1; I<=sz;++I) {
    for(int J=int(quadMax_[I-1]); J>=int(quadMin_[I-1]); --J) {
      ny += 1;
      icopy += 1;
      LogDebug("SFGeom")<< "I=" << I << " J="  << J << endl;
      ypos = -1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      xpos = (sz-int(I))*waf_active + wedge_length/2. + 0.05*cm;
      zpos = zlead2_ + wedge_offset;
      // place the wedge
      DDpos(DDLogicalPart("esalgo:SWED"), parent(), icopy+nx, DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1301"));

    LogDebug("SFGeom")<<" debug : SWED, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SWED")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1301")<< endl;

      zpos = zlead2_ + zwedge_ceramic_diff;
      xpos = xpos + ywedge_ceramic_diff;
      DDpos(DDLogicalPart("esalgo:SFBY"), parent(), icopy, DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1300"));

    LogDebug("SFGeom")<<" debug : SFBY, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SFBY")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1300")<< endl;
    }
    for(int J=int(quadMin_[I-1]); J<=int(quadMax_[I-1]); ++J) {
      ny += 1;
      icopy += 1;
      LogDebug("SFGeom")<< "I=" << I << " J="  << J << endl;
      ypos = 1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      xpos = (sz-int(I))*waf_active + wedge_length/2. + 0.05*cm;
      zpos = zlead2_ + wedge_offset;
      // place the wedge
     DDpos(DDLogicalPart("esalgo:SWED"), parent(), icopy+nx, DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1301"));
    LogDebug("SFGeom")<<" debug : SWED, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SWED")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1301")<< endl;
      zpos = zlead2_ + zwedge_ceramic_diff;
      xpos = xpos + ywedge_ceramic_diff;
      DDpos(DDLogicalPart("esalgo:SFBY"), parent(), icopy, DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1300"));

    LogDebug("SFGeom")<<" debug : SFBY, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SFBY")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1300")<< endl;
    }
  }
  for(int I=1; I<=sz;++I) {
    for(int J=int(quadMax_[I-1]); J>=int(quadMin_[I-1]); --J) {
      ny += 1;
      icopy += 1;
      LogDebug("SFGeom")<< "I=" << I << " J="  << J << endl;
      ypos = -1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      xpos = -1.*(sz-int(I))*waf_active - wedge_length/2. - 0.05*cm;
      zpos = zlead2_ + wedge_offset;
      // place the wedge
      DDpos(DDLogicalPart("esalgo:SWED"), parent(), icopy+nx, DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1305"));

    LogDebug("SFGeom")<<" debug : SWED, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SWED")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1305")<< endl;

      zpos = zlead2_ + zwedge_ceramic_diff;
      xpos = xpos - ywedge_ceramic_diff;
      DDpos(DDLogicalPart("esalgo:SFBY"), parent(), icopy, DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1304"));

    LogDebug("SFGeom")<<" debug : SFBY, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SFBY")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1304")<< endl;


    }
    for(int J=int(quadMin_[I-1]); J<=int(quadMax_[I-1]); ++J) {
      ny += 1;
      icopy += 1;
      LogDebug("SFGeom")<< "I=" << I << " J="  << J << endl;
      ypos = 1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      xpos = -1.*(sz-int(I))*waf_active - wedge_length/2. - 0.05*cm;
      zpos = zlead2_ + wedge_offset;
      // place the wedge
      DDpos(DDLogicalPart("esalgo:SWED"), parent(), icopy+nx, DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1305"));

    LogDebug("SFGeom")<<" debug : SWED, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SWED")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1305")<< endl;
      zpos = zlead2_ + zwedge_ceramic_diff;
      xpos = xpos - ywedge_ceramic_diff;
      DDpos(DDLogicalPart("esalgo:SFBY"), parent(), icopy, DDTranslation(xpos,ypos,zpos),
            DDRotation("rotations:RM1304"));

    LogDebug("SFGeom")<<" debug : SFBY, copy = "<< icopy<<" Logical part "<<DDLogicalPart("esalgo:SFBY")<<"translation "<<DDTranslation(xpos,ypos,zpos)<<" rotation "<<DDRotation("rotations:RM1304")<< endl;
    }
  }
}

void DDEcalPreshowerAlgo::doSens()
{
   double xpos(0), ypos(0);
  for(size_t i = 0; i<32; ++i) 
    {
     xpos = -waf_active/2. + i*waf_active/32. + waf_active/64.;
     DDpos(DDLogicalPart("esalgo:SFSX"), DDName("SFAX","esalgo"), i+1, DDTranslation(xpos,0., 0.),DDRotation());

     LogDebug("SFGeom")<<" debug : SFSX, Logical part: "<<DDLogicalPart("esalgo:SFSX")<<endl<<" translation "<<DDTranslation(xpos,0.,0.)<<" rotation "<<DDRotation()<<" copy number= " <<i<<endl;
   
     ypos = -waf_active/2. + i*waf_active/32. + waf_active/64.;
     DDpos(DDLogicalPart("esalgo:SFSY"),DDName("SFAY","esalgo"), i+1, DDTranslation(0.,ypos, 0.), DDRotation());

     LogDebug("SFGeom")<<" debug : SFSY, Logical part: "<<DDLogicalPart("esalgo:SFSY")<<endl<<" translation "<<DDTranslation(0.,ypos,0.)<<" rotation "<<DDRotation()<< " copy number= " <<i<< endl;

  }

}
