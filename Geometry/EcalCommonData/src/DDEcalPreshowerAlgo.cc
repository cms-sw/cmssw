#include "Geometry/EcalCommonData/interface/DDEcalPreshowerAlgo.h"

#include <cmath>
#include <algorithm>

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

DDEcalPreshowerAlgo::DDEcalPreshowerAlgo() : DDAlgorithm() {
  LogDebug("EcalGeom") << "DDEcalPreshowerAlgo info: Creating an instance" ;

}

void DDEcalPreshowerAlgo::initialize(const DDNumericArguments & nArgs,
				     const DDVectorArguments & vArgs,
				     const DDMapArguments & mArgs,
				     const DDStringArguments & sArgs,
				     const DDStringVectorArguments & vsArgs)
{

  LogDebug("EcalGeom") << "DDEcalPreshowerAlgo info: Initialize" ;
 
  asym_ladd_ = vArgs["ASYMETRIC_LADDER"];
  types_l5_ = vsArgs["TYPES_OF_LADD_L5"];
  types_l4_ = vsArgs["TYPES_OF_LADD_L4"];
  ladd_l5_map_ = vArgs["LADD_L5_MAP"];
  ladd_l4_map_ = vArgs["LADD_L4_MAP"];
  noLaddInCol_ = vArgs["NUMB_OF_LADD_IN_COL"];
  startOfFirstLadd_ = vArgs["START_OF_1ST_LADD"];
  typeOfLaddRow0 = vsArgs["TYPE_OF_LADD_1"];
  typeOfLaddRow1 = vsArgs["TYPE_OF_LADD_2"];
  typeOfLaddRow2 = vsArgs["TYPE_OF_LADD_3"];
  typeOfLaddRow3 = vsArgs["TYPE_OF_LADD_4"];
  thickLayers_ = vArgs["Layers"];
  thickness_ = double(nArgs["PRESH_Z_TOTAL"]);
  materials_ = vsArgs["LayMat"];
  layName_ = vsArgs["LayName"];
  rmaxVec = vArgs["R_MAX"]; // inner radii
  rminVec = vArgs["R_MIN"]; // outer radii
  waf_intra_col_sep = double(nArgs["waf_intra_col_sep"]);
  waf_inter_col_sep = double(nArgs["waf_inter_col_sep"]);
  waf_active = double(nArgs["waf_active"]);
  wedge_length = double(nArgs["wedge_length"]);
  wedge_offset = double(nArgs["wedge_offset"]);
  zwedge_ceramic_diff = double(nArgs["zwedge_ceramic_diff"]);
  ywedge_ceramic_diff = double(nArgs["ywedge_ceramic_diff"]);
  wedge_angle = double(nArgs["wedge_angle"]);
  ladder_thick = double(nArgs["ladder_thick"]);
  yladder_1stwedge_diff = double(nArgs["yladder_1stwedge_diff"]);
  ladder_width = double(nArgs["ladder_width"]);
  micromodule_length = double(nArgs["micromodule_length"]);
  subtr_length = double(nArgs["subtr_length"]);
  subtr_width = double(nArgs["subtr_width"]);
  subtr_thick = double(nArgs["subtr_thick"]);
  box_thick = double(nArgs["box_thick"]);
  ysubtr1_ladder_diff = double(nArgs["ysubtr1_ladder_diff"]);
  zsubtr1_ladder_diff = double(nArgs["zsubtr1_ladder_diff"]);
  ysubtr2_ladder_diff = double(nArgs["ysubtr2_ladder_diff"]);
  zsubtr2_ladder_diff = double(nArgs["zsubtr2_ladder_diff"]);
  abs1stx = vArgs["1ST_ABSX"];
  abs1sty = vArgs["1ST_ABSY"];
  abs2ndx = vArgs["2ND_ABSX"];
  abs2ndy = vArgs["2ND_ABSY"];
  ladPfx_ = vsArgs["LadPrefix"];
  LaddMaterial_ =  sArgs["LadderMaterial"];
}

void DDEcalPreshowerAlgo::execute()
{
  LogDebug("EcalGeom") << "******** DDEcalPreshowerAlgo execute!";

  // creates all the tube-like layers of the preshower
  doLayers();
  // creates and places the ladders
  doLadders();  
  // places the slicon strips in active silicon wafers
  doSens();
  
}

void DDEcalPreshowerAlgo::doLayers()
{

  double zpos = -thickness_/2., sdx(0), sdy(0), bdx(0),bdy(0);;

  for(size_t i = 0; i<thickLayers_.size(); ++i) {
    
    int I = int(i)+1; // FOTRAN I (offset +1)
    
    double rIn(0), rOut(0), zHalf(0);
    
    // create the name
    DDName ddname(getLayName(i),"esalgo");   // namespace:name

    // cone dimensions
    rIn   = rminVec[i];
    rOut  = rmaxVec[i];
    zHalf = thickLayers_[i]/2.;

    // create a logical part representing a single layer in the preshower
    DDSolid solid = DDSolidFactory::tubs(ddname, zHalf, rIn, rOut, 0.,360.*deg);

    DDLogicalPart layer = DDLogicalPart(ddname,getMaterial(i),solid);
    
    // position the logical part w.r.t. the parent volume
    zpos += zHalf;
    
    // create a logical part representing a single layer in the preshower
    // skip layers with detectors, front and rear window 
    if (I==2 || I==28 || I==13 || I==23) {
      zpos += zHalf;
      continue;
    }
    
    if ( I==12 ) {
      zlead1_ = zpos + zHalf;
    }
    if ( I==21 ) {
      zlead2_ = zpos + zHalf;
    }
    
    if (I==10 || I==20) { // New lead shape
      
      int absz=0;
      absz = int(abs1stx.size()); 
      if ( I==20) absz = int(abs2ndx.size()); 
      int cutabsx=-1;
      
      for (int L=0; L<absz; ++L) {
	
	int K=L; 
	ostringstream  tmp_name_b;
	tmp_name_b << getLayName(i) << "L" << K; 
	DDName dd_tmp_name_b(tmp_name_b.str(),"esalgo");
	
	bdx = abs(abs1stx[K]-abs1stx[K-1])/2; bdy=abs1sty[K];
	if(abs1stx[K] < rIn+20*cm) { bdy=abs1sty[K]/2 - 30*cm; cutabsx = K;}
	
	if(I==20) { bdx = abs(abs2ndx[K]-abs2ndx[K-1])/2; bdy=abs2ndy[K];}
	
	if((abs2ndx[K] < rIn+20*cm) && I==20) { bdy=abs2ndy[K]/2 - 30*cm; cutabsx = K;}
	
	DDSolid solid_b = DDSolidFactory::box(dd_tmp_name_b,bdx,bdy,zHalf);
	
	sdx = abs1stx[K]-bdx ; sdy = 0;
	if(abs1stx[K] < rIn+20*cm ) sdy = abs1sty[K]-bdy ; 
	
	if(I==20) {sdx = abs2ndx[K]-bdx ; sdy = 0;}
	if((abs2ndx[K] < rIn+20*cm) && I==20 ) sdy = abs2ndy[K]-bdy ;
		
	DDLogicalPart layer = DDLogicalPart(dd_tmp_name_b,getMaterial(i),solid_b);
	
	DDpos(layer, parent(), 1, DDTranslation(sdx,sdy, zpos), DDRotation());
	DDpos(layer, parent(), 2, DDTranslation(-sdx,sdy, zpos), DDRotation());
	if(((abs1stx[K] < rIn+20*cm) && I==10) || ((abs2ndx[K] < rIn+20*cm) && I==20) ) { 
	  DDpos(layer, parent(), 3, DDTranslation(sdx,-sdy, zpos), DDRotation());
	  DDpos(layer, parent(), 4, DDTranslation(-sdx,-sdy, zpos), DDRotation());
	  
	}		
      }  
      
      DDName dd_tmp_name_b(getLayName(i)+"Lcut","esalgo");
      DDName dd_tmp_name_c(getLayName(i)+"tmpb","esalgo");
      DDName dd_tmp_name_d(getLayName(i)+"LinAl","esalgo");
      
      bdx = abs1stx[cutabsx]; bdy=2*30*cm;
      if(I==20) bdx = abs2ndx[cutabsx]; bdy=2*30*cm;
      
      DDSolid solidcut = DDSolidFactory::box(dd_tmp_name_b,bdx,bdy,zHalf);
      DDSolid iner = DDSolidFactory::tubs(dd_tmp_name_c,zHalf+0.1*mm,0,rIn,0.,360.*deg); 
      DDSolid final = DDSolidFactory::subtraction(dd_tmp_name_d,solidcut,iner,DDTranslation(0,0,0),DDRotation());
      
      DDLogicalPart layer = DDLogicalPart(dd_tmp_name_d,getMaterial(i),final);
      DDpos(layer, parent(), 1, DDTranslation(0,0, zpos), DDRotation());
      
    } else {
      
      DDpos(layer, parent(), 1, DDTranslation(0.,0., zpos), DDRotation());
      
      LogDebug("SFGeom")<<" debug : tubs, Logical part: "<<DDLogicalPart(ddname,getMaterial(i),solid)<<endl<<" translation "<<DDTranslation(0.,0.,zpos)<<" rotation "<<DDRotation()<< endl;
    } 
    
    zpos += zHalf; 
  }
  
}

void DDEcalPreshowerAlgo::doLadders() {
  
  double xpos(0), ypos(0), zpos(0), sdx(0), sdy(0), sdz(0);
  double prev_length_(0), ladder_new_length_(0), ladd_shift_(0), ladder_length (0);
  int enb(0); double sdxe[50] = {0}, sdye[50] = {0}, sdze[50] = {0};
  
  for (int M=0; M<int(types_l5_.size() + types_l4_.size()); M++) {
    int scopy(0);

    if(M<int(types_l5_.size())) {
      DDName ddname(getLadPrefix(0)+types_l5_[M],"esalgo");
      ladder_length = micromodule_length + 4*waf_active;
      
      DDName dd_tmp_name_5a(getLadPrefix(2),"esalgo");
      DDName dd_tmp_name_5b(getLadPrefix(3),"esalgo");
      DDName dd_tmp_name_5c(getLadPrefix(4),"esalgo");
      DDName dd_tmp_name_5e(getLadPrefix(6),"esalgo");
      
      DDSolid solid_5a = DDSolidFactory::box(dd_tmp_name_5a,ladder_width/2,ladder_length/2,ladder_thick/2.);
      DDSolid solid_5b = DDSolidFactory::box(dd_tmp_name_5b,ladder_width/2+0.1*mm,subtr_length/2,subtr_thick/2.);
      sdx = 0; sdy= -ladder_length/2 + ysubtr1_ladder_diff; sdz = zsubtr1_ladder_diff;
      DDSolid solid_5c = DDSolidFactory::subtraction(dd_tmp_name_5c,solid_5a,solid_5b,DDTranslation(sdx,sdy,sdz),DDRotation("esalgo:RM1298"));
      
      int change =0; 
      DDSolid solid_5e = DDSolidFactory::box(dd_tmp_name_5e,ladder_width/4+0.01*mm,waf_active/2+0.01*mm,ladder_thick/2+0.1*mm);
      
      for (int i=0; i<=1; i++) {
	for (int j=0; j<=4; j++) {
	  if(ladd_l5_map_[(i+j*2+M*10)]!=1 ){  
	    enb++; 
	    ostringstream  tmp_name_5c, tmp_name_5d, tmp_name_5e ;
	    if(change==0) tmp_name_5c << getLadPrefix(4);
	    if(change==1) tmp_name_5c <<getLadPrefix(5)<< enb-1;
	    DDName dd_tmp_name_5c(tmp_name_5c.str(),"esalgo");	    
	    tmp_name_5d << getLadPrefix(5) << enb;
	    DDName dd_tmp_name_5d(tmp_name_5d.str(),"esalgo");
	    tmp_name_5e << getLadPrefix(6) << enb;
	    DDName dd_tmp_name_5e(tmp_name_5e.str(),"esalgo");	    
	    change=1;
	    sdxe[enb] = i*(ladder_width/2.) - ladder_width/4.; 
	    sdye[enb] = - ladder_length/2. + j*waf_active + waf_active/2. - 0.2*mm; 
	    if(j>1) sdye[enb] = ladder_length/2. - (3-j)*waf_active - waf_active/4. + 0.2*mm; 
	    sdze[enb] = 0.;
	    DDSolid solid_5c = DDSolid(dd_tmp_name_5c);
	    DDSolid solid_5d = DDSolidFactory::subtraction(dd_tmp_name_5d,solid_5c,solid_5e,DDTranslation(sdxe[enb],sdye[enb],sdze[enb]),DDRotation());	    
	  }
	}
      }            
      ostringstream  tmp_name_5d;
      if(change==0) tmp_name_5d << getLadPrefix(4);      
      if(change==1) tmp_name_5d << getLadPrefix(5) << enb;
      DDName dd_tmp_name_5d(tmp_name_5d.str(),"esalgo");
      DDSolid solid_5d = DDSolid(dd_tmp_name_5d);
      sdx = 0; sdy = ladder_length/2 + ysubtr2_ladder_diff; sdz = zsubtr2_ladder_diff;      
      DDSolid solid = DDSolidFactory::subtraction(ddname,solid_5d,solid_5b,DDTranslation(sdx,sdy,sdz),DDRotation("esalgo:RM1298"));      
      DDLogicalPart ladder = DDLogicalPart(ddname,getLaddMaterial(),solid);      
      DDName ddname2(getLadPrefix(1)+types_l5_[M],"esalgo");      
      DDLogicalPart ladder2 = DDLogicalPart(ddname2,getLaddMaterial(),solid);      
    }
    if( M >= int(types_l5_.size()) ) {
      int d = M - types_l5_.size();
      DDName ddname(getLadPrefix(0)+types_l4_[d],"esalgo");      
      ladder_length = micromodule_length + 3*waf_active;      
      DDName dd_tmp_name_a(getLadPrefix(7),"esalgo");
      DDName dd_tmp_name_b(getLadPrefix(8),"esalgo");
      DDName dd_tmp_name_c(getLadPrefix(9),"esalgo");      
      DDName dd_tmp_name_e(getLadPrefix(11),"esalgo");           
      DDSolid solid_a = DDSolidFactory::box(dd_tmp_name_a,ladder_width/2,ladder_length/2,ladder_thick/2.);
      DDSolid solid_b = DDSolidFactory::box(dd_tmp_name_b,ladder_width/2+0.1*mm,subtr_length/2,subtr_thick/2.);
      sdx = 0; sdy= -ladder_length/2 + ysubtr1_ladder_diff; sdz = zsubtr1_ladder_diff;
      DDSolid solid_c = DDSolidFactory::subtraction(dd_tmp_name_c,solid_a,solid_b,DDTranslation(sdx,sdy,sdz),DDRotation("esalgo:RM1298"));
      double sdxe[20] = {0}, sdye[20] = {0}, sdze[20] = {0};
      int change =0; 
      DDSolid solid_e = DDSolidFactory::box(dd_tmp_name_e,ladder_width/4+0.01*mm,waf_active/2+0.01*mm,ladder_thick/2+0.01*mm);
      for (int i=0; i<=1; i++) {
	for (int j=0; j<=3; j++) {
	  if(ladd_l4_map_[(i+j*2+(M-types_l5_.size())*8)]!=1 ){  
	    ostringstream  tmp_name_c, tmp_name_d, tmp_name_e ;
	    enb++; 
	    if(change==0) tmp_name_c << getLadPrefix(9);
	    if(change==1) tmp_name_c <<getLadPrefix(10) << enb-1;
	    DDName dd_tmp_name_c(tmp_name_c.str(),"esalgo");	    
	    tmp_name_d << getLadPrefix(10) << enb;
	    DDName dd_tmp_name_d(tmp_name_d.str(),"esalgo");
	    tmp_name_e << getLadPrefix(11) << enb;
	    DDName dd_tmp_name_e(tmp_name_e.str(),"esalgo");
	    change=1;
	    sdxe[enb] = i*(ladder_width/2.) - ladder_width/4.; 
	    sdye[enb] = - ladder_length/2. + j*waf_active + waf_active/2.; 
	    if(j>1) sdye[enb] = ladder_length/2. - (3-j)*waf_active - waf_active/4.; 
	    sdze[enb] = 0.;
	    DDSolid solid_c = DDSolid(dd_tmp_name_c);
	    DDSolid solid_d = DDSolidFactory::subtraction(dd_tmp_name_d,solid_c,solid_e,DDTranslation(sdxe[enb],sdye[enb],sdze[enb]),DDRotation());
	  }
	}
      }
      ostringstream tmp_name_d;   
      if(change==0) tmp_name_d << getLadPrefix(9);
      if(change==1) tmp_name_d <<getLadPrefix(10) << enb;
      DDName dd_tmp_name_d(tmp_name_d.str(),"esalgo");	    
      DDSolid solid_d = DDSolid(dd_tmp_name_d);
      sdx = 0; sdy = ladder_length/2 + ysubtr2_ladder_diff; sdz = zsubtr2_ladder_diff;
      DDSolid solid = DDSolidFactory::subtraction(ddname,solid_d,solid_b,DDTranslation(sdx,sdy,sdz),DDRotation("esalgo:RM1298"));
      DDLogicalPart ladder = DDLogicalPart(ddname,getLaddMaterial(),solid);
      DDName ddname2(getLadPrefix(1)+types_l4_[d],"esalgo"); 
      DDLogicalPart ladder2 = DDLogicalPart(ddname2,getLaddMaterial(),solid);
    }
    if(M<int(types_l5_.size())) {
      DDName ddname(getLadPrefix(0)+types_l5_[M],"esalgo");
      DDName ddname2(getLadPrefix(1)+types_l5_[M],"esalgo"); 
      for (int i=0; i<=1; i++) {
	for (int j=0; j<=4; j++) {
	  xpos = (i*2-1)*waf_intra_col_sep/2.; ypos = -ladder_length/2. + wedge_length/2. + j*waf_active;
	  zpos = -ladder_thick/2. + wedge_offset;
	  if(ladd_l5_map_[(i+j*2+M*10)]==1) {
	    scopy ++;
	    DDpos(DDLogicalPart("esalgo:SWED"), ddname, scopy, DDTranslation(xpos,ypos,zpos), DDRotation("esalgo:RM1299"));
	    DDpos(DDLogicalPart("esalgo:SWED"), ddname2, scopy, DDTranslation(xpos,ypos,zpos), DDRotation("esalgo:RM1299"));

	    ypos = ypos + ywedge_ceramic_diff; zpos = -ladder_thick/2. + zwedge_ceramic_diff;
	    DDpos(DDLogicalPart("esalgo:SFBX"), ddname, scopy, DDTranslation(xpos,ypos,zpos), DDRotation("esalgo:RM1298"));
	    DDpos(DDLogicalPart("esalgo:SFBY"), ddname2, scopy, DDTranslation(xpos,ypos,zpos), DDRotation("esalgo:RM1300A")); 
	  }
	}
      }
    }
    else
      {
	int d = M - types_l5_.size();
	DDName ddname(getLadPrefix(0)+types_l4_[d],"esalgo"); 
	DDName ddname2(getLadPrefix(1)+types_l4_[d],"esalgo"); 	
	for (int i=0; i<=1; i++) {
	  for (int j=0; j<=3; j++) {
	    xpos = (i*2-1)*waf_intra_col_sep/2.; ypos = -ladder_length/2. + wedge_length/2. + j*waf_active;
	    zpos = -ladder_thick/2. + wedge_offset;
	    if(ladd_l4_map_[(i+j*2+(M-types_l5_.size())*8)]==1) {
	      scopy ++;
	      DDpos(DDLogicalPart("esalgo:SWED"), ddname, scopy, DDTranslation(xpos,ypos,zpos), DDRotation("esalgo:RM1299"));
	      DDpos(DDLogicalPart("esalgo:SWED"), ddname2, scopy, DDTranslation(xpos,ypos,zpos), DDRotation("esalgo:RM1299"));

	      ypos = ypos + ywedge_ceramic_diff; zpos = -ladder_thick/2. + zwedge_ceramic_diff;
	      DDpos(DDLogicalPart("esalgo:SFBX"), ddname, scopy, DDTranslation(xpos,ypos,zpos), DDRotation("esalgo:RM1298"));
	      DDpos(DDLogicalPart("esalgo:SFBY"), ddname2, scopy, DDTranslation(xpos,ypos,zpos), DDRotation("esalgo:RM1300A"));
	    }
	  }
	}
      }
  }
  string type;
  int icopy[100] = {0};

  for(int I=-9; I<=9;++I) {
    prev_length_=0; int J=abs(I);
    for (int K=0; K<noLaddInCol_[J]; K++) {
      string type;

      ladder_new_length_ = micromodule_length + 3*waf_active;
      ladd_shift_ = 4*waf_active;

      if(K==0) type = typeOfLaddRow0[J]; 
      if(K==1) type = typeOfLaddRow1[J];
      if(K==2) type = typeOfLaddRow2[J];
      if(K==3) type = typeOfLaddRow3[J]; 

      for(int i=0;i<int(types_l5_.size());i++) if(type == types_l5_[i]) {
	ladder_new_length_ = micromodule_length + 4*waf_active; 
	ladd_shift_ = 5*waf_active;}
      
      int j = 0;
      
      for(int t=0;t<int(types_l5_.size());t++) if(type == types_l5_[t]) {j = t; if(I<0 && asym_ladd_[t] == 1 ) {j = j + 1;  type = types_l5_[j];}} 
      for(int t=0;t<int(types_l4_.size());t++) if(type == types_l4_[t]) {j = t+types_l5_.size(); if(I<0 && asym_ladd_[(t+types_l5_.size())] == 1 ) {j = j + 1; type = types_l4_[j-types_l5_.size()];}}
      
      xpos = I*(2*waf_intra_col_sep + waf_inter_col_sep);
      int sz = 20;    
      ypos = (sz-int(startOfFirstLadd_[J]))*waf_active - ladder_new_length_/2. + micromodule_length + 0.05*cm - prev_length_;
      prev_length_ += ladd_shift_;        
      
      zpos = zlead1_ + ladder_thick/2.;
      icopy[j] +=1;
      DDName ddname(getLadPrefix(0)+type,"esalgo"); 
      
      DDpos(DDLogicalPart(ddname), DDName("SF","esalgo"), icopy[j], DDTranslation(xpos,ypos,zpos), DDRotation());
      DDName ddname2(getLadPrefix(1)+type,"esalgo"); 
      
      DDpos(DDLogicalPart(ddname2), DDName("SF","esalgo"), icopy[j], DDTranslation(ypos,-xpos,zpos-zlead1_+zlead2_), DDRotation("esalgo:R270"));	
      
      int changed = 0;
      for(int t=0;t<int(types_l5_.size());t++) if(type == types_l5_[t]) {j = t; if(asym_ladd_[t] == 2 && !changed) { j = j - 1; changed = 1;} if(asym_ladd_[t] == 1 && !changed) { j = j + 1; changed = 1;}  type = types_l5_[j];} 
      for(int t=0;t<int(types_l4_.size());t++) if(type == types_l4_[t]) {j = t+types_l5_.size(); if(asym_ladd_[(t+types_l5_.size())] == 2 && !changed) { j = j - 1; changed = 1;} if(asym_ladd_[(t+types_l5_.size())] == 1 && !changed) { j = j + 1; changed = 1;} type = types_l4_[j-types_l5_.size()]; }
      
      icopy[j] +=1;
      DDName ddname3(getLadPrefix(0)+type,"esalgo");       
      DDpos(DDLogicalPart(ddname3), DDName("SF","esalgo"), icopy[j], DDTranslation(xpos,-ypos,zpos), DDRotation("esalgo:R180"));
      DDName ddname4(getLadPrefix(1)+type,"esalgo"); 
      DDpos(DDLogicalPart(ddname4), DDName("SF","esalgo"), icopy[j], DDTranslation(-ypos,-xpos,zpos-zlead1_+zlead2_), DDRotation("esalgo:R090"));
    }
  } 
}

void DDEcalPreshowerAlgo::doSens() {
  
  double xpos(0), ypos(0);
  for(size_t i = 0; i<32; ++i)
    {
      xpos = -waf_active/2. + i*waf_active/32. + waf_active/64.;
      DDpos(DDLogicalPart("esalgo:SFSX"), DDName("SFWX","esalgo"), i+1, DDTranslation(xpos,0., 0.),DDRotation());
      
      LogDebug("SFGeom")<<" debug : SFSX, Logical part: "<<DDLogicalPart("esalgo:SFSX")<<endl<<" translation "<<DDTranslation(xpos,0.,0.)<<" rotation "<<DDRotation()<<" copy number= " <<i<<endl;
      
      ypos = -waf_active/2. + i*waf_active/32. + waf_active/64.;
      DDpos(DDLogicalPart("esalgo:SFSY"),DDName("SFWY","esalgo"), i+1, DDTranslation(0.,ypos, 0.), DDRotation());
      
      LogDebug("SFGeom")<<" debug : SFSY, Logical part: "<<DDLogicalPart("esalgo:SFSY")<<endl<<" translation "<<DDTranslation(0.,ypos,0.)<<" rotation "<<DDRotation()<< " copy number= " <<i<< endl;
      
    }
}
