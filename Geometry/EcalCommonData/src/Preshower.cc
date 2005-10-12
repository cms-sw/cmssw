#include "Geometry/EcalCommonData/interface/Preshower.h"

//#include "DDD/DDCore/interface/DDD.h"
#include <iostream>
#include <sstream>
using namespace DD;
using namespace std;

Preshower::Preshower() : DDAlgorithm() {}

void Preshower::initialize(const DDNumericArguments & nArgs,
                           const DDVectorArguments & vArgs,
                           const DDMapArguments & mArgs,
			   const DDStringArguments & sArgs,
			   const DDStringVectorArguments & vsArgs)
{
  cout << "Preshower initializing:" << endl <<  vArgs; 
  quadMin_ = vArgs["IQUAD_MIN"];
  quadMax_ = vArgs["IQUAD_MAX"];   
  thickLayers_ = Vector("presh:Layers");
  thickness_ = (Constant("presh:PRESH_Z_TOTAL").val()),
  /* Set up the preshower materials; 
     the materials are supposed to be defined either in a document materials.xml 
     or created in the namespace 'materials' using the DDD API */
  nmat_ = thickLayers_.size();
  // for the moment we use only Air
  for ( int i=0; i<nmat_; ++i) {
     materials_.push_back(Material("materials:Lead"));
  }
  // some lead, note: FORTRAN offset=1, C++ offset=0;
  materials_[6]  = Material("materials:Lead");
  materials_[16] = materials_[6];
}

void Preshower::execute()
{
  // creates all the tube-like layers of the preshower
  doLayers();
  // places the wedges and silicon strip detectors in their x and y layer
  doWedges();
}

void Preshower::doLayers()
{
  cout << "doLayers()" << endl;
  cout << "Parent:" << parent() << endl;
  Vector rmaxVec = Vector("presh:R_MAX"); // inner radii
  Vector rminVec = Vector("presh:R_MIN"); // outer radii
  cout << rminVec << endl; 
  double sum_z=0;
  double zpos = -thickness_/2.;
  for(size_t i = 0; i<rminVec.size(); ++i) {
    int I = int(i)+1; // FOTRAN I (offset +1)
    
    double rIn(0),rOut(0),zHalf(0);
    
    // create the name
    ostringstream name;
    name << "presh:SF";
    name << int((50+I)/10) << I-int(I/10)*10;
    Name ddname(name.str()); // namespace:name
    
    // tube dimensions
    rIn = rmaxVec[i];
    rOut = rminVec[i];
    zHalf = thickLayers_[i]/2.;

    // create a logical part representing a single layer in the preshower
    Solid solid = SolidFactory::tubs(ddname,zHalf,rIn,rOut,0.,360.*deg);
    Material material = materials_[i]; 
    LogicalPart layer = LogicalPart(ddname,material,solid);
    cout << layer << endl;
   
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
          Translation(0.,0., zpos),
          Rotation());
    cout << "pos of " << layer.name() << " z=" << zpos << endl;
    cout << "sum_z=" << sum_z << " presh_z_total=" << thickness_ << endl;
    zpos += zHalf; 
  }

}

void Preshower::doWedges()
{
  cout << "doWedges()" << endl;
  int nx(0), ny(0), icopy(0);
  double xpos(0), ypos(0), zpos(0);// zposY(0);
  double waf_intra_col_sep(Constant("presh:waf_intra_col_sep").val()),
    waf_inter_col_sep(Constant("presh:waf_inter_col_sep").val()),
    waf_active(Constant("presh:waf_active").val()),
    wedge_length(Constant("presh:wedge_length").val()),
    wedge_offset(Constant("presh:wedge_offset").val()),
    zwedge_ceramic_diff(Constant("presh:zwedge_ceramic_diff").val()),
    ywedge_ceramic_diff(Constant("presh:ywedge_ceramic_diff").val());


  cout << waf_intra_col_sep << endl;
  int sz = int(quadMax_.size());
  for(int I=1; I<=sz;++I) {
    for(int J=int(quadMax_[I-1]); J>=int(quadMin_[I-1]); --J) {
      nx += 1;
      icopy += 1;
      cout << "I=" << I << " J="  << J << endl;
      xpos = -1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      ypos = (sz-int(I))*waf_active + wedge_length/2. + 0.05*cm;
      zpos = zlead1_ + wedge_offset;
      // place the wedge
      DDpos(LogicalPart("presh:SWED"), parent(), icopy, Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_1001"));
      cout << "Wedge: icopy=" << icopy << " trans=" << Translation(xpos,ypos,zpos) << endl;
      zpos = zlead1_ + zwedge_ceramic_diff;
      ypos = ypos + ywedge_ceramic_diff;
      DDpos(LogicalPart("presh:SFBX"), parent(), icopy, Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_1000"));
    }
    for(int J=int(quadMin_[I-1]); J<=int(quadMax_[I-1]); ++J) {
      nx += 1;
      icopy += 1;
      xpos = 1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      ypos = (sz-int(I))*waf_active + wedge_length/2. + 0.05*cm;
      zpos = zlead1_ + wedge_offset;      
      DDpos(LogicalPart("presh:SWED"), parent(), icopy, Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_1001"));
      cout << "Wedge: icopy=" << icopy << " trans=" << Translation(xpos,ypos,zpos) << endl;
      zpos = zlead1_ + zwedge_ceramic_diff;
      ypos = ypos + ywedge_ceramic_diff;
      DDpos(LogicalPart("presh:SFBX"), parent(),icopy,Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_1000"));
    }

  }

  // mirror image system
  for(int I=sz; I>=1;--I) {
    for(int J=int(quadMax_[I-1]); J>=int(quadMin_[I-1]); --J) {
      nx += 1;
      icopy += 1;
      cout << "I=" << I << " J="  << J << endl;
      xpos = -1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      ypos = -1.*(sz-int(I))*waf_active - wedge_length/2. - 0.05*cm;
      zpos = zlead1_ + wedge_offset;
      // place the wedge
      DDpos(LogicalPart("presh:SWED"), 
            parent(),
            icopy,
            Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_2001"));
      cout << "Wedge: icopy=" << icopy << " trans=" << Translation(xpos,ypos,zpos) << endl;
      zpos = zlead1_ + zwedge_ceramic_diff;
      ypos = ypos - ywedge_ceramic_diff;
      DDpos(LogicalPart("presh:SFBX"), 
            parent(),
            icopy,
            Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_2000"));
    }
    for(int J=int(quadMin_[I-1]); J<=int(quadMax_[I-1]); ++J) {
      nx += 1;
      icopy += 1;
      xpos = 1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      ypos = -1.*(sz-int(I))*waf_active - wedge_length/2. - 0.05*cm;
      zpos = zlead1_ + wedge_offset;      
      DDpos(LogicalPart("presh:SWED"), 
            parent(),
            icopy,
            Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_2001"));
      cout << "Wedge: icopy=" << icopy << " trans=" << Translation(xpos,ypos,zpos) << endl;
      zpos = zlead1_ + zwedge_ceramic_diff;
      ypos = ypos - ywedge_ceramic_diff;
      DDpos(LogicalPart("presh:SFBX"), 
            parent(),
            icopy,
            Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_2000"));
    }
  }

  // Second Plane (Y Plane)
  // Top half first
  icopy =  0;
  for(int I=1; I<=sz;++I) {
    for(int J=int(quadMax_[I-1]); J>=int(quadMin_[I-1]); --J) {
      ny += 1;
      icopy += 1;
      cout << "I=" << I << " J="  << J << endl;
      ypos = -1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      xpos = (sz-int(I))*waf_active + wedge_length/2. + 0.05*cm;
      zpos = zlead2_ + wedge_offset;
      // place the wedge
      DDpos(LogicalPart("presh:SWED"), parent(), icopy+nx, Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_1003"));
      cout << "Wedge: icopy=" << icopy << " trans=" << Translation(xpos,ypos,zpos) << endl;
      zpos = zlead2_ + zwedge_ceramic_diff;
      xpos = xpos + ywedge_ceramic_diff;
      DDpos(LogicalPart("presh:SFBY"), parent(), icopy, Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_1002"));
    }
    for(int J=int(quadMin_[I-1]); J<=int(quadMax_[I-1]); ++J) {
      ny += 1;
      icopy += 1;
      cout << "I=" << I << " J="  << J << endl;
      ypos = 1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      xpos = (sz-int(I))*waf_active + wedge_length/2. + 0.05*cm;
      zpos = zlead2_ + wedge_offset;
      // place the wedge
      DDpos(LogicalPart("presh:SWED"), parent(), icopy+nx, Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_1003"));
      cout << "Wedge: icopy=" << icopy << " trans=" << Translation(xpos,ypos,zpos) << endl;
      zpos = zlead2_ + zwedge_ceramic_diff;
      xpos = xpos + ywedge_ceramic_diff;
      DDpos(LogicalPart("presh:SFBY"), parent(), icopy, Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_1002"));
    }
  }
  for(int I=1; I<=sz;++I) {
    for(int J=int(quadMax_[I-1]); J>=int(quadMin_[I-1]); --J) {
      ny += 1;
      icopy += 1;
      cout << "I=" << I << " J="  << J << endl;
      ypos = -1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      xpos = -1.*(sz-int(I))*waf_active - wedge_length/2. - 0.05*cm;
      zpos = zlead2_ + wedge_offset;
      // place the wedge
      DDpos(LogicalPart("presh:SWED"), parent(), icopy+nx, Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_2003"));
      cout << "Wedge: icopy=" << icopy << " trans=" << Translation(xpos,ypos,zpos) << endl;
      zpos = zlead2_ + zwedge_ceramic_diff;
      xpos = xpos - ywedge_ceramic_diff;
      DDpos(LogicalPart("presh:SFBY"), parent(), icopy, Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_2002"));
    }
    for(int J=int(quadMin_[I-1]); J<=int(quadMax_[I-1]); ++J) {
      ny += 1;
      icopy += 1;
      cout << "I=" << I << " J="  << J << endl;
      ypos = 1.*(J*waf_intra_col_sep + (int(J/2))*waf_inter_col_sep - waf_intra_col_sep/2.);
      xpos = -1.*(sz-int(I))*waf_active - wedge_length/2. - 0.05*cm;
      zpos = zlead2_ + wedge_offset;
      // place the wedge
      DDpos(LogicalPart("presh:SWED"), parent(), icopy+nx, Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_2003"));
      cout << "Wedge: icopy=" << icopy << " trans=" << Translation(xpos,ypos,zpos) << endl;
      zpos = zlead2_ + zwedge_ceramic_diff;
      xpos = xpos - ywedge_ceramic_diff;
      DDpos(LogicalPart("presh:SFBY"), parent(), icopy, Translation(xpos,ypos,zpos),
            Rotation("presh:MROT_2002"));
    }
  }

}
