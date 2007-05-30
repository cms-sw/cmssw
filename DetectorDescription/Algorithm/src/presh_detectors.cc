#include "DetectorDescription/Algorithm/src/presh_detectors.h"

#include <vector>
#include <cmath>

#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"

#include "CLHEP/Units/SystemOfUnits.h"

typedef std::vector<double> dbl_t;
void preshPrepareTubs();
//void preshDefineConstants();

presh_detectors::presh_detectors(AlgoPos* p,std::string label) 
 : AlgoImpl(p,label)
{ 
}
  
//! your comment here ...
presh_detectors::~presh_detectors()
  { }
  
bool presh_detectors::checkParameters() 
{
  static bool once_ = false;
  if (!once_) {
   // preshPrepareTubs();
    defineConstants();
    once_=true;
  }
  
  topHalfXPlane();
  return true;
}
  
  
DDTranslation presh_detectors::translation()
{
  // pseudo-implementation
  return DDTranslation();
}
  
  
DDRotationMatrix presh_detectors::rotation()
{
  // pseudo-implementation
  return DDRotationMatrix();
}


int presh_detectors::copyno() const
{
  return 0;
}
  

void presh_detectors::topHalfXPlane()
{
   int NX(0), icopy(0);
  
   DDLogicalPart SF(DDName("SF","preshower"));
   
   DDLogicalPart SWED(DDName("SWED","preshower"));
   DDLogicalPart BOX(DDName("SFBX","preshower"));
  
   DDRotation w_rot(DDName("MROT_1001","preshower"));
   DDRotation b_rot(DDName("MROT_1000","preshower"));
   double zlead1=0;
   for(int I=1; I<=20; ++I) {
     for(int J=IQUAD_MAX_[I]; J>=IQUAD_MIN_[I]; --J) {
       ++NX;
       ++icopy;
       double XPOS = -1.*(J*waf_intra_col_sep_+(int(J/2))*waf_inter_col_sep_ - waf_intra_col_sep_/2.);
       double YPOS = (20-I)*waf_active_ + wedge_length_/2. + 0.05*cm;
       DDTranslation trans(XPOS,YPOS,zlead1+wedge_offset_);
       DDpos(SWED,SF,icopy,trans,w_rot);
       trans = DDTranslation(XPOS,YPOS,zlead1+zwedge_ceramic_diff_);
       DDpos(BOX ,SF,icopy,trans,b_rot);
     }
     for(int J=IQUAD_MIN_[I]; J<=IQUAD_MAX_[I]; ++J) {
       ++NX;
       ++icopy;
       // = -XPOS from above
       double XPOS = (J*waf_intra_col_sep_ + (int(J/2))*waf_inter_col_sep_ - waf_intra_col_sep_/2.);              
       // = same as YPOS from above
       double YPOS = (20-I)*waf_active_ + wedge_length_/2. + 0.05*cm;
       DDTranslation trans(XPOS,YPOS,zlead1+wedge_offset_);
       DDpos(SWED,SF,icopy,trans,w_rot);
       trans = DDTranslation(XPOS,YPOS,zlead1+zwedge_ceramic_diff_);
       DDpos(BOX ,SF,icopy,trans,b_rot);
     }
   }
   
}
    
void presh_detectors::stream(std::ostream & os) const
{
  os << "algorithm to place wedges & silicon strip detectors of the ecal preshower";
}


// The constants defined here should come from the XML ....
void presh_detectors::defineConstants()
{
   std::vector<double> & gmx = *(new std::vector<double>(21,19));
   std::vector<double> & gmn = *(new std::vector<double>(21,1));
   
   gmx[1] = 5;
   gmx[2] = 7;
   gmx[3] = 10;
   gmx[4] = 11;
   gmx[5] = 13;
   gmx[6] = 13;
   gmx[7] = 14;
   gmx[8] = 15;
   gmx[9] = 16;
   gmx[10] = 17;
   gmx[11] = 17;
   gmx[12] = 17;
   gmx[13] = 18;
   
   gmn[14] = 4;
   gmn[15] = 4;
   gmn[16] = 6;
   gmn[17] = 6;
   gmn[18] = 8;
   gmn[19] = 8;
   gmn[20] = 8;
   
   /* These are 'global' parameters in the sense that they are always the
      same in subsequent usage of the algorithm */
   DDVector iquad_max(DDName("IQUAD_MAX","preshower"),&gmx);
   DDVector iquad_min(DDName("IQUAD_MIN","preshower"),&gmn);
   
   waf_intra_col_sep_ = DDConstant(DDName("waf_intra_col_sep","preshower"));
   waf_inter_col_sep_ = DDConstant(DDName("waf_inter_col_sep","preshower"));
   waf_active_   = DDConstant(DDName("waf_active","preshower"));
   wedge_length_ = DDConstant(DDName("wedge_length","preshower")); 
   wedge_offset_ = DDConstant(DDName("wedge_offset","preshower")); 
   zwedge_ceramic_diff_ = DDConstant(DDName("zwedge_ceramic_diff","preshower"));
   
   IQUAD_MAX_ = DDVector(DDName("IQUAD_MAX","preshower"));
   IQUAD_MIN_ = DDVector(DDName("IQUAD_MIN","preshower"));
}
    
void preshPrepareTubs()
{
  DDCurrentNamespace::ns() = "presh";
 
  /* all constants from the titles file esfx.tz;
     in future: they'll be defined in XML */
  //preshDefineConstants();
  
  
  dbl_t PAR(21);
  double PRESH_Z_TOTAL = DDConstant("PRESH_Z_TOTAL");
  double PRES_Z = DDConstant("PRESH_Z");
  double PRESH_Z = PRES_Z;

  PAR[3] = PRESH_Z_TOTAL /2.;
  double THETA_MIN = 2.*atan(exp(-double(DDConstant("PRE_ETA_MIN"))));
  double THETA_MAX = 2.*atan(exp(-double(DDConstant("PRE_ETA_MAX"))));
  DCOUT('E', "THETA_MIN=" << THETA_MIN/deg << "THETA_MAX=" << THETA_MAX/deg);

  double ECAL_Z = DDConstant("ECAL_Z");
  double R_MIN = ECAL_Z * tan(THETA_MIN);
  double R_MAX = PRESH_Z*tan(THETA_MAX);
  DCOUT('E', "R_MIN=" << R_MIN/cm << "cm   R_MAX=" << R_MAX/cm << "cm");


} 
    

// Check constraints of input parameters
presh_detectorsChecker::presh_detectorsChecker()
{
}

presh_detectorsChecker::~presh_detectorsChecker()
{
}
