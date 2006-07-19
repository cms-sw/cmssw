//-------------------------------------------------
//
//   Class: DTConfig
//
//   Description: Configurable parameters and constants for Level1 Mu DT Trigger
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   23/X/02 Sara Vanini: AC1,AC2,ACH,ACL parameters added
//   12/XI/02 Sara Vanini: 4ST3 and 4RE3 parameters added instead of tmax
//   24/III/03 Sara Vanini: traco geometry configuration parameters added
//   10/III/04 Stefano Marcellini: sector collector and TSM back up mode added
//   22/VI/04 SV: last trigger code update
//   17/VI/05 SV: bti mask in traco 
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTUtilities/interface/DTConfig.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/DTUtilities/interface/DTParameterValue.h"
#include "VisFramework/VisUtilities/interface/SimpleConfigurable.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"
// #include "CARF/G3Interface/interface/CMSIMHead.h"
// #include "Utilities/Notification/interface/TimingReport.h"

//---------------
// C++ Headers --
//---------------
#include <cstdlib>
#include <string>  
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iomanip>
               
//----------------
// Constructors --
//----------------
DTConfig::DTConfig() { 

  /*
  // reserve the appropriate ammount of space for parameters vectors
  intParam_.reserve(100);
  floatParam_.reserve(5);
  */

  for(int isl=0;isl<3;isl++){
    for(int ic=0;ic<72;ic++){
      LLvec[ic][isl]=0;
      LHvec[ic][isl]=64;
      CLvec[ic][isl]=0;
      CHvec[ic][isl]=64;
      RLvec[ic][isl]=0;
      RHvec[ic][isl]=64;
    }
  }


  createParametersGeneral();
  createParametersBTI();
  createParametersTRACO();
  createParametersTS();
  createParametersBTInew();    //SV BTI parameters added
  createParametersTRACOnew();  //SV TRACO parameters added
  createParametersSC();       //SM Sector Collector parameters (S. Marcellini)
  readParameters();

//pz   if( trigSetupGeom() ){
//pz     //input acceptance of bti's from hardware setup file
//pz     std::string name = SimpleConfigurable<std::string>( "setup",
//pz                 "Trigger:DTConfig:SetupFileName" );
//pz     inputBtiAccep(name.c_str());
//pz   }
}

//--------------
// Destructor --
//--------------
DTConfig::~DTConfig() {}

//--------------
// Operations --
//--------------

void
DTConfig::createParametersGeneral() {

  // create all the parameters and store them in the vectors
  DTParameter par;

  // Debug flag - 0
  par.setName("dttrigdbg","Debugging level");
  par.addValidParam("no_debugging"                ,0);
  par.addValidParam("input_to_MTTF"               ,1);
  par.addValidParam("BTI_TRACO_TS"                ,2);
  par.addValidParam("fullBTI"                     ,3);
  par.addValidParam("fullTRACO"                   ,4);
  par.addValidParam("fullTS"                      ,5);
  addParam(par);
  par.clear();

  // max drift time in 25ns steps - 1                          (-XDTBX(1))
  par.setName("tmax","Max drift time in 25 ns steps");
  par.addValidParam("Numerical Value",16);
  addParam(par);
  par.clear();
}
  
void
DTConfig::createParametersBTI() {

  // create all the parameters and store them in the vectors
  DTParameter par;

  // Time indep. K equation suppression (XON)  - 2                     (XMTFL(3))
  par.setName("timindkeqsupp","Time ind. K-eq. supp.");
  par.addValidParam("Theta_and_phi",2);
  par.addValidParam("Theta_only"   ,1);
  par.addValidParam("Not_enabled"  ,0);
  addParam(par);
  par.clear();

  // superlayer LTS flag - 3                             (MOD(XMTFL(4),10))
  par.setName("sllts","LTS, adjacent LTRIG supp. in SL");
  par.addValidParam("Theta_and_phi"   ,3);
  par.addValidParam("Not_enabled"     ,0);
  par.addValidParam("Phi_only"        ,1);
  par.addValidParam("Theta_only"      ,2);
  addParam(par);
  par.clear();

  // side LTS flag - 4                                  (INT(XMTFL(4)/10-2)
  // NB low mean before bx, high means after bx! SV
  par.setName("sidelts","LTS, suppression at side");
  par.addValidParam("Both"                    ,2-2);
  par.addValidParam("Not_enabled"             ,0);
  par.addValidParam("Low"                     ,1-2);
  par.addValidParam("High"                    ,3-2);
  par.addValidParam("Low(phi)_and_high(theta)",4-2);
  addParam(par);
  par.clear();

  // number of bx suppressed at high side with LTS for phi SL - 5 (XMTFL(5))
  par.setName("nbxltsphi","LTS, n. of BX supp. at high side, phi SL");
  par.addValidParam("Numerical Value",8);
  addParam(par);
  par.clear();

  // number of bx suppressed at high side with LTS for theta SL - 6 (XMTFL(5))
  par.setName("nbxltstheta","LTS, n. of BX supp. at high side, theta SL");
  par.addValidParam("Numerical Value",8);
  addParam(par);
  par.clear();

  // suppr. of LTRIG in BTI adj. to HTRIG - 7                  (! XMTFL(7))
  par.setName("adjbtilts","LTS on BTI adjacent to a BTI with HTRIG");
  par.addValidParam("Enabled",    0);
  par.addValidParam("Not_enabled",1);
  addParam(par);
  par.clear();

  // BTI setup time - 8                                          (XMTCT(1))
  par.setName("setuptime","BTI setup time");
  par.addValidParam("Numerical Value",0.);
  addParam(par);
  par.clear();

  // large angle BTI corr - 9                                    (XMTCT(2))
  par.setName("langbticorr","Large angle BTIcorrection");
  par.addValidParam("Numerical Value",0.0);
  addParam(par);
  par.clear();

  // Max K param accepted in phi view - 10          (XMTCT(3) and XMTCT(7))
  par.setName("kcutphi","Maximum K-parameter accepted in phi view");
  par.addValidParam("Numerical Value",64);
  addParam(par);
  par.clear();

  // Max K param accepted in theta view - 11        (XMTCT(3) and XMTCT(7))
  par.setName("kcuttheta","Maximum K-parameter accepted in theta view");
  par.addValidParam("Numerical Value",64);
  addParam(par);
  par.clear();

  // acceptance pattern A - 12                                   (XMTCT(4))
  par.setName("accpatta","Acceptance pattern A");
  par.addValidParam("Numerical Value",2);
  addParam(par);
  par.clear();

  // acceptance pattern B - 13                                   (XMTCT(5))
  par.setName("accpattb","Acceptance pattern B");
  par.addValidParam("Numerical Value",1);
  addParam(par);
  par.clear();


  // BTI angular acceptance in theta view - 14                   (XMTCT(8))
  par.setName("kacctheta","BTI angular acceptance in theta view");
  par.addValidParam("Numerical Value",1);
  addParam(par);
  par.clear();

  // bending angle cut in all stations - 15 (XMTCT(10))
  // SV fix X/03
  par.setName("bendinganglecut","Bending angle cut in chambers");
  par.addValidParam("Numerical Value",255);
  par.addValidParam("Not_enabled",-1);
  // BTIParams_[16]=sel.toInt()+1;
  addParam(par);
  par.clear();

}
  
void
DTConfig::createParametersTRACO() {

  // create all the parameters and store them in the vectors
  DTParameter par;

  // ascend. order for K sorting first tracks - 16            (! XMTCR(1))
  par.setName("sortkascend1","Sorting order for K, first tracks");
  par.addValidParam("Ascending", 0);
  par.addValidParam("Descending",1);
  addParam(par);
  par.clear();

  // ascend. order for K sorting second tracks - 17            (! XMTCR(1))
  par.setName("sortkascend2","Sorting order for K, second tracks");
  par.addValidParam("Ascending", 0);
  par.addValidParam("Descending",1);
  addParam(par);
  par.clear();

  // preference to HTRIG on first tracks - 18                     (XMTCR(2))
  par.setName("prefhtrig1","Preference to HTRIG, first tracks");
  par.addValidParam("Enabled",    1);
  par.addValidParam("Not_enabled",0);
  addParam(par);
  par.clear();

  // preference to HTRIG on second tracks -19                     (XMTCR(2))
  par.setName("prefhtrig2","Preference to HTRIG, second tracks");
  par.addValidParam("Enabled",    1);
  par.addValidParam("Not_enabled",0);
  addParam(par);
  par.clear();

  // single HTRIG enabling on first tracks - 20                   (XMTCR(3))
  par.setName("singlehflag1","Single HTRIG, first tracks");
  par.addValidParam("Always_accepted",            0);
  par.addValidParam("Only_with_theta_coincidence",1);
  addParam(par);
  par.clear();

  // single HTRIG enabling on second tracks - 21                   (XMTCR(3))
  par.setName("singlehflag2","Single HTRIG, second tracks");
  par.addValidParam("Always_accepted",            0);
  par.addValidParam("Only_with_theta_coincidence",1);
  addParam(par);
  par.clear();

  // single LTRIG enabling on first tracks - 22                  (XMTCR(4))
  // single LTRIG accept enabling                                (XMTCR(8))
  //@@ to be checked: find a better way to set this parameter
  par.setName("singlelflag1","Single LTRIG, first tracks");
  par.addValidParam("Only_with_theta_coincidence", 0);
  par.addValidParam("Never_accepted",              1);
  par.addValidParam("Only_with_theta_HTRIG_coinc.",2);  //not in hardware! SV
  par.addValidParam("Always_accepted",             3);
  addParam(par);
  par.clear();

  // single LTRIG enabling on second tracks - 23                 (XMTCR(4))
  // single LTRIG accept enabling                                (XMTCR(8))
  //@@ to be checked: find a better way to set this parameter
  par.setName("singlelflag2","Single LTRIG, second tracks");
  par.addValidParam("Only_with_theta_coincidence", 0);
  par.addValidParam("Never_accepted",              1);
  par.addValidParam("Only_with_theta_HTRIG_coinc.",2); //not in harware! SV
  par.addValidParam("Always_accepted",             3);
  addParam(par);
  par.clear();

  // preference to inner on first tracks - 24                   (!XMTCR(5))
  par.setName("prefinner1","Single trigger preference to SL, first tracks");
  par.addValidParam("Inner", 0);
  par.addValidParam("Outer", 1);
  addParam(par);
  par.clear();

  // preference to inner on second tracks - 25                   (!XMTCR(5))
  par.setName("prefinner2","Single trigger preference to SL, second tracks");
  par.addValidParam("Inner", 0);
  par.addValidParam("Outer", 1);
  addParam(par);
  par.clear();

  // K tollerance for corr. in TRACO for first tracks ( XMTCR(6) and XMTCR(7))
  //                                                  - 26
  par.setName("tcktoll1","K tollerance for correlation, first tracks");
  par.addValidParam("Numerical Value",2);
  addParam(par);
  par.clear();

  // K tollerance for corr. in TRACO for second tracks ( XMTCR(6) and XMTCR(7))
  //                                                   - 27
  par.setName("tcktoll2","K tollerance for correlation, second tracks");
  par.addValidParam("Numerical Value",2);
  addParam(par);
  par.clear();

  // suppr. of LTRIG in 4 BX before HTRIG - 28                   (XMTCR(9))
  par.setName("tcbxlts","Single LTRIG suppression in 4 BX low side");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();
  
  // recycling of TRACO cand. in inner SL -29                   ( XMTFL(8))
  par.setName("tcreuse1","Recycling of candidates in inner SL");
  par.addValidParam("Enabled",    1);
  par.addValidParam("Not_enabled",0);
  addParam(par);
  par.clear();
  
  // recycling of TRACO cand. in outer SL - 30                   ( XMTFL(8))
  par.setName("tcreuse2","Recycling of candidates in outer SL");
  par.addValidParam("Enabled",    1);
  par.addValidParam("Not_enabled",0);
  addParam(par);
  par.clear();

  // maximum number of TRACO output candidates - 31 (not present in FORTRAN)
  par.setName("nmaxoutcand","Number of tracks passed to TS");
  par.addValidParam("Numerical Value",2);
  addParam(par);
  par.clear();

}
  
void
DTConfig::createParametersTS() {

  // create all the parameters and store them in the vectors
  DTParameter par;

 // Order of quality bits in TSS for sort1 - 32   (not present in FORTRAN)
  par.setName("tssmasking1","Priority in TSS for first track selection");
  par.addValidParam("Corr/NotC,H/L,In/Out", 312);
  par.addValidParam("H/L,In/Out,Corr/NotC", 123);
  par.addValidParam("H/L,Corr/NotC,In/Out", 132);
  par.addValidParam("In/Out,H/L,Corr/NotC", 213);
  par.addValidParam("In/Out,Corr/NotC,H/L", 231);
  par.addValidParam("Corr/NotC,In/Out,H/L", 321);
  addParam(par);
  par.clear();

 // Order of quality bits in TSS for sort2 - 33    (not present in FORTRAN)
  par.setName("tssmasking2","Priority in TSS for second track selection");
  par.addValidParam("Corr/NotC,H/L,In/Out", 312);
  par.addValidParam("H/L,In/Out,Corr/NotC", 123);
  par.addValidParam("H/L,Corr/NotC,In/Out", 132);
  par.addValidParam("In/Out,H/L,Corr/NotC", 213);
  par.addValidParam("In/Out,Corr/NotC,H/L", 231);
  par.addValidParam("Corr/NotC,In/Out,H/L", 321);
  addParam(par);
  par.clear();

  // enable Htrig checking in TSS for sort1 - 34  (not present in FORTRAN)
  par.setName("tsshtrigena1","Pref. to HTRIG in TSS for first track selection");
  par.addValidParam("Enabled",    1);
  par.addValidParam("Not_enabled",0);
  addParam(par);
  par.clear();
  
  // enable Htrig checking in TSS for sort2 - 35 (not present in FORTRAN)
  par.setName("tsshtrigena2","Pref. to HTRIG in TSS for second track selection");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();
  
  // enable Htrig checking in TSS for carry - 36 (not present in FORTRAN)
  par.setName("tsshtrigenac","Pref. to HTRIG in TSS for carry");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();
  
  // enable Inner SL checking in TSS for sort1 - 37 (not present in FORTRAN)
  par.setName("tssinoutena1","Pref. to Inner in TSS for first track selection");
  par.addValidParam("Enabled",    1);
  par.addValidParam("Not_enabled",0);
  addParam(par);
  par.clear();
  
  // enable Inner SL checking in TSS for sort2 -38 (not present in FORTRAN)
  par.setName("tssinoutena2","Pref. to Inner in TSS for second track selection");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();
  
  // enable Inner SL checking in TSS for carry -39 (not present in FORTRAN)
  par.setName("tssinoutenac","Pref. to Inner in TSS for carry");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();

  // enable Correlation checking in TSS for sort1 - 40 (not present in FORTRAN)
  par.setName("tsscorrena1","Pref. to Corr. in TSS for first track selection");
  par.addValidParam("Enabled",    1);
  par.addValidParam("Not_enabled",0);
  addParam(par);
  par.clear();
  
  // enable Correlation checking in TSS for sort2 - 41 (not present in FORTRAN)
  par.setName("tsscorrena2","Pref. to Corr. in TSS for second track selection");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();
  
  // enable Correlation checking in TSS for carry - 42 (not present in FORTRAN)
  par.setName("tsscorrenac","Pref. to Corr. in TSS for carry");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();

 // Order of quality bits in TSM for sort1 - 43       (not present in FORTRAN)
  par.setName("tsmmasking1","Priority in TSM for first track selection");
  par.addValidParam("Corr/NotC,H/L,In/Out", 312);
  par.addValidParam("H/L,In/Out,Corr/NotC", 123);
  par.addValidParam("H/L,Corr/NotC,In/Out", 132);
  par.addValidParam("In/Out,H/L,Corr/NotC", 213);
  par.addValidParam("In/Out,Corr/NotC,H/L", 231);
  par.addValidParam("Corr/NotC,In/Out,H/L", 321);
  addParam(par);
  par.clear();

 // Order of quality bits in TSM for sort2 - 44   (not present in FORTRAN)
  par.setName("tsmmasking2","Priority in TSM for second track selection");
  par.addValidParam("Corr/NotC,H/L,In/Out", 312);
  par.addValidParam("H/L,In/Out,Corr/NotC", 123);
  par.addValidParam("H/L,Corr/NotC,In/Out", 132);
  par.addValidParam("In/Out,H/L,Corr/NotC", 213);
  par.addValidParam("In/Out,Corr/NotC,H/L", 231);
  par.addValidParam("Corr/NotC,In/Out,H/L", 321);
  addParam(par);
  par.clear();

  // enable Htrig checking in TSM for sort1 - 45 (not present in FORTRAN)
  par.setName("tsmhtrigena1","Pref. to HTRIG in TSM for first track selection");
  par.addValidParam("Enabled",    1);
  par.addValidParam("Not_enabled",0);
  addParam(par);
  par.clear();
  
  // enable Htrig checking in TSM for sort2 - 46 (not present in FORTRAN)
  par.setName("tsmhtrigena2","Pref. to HTRIG in TSM for second track selection");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();
  
  // enable Htrig checking in TSM for carry - 47 (not present in FORTRAN)
  par.setName("tsmhtrigenac","Pref. to HTRIG in TSM for carry");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();
  
  // enable Inner SL checking in TSM for sort1 - 48 (not present in FORTRAN)
  par.setName("tsminoutena1","Pref. to Inner in TSM for first track selection");
  par.addValidParam("Enabled",    1);
  par.addValidParam("Not_enabled",0);
  addParam(par);
  par.clear();
  
  // enable Inner SL checking in TSM for sort2 - 49 (not present in FORTRAN)
  par.setName("tsminoutena2","Pref. to Inner in TSM for second track selection");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();
  
  // enable Inner SL checking in TSM for carry - 50 (not present in FORTRAN)
  par.setName("tsminoutenac","Pref. to Inner in TSM for carry");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();
  

  // enable Correlation checking in TSM for sort1 - 51 (not present in FORTRAN)
  par.setName("tsmcorrena1","Pref. to Corr. in TSM for first track selection");
  par.addValidParam("Enabled",    1);
  par.addValidParam("Not_enabled",0);
  addParam(par);
  par.clear();
  
  // enable Correlation checking in TSM for sort2 - 52 (not present in FORTRAN)
  par.setName("tsmcorrena2","Pref. to Corr. in TSM for second track selection");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();
  
  // enable Correlation checking in TSM for carry - 53 (not present in FORTRAN)
  par.setName("tsmcorrenac","Pref. to Corr. in TSM for carry");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();

  // ghost 1 suppression option in TSS - 54         (XMTSV(8) and XMTSV(9))
  par.setName("tssghost1flag","Carry suppression in TSS (Ghost1)");
  par.addValidParam("If_Outer_adj_to_1st_tr", 1);
  par.addValidParam("Never",0);
  par.addValidParam("Always",2);
  addParam(par);
  par.clear();

  // ghost 2 suppression option in TSS - 55         (XMTSV(8) and XMTSV(9))
  par.setName("tssghost2flag","2nd track suppression in TSS (Ghost2)");
  par.addValidParam("If_Outer_same_TRACO_of_uncorr_1st_tr", 1);
  par.addValidParam("If_Outer_same_TRACO_of_1st_tr",2);
  par.addValidParam("Always",3);
  par.addValidParam("If_Outer_same_TRACO_of_inner_1st_tr",4);
  par.addValidParam("Never",0);
  addParam(par);
  par.clear();

  // ghost 1 suppression option in TSM - 56         (XMTSV(8) and XMTSV(9))
  par.setName("tsmghost1flag","Carry suppression in TSM (Ghost1)");
  par.addValidParam("If_Outer_adj_to_1st_tr", 1);
  par.addValidParam("Never",0);
  par.addValidParam("Always",2);
  addParam(par);
  par.clear();

  // ghost 2 suppression option in TSM - 57         (XMTSV(8) and XMTSV(9))
  par.setName("tsmghost2flag","2nd track suppression in TSM (Ghost2)");
  par.addValidParam("If_Outer_same_TRACO_of_uncorr_1st_tr", 1);
  par.addValidParam("If_Outer_same_TRACO_of_1st_tr",2);
  par.addValidParam("Always",3);
  par.addValidParam("If_Outer_same_TRACO_of_inner_1st_tr",4);
  par.addValidParam("Never",0);
  addParam(par);
  par.clear();

  // correlated ghost 1 suppression option in TSS - 58
  par.setName("tssghost1corr","correlated carry suppression in TSS (Ghost1)");
  par.addValidParam("Accepted_if_correlated",1);
  par.addValidParam("Rejected_also_if_correlated",0);
  addParam(par);
  par.clear();

  // correlated ghost 2 suppression option in TSS - 59
  par.setName("tssghost2corr","correlated carry suppression in TSS (Ghost2)");
  par.addValidParam("Accepted_if_correlated",1);
  par.addValidParam("Rejected_also_if_correlated",0);
  addParam(par);
  par.clear();

  // correlated ghost 1 suppression option in TSM - 60
  par.setName("tsmghost1corr","correlated carry suppression in TSM (Ghost1)");
  par.addValidParam("Accepted_if_correlated",1);
  par.addValidParam("Rejected_also_if_correlated",0);
  addParam(par);
  par.clear();

  // correlated ghost 2 suppression option in TSM - 61
  par.setName("tsmghost2corr","correlated carry suppression in TSM (Ghost2)");
  par.addValidParam("Accepted_if_correlated",1);
  par.addValidParam("Rejected_also_if_correlated",0);
  addParam(par);
  par.clear();

  // Handling of second track (carry) in case of pile-up, in TSM - 62
  par.setName("tsmgetcarryflag","second track handling in case of pile-up in TSM");
  par.addValidParam("get_best_2ndprevBX_1st",1);
  par.addValidParam("get_best_2ndprevBX_if_1stisLow",2);
  par.addValidParam("Rejected_2ndtrack",0);
  addParam(par);
  par.clear();

}


void
DTConfig::createParametersBTInew() {

  // create all the parameters and store them in the vectors
  DTParameter par;

  // acceptance pattern AC1  -  63
  par.setName("accpattac1","Acceptance pattern AC1");
  par.addValidParam("Numerical Value",0);
  addParam(par);
  par.clear();

  // acceptance pattern AC2  -  64
  par.setName("accpattac2","Acceptance pattern AC2");
  par.addValidParam("Numerical Value",3);
  addParam(par);
  par.clear();

  // acceptance pattern ACH  -  65
  par.setName("accpattach","Acceptance pattern ACH");
  par.addValidParam("Numerical Value",1);
  addParam(par);
  par.clear();

  // acceptance pattern ACL  -  66
  par.setName("accpattacl","Acceptance pattern ACL");
  par.addValidParam("Numerical Value",2);
  addParam(par);
  par.clear();

  // redundant patterns flag RON -  67
  par.setName("RON","Redundant patterns flag RON");
  par.addValidParam("Numerical Value",1);
  addParam(par);
  par.clear();

  // pattern mask flag PTMS - from 68 to 99
  std::string lab = "PTMS";
  std::string nam = "Mask flag PTMS for pattern ";
  for(int patt=0; patt<32; patt++){
   char patt0 = (patt/10)+'0';
   char patt1 = (patt%10)+'0';
   std::string label = lab;
   std::string name  = nam;
   if ( patt0 != '0' ) {
     label = label + patt0;
     name  = name  + patt0;
   }
   label = label + patt1;
   name  = name  + patt1;
   par.setName(label,name);
   if(patt==0 || patt==1 || patt==2 || patt==3 || patt==30 || patt==31)
     par.addValidParam("Numerical Value",0);
   else
     par.addValidParam("Numerical Value",1);
   addParam(par);
   par.clear();
  }




  // wire mask flag WEN - from 100 to 108
  std::string wlab = "WEN";
  std::string wnam = "Mask flag WEN for wire ";
  for(int wire=1; wire<10; wire++){
   char wires = wire+'0';
   std::string wlabel = wlab + wires;
   std::string wname = wnam + wires;
   par.setName(wlabel,wname);
   par.addValidParam("Numerical Value",1);
   addParam(par);
   par.clear();
  }

  // angular window limits for left traco: LL - 109
  par.setName("anglimLL","Angular limit for left traco: LL");
  par.addValidParam("Numerical Value",2);
  addParam(par);
  par.clear();
  
  // angular window limits for left traco: LH - 110
  par.setName("anglimLH","Angular limit for left traco: LH");
  par.addValidParam("Numerical Value",21);
  addParam(par);
  par.clear();
    
  // angular window limits for center traco: CL - 111
  par.setName("anglimCL","Angular limit for center traco: CL");
  par.addValidParam("Numerical Value",22);
  addParam(par);
  par.clear();
  
  // angular window limits for center traco: CH - 112
  par.setName("anglimCH","Angular limit for center traco: CH");
  par.addValidParam("Numerical Value",41);
  addParam(par);
  par.clear();
    
  // angular window limits for right traco: RL - 113
  par.setName("anglimRL","Angular limit for right traco: RL");
  par.addValidParam("Numerical Value",42);
  addParam(par);
  par.clear();
  
  // angular window limits for right traco: RH - 114
  par.setName("anglimRH","Angular limit for right traco: RH");
  par.addValidParam("Numerical Value",61);
  addParam(par);
  par.clear();

  // drift velocity parameter ST - 115
  par.setName("4ST3","Drift velocity parameter ST");
  par.addValidParam("Numerical Value",42);
  addParam(par);
  par.clear();

  // drift velocity parameter RE - 116
  par.setName("4RE3","Drift velocity parameter RE");
  par.addValidParam("Numerical Value",2);
  addParam(par);
  par.clear();

  // wire DEAD time parameter - 117
  par.setName("DEAD","Wire DEAD time parameter");
  par.addValidParam("Numerical Value",31);
  addParam(par);
  par.clear();

}

void
DTConfig::createParametersTRACOnew() {

  // create all the parameters and store them in the vectors
  DTParameter par;

  // flag for L acceptance - 118
  par.setName("LVALIDIFH","L if H flag");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();

  // KRAD parameter - 119
  // SV: always rad=0 for hardware bug
  par.setName("RAD","K parameter of radial tracks");
  par.addValidParam("Numerical Value",0);
  addParam(par);
  par.clear();

  // BTIC parameter - 120
  par.setName("BTIC","Maximum drift time of bti - ST");
  par.addValidParam("Numerical Value",32);
  addParam(par);
  par.clear();

  // IBTIOFF parameter - 121
  par.setName("IBTIOFF","First superlayer addition shift");
  par.addValidParam("Numerical Value",0);
  addParam(par);
  par.clear();

  // DD parameter - 122
  par.setName("DD","Ratio between distance of superlayers and distance of wires");
  par.addValidParam("Numerical Value",18);
  addParam(par);
  par.clear();

  // ltf flag : in logic or with theta, first and second track - 123
  par.setName("ltf","Force low trigs: in logic or with theta bit");
  par.addValidParam("Not_enabled",0);
  par.addValidParam("Enabled",    1);
  addParam(par);
  par.clear();

 // output from a single traco only   - 124
 par.setName("usedTraco","Single Traco to be used for debug ");
 par.addValidParam("All tracos",-1);
 par.addValidParam("Numerical Value",0xFFF0);   //mask for connected tracos
 addParam(par);
 par.clear();


 // orca geometry or trigger parametrs    - 125
 par.setName("triggerGeometry","Geometry parameters");
 par.addValidParam("orca",0);
 par.addValidParam("trigger_setup",1);
 par.addValidParam("trigger_setup_TB04",2);
 addParam(par);
 par.clear();



}

void
DTConfig::createParametersSC() {

  // create all the parameters and store them in the vectors
  DTParameter par;

  //  Enabling Carry in Sector Collector (1 means enabled, 0 disabled) - 126
  par.setName("scgetcarryflag","enabling carry in SC");
  par.addValidParam("carry_disabled",0);
  par.addValidParam("carry_enabled",1);
  addParam(par);
  par.clear();

  //SV only for Testbeam 2004 purpose....
  //Setuptime for MB1   - 127
  par.setName("setuptimeMB1","MB1 station bti setup time");
  par.addValidParam("Numerical Value",0);
  addParam(par);
  par.clear();
  //SetupTime for MB3   - 128
  par.setName("setuptimeMB3","MB1 station bti setup time");
  par.addValidParam("Numerical Value",0);
  addParam(par);
  par.clear();
  //SetupTime for theta   - 129
  par.setName("setuptimeTHETA","Theta superlayer bti setup time");
  par.addValidParam("Numerical Value",0);
  addParam(par);
  par.clear();
  // end testbeam code

 // connected btis in traco   - 130
 par.setName("usedBtiInTraco","Connected btis in traco ");
 par.addValidParam("All btis",-1);
 par.addValidParam("Numerical Value",0xFFFF);   //mask for connected btis
 addParam(par);
 par.clear();


}


void 
DTConfig::readParameters() {

  // Loop on parameters:
  ParamIterator pi;
  for(pi=param_.begin();pi!=param_.end();pi++) {
    if((*pi).currentMeaning()=="Undefined") {
      // if the parameter doesn't have any option, skip it!
      std::cout << "DTConfig::readParameters: parameter " << (*pi).name();
      std::cout << " doesn't have any valid option!!!" << std::endl;
    } else {
      //std::cout<<"(*pi).currentMeaning()="<<(*pi).currentMeaning()<<std::endl;
      //std::cout<<"(*pi).label()="<<(*pi).label()<<std::endl;
      //std::cout<<"(*pi).name()="<<(*pi).name()<<std::endl;
      //std::cout<<"(*pi).currentValue()="<<(*pi).currentValue()<<std::endl;
      // Note that if the default is a numerical value then:
      //    param((*pf).currentMeaning()=="Numerical Value"
      //    if in .orcarc the parameter is not found then
      //       (*pf) = param.value()  doesn't do anything
//pz      SimpleConfigurable<std::string> 
//pz        param((*pi).currentMeaning(),"L1DTTrigger:"+(*pi).label());
//pz      (*pi) = param.value();
    }
  }

  // print out configuration if any debug option is chosen
  if(debug())print();
}

void 
DTConfig::print() const {
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*              DTTrigger configuration                                   *" << std::endl;
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*                                                                            *" << std::endl;

  ParamConstIterator pi;
  for(pi=param_.begin();pi!=param_.end();pi++) {
    std::cout << "* " << (*pi).name();
    std::cout << ": " << (*pi).currentMeaning();
    std::cout << " (" << (*pi).currentValue() << ")" << std::endl;
  }


  std::cout << "******************************************************************************" << std::endl;

}

BitArray<8>
DTConfig::TsmStatus(int stat, int sect, int whee) {

  // TsmWord:
  //  bit numbering 7 6 5 4 3 2 1 0
  //
  //  bit 0 = 1  --> TSMS OK     => normal mode (default)
  //  bit 0 = 0  --> TSMS broken => back-up mode (see example a)
  //  bits 1-6 = 0 --> broken TSS (see example b)
  //  bits 1-6 = 1 --> working TSS (default)
  
  _TsmWord.one();
  //------------------------------------------------------------
  // Example a): Set TSMS broken (back up mode option for TSM)
  //if(stat==1 || stat==2 || stat==3 || stat==4) {  // set back-up mode on by hand choosing station, wheel sector of where the failure occurs
    //
    // _TsmWord.unset(0);}
  //------------------------------------------------------------ 
  // Example b): Set a TSS with a failure.
  // In this case essentially half a chamber is blind, for what concerns the DT trigger.
  //
  //   if(stat==1 || stat==2 || stat==3 ||  stat==4) { //chose where the failure occurs
  //_TsmWord.unset(1);    // these locations depend on where the failure occurs,
  //_TsmWord.unset(2);    // as the connections to the TSSs varies from chamber to chamber.
  //_TsmWord.unset(3);    // it could be 1,2,3, or 4,5,6, it depends where the failure occurs
  //       _TsmWord.unset(4);    
  //       _TsmWord.unset(5);    
  //       _TsmWord.unset(6);    
  // }
  // -----------------------------
  
  
  return _TsmWord;
  
}
// }

int 
DTConfig::TSSinTSMD(int stat, int sect) {

  // Number of TSS for each TSMD (it changes from station to station) The DT stations are indicated in parenthesis
  // in the DT column.
  //
  //      MB                    nb.TSS        nb.TTS per TSMD
  //      1                       3             2   
  //      2                       4             2  
  //      3                       5             3  
  //      4(1,2,3,5,6,7)          6             3   
  //      4(8,12)                 6             3   
  //      4(9,11)                 3             2     
  //      4(4L)                   5             3    
  //      4(4R)                   5             3    
  //      4(10L)                  4             2     
  //      4(10R)                  4             2     
  
  if( stat==1 ||
      stat==2 ||
      ( stat==4 && (sect==9 || sect==11 ||
		    sect==10))) {
    _ntsstsmd = 2;
  } else {
    _ntsstsmd = 3;
  }

  return _ntsstsmd;
}

void 
DTConfig::setParam(std::string name, std::string val) {

  std::cout << "DTConfig::setParam: superseding value of " << name;

  // loop on available parameters and check parameter name...
  ParamIterator pi;
  for(pi=param_.begin();pi!=param_.end();pi++) {
    if((*pi).name()==name) {
      (*pi)=val;
      std::cout << " to " << (*pi).currentMeaning();
      std::cout << " (" << (*pi).currentValue() << ")" << std::endl;
      return;
    }
  }
  std::cout << " parameter not found!" << std::endl;

}


void 
DTConfig::setParamValue(std::string name, std::string valName, float val) {

  std::cout << "DTConfig::setParam: superseding value of " << name;

  // loop on available parameters and check parameter name...
  ParamIterator pi;
  for(pi=param_.begin();pi!=param_.end();pi++) {
    if((*pi).name()==name) {
      (*pi).addValidParam(valName,val); 
      (*pi) = valName;
      std::cout << " to " << (*pi).currentMeaning();
      std::cout << " (" << (*pi).currentValue() << ")" << std::endl;
      return;
    }
  }
  std::cout << " parameter not found!" << std::endl;

}




void 
DTConfig::addParam(DTParameter par) {
  param_.push_back(par);
}

//  std::string
//  DTConfig::paramMeaning(unsigned index) const {
//    if(index>param_.size()){
//      std::cout << "DTConfig::paramValue: parameter not found: " << index << std::endl;
//      return "Undefined";
//    }
//    return param_[index].currentMeaning();
//  }

//  double 
//  DTConfig::paramValue(unsigned index) const {
//    //  TimeMe t("DTConfig::paramValue");
//    if(index>param_.size()){
//      std::cout << "DTConfig::paramValue: parameter not found: " << index << std::endl;
//      return 0;
//    }
//    return param_[index].currentValue();
//  }

void 
DTConfig::inputBtiAccep(const char* setupfile) {
  std::ifstream filein(setupfile);
  if (! filein.is_open()){
    std::cout << "Error opening file " << setupfile << std::endl; 
    exit (1); 
  }
 
  //SV TB2003: NB first trigger board isn't connected, start from 1... 
  for(int tb=1;tb<4;tb++){
    unsigned short int id, board, chip_num, traco_num, tss_num;
    unsigned short int memory[38];
    unsigned short int memory_tss[27];
    unsigned short int memory_bti[31];

      for(int traco=0;traco<4;traco++){
        for(int bti=0;bti<4;bti++){
          filein >> std::hex >> id >> board >> chip_num;
            //if(id==0x14)
            //std::cout << "BTI " << chip_num << " in board " << board << "-->";
              for(int i=0;i<31;i++){
                filein >> std::hex >> memory_bti[i];
                //std::cout << std::hex << memory_bti[i] << "  ";
              }
              int st43 = memory_bti[0] & 0x3f;
    	      int re43 = (memory_bti[1] >> 5) & 0x03;
              int dead = memory_bti[3] & 0x3F;
              int LH =  memory_bti[4] & 0x3F;
              int LL =  memory_bti[5] & 0x3F;
              int CH =  memory_bti[6] & 0x3F;
              int CL =  memory_bti[7] & 0x3F;
              int RH =  memory_bti[8] & 0x3F;
              int RL =  memory_bti[9] & 0x3F;
              int tston = ( memory_bti[10] & 0x20 ) != 0 ;
              int test = ( memory_bti[10] & 0x10 ) != 0 ;
              int ten = ( memory_bti[10] & 0x8 ) != 0 ;
              int xon = ( memory_bti[10] & 0x2 ) != 0 ;
              int ron = ( memory_bti[10] & 0x1 ) != 0 ;
              int set = ( memory_bti[11] & 0x38 ) >> 3 ;
              int lts = ( memory_bti[11] & 0x6 ) >> 1 ;
              int ac1 = ( memory_bti[12] & 0x30 ) >> 4 ;
              int ac2 = ( memory_bti[12] & 0xc ) >> 2 ;
              int acl = ( memory_bti[12] & 0x3 ) ;
              int ach = ( memory_bti[13] & 0x30 ) >> 4 ;

    
              if(debug()>3)
               std::cout << std::dec << "st43=" << st43 
        	<< " re43=" << re43 
       		<< " dead=" << dead 
       		<< " LH=" << LH 
       		<< " LL=" << LL 
       		<< " CH=" << CH 
       		<< " CL=" << CL 
       		<< " RH=" << RH 
       		<< " RL=" << RL 
                << " tston=" << tston
                << " test=" << test 
                << " ten=" << ten
                << " xon=" << xon
                << " ron=" << ron
                << " set=" << set 
                << " lts=" << lts
                << " ac1=" << ac1
                << " ac2=" << ac2
                << " acl=" << acl
                << " ach=" << ach
                << std::endl; 
	     
              //SV TESTBEAM 2003: ATTENTION: THE FOLLOWING DEPENDS ON TB HW CONFIGURATION!!! 
              int sl=3;
              int cell;
              if(traco==0)
                cell = chip_num+1 + (board+1)*16 - 4;
              if(traco!=0)
                cell= static_cast<int>(traco*4 + 1 + fmod(float(chip_num),float(std::max(traco*8,1))) + (board+1)*16) - 4;
              //END TESTBEAM 2003

              if(debug()>2)
                 std::cout << "Setting K acceptance for cell " << cell << " sl " << sl << std::endl;
              setLH_bti(cell,sl,LH);
              setLL_bti(cell,sl,LL);
              setCH_bti(cell,sl,CH);
              setCL_bti(cell,sl,CL);
              setRH_bti(cell,sl,RH);
              setRL_bti(cell,sl,RL);
            }


            filein >> std::hex >> id >> board >> traco_num;
              if(id==0x15){
                //std::cout << "TRACO ";
                //std::cout << traco_num << "  in board " << board << std::endl; 
              }
            for(int i=0;i<38;i++){
              filein >> std::hex >> memory[i];
              //std::cout << std::hex << memory[i];
            }
            int btic = memory[0] & 0x3f ;
            int rad = ( ( memory[0] & 0xc0 ) >> 6 ) | ( ( memory[1] & 0x7 ) << 2 ) ;
            int dd = ( memory[1] & 0xf8 ) >> 3 ;
            int fprgcomp = memory[2] & 0x3 ;
            int sprgcomp = memory[3] & 0x3 ;
            int fhism = ( memory[2] & 0x4 ) != 0 ;
            int fhtprf = ( memory[2] & 0x8 ) != 0 ;
            int fslmsk = ( memory[2] & 0x10 ) != 0 ;
            int fltmsk = ( memory[2] & 0x20 ) != 0 ;
            int fhtmsk = ( memory[2] & 0x40 ) != 0 ;
            int shism = ( memory[3] & 0x4 ) != 0 ;
            int shtprf = ( memory[3] & 0x8 ) != 0 ;
            int sslmsk = ( memory[3] & 0x10 ) != 0 ;
            int sltmsk = ( memory[3] & 0x20 ) != 0 ;
            int shtmsk = ( memory[3] & 0x40 ) != 0 ;
            int reusei = ( memory[2] & 0x80 ) != 0 ;
            int reuseo = ( memory[3] & 0x80 ) != 0 ;
            int ltf = ( memory[4] & 1 ) != 0 ;
            int lts = ( memory[4] & 2 ) != 0 ;
            int prgdel = ( memory[4] & 0x1c ) >> 2 ;
            int snapcor = ( memory[4] & 0xe0 ) >> 5 ;
            int trgenb = ( memory[5] & 0xff ) | ( ( memory[6] & 0xff ) << 8 ) ;
            int trgadel = memory[7] & 0x3 ;
            int ibtioff = ( memory[7] & 0xfc ) >> 2 ;
            int kprgcom = ( memory[8] & 0xff ) ;
            int testmode = ( memory[9] & 1 ) != 0 ;
            int starttest = ( memory[9] & 2 ) != 0 ;
            int prvsignmux = ( memory[9] & 4 ) != 0 ;
            int lth = ( memory[9] & 8 ) != 0 ;

            if(debug()>3)
              std::cout 
              << "btic=" << btic
              << " rad=" << rad
              << " dd=" << dd
              << " fprgcomp=" << fprgcomp
              << " sprgcomp=" << sprgcomp
              << " fhism=" << fhism
              << " fhtprf=" << fhtprf
              << " fslmsk=" << fslmsk
             << " fltmsk=" << fltmsk 
             << " fhtmsk=" <<  fhtmsk
             << " shism=" <<  shism 
             << " shtprf=" <<  shtprf 
             << " sslmsk=" <<  sslmsk
             << " sltmsk=" <<  sltmsk 
             << " shtmsk=" <<  shtmsk 
             << " reusei=" <<  reusei 
             << " reuseo=" <<  reuseo
             << " ltf=" <<  ltf
             << " lts=" <<  lts
             << " prgdel=" <<  prgdel
             << " snapcor=" <<  snapcor
             << " trgenb=" <<  trgenb
             << " trgadel=" <<  trgadel
             << " ibtioff=" <<  ibtioff
             << " kprgcom=" <<  kprgcom
             << " testmode=" <<  testmode
             << " starttest=" <<  starttest
             << " prvsignmux=" <<  prvsignmux
             << " lth=" <<  lth << std::endl;

            for(int bti=0;bti<4;bti++){
              filein >> std::hex >> id >> board >> chip_num;
              //if(id==0x14)
                //std::cout << "BTI " << chip_num << " in board " << board << "-->";
              for(int i=0;i<31;i++){
                filein >> std::hex >> memory_bti[i];
                //std::cout << std::hex << memory_bti[i] << "  ";
              }

              int st43 = memory_bti[0] & 0x3f;
              int re43 = (memory_bti[1] >> 5) & 0x03;
              int dead = memory_bti[3] & 0x3F;
              int LH =  memory_bti[4] & 0x3F;
              int LL =  memory_bti[5] & 0x3F;
              int CH =  memory_bti[6] & 0x3F;
              int CL =  memory_bti[7] & 0x3F;
              int RH =  memory_bti[8] & 0x3F;
              int RL =  memory_bti[9] & 0x3F;
              int tston = ( memory_bti[10] & 0x20 ) != 0 ;
              int test = ( memory_bti[10] & 0x10 ) != 0 ;
              int ten = ( memory_bti[10] & 0x8 ) != 0 ;
              int xon = ( memory_bti[10] & 0x2 ) != 0 ;
              int ron = ( memory_bti[10] & 0x1 ) != 0 ;
              int set = ( memory_bti[11] & 0x38 ) >> 3 ;
              int lts = ( memory_bti[11] & 0x6 ) >> 1 ;
              int ac1 = ( memory_bti[12] & 0x30 ) >> 4 ;
              int ac2 = ( memory_bti[12] & 0xc ) >> 2 ;
              int acl = ( memory_bti[12] & 0x3 ) ;
              int ach = ( memory_bti[13] & 0x30 ) >> 4 ;

    
              if(debug()>3)
               std::cout << std::dec << "st43=" << st43 
        	<< " re43=" << re43 
       		<< " dead=" << dead 
       		<< " LH=" << LH 
       		<< " LL=" << LL 
       		<< " CH=" << CH 
       		<< " CL=" << CL 
       		<< " RH=" << RH 
       		<< " RL=" << RL 
                << " tston=" << tston
                << " test=" << test 
                << " ten=" << ten
                << " xon=" << xon
                << " ron=" << ron
                << " set=" << set 
                << " lts=" << lts
                << " ac1=" << ac1
                << " ac2=" << ac2
                << " acl=" << acl
                << " ach=" << ach
                << std::endl; 
	
              //SV TESTBEAM 2003: ATTENTION: THE FOLLOWING DEPENDS ON TB HW CONFIGURATION!!! 
              int sl=1;
              int cell;
              if(traco==0)
                cell = chip_num+1 + (board+1)*16;
              if(traco!=0)
                cell= static_cast<int>(traco*4 + 1 + fmod(float(chip_num),float(std::max(traco*8,1))) + (board+1)*16);
              //END TB2003
              if(debug()>2)
                 std::cout << "Setting K acceptance for cell " << cell << " sl " << sl << std::endl;

              setLH_bti(cell,sl,LH);
              setLL_bti(cell,sl,LL);
              setCH_bti(cell,sl,CH);
              setCL_bti(cell,sl,CL);
              setRH_bti(cell,sl,RH);
              setRL_bti(cell,sl,RL);
         }
       }

       filein >> std::hex >> id >> board >> tss_num;
       if(id==0x16){
         //std::cout << " TSS ";
         //std::cout << tss_num << "  in board " << board << "-->"; 
       }
 
       for(int ts=0;ts<26;ts++){
         filein >> std::hex >> memory_tss[ts];
         //std::cout << std::hex << memory_tss[ts] << " ";
       }
       
       //std::cout << std::endl;
     }

    //theta btis
    unsigned short int id,board,chip_num;
    unsigned short int memory_bti[31];
    for(int bti=0;bti<64;bti++){
      filein >> std::hex >> id >> board >> chip_num;
      //if(id==0x14)
        //std::cout << "BTI " << chip_num << " in board " << board << "-->";
      for(int i=0;i<31;i++){
       filein >> std::hex >> memory_bti[i];
       //std::cout << std::hex << memory_bti[i] << "  ";
      }

    int st43 = memory_bti[0] & 0x3f;
    int re43 = (memory_bti[1] >> 5) & 0x03;
    int dead = memory_bti[3] & 0x3F;
    int LH =  memory_bti[4] & 0x3F;
    int LL =  memory_bti[5] & 0x3F;
    int CH =  memory_bti[6] & 0x3F;
    int CL =  memory_bti[7] & 0x3F;
    int RH =  memory_bti[8] & 0x3F;
    int RL =  memory_bti[9] & 0x3F;
    int tston = ( memory_bti[10] & 0x20 ) != 0 ;
    int test = ( memory_bti[10] & 0x10 ) != 0 ;
    int ten = ( memory_bti[10] & 0x8 ) != 0 ;
    int xon = ( memory_bti[10] & 0x2 ) != 0 ;
    int ron = ( memory_bti[10] & 0x1 ) != 0 ;
    int set = ( memory_bti[11] & 0x38 ) >> 3 ;
    int lts = ( memory_bti[11] & 0x6 ) >> 1 ;
    int ac1 = ( memory_bti[12] & 0x30 ) >> 4 ;
    int ac2 = ( memory_bti[12] & 0xc ) >> 2 ;
    int acl = ( memory_bti[12] & 0x3 ) ;
    int ach = ( memory_bti[13] & 0x30 ) >> 4 ;

    
    if(debug()>3)
     std::cout << std::dec << "st43=" << st43 
     	<< " re43=" << re43 
	<< " dead=" << dead 
      	<< " LH=" << LH 
       	<< " LL=" << LL 
       	<< " CH=" << CH 
       	<< " CL=" << CL 
       	<< " RH=" << RH 
       	<< " RL=" << RL 
        << " tston=" << tston
        << " test=" << test 
        << " ten=" << ten
        << " xon=" << xon
        << " ron=" << ron
        << " set=" << set 
        << " lts=" << lts
        << " ac1=" << ac1
        << " ac2=" << ac2
        << " acl=" << acl
        << " ach=" << ach
        << std::endl; 
	

    int sl=2;	
    int cell=bti+1;
    if(debug()>2)
      std::cout << "Setting K acceptance for cell " << cell << " sl " << sl << std::endl;

    setLH_bti(cell,sl,LH);
    setLL_bti(cell,sl,LL);
    setCH_bti(cell,sl,CH);
    setCL_bti(cell,sl,CL);
    setRH_bti(cell,sl,RH);
    setRL_bti(cell,sl,RL);


  }
}
 
