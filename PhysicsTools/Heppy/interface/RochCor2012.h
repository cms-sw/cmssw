#ifndef PhysicsTools_Heppy_RochCor2012_h
#define PhysicsTools_Heppy_RochCor2012_h

////  VERSION for 2012 received from Jiyeon on 30 september 2012
////  moved static const float from .h to .cc to make the gcc happy

#include <iostream>

#include <TChain.h>
#include <TClonesArray.h>
#include <TString.h>
#include <map>

#include <TSystem.h>
#include <TROOT.h>
#include <TMath.h>
#include <TLorentzVector.h>
#include <TRandom3.h>

namespace heppy {

class RochCor2012 {
 public:
  RochCor2012();
  RochCor2012(int seed);
  ~RochCor2012();
  
  void momcor_mc(TLorentzVector&, float, float, int);
  void momcor_data(TLorentzVector&, float, float, int);
  
  void musclefit_data(TLorentzVector& , TLorentzVector&);
  
  float zptcor(float);
  int etabin(float);
  int phibin(float);
  
 private:
  
  TRandom3 eran;
  TRandom3 sran;
  
  //  static float netabin[9] = {-2.4,-2.1,-1.4,-0.7,0.0,0.7,1.4,2.1,2.4};
  static const float netabin[9];
////^^^^^------------ GP BEGIN 
  static const double pi;
  
  static const float genm_smr;
  static const float genm;
  
  static const float mrecm;
  static const float drecm;
  static const float mgscl_stat;
  static const float mgscl_syst;
  static const float dgscl_stat;
  static const float dgscl_syst;
  
  //iteration2 after FSR : after Z Pt correction
  static const float delta;
  static const float delta_stat;
  static const float delta_syst;
  
  static const float sf;
  static const float sf_stat;
  static const float sf_syst;
  
  static const float apar;
  static const float bpar;
  static const float cpar;
  static const float d0par;
  static const float e0par;
  static const float d1par;
  static const float e1par;
  static const float d2par;
  static const float e2par;
////^^^^^------------ GP END 
 
  //---------------------------------------------------------------------------------------------
  
  static const float dcor_bf[8][8];  
  static const float dcor_ma[8][8];
  static const float mcor_bf[8][8];
  static const float mcor_ma[8][8];
  static const float dcor_bfer[8][8];  
  static const float dcor_maer[8][8];
  static const float mcor_bfer[8][8];
  static const float mcor_maer[8][8];

  //=======================================================================================================
  
  static const float dmavg[8][8];  
  static const float dpavg[8][8];  
  static const float mmavg[8][8];  
  static const float mpavg[8][8];

  //===============================================================================================
  //parameters for Z pt correction
  static const int nptbins=84;
  static const float ptlow[85];    
  
  static const float zptscl[84];
  static const float zptscler[84];

  float mptsys_mc_dm[8][8];
  float mptsys_mc_da[8][8];
  float mptsys_da_dm[8][8];
  float mptsys_da_da[8][8];

  float gscler_mc_dev;
  float gscler_da_dev;
};
}
  
#endif
