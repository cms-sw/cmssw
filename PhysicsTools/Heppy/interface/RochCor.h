#ifndef PhysicsTools_Heppy_RochCor_h
#define PhysicsTools_Heppy_RochCor_h

////  VERSION 4, taken from http://www-cdf.fnal.gov/~jyhan/cms_momscl/cms_RochCor_manual.html on 19 september 2012
////  moved static const float from .h to .cc to make the gcc434 happy

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

class RochCor {
 public:
  RochCor();
  RochCor(int seed);
  ~RochCor();
  
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
  static const float genm_smr; //gen mass peak with eta dependent gaussian smearing => better match in Z mass profile vs. eta/phi
  static const float genm; //gen mass peak without smearing => Z mass profile vs. eta/phi in CMS note
  
  static const float recmA; //rec mass peak in MC (2011A)
  static const float drecmA; //rec mass peak in data (2011A)
  static const float mgsclA_stat; //stat. error of global factor for mass peak in MC (2011A)  
  static const float mgsclA_syst; //syst. error of global factor for mass peak in MC (2011A)  
  static const float dgsclA_stat; //stat. error of global factor for mass peak in data (2011A)
  static const float dgsclA_syst; //syst. error of global factor for mass peak in data (2011A)
  static const float recmB; //rec mass peak in MC (2011B)
  static const float drecmB; //rec mass peak in data (2011B)
  static const float mgsclB_stat; //stat. error of global factor for mass peak in MC (2011B)  
  static const float mgsclB_syst; //syst. error of global factor for mass peak in MC (2011B)  
  static const float dgsclB_stat; //stat. error of global factor for mass peak in data (2011B)
  static const float dgsclB_syst; //syst. error of global factor for mass peak in data (2011B)
  
  //iteration2 after FSR : after Z Pt correction
  static const float deltaA;
  static const float deltaA_stat;
  static const float deltaA_syst;
  
  static const float sfA;
  static const float sfA_stat;
  static const float sfA_syst;
  
  static const float deltaB;
  static const float deltaB_stat;
  static const float deltaB_syst;
  
  static const float sfB;
  static const float sfB_stat;
  static const float sfB_syst;

  static const float apar; //+- 0.002
  static const float bpar; //+- 1.57968e-06
  static const float cpar; //+- 1.92775e-06
  static const float d0par; //+- 3.16301e-06
  static const float e0par; //+- 0.0249021
  static const float d1par; //+- 1.12386e-05
  static const float e1par; //+- 0.17896
  static const float d2par; //+- 5.68386e-06
  static const float e2par; //+- 0.0431732
////^^^^^------------ GP END 
 
  //---------------------------------------------------------------------------------------------
  
  static const float dcor_bfA[8][8];  
  static const float dcor_maA[8][8];
  static const float mcor_bfA[8][8];
  static const float mcor_maA[8][8];
  static const float dcor_bfAer[8][8];  
  static const float dcor_maAer[8][8];
  static const float mcor_bfAer[8][8];
  static const float mcor_maAer[8][8];

  static const float dcor_bfB[8][8];  
  static const float dcor_maB[8][8];
  static const float mcor_bfB[8][8];
  static const float mcor_maB[8][8];
  static const float dcor_bfBer[8][8];  
  static const float dcor_maBer[8][8];
  static const float mcor_bfBer[8][8];
  static const float mcor_maBer[8][8];

  //=======================================================================================================
  
  static const float dmavgA[8][8];  
  static const float dpavgA[8][8];  
  static const float mmavgA[8][8];  
  static const float mpavgA[8][8];

  static const float dmavgB[8][8];  
  static const float dpavgB[8][8];  
  static const float mmavgB[8][8];  
  static const float mpavgB[8][8];
  
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

};
}

#endif  
