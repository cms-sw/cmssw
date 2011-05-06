#ifndef UEAnalysisJets_h
#define UEAnalysisJets_h

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <TFile.h>

#include<TLorentzVector.h>

#include <TH1F.h>
#include <TProfile.h>

#include <TClonesArray.h>

class UEAnalysisJets {
 public :

  UEAnalysisJets();
  ~UEAnalysisJets(){}
  void jetCalibAnalysis(float ,float,TClonesArray *,TClonesArray *,TClonesArray *,TClonesArray *);
  void writeToFile(TFile *);

  void Begin(TFile *);

  //Charged Jet caharacterization
  TH1F* dr_chgcalo;
  TH1F* dr_chginc;
  TH1F* dr_chgmcreco;
  TH1F* dr_caloinc;
  TH1F* numb_cal;
  TH1F* pT_cal;
  TH1F* eta_cal;
  TH1F* eta_cal_res;
  TH1F* phi_cal;
  TH1F* phi_cal_res;
  TH1F* numb_chgmc;
  TH1F* pT_chgmc;
  TH1F* eta_chgmc;
  TH1F* eta_chgmc_res;
  TH1F* phi_chgmc;
  TH1F* phi_chgmc_res;
  TH1F* numb_chgreco;
  TH1F* pT_chgreco;
  TH1F* eta_chgreco;
  TH1F* eta_chgreco_res;
  TH1F* phi_chgreco;
  TH1F* phi_chgreco_res;
  TH1F* numb_inc;
  TH1F* pT_inc;
  TH1F* eta_inc;
  TH1F* phi_inc;
  TProfile* calib_chgcalo;
  TProfile* calib_chginc;
  TProfile* calib_chgmcreco;
  TProfile* calib_caloinc;
  TProfile* calib_chgcalo_eta;
  TProfile* calib_chginc_eta;
  TProfile* calib_chgmcreco_eta;
  TProfile* calib_caloinc_eta;
  TProfile* calib_chgcalo_phi;
  TProfile* calib_chginc_phi;
  TProfile* calib_chgmcreco_phi;
  TProfile* calib_caloinc_phi;

  float piG;
};

#endif
