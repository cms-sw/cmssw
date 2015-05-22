#ifndef CMGTools_Utilities_RecoilCorrector_H
#define CMGTools_Utilities_RecoilCorrector_H


#include <vector>
#include <sstream>
#include <string>
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TProfile.h"
#include "TF1.h"
#include "TMath.h"
#include "TRandom3.h"

//
// ** apply recoil corrections **
// 
// usage: 
//    double met=rawMetValue;
//    double metphi=rawMetPhiValue;
//    RecoilCorrector corrector;
//    corrector->Correct(met,metphi,GenZPt,GenZPhi,leptonPt,leptonPhi);
//  
//
// where leptonPt, leptonPhi are dilepton kinematics for z->ll and single lepton kinematics for w->lnu
//

using namespace std;

class RecoilCorrector
{
  
public:
  RecoilCorrector(string iNameZDat, int iSeed=0xDEADBEEF);
  RecoilCorrector(string iNameZDat1, string iPrefix, int iSeed=0xDEADBEEF);
  ~RecoilCorrector();
  void CorrectAll(double &met, double &metphi, double iGenPt, double iGenPhi, double iLepPt, double iLepPhi,double &iU1,double &iU2,double iFluc,double iScale=0,int njet=0);
  void Correct(double &pfmet, double &pfmetphi, double &trkmet, double &trkmetphi, 
	       double iGenPt, double iGenPhi, double iLepPt, double iLepPhi,double iFluc    ,double iScale=0,int njet=0);
  void CorrectType1(double &pfmet, double &pfmetphi,double iGenPt,double iGenPhi,double iLepPt,double iLepPhi,double &iU1,double &iU2,double iFlucU2,double iFlucU1,double iScale=0,int njet=0);
  void CorrectType2(double &pfmet, double &pfmetphi,double iGenPt,double iGenPhi,double iLepPt,double iLepPhi,double &iU1,double &iU2,double iFlucU2,double iFlucU1,double iScale=0,int njet=0,bool doSingleGauss=false);
  void CorrectU1U2(double &pfu1, double &pfu2, double &trku1, double &trku2, 
		   double iGenPt, double iGenPhi, double iLepPt, double iLepPhi,double iFluc,double iScale=0,int njet=0);
  void addDataFile(std::string iNameDat);
  void addMCFile  (std::string iNameMC);
protected:
  enum Recoil { 
    PFU1,
    PFU2,
    PFMSU1,
    PFMSU2,
    PFS1U1,
    PFS2U1,
    PFS1U2,
    PFS2U2,
    TKU1,
    TKU2,
    TKMSU1,
    TKMSU2,
    TKS1U1,
    TKS2U1,
    TKS1U2,
    TKS2U2
  };

  void readRecoil(std::vector<TF1*> &iU1Fit,std::vector<TF1*> &iU1MRMSFit,std::vector<TF1*> &iU1RMS1Fit,std::vector<TF1*> &iU1RMS2Fit,
		  std::vector<TF1*> &iU2Fit,std::vector<TF1*> &iU2MRMSFit,std::vector<TF1*> &iU2RMS1Fit,std::vector<TF1*> &iU2RMS2Fit,
		  std::string iFName,std::string iPrefix); 

  void readCorr(std::string iName,//int iType=2,
		std::vector<TF1*> &iF1U1U2Corr  ,std::vector<TF1*> &iF2U1U2Corr,std::vector<TF1*> &iF1F2U1Corr,std::vector<TF1*> &iF1F2U2Corr,
		std::vector<TF1*> &iF1F2U1U2Corr,std::vector<TF1*> &iF1F2U2U1Corr,int iType=2);

  void metDistribution(double &iMet,double &iMPhi,double iGenPt,double iGenPhi,
		       double iLepPt,double iLepPhi,TRandom3 *iRand,
		       TF1 *iU1RZFit, 
		       TF1 *iU1MSZFit, 
		       TF1 *iU1S1ZFit,
		       TF1 *iU1S2ZFit,
		       TF1 *iU2MSZFit,		   
		       TF1 *iU2S1ZFit, 
		       TF1 *iU2S2ZFit, 
		       TF1 *iU1U2Corr, 
		       double &iU1,double &iU2,double iFluc=0,double iScale=0);

  void metDistribution(double &iPFMet,double &iPFMPhi,double &iTKMet,double &iTKMPhi,
		       double iGenPt,double iGenPhi,
		       double iLepPt,double iLepPhi,TRandom3 *iRand,
		       TF1 *iU1RZPFFit,  TF1 *iU1RZTKFit, 
		       TF1 *iU1MSZPFFit, TF1 *iU1MSZTKFit, 
		       TF1 *iU1S1ZPFFit, TF1 *iU1S1ZTKFit,
		       TF1 *iU1S2ZPFFit, TF1 *iU1S2ZTKFit,
		       TF1 *iU2MSZPFFit, TF1 *iU2MSZTKFit,		   
		       TF1 *iU2S1ZPFFit, TF1 *iU2S1ZTKFit, 
		       TF1 *iU2S2ZPFFit, TF1 *iU2S2ZTKFit, 
		       TF1 *iPFU1U2Corr, TF1 *iTKU1U2Corr,
		       TF1 *iPFTKU1Corr, TF1 *iPFTKU2Corr,
		       TF1 *iPFTKU1MCorr,TF1 *iPFTKU2MCorr,
		       double &iU1,double &iU2,double iFluc=0,double iScale=0);

  void metDistributionType1(double &iMet,double &iMPhi,double iGenPt,double iGenPhi,
			    double iLepPt,double iLepPhi,TRandom3 *iRand,
			    TF1 *iU1RZDatFit,  TF1 *iU1RZMCFit,
			    TF1 *iU1MSZDatFit, TF1 *iU1MSZMCFit, 
			    TF1 *iU2MSZDatFit, TF1 *iU2MSZMCFit,
			    double &iU1,double &iU2,double iFlucU2=0,double iFlucU1=0,double iScale=0);

  void metDistributionType2(double &iMet,double &iMPhi,double iGenPt,double iGenPhi,
			    double iLepPt,double iLepPhi,
			    TF1 *iU1Default,
			    TF1 *iU1RZDatFit,  TF1 *iU1RZMCFit,
			    TF1 *iU1MSZDatFit, TF1 *iU1MSZMCFit, 
			    TF1 *iU1S1ZDatFit, TF1 *iU1S1ZMCFit, 
			    TF1 *iU1S2ZDatFit, TF1 *iU1S2ZMCFit, 
			    TF1 *iU2MSZDatFit, TF1 *iU2MSZMCFit,
			    TF1 *iU2S1ZDatFit, TF1 *iU2S1ZMCFit,	   
			    TF1 *iU2S2ZDatFit, TF1 *iU2S2ZMCFit,	   
			    //			    TF1 *iU1U2ZDatCorr,TF1 *iU1U2ZMCCorr, // MARIA comment for now since not used
			    double &iU1, double &iU2,double iFlucU2=0,double iFlucU1=0,double iScale=0,
			    bool doSingleGauss=false);

  void metDistributionType2CorrU(double &iMet,double &iMPhi,double iGenPt,double iGenPhi,
				 double iLepPt,double iLepPhi,
				 TF1 *iU1Default,
				 TF1 *iU1RZDatFit,  TF1 *iU1RZMCFit,
				 TF1 *iU1MSZDatFit, TF1 *iU1MSZMCFit,
				 TF1 *iU1S1ZDatFit, TF1 *iU1S1ZMCFit,
				 TF1 *iU1S2ZDatFit, TF1 *iU1S2ZMCFit,
				 TF1 *iU2MSZDatFit, TF1 *iU2MSZMCFit,
				 TF1 *iU2S1ZDatFit, TF1 *iU2S1ZMCFit,
				 TF1 *iU2S2ZDatFit, TF1 *iU2S2ZMCFit,
				 //                                         TF1 *iU1U2ZDatCorr,TF1 *iU1U2ZMCCorr, // MARIA comment for now                              
				 double &iU1,double &iU2,double iFlucU2, double iFlucU1, double iScale,
				 bool doSingleGauss);

  double diGausPVal    (double iVal, double iFrac,double iSimga1,double iSigma2);
  double diGausPInverse(double iPVal,double iFrac,double iSigma1,double iSigma2);
  double oneGausPInverse(double iPVal,double iFrac,double iSigma1,double iSigma2);
  double calculate(int iMet,double iEPt,double iEPhi,double iWPhi,double iU1,double iU2);
  double getError(double iVal,TF1 *iZDatFit,Recoil iType);
  double getError2(double iVal,TF1 *iFit);
  double getCorError2(double iVal,TF1 *iFit);
  double mag(double iV0,double iV1,double iV2,double iV3);
  double correlatedSeed(double iVal, double iCorr1,double iCorr2,double iCorr3,double iSeed0,double iSeed1,double iSeed2,double iSeed3);
  double deCorrelate   (double iVal, double iCorr1,double iCorr2,double iCorr3,double iSeed0,double iSeed1,double iSeed2,double iSeed3);
  TF1*   getFunc(bool iMC, Recoil iType);
  double CorrVal(double iPt,double iVal,Recoil iType);
  //void   Correct(double &met, double &metphi, double lGenPt, double lGenPhi, double lepPt, double lepPhi,double iFluc,int njet);

  TRandom3 *fRandom; 
  vector<TF1*> fF1U1Fit; vector<TF1*> fF1U1RMSSMFit; vector<TF1*> fF1U1RMS1Fit; vector<TF1*> fF1U1RMS2Fit; 
  vector<TF1*> fF1U2Fit; vector<TF1*> fF1U2RMSSMFit; vector<TF1*> fF1U2RMS1Fit; vector<TF1*> fF1U2RMS2Fit; 
  vector<TF1*> fF2U1Fit; vector<TF1*> fF2U1RMSSMFit; vector<TF1*> fF2U1RMS1Fit; vector<TF1*> fF2U1RMS2Fit; 
  vector<TF1*> fF2U2Fit; vector<TF1*> fF2U2RMSSMFit; vector<TF1*> fF2U2RMS1Fit; vector<TF1*> fF2U2RMS2Fit; 

  vector<TF1*> fD1U1Fit; vector<TF1*> fD1U1RMSSMFit; vector<TF1*> fD1U1RMS1Fit; vector<TF1*> fD1U1RMS2Fit; 
  vector<TF1*> fD1U2Fit; vector<TF1*> fD1U2RMSSMFit; vector<TF1*> fD1U2RMS1Fit; vector<TF1*> fD1U2RMS2Fit; 
  vector<TF1*> fD2U1Fit; vector<TF1*> fD2U1RMSSMFit; vector<TF1*> fD2U1RMS1Fit; vector<TF1*> fD2U1RMS2Fit; 
  vector<TF1*> fD2U2Fit; vector<TF1*> fD2U2RMSSMFit; vector<TF1*> fD2U2RMS1Fit; vector<TF1*> fD2U2RMS2Fit; 

  vector<TF1*> fM1U1Fit; vector<TF1*> fM1U1RMSSMFit; vector<TF1*> fM1U1RMS1Fit; vector<TF1*> fM1U1RMS2Fit; 
  vector<TF1*> fM1U2Fit; vector<TF1*> fM1U2RMSSMFit; vector<TF1*> fM1U2RMS1Fit; vector<TF1*> fM1U2RMS2Fit; 
  vector<TF1*> fM2U1Fit; vector<TF1*> fM2U1RMSSMFit; vector<TF1*> fM2U1RMS1Fit; vector<TF1*> fM2U1RMS2Fit; 
  vector<TF1*> fM2U2Fit; vector<TF1*> fM2U2RMSSMFit; vector<TF1*> fM2U2RMS1Fit; vector<TF1*> fM2U2RMS2Fit; 

  vector<TF1*> fF1U1U2Corr;     vector<TF1*> fF2U1U2Corr;
  vector<TF1*> fF1F2U1Corr;     vector<TF1*> fF1F2U2Corr;
  vector<TF1*> fF1F2U1U2Corr;   vector<TF1*> fF1F2U2U1Corr;

  vector<TF1*> fM1U1U2Corr;     vector<TF1*> fM2U1U2Corr;
  vector<TF1*> fM1M2U1Corr;     vector<TF1*> fM1M2U2Corr;
  vector<TF1*> fM1M2U1U2Corr;   vector<TF1*> fM1M2U2U1Corr;
  int fId; int fJet;
};



#endif
