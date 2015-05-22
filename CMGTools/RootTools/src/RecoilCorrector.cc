#include "CMGTools/RootTools/interface/RecoilCorrector.h"

//-----------------------------------------------------------------------------------------------------------------------------------------
RecoilCorrector::RecoilCorrector(string iNameZDat,std::string iPrefix, int iSeed) {

  fRandom = new TRandom3(iSeed);

  // get fits for Z data
  readRecoil(fF1U1Fit,fF1U1RMSSMFit,fF1U1RMS1Fit,fF1U1RMS2Fit,fF1U2Fit,fF1U2RMSSMFit,fF1U2RMS1Fit,fF1U2RMS2Fit,iNameZDat,iPrefix);
  if(iPrefix == "PF") readCorr  (iNameZDat,fF1U1U2Corr,fF2U1U2Corr,fF1F2U1Corr,fF1F2U2Corr,fF1F2U1U2Corr,fF1F2U2U1Corr,0);
  if(iPrefix == "TK") readCorr  (iNameZDat,fF1U1U2Corr,fF2U1U2Corr,fF1F2U1Corr,fF1F2U2Corr,fF1F2U1U2Corr,fF1F2U2U1Corr,1);  
  fId = 0; fJet = 0;
}

RecoilCorrector::RecoilCorrector(string iNameZ, int iSeed) {

  fRandom = new TRandom3(iSeed);
  // get fits for Z data
  readRecoil(fF1U1Fit,fF1U1RMSSMFit,fF1U1RMS1Fit,fF1U1RMS2Fit,fF1U2Fit,fF1U2RMSSMFit,fF1U2RMS1Fit,fF1U2RMS2Fit,iNameZ,"PF");
  readRecoil(fF2U1Fit,fF2U1RMSSMFit,fF2U1RMS1Fit,fF2U1RMS2Fit,fF2U2Fit,fF2U2RMSSMFit,fF2U2RMS1Fit,fF2U2RMS2Fit,iNameZ,"TK");
  readCorr  (iNameZ  ,fF1U1U2Corr,fF2U1U2Corr,fF1F2U1Corr,fF1F2U2Corr,fF1F2U1U2Corr,fF1F2U2U1Corr);
  fId = 0; fJet = 0;
}

RecoilCorrector::~RecoilCorrector() {

  delete fRandom;

}

//-----------------------------------------------------------------------------------------------------------------------------------------
void RecoilCorrector::addDataFile(std::string iNameData) { 
  readRecoil(fD1U1Fit,fD1U1RMSSMFit,fD1U1RMS1Fit,fD1U1RMS2Fit,fD1U2Fit,fD1U2RMSSMFit,fD1U2RMS1Fit,fD1U2RMS2Fit,iNameData,"PF");
  readRecoil(fD2U1Fit,fD2U1RMSSMFit,fD2U1RMS1Fit,fD2U1RMS2Fit,fD2U2Fit,fD2U2RMSSMFit,fD2U2RMS1Fit,fD2U2RMS2Fit,iNameData,"TK");
  //readCorr(iNameData);
  fId++;   
}

void RecoilCorrector::addMCFile  (std::string iNameMC) { 
  fId++;
  readRecoil(fM1U1Fit,fM1U1RMSSMFit,fM1U1RMS1Fit,fM1U1RMS2Fit,fM1U2Fit,fM1U2RMSSMFit,fM1U2RMS1Fit,fM1U2RMS2Fit,iNameMC,"PF");
  readRecoil(fM2U1Fit,fM2U1RMSSMFit,fM2U1RMS1Fit,fM2U1RMS2Fit,fM2U2Fit,fM2U2RMSSMFit,fM2U2RMS1Fit,fM2U2RMS2Fit,iNameMC,"TK");
  readCorr  (iNameMC ,fM1U1U2Corr,fM2U1U2Corr,fM1M2U1Corr,fM1M2U2Corr,fM1M2U1U2Corr,fM1M2U2U1Corr);
}

void RecoilCorrector::CorrectAll(double &met, double &metphi, double lGenPt, double lGenPhi, double lepPt, double lepPhi,double &iU1,double &iU2,double iFluc,double iScale,int njet) {

  fJet = njet; 
  
  //  if(njet > 2) fJet = 2;  
  //  if(fJet >= int(fF1U1Fit.size())) fJet = 1; 
  if(fJet >= int(fF1U1Fit.size())) fJet = int(fF1U1Fit.size()) - 1; 

  fRandom->SetSeed((int)((lGenPhi+4)*100000));
  metDistribution(met,metphi,lGenPt,lGenPhi,lepPt,lepPhi,fRandom,
		  fF1U1Fit     [fJet],
		  fF1U1RMSSMFit[fJet],
		  fF1U1RMS1Fit [fJet],
		  fF1U1RMS2Fit [fJet],
		  fF1U2RMSSMFit[fJet],
		  fF1U2RMS1Fit [fJet],
		  fF1U2RMS2Fit [fJet],
		  fF1U1U2Corr  [fJet],
		  iU1, iU2,iFluc,iScale
		  );
}

void RecoilCorrector::CorrectType1(double &met, double &metphi, double lGenPt, double lGenPhi, double lepPt, double lepPhi,double &iU1,double &iU2,double iFlucU2,double iFlucU1,double iScale,int njet) {

  //  cout << "TYPE1: nVTX " << njet << " fId " << fId << " function size "<< fF1U1Fit.size() << endl;

  fJet = njet; 

  //  if(njet > 2) fJet = 2;  
  //  if(fJet >= int(fF1U1Fit.size())) fJet = 1; 
  if(fJet >= int(fF1U1Fit.size())) fJet = int(fF1U1Fit.size()) - 1; 

  fRandom->SetSeed((int)((lGenPhi+4)*100000));
  metDistributionType1(met,metphi,lGenPt,lGenPhi,lepPt,lepPhi,fRandom,
		       fD1U1Fit     [fJet],fM1U1Fit     [fJet],
		       fD1U1RMSSMFit[fJet],fM1U1RMSSMFit[fJet],
		       fD1U2RMSSMFit[fJet],fM1U2RMSSMFit[fJet],
		       //fF1U1U2Corr  [fJet],fM1U1U2Corr  [fJet], ===> For the future
		       iU1,iU2,iFlucU2,iFlucU1,iScale
		       );
}

void RecoilCorrector::CorrectType2(double &met, double &metphi, double lGenPt, double lGenPhi, double lepPt, double lepPhi,double &iU1,double &iU2,double iFlucU2,double iFlucU1,double iScale,int njet, bool doSingleGauss) {

  //  cout << "TYPE2: nVTX " << njet << " function size "<< fD1U1Fit.size() << endl;
  
  fJet = njet; 

  //  if(njet > 2) fJet = 2;  
  //  if(fJet >= int(fF1U1Fit.size())) fJet = 1; 
  if(fJet >= int(fF1U1Fit.size())) fJet = int(fF1U1Fit.size()) - 1; 

  metDistributionType2(met,metphi,lGenPt,lGenPhi,lepPt,lepPhi,fF1U1Fit[fJet],
		       fD1U1Fit     [fJet],fM1U1Fit     [fJet],
		       fD1U1RMSSMFit[fJet],fM1U1RMSSMFit[fJet],
		       fD1U1RMS1Fit [fJet],fM1U1RMS1Fit [fJet],
		       fD1U1RMS2Fit [fJet],fM1U1RMS2Fit [fJet],
		       fD1U2RMSSMFit[fJet],fM1U2RMSSMFit[fJet],
		       fD1U2RMS1Fit [fJet],fM1U2RMS1Fit [fJet],
		       fD1U2RMS2Fit [fJet],fM1U2RMS2Fit [fJet],
		       //		       fF1U1U2Corr  [fJet],fM1U1U2Corr  [fJet],  // MARIA comment for now since not used
		       iU1,iU2,iFlucU2,iFlucU1,iScale,
		       doSingleGauss );
}

void RecoilCorrector::Correct(double &pfmet, double &pfmetphi, double &trkmet, double &trkmetphi, 
                              double lGenPt, double lGenPhi, double lepPt, double lepPhi,double iFluc,double iScale,int njet) {
  double lU1 = 0; double lU2 = 0;
  fJet = njet; 

  //    if(njet > 2) fJet = 2;  
  //  if(fJet > int(fF1U1Fit.size())) fJet = 1; 
  if(fJet >= int(fF1U1Fit.size())) fJet = int(fF1U1Fit.size()) - 1; 

  fRandom->SetSeed((int)((lGenPhi+4)*100000));
  metDistribution(pfmet,pfmetphi,trkmet,trkmetphi,lGenPt,lGenPhi,lepPt,lepPhi,fRandom,
		  fF1U1Fit     [fJet],fF2U1Fit     [fJet],
		  fF1U1RMSSMFit[fJet],fF2U1RMSSMFit[fJet],
		  fF1U1RMS1Fit [fJet],fF2U1RMS1Fit [fJet],
		  fF1U1RMS2Fit [fJet],fF2U1RMS2Fit [fJet],
		  fF1U2RMSSMFit[fJet],fF2U2RMSSMFit[fJet],
		  fF1U2RMS1Fit [fJet],fF2U2RMS1Fit [fJet],
		  fF1U2RMS2Fit [fJet],fF2U2RMS2Fit [fJet],
		  fF1U1U2Corr  [fJet],fF2U1U2Corr  [fJet],
		  fF1F2U1Corr  [fJet],fF1F2U2Corr  [fJet],
		  fF1F2U1U2Corr[fJet],fF1F2U2U1Corr[fJet],
		  lU1                ,lU2,
		  iFluc,              iScale
		  );
}
void RecoilCorrector::CorrectU1U2(double &iPFU1, double &iPFU2, double &iTKU1, double &iTKU2, 
				  double lGenPt, double lGenPhi, double lepPt, double lepPhi,double iFluc,double iScale,int njet) {
  double pfmet = 0; double pfmetphi = 0; //double trkmet = 0; double trkmetphi = 0;
  fJet = njet; 

  //   if(njet > 2) fJet = 2;  
  //    if(fJet > int(fF1U1Fit.size())) fJet = 1; 
  if(fJet >= int(fF1U1Fit.size())) fJet = int(fF1U1Fit.size()) - 1; 

  metDistribution(pfmet,pfmetphi,iTKU1,iTKU2,lGenPt,lGenPhi,lepPt,lepPhi,fRandom,
		  fF1U1Fit     [fJet],fF2U1Fit     [fJet],
		  fF1U1RMSSMFit[fJet],fF2U1RMSSMFit[fJet],
		  fF1U1RMS1Fit [fJet],fF2U1RMS1Fit [fJet],
		  fF1U1RMS2Fit [fJet],fF2U1RMS2Fit [fJet],
		  fF1U2RMSSMFit[fJet],fF2U2RMSSMFit[fJet],
		  fF1U2RMS1Fit [fJet],fF2U2RMS1Fit [fJet],
		  fF1U2RMS2Fit [fJet],fF2U2RMS2Fit [fJet],
		  fF1U1U2Corr  [fJet],fF2U1U2Corr  [fJet],
		  fF1F2U1Corr  [fJet],fF1F2U2Corr  [fJet],
		  fF1F2U1U2Corr[fJet],fF1F2U2U1Corr[fJet],
		  iPFU1              ,iPFU2,
		  iFluc,              iScale
		  );
  //iTKU1 = 0; iTKU2 = 0;
}
double RecoilCorrector::CorrVal(double iPt, double iVal, Recoil iType) { 
  if(fId == 0 || fId == 1) return iVal;
  switch(iType) {
  case PFU1   : return iVal*(fD1U1Fit     [fJet]->Eval(iPt)/fM1U1Fit     [fJet]->Eval(iPt));
  case PFMSU1 : return iVal*(fD1U1RMSSMFit[fJet]->Eval(iPt)/fM1U1RMSSMFit[fJet]->Eval(iPt));
  case PFS1U1 : return iVal*(fD1U1RMS1Fit [fJet]->Eval(iPt)/fM1U1RMS1Fit [fJet]->Eval(iPt));
  case PFS2U1 : return iVal*(fD1U1RMS2Fit [fJet]->Eval(iPt)/fM1U1RMS2Fit [fJet]->Eval(iPt));
  case PFU2   : return 0;
  case PFMSU2 : return iVal*(fD1U2RMSSMFit[fJet]->Eval(iPt)/fM1U2RMSSMFit[fJet]->Eval(iPt));
  case PFS1U2 : return iVal*(fD1U2RMS1Fit [fJet]->Eval(iPt) /fM1U2RMS1Fit[fJet]->Eval(iPt));
  case PFS2U2 : return iVal*(fD1U2RMS2Fit [fJet]->Eval(iPt) /fM1U2RMS2Fit[fJet]->Eval(iPt));
  case TKU1   : return iVal*(fD2U1Fit     [fJet]->Eval(iPt)/fM2U1Fit     [fJet]->Eval(iPt));
  case TKMSU1 : return iVal*(fD2U1RMSSMFit[fJet]->Eval(iPt)/fM2U1RMSSMFit[fJet]->Eval(iPt));
  case TKS1U1 : return iVal*(fD2U1RMS1Fit [fJet]->Eval(iPt) /fM2U1RMS1Fit[fJet]->Eval(iPt));
  case TKS2U1 : return iVal*(fD2U1RMS2Fit [fJet]->Eval(iPt) /fM2U1RMS2Fit[fJet]->Eval(iPt));
  case TKU2   : return 0;
  case TKMSU2 : return iVal*(fD2U2RMSSMFit[fJet]->Eval(iPt)/fM2U2RMSSMFit[fJet]->Eval(iPt));
  case TKS1U2 : return iVal*(fD2U2RMS1Fit [fJet]->Eval(iPt) /fM2U2RMS1Fit[fJet]->Eval(iPt));
  case TKS2U2 : return iVal*(fD2U2RMS2Fit [fJet]->Eval(iPt) /fM2U2RMS2Fit[fJet]->Eval(iPt));
  }
  return iVal;
}
TF1* RecoilCorrector::getFunc(bool iMC, Recoil iType) { 
  if(fId == 0 || fId == 1) return 0;
  switch(iType) {
  case PFU1   : if(iMC) {return fD1U1Fit     [fJet];} return fM1U1Fit     [fJet];
  case PFMSU1 : if(iMC) {return fD1U1RMSSMFit[fJet];} return fM1U1RMSSMFit[fJet];
  case PFS1U1 : if(iMC) {return fD1U1RMS1Fit [fJet];} return fM1U1RMS1Fit [fJet];
  case PFS2U1 : if(iMC) {return fD1U1RMS2Fit [fJet];} return fM1U1RMS2Fit [fJet];
  case PFU2   : return 0;
  case PFMSU2 : if(iMC) {return fD1U2RMSSMFit[fJet];} return fM1U2RMSSMFit[fJet];
  case PFS1U2 : if(iMC) {return fD1U2RMS1Fit [fJet];} return fM1U2RMS1Fit [fJet];
  case PFS2U2 : if(iMC) {return fD1U2RMS2Fit [fJet];} return fM1U2RMS2Fit [fJet];
  case TKU1   : if(iMC) {return fD2U1Fit     [fJet];} return fM2U1Fit     [fJet];
  case TKMSU1 : if(iMC) {return fD2U1RMSSMFit[fJet];} return fM2U1RMSSMFit[fJet];
  case TKS1U1 : if(iMC) {return fD2U1RMS1Fit [fJet];} return fM2U1RMS1Fit [fJet];
  case TKS2U1 : if(iMC) {return fD2U1RMS2Fit [fJet];} return fM2U1RMS2Fit [fJet];
  case TKU2   : return 0;
  case TKMSU2 : if(iMC) {return fD2U2RMSSMFit[fJet];} return fM2U2RMSSMFit[fJet];
  case TKS1U2 : if(iMC) {return fD2U2RMS1Fit [fJet];} return fM2U2RMS1Fit [fJet];
  case TKS2U2 : if(iMC) {return fD2U2RMS2Fit [fJet];} return fM2U2RMS2Fit [fJet];
  }
  return 0;
}
//-----------------------------------------------------------------------------------------------------------------------------------------
void RecoilCorrector::readRecoil(std::vector<TF1*> &iU1Fit,std::vector<TF1*> &iU1MRMSFit,std::vector<TF1*> &iU1RMS1Fit,std::vector<TF1*> &iU1RMS2Fit,
		                 std::vector<TF1*> &iU2Fit,std::vector<TF1*> &iU2MRMSFit,std::vector<TF1*> &iU2RMS1Fit,std::vector<TF1*> &iU2RMS2Fit,
		                 std::string iFName,std::string iPrefix) {
//  if(!getenv("CMSSW_BASE")) {
//    printf("error! RecoilCorrector called without input files. Define CMSSW_BASE or add by hand.\n");
//    assert(0);
//  }
  TFile *lFile  = new TFile(iFName.c_str());

  cout << "reading file "<< iFName.c_str() << endl; 
  // lFile->ls();

  int lNJet = 1; // this is for the nvtx or rapidity binned
  //   int lNJet = -1; // this is for inclusive nvtx 
  std::stringstream lSS; lSS << iPrefix << "u1Mean_" << lNJet;
  while(lFile->FindObjectAny(lSS.str().c_str()) != 0) { lSS.str("");
    //     cout << lNJet << endl;

    lSS << iPrefix << "u1Mean_"    << lNJet; iU1Fit.push_back    ( (TF1*) lFile->FindObjectAny(lSS.str().c_str())); lSS.str("");
    lSS << iPrefix << "u1MeanRMS_" << lNJet; iU1MRMSFit.push_back( (TF1*) lFile->FindObjectAny(lSS.str().c_str())); lSS.str(""); 
    lSS << iPrefix << "u1RMS1_"    << lNJet; iU1RMS1Fit.push_back( (TF1*) lFile->FindObjectAny(lSS.str().c_str())); lSS.str(""); 
    lSS << iPrefix << "u1RMS2_"    << lNJet; iU1RMS2Fit.push_back( (TF1*) lFile->FindObjectAny(lSS.str().c_str())); lSS.str(""); 
    lSS << iPrefix << "u2Mean_"    << lNJet; iU2Fit    .push_back( (TF1*) lFile->FindObjectAny(lSS.str().c_str())); lSS.str("");
    lSS << iPrefix << "u2MeanRMS_" << lNJet; iU2MRMSFit.push_back( (TF1*) lFile->FindObjectAny(lSS.str().c_str())); lSS.str("");
    lSS << iPrefix << "u2RMS1_"    << lNJet; iU2RMS1Fit.push_back( (TF1*) lFile->FindObjectAny(lSS.str().c_str())); lSS.str("");
    lSS << iPrefix << "u2RMS2_"    << lNJet; iU2RMS2Fit.push_back( (TF1*) lFile->FindObjectAny(lSS.str().c_str())); lSS.str("");
    lSS << iPrefix << "u2RMS2_"    << lNJet; iU2RMS2Fit.push_back( (TF1*) lFile->FindObjectAny(lSS.str().c_str())); lSS.str("");
    lNJet++; lSS << iPrefix << "u1Mean_" << lNJet;

    //    cout << "Filename " << iFName.c_str() << " lNJet " << lNJet << endl;

  }

  lFile->Close();
}
//-----------------------------------------------------------------------------------------------------------------------------------------
void RecoilCorrector::readCorr(std::string iName,
			       std::vector<TF1*> &iF1U1U2Corr  ,std::vector<TF1*> &iF2U1U2Corr,std::vector<TF1*> &iF1F2U1Corr,std::vector<TF1*> &iF1F2U2Corr,
			       std::vector<TF1*> &iF1F2U1U2Corr,std::vector<TF1*> &iF1F2U2U1Corr,int iType) {
  TFile *lFile = new TFile(iName.c_str());
  std::stringstream pSS1,pSS2,pSS3,pSS4,pSS5,pSS6;
  int lNJet = 0;
  std::stringstream lSS; lSS << "PFu1Mean_" << lNJet;
  while(lFile->FindObjectAny(lSS.str().c_str()) != 0) { 
    lSS.str(""); pSS1.str(""); pSS2.str(""); pSS3.str(""); pSS4.str(""); pSS5.str(""); pSS6.str("");
    if(iType != 1) {pSS1  << "u1u2pfCorr_" << lNJet;   iF1U1U2Corr.push_back((TF1*) lFile->FindObjectAny(pSS1.str().c_str())); }
    if(iType != 0) {pSS2  << "u1u2tkCorr_" << lNJet;   iF2U1U2Corr.push_back((TF1*) lFile->FindObjectAny(pSS2.str().c_str())); }
    if(iType <  2) { lNJet++; lSS <<  "PFu1Mean_" << lNJet;  continue;}
    pSS3  << "pftku1Corr_"  << lNJet;   iF1F2U1Corr   .push_back((TF1*) lFile->FindObjectAny(pSS3.str().c_str()));
    pSS4  << "pftku2Corr_"  << lNJet;   iF1F2U2Corr   .push_back((TF1*) lFile->FindObjectAny(pSS4.str().c_str()));
    pSS5  << "pftkum1Corr_" << lNJet;   iF1F2U1U2Corr .push_back((TF1*) lFile->FindObjectAny(pSS5.str().c_str()));
    pSS6  << "pftkum2Corr_" << lNJet;   iF1F2U2U1Corr .push_back((TF1*) lFile->FindObjectAny(pSS6.str().c_str()));
    lNJet++; lSS   << "PFu1Mean_" << lNJet;
  }
  lFile->Close();
}
//-----------------------------------------------------------------------------------------------------------------------------------------
void RecoilCorrector::metDistribution(double &iMet,double &iMPhi,double iGenPt,double iGenPhi,
		                      double iLepPt,double iLepPhi,TRandom3 *iRand,
		                      TF1 *iU1RZDatFit,
		                      TF1 *iU1MSZDatFit, 
		                      TF1 *iU1S1ZDatFit,
		                      TF1 *iU1S2ZDatFit,
		                      TF1 *iU2MSZDatFit, 		   
		                      TF1 *iU2S1ZDatFit, 
		                      TF1 *iU2S2ZDatFit, 		                      
		                      TF1 *iU1U2Corr,
				      double &iU1, double &iU2,
		                      double iFluc,double iScale) {
  double lRescale  = sqrt((TMath::Pi())/2.);		     
  double pU1       = CorrVal(iGenPt,iU1RZDatFit->Eval(iGenPt),PFU1); //iU1RZDatFit->Eval(iGenPt);//CorrVal(iGenPt,iU1RZDatFit->Eval(iGenPt),PFU1);
  double pU2       = 0; //Right guys are for cumulants => code deleted
  double pFrac1    = CorrVal(iGenPt,iU1MSZDatFit->Eval(iGenPt),PFMSU1)*lRescale;
  double pFrac2    = CorrVal(iGenPt,iU2MSZDatFit->Eval(iGenPt),PFMSU2)*lRescale;
  double pSigma1_1 = CorrVal(iGenPt,iU1S1ZDatFit->Eval(iGenPt),PFS1U1)*lRescale*CorrVal(iGenPt,iU1MSZDatFit->Eval(iGenPt),PFMSU1);
  double pSigma1_2 = CorrVal(iGenPt,iU1S2ZDatFit->Eval(iGenPt),PFS2U1)*lRescale*CorrVal(iGenPt,iU1MSZDatFit->Eval(iGenPt),PFMSU1);
  double pSigma2_1 = CorrVal(iGenPt,iU2S1ZDatFit->Eval(iGenPt),PFS1U2)*lRescale*CorrVal(iGenPt,iU2MSZDatFit->Eval(iGenPt),PFS1U2);
  double pSigma2_2 = CorrVal(iGenPt,iU2S2ZDatFit->Eval(iGenPt),PFS2U2)*lRescale*CorrVal(iGenPt,iU2MSZDatFit->Eval(iGenPt),PFS2U2);
  //double pMU1      = fabs(iU1RZDatFit->GetParameter(1));
  
  //Uncertainty propagation
  if(iFluc != 0 || iScale != 0) { 
    double lEUR1    = getError(iGenPt,iU1RZDatFit ,PFU1);
    double lEUS1_1  = getError(iGenPt,iU1S1ZDatFit,PFS1U1);
    double lEUS1_2  = getError(iGenPt,iU1S2ZDatFit,PFS2U1);
    double lEU1Frac = getError(iGenPt,iU1MSZDatFit,PFMSU1);
    double lEUS2_1  = getError(iGenPt,iU2S1ZDatFit,PFS1U2);
    double lEUS2_2  = getError(iGenPt,iU2S2ZDatFit,PFS2U2);
    double lEU2Frac = getError(iGenPt,iU2MSZDatFit,PFMSU2);

    //Modify all the different parameters the choice of signs makes it maximal
    pU1       = pU1       + iScale*lEUR1;           //Recoil
    pFrac1    = pFrac1    + iFluc*(lEU1Frac);        //Mean RMS 
    pSigma1_1 = pSigma1_1 - iFluc*lEUS1_1*pFrac1;    //Sigma 1 smalles sigma
    pSigma1_2 = pSigma1_2 + iFluc*lEUS1_2*pFrac1;    //Sigma 2 (Maximal when oppsite sigma 1)
    pFrac2    = pFrac2    + iFluc*(lEU2Frac);        //Mean RMS for U2
    pSigma2_1 = pSigma2_1 - iFluc*lEUS2_1*pFrac2;    //Sigma 1 U2
    pSigma2_2 = pSigma2_2 + iFluc*(lEUS2_2)*pFrac2;
  }
  //Caculat the proper fraction
  pFrac1 = (pFrac1-pSigma1_2)/(pSigma1_1-pSigma1_2);
  pFrac2 = (pFrac2-pSigma2_2)/(pSigma2_1-pSigma2_2);

  //Now sample for the MET distribution
  double pVal0  = iRand->Uniform(0,1);
  double pVal1  = iRand->Uniform(0,1);
  double pCorr1 = iRand->Gaus(0,1);     
  double pCorr2 = iRand->Gaus(0,1);  
  //double pCorrT1    = iRand->Gaus(0,1);     double pCorrT2    = iRand->Gaus(0,1);  
  pSigma1_1 = ((pVal0 < pFrac1)*(pSigma1_1)+(pVal0 > pFrac1)*(pSigma1_2)); 
  pSigma2_1 = ((pVal1 < pFrac2)*(pSigma2_1)+(pVal1 > pFrac2)*(pSigma2_2)); 
  
  double lU1U2   = iU1U2Corr->Eval(iGenPt)*0.5;
  //cout << "===> " << lU1U2 << " -- " << iGenPt << endl;
  double pVal1_1 = correlatedSeed(pSigma1_1,lU1U2,0.,0.,pCorr1,pCorr2,0.,0.);
  double pVal2_1 = correlatedSeed(pSigma2_1,lU1U2,0.,0.,pCorr2,pCorr1,0.,0.);
  //double pVal2_1 = correlatedSeed(pSigma2_1,lU1U2,0.,0.,pCorr2,pCorr1,0.,0.);
  //double pVal2_2 = correlatedSeed(pSigma2_2,lU1U2,0.,0.,pCorrT2,pCorrT1,0.,0.);


  pU1   = (pVal1_1+pU1);//(pVal0 < pFrac1)*(pVal1_1+pU1)+(pVal0 > pFrac1)*(pVal1_2+pU1);
  pU2   = (pVal2_1+pU2);//(pVal1 < pFrac2)*(pVal2_1+pU2)+(pVal1 > pFrac2)*(pVal2_2+pU2);

  //pU1   = (lVal0 < pFrac1)*iRand->Gaus(pU1,pSigma1_1)+(lVal0 > pFrac1)*iRand->Gaus(pU1,pSigma1_2);
  //pU2   = (lVal1 < pFrac2)*iRand->Gaus(pU2,pSigma2_1)+(lVal1 > pFrac2)*iRand->Gaus(pU2,pSigma2_2);
  iMet  = calculate(0,iLepPt,iLepPhi,iGenPhi,pU1,pU2);
  iMPhi = calculate(1,iLepPt,iLepPhi,iGenPhi,pU1,pU2);

  iU1   = pU1; 
  iU2   = pU2;
  return;
}

void RecoilCorrector::metDistributionType1(double &iMet,double &iMPhi,double iGenPt,double iGenPhi,
					   double iLepPt,double iLepPhi,TRandom3 *iRand,
					   TF1 *iU1RZDatFit,  TF1 *iU1RZMCFit,
					   TF1 *iU1MSZDatFit, TF1 *iU1MSZMCFit, 
					   TF1 *iU2MSZDatFit, TF1 *iU2MSZMCFit, 		   		   
					   //TF1 *iU1U2ZDatCorr,TF1 *iU1U2ZMCCorr,
					   double &iU1,double &iU2, double iFlucU2, double iFlucU1, double iScale) {

  // MARIA : commented this one
  //  if(iLepPt < 4) return;

  double lRescale  = sqrt((TMath::Pi())/2.);		     
  double pU1       = iU1RZDatFit->Eval(iGenPt)/iU1RZMCFit->Eval(iGenPt);
  double pU2       = 0; //Right guys are for cumulants => code deleted
  double pFrac1    = max( iU1MSZDatFit->Eval(iGenPt)*iU1MSZDatFit->Eval(iGenPt)
			  -(pU1*pU1)*iU1MSZMCFit->Eval(iGenPt)*iU1MSZMCFit ->Eval(iGenPt),0.);
  double pFrac2    = max( iU2MSZDatFit->Eval(iGenPt)*iU2MSZDatFit->Eval(iGenPt)
			  -iU2MSZMCFit->Eval(iGenPt)*iU2MSZMCFit ->Eval(iGenPt),0.);
  pFrac1 = sqrt(pFrac1)*lRescale;
  pFrac2 = sqrt(pFrac2)*lRescale;
 
  //  cout << "===> " << pU1 << " -- " << pFrac1 << " -- " << pFrac2 << " -- "  << iU1MSZDatFit->Eval(iGenPt) << " - " << iU1MSZMCFit->Eval(iGenPt) << endl;
  
  //Uncertainty propagation
  if(iFlucU2 != 0 || iFlucU1 != 0 || iScale != 0) { 
    /*
    /// ORIGIANL ERROR FROM PHIL

    double lEUR1    = getError(iGenPt,iU1RZDatFit ,PFU1);
    double lEU1Frac = getError(iGenPt,iU1MSZDatFit,PFMSU1);
    double lEU2Frac = getError(iGenPt,iU2MSZDatFit,PFMSU2);

    //Modify all the different parameters the choice of signs makes it maximal
    pU1       = pU1       + iScale*lEUR1;             //Recoil
    pFrac1    = pFrac1    + iFlucU1*(lEU1Frac);        //Mean RMS 
    pFrac2    = pFrac2    + iFlucU2*(lEU2Frac);        //Mean RMS for U2
    */

    /// MARIA IMPLEMENTATION: should be the same as Phil and original function getError
    double lEUR1    = sqrt(getError2(iGenPt, iU1RZDatFit));
    double lEUR1mc  = sqrt(getError2(iGenPt, iU1RZMCFit));

    double lEU1Frac = sqrt(getError2(iGenPt,iU1MSZDatFit));
    double lEU2Frac = sqrt(getError2(iGenPt,iU2MSZDatFit));

    double lEU1FracMC = sqrt(getError2(iGenPt,iU1MSZMCFit));
    double lEU2FracMC = sqrt(getError2(iGenPt,iU2MSZMCFit));

    double errorScale= pU1 * sqrt((lEUR1*lEUR1)/(iU1RZDatFit->Eval(iGenPt)*iU1RZDatFit->Eval(iGenPt))+(lEUR1mc*lEUR1mc)/(iU1RZMCFit->Eval(iGenPt)*iU1RZMCFit->Eval(iGenPt)));
    //    double errorResU1= lRescale * sqrt(lEU1Frac*lEU1Frac + (pU1*pU1)*lEU1FracMC*lEU1FracMC ) ; // for this we keep the scale constant
    //    double errorResU2= lRescale * sqrt(lEU2Frac*lEU2Frac + lEU2FracMC*lEU2FracMC ) ;

    double errorResU1=0;
    double errorResU2=0;

    if(pFrac1!=0) errorResU1 = (lRescale*lRescale/pFrac1) * sqrt(lEU1Frac*lEU1Frac*iU1MSZDatFit->Eval(iGenPt)*iU1MSZDatFit->Eval(iGenPt) + pU1*pU1*pU1*pU1*lEU1FracMC*lEU1FracMC*iU1MSZMCFit->Eval(iGenPt)*iU1MSZMCFit->Eval(iGenPt)); // for this we keep the scale constant                                                                                                          
    if(pFrac2!=0) errorResU2 = (lRescale*lRescale/pFrac2) * sqrt(lEU2Frac*lEU2Frac*iU2MSZDatFit->Eval(iGenPt)*iU2MSZDatFit->Eval(iGenPt) + lEU2FracMC*lEU2FracMC*iU2MSZMCFit->Eval(iGenPt)*iU2MSZMCFit->Eval(iGenPt)) ;

    pU1       = pU1       + iScale*errorScale;         //Recoil 
    pFrac1    = pFrac1    + iFlucU1*(errorResU1);        //Mean RMS 
    pFrac2    = pFrac2    + iFlucU2*(errorResU2);        //Mean RMS for U2 

  }

  //For the future
  //double lU1U2   = iU1U2Corr->Eval(iGenPt);
  //double pVal1_1 = correlatedSeed(pSigma1_1,lU1U2,0.,0.,pCorr1,pCorr2,0.,0.);
  //double pVal2_1 = correlatedSeed(pSigma2_1,lU1U2,0.,0.,pCorr2,pCorr1,0.,0.);

  double pUX  = iMet*cos(iMPhi) + iLepPt*cos(iLepPhi);
  double pUY  = iMet*sin(iMPhi) + iLepPt*sin(iLepPhi);
  double pU   = sqrt(pUX*pUX+pUY*pUY);

  // default
  double pCos = - (pUX*cos(iGenPhi) + pUY*sin(iGenPhi))/pU;
  double pSin =   (pUX*sin(iGenPhi) - pUY*cos(iGenPhi))/pU;

  pU1   = pU*pCos*pU1;//*(pU1*(iGenPt > 10) + (iGenPt > 10)*((1.-iGenPt/10.)*(pU1-1.)+1.));
  pU2   = pU*pSin;
  pU1   = iRand->Gaus(pU1,pFrac1);
  pU2   = iRand->Gaus(pU2,pFrac2);
  iMet  = calculate(0,iLepPt,iLepPhi,iGenPhi,pU1,pU2);
  iMPhi = calculate(1,iLepPt,iLepPhi,iGenPhi,pU1,pU2);
  iU1   = pU1; 
  iU2   = pU2;
  return;
}

void RecoilCorrector::metDistributionType2CorrU(double &iMet,double &iMPhi,double iGenPt,double iGenPhi,
						double iLepPt,double iLepPhi,
						TF1 *iU1Default,
						TF1 *iU1RZDatFit,  TF1 *iU1RZMCFit,
						TF1 *iU1MSZDatFit, TF1 *iU1MSZMCFit, 
						TF1 *iU1S1ZDatFit, TF1 *iU1S1ZMCFit, 
						TF1 *iU1S2ZDatFit, TF1 *iU1S2ZMCFit, 
						TF1 *iU2MSZDatFit, TF1 *iU2MSZMCFit,
						TF1 *iU2S1ZDatFit, TF1 *iU2S1ZMCFit,  		   		   
						TF1 *iU2S2ZDatFit, TF1 *iU2S2ZMCFit,  		   		   
						//					   TF1 *iU1U2ZDatCorr,TF1 *iU1U2ZMCCorr, // MARIA comment for now
						double &iU1,double &iU2,double iFlucU2, double iFlucU1, double iScale,
						bool doSingleGauss) {
  
  double pDefU1    = iU1Default->Eval(iGenPt);
  double lRescale  = sqrt((TMath::Pi())/2.);
  double pDU1       = iU1RZDatFit ->Eval(iGenPt);
  //double pDU2       = 0; sPM                                                                                                                                                         
  double pDFrac1    = iU1MSZDatFit->Eval(iGenPt)*lRescale;
  double pDSigma1_1 = iU1S1ZDatFit->Eval(iGenPt)*pDFrac1;
  double pDSigma1_2 = iU1S2ZDatFit->Eval(iGenPt)*pDFrac1;

  double pDFrac2    = iU2MSZDatFit->Eval(iGenPt)*lRescale;
  double pDSigma2_1 = iU2S1ZDatFit->Eval(iGenPt)*pDFrac2;
  double pDSigma2_2 = iU2S2ZDatFit->Eval(iGenPt)*pDFrac2;
  //double pDMean1    = pDFrac1;                                                                                                                                                       
  //double pDMean2    = pDFrac2;                                                                                                                                                       

  double pMU1       = iU1RZMCFit  ->Eval(iGenPt);
  //  double pMU2       = 0;                                                                                                                                                           
  double pMFrac1    = iU1MSZMCFit ->Eval(iGenPt)*lRescale;
  double pMSigma1_1 = iU1S1ZMCFit ->Eval(iGenPt)*pMFrac1;
  double pMSigma1_2 = iU1S2ZMCFit ->Eval(iGenPt)*pMFrac1;

  double pMFrac2    = iU2MSZMCFit ->Eval(iGenPt)*lRescale;
  double pMSigma2_1 = iU2S1ZMCFit ->Eval(iGenPt)*pMFrac2;
  double pMSigma2_2 = iU2S2ZMCFit ->Eval(iGenPt)*pMFrac2;

  //double pMMean1    = pMFrac1;                                                                                                                                                       
  //double pMMean2    = pMFrac2;                                

  //double pMMean1    = pMFrac1;                                                                                                                                                       
  //double pMMean2    = pMFrac2;                                                                                                                                                       
  //Uncertainty propagation                                                                                                                                                            
  pDFrac1     = (pDFrac1-pDSigma1_2)/(pDSigma1_1-pDSigma1_2);
  pDFrac2     = (pDFrac2-pDSigma2_2)/(pDSigma2_1-pDSigma2_2);
  pMFrac1     = (pMFrac1-pMSigma1_2)/(pMSigma1_1-pMSigma1_2);
  pMFrac2     = (pMFrac2-pMSigma2_2)/(pMSigma2_1-pMSigma2_2);

  /////                                                                                                                                                                                
  /////                                                                                                                                                                                
  /////                                                                                                                                                                                

  double pUX   = iMet*cos(iMPhi) + iLepPt*cos(iLepPhi);
  double pUY   = iMet*sin(iMPhi) + iLepPt*sin(iLepPhi);
  double pU    = sqrt(pUX*pUX+pUY*pUY);

  double pCos  = - (pUX*cos(iGenPhi) + pUY*sin(iGenPhi))/pU;
  double pSin  =   (pUX*sin(iGenPhi) - pUY*cos(iGenPhi))/pU;

  double offset = iU1RZMCFit->Eval(iGenPt);

  bool scaleU2=true;

  double normSigmaM = iU2MSZMCFit ->Eval(iGenPt)/iU1MSZMCFit ->Eval(iGenPt);
  if(!scaleU2) normSigmaM = iU1MSZMCFit ->Eval(iGenPt)/iU2MSZMCFit ->Eval(iGenPt);
  double normSigmaD = iU2MSZDatFit ->Eval(iGenPt)/iU1MSZDatFit ->Eval(iGenPt);
  if(!scaleU2) normSigmaD = iU1MSZDatFit ->Eval(iGenPt)/iU2MSZDatFit ->Eval(iGenPt);

  double pU1   = pU*pCos;
  double pU2   = pU*pSin;
  double pU1Diff  = pU1-offset; // pU1 of the event ; pDefU1 is the scale iU1RZDatFit ->Eval(iGenPt);                                                                                  
  double pU2Diff  = pU2;

  pU1Diff*=normSigmaM; 

  double recoil = sqrt(pU1Diff*pU1Diff+pU2Diff*pU2Diff); 

  //  double pU1ValM = 0;
  //  double pU2ValM = 0 ;
  double pU1ValD = 0 ;
  double pU2ValD = 0;
  double pUValM = 0;
  double pUValD = 0 ;

  if(doSingleGauss) {

    if(scaleU2) pUValM         = diGausPVal(fabs(recoil),1,iU2MSZMCFit ->Eval(iGenPt)*lRescale,0);
    if(scaleU2) pUValD         = oneGausPInverse(pUValM ,1,iU2MSZDatFit->Eval(iGenPt)*lRescale,0);

    if(!scaleU2) pUValM         = diGausPVal(fabs(recoil),1,iU1MSZMCFit ->Eval(iGenPt)*lRescale,0);
    if(!scaleU2) pUValD         = oneGausPInverse(pUValM ,1,iU1MSZDatFit->Eval(iGenPt)*lRescale,0);

  } else {

    if(scaleU2) pUValM         = diGausPVal(fabs(recoil),pMFrac2,pMSigma2_1,pMSigma2_2);
    if(scaleU2) pUValD         = diGausPInverse(pUValM  ,pDFrac2,pDSigma2_1,pDSigma2_2);

    if(!scaleU2) pUValM         = diGausPVal(fabs(recoil),pMFrac1,pMSigma1_1,pMSigma1_2);
    if(!scaleU2) pUValD         = diGausPInverse(pUValM  ,pDFrac1,pDSigma1_1,pDSigma1_2);

  }

  if(pUValM==fabs(recoil)) pUValD=fabs(recoil); // in those cases do nothing  since Erf is zero  

  pU1ValD=pUValD*(pU1Diff/recoil)*(1/normSigmaD);
  pU2ValD=pUValD*(pU2Diff/recoil);
  pDefU1 *= (pDU1/pMU1);


  pU1   = pDefU1       +       pU1ValD;
  pU2   =                      pU2ValD;
  iMet  = calculate(0,iLepPt,iLepPhi,iGenPhi,pU1,pU2);
  iMPhi = calculate(1,iLepPt,iLepPhi,iGenPhi,pU1,pU2);
  iU1   = pU1;                                                                                                                                                                     
  iU2   = pU2;                                                                                                                                                                     

  return;

}

double RecoilCorrector::diGausPVal(double iVal,double iFrac,double iSigma1,double iSigma2) { 

  double lVal=iFrac*TMath::Erf(iVal/iSigma1) + (1-iFrac)*TMath::Erf(iVal/iSigma2);
  if(TMath::ErfInverse(lVal)==0) return iVal;
  return lVal;


}
double RecoilCorrector::diGausPInverse(double iPVal,double iFrac,double iSigma1,double iSigma2) { 
  double lVal = TMath::ErfInverse(iPVal);
  double lMin = lVal * ((iSigma1 < iSigma2) ? iSigma1 : iSigma2);
  double lMax = lVal * ((iSigma1 < iSigma2) ? iSigma2 : iSigma1);
  //  cout << "-- Min - " << lMin <<  " -> " << lMax << " -- " << iSigma1 << " -- " << iSigma2 << endl;
  double lDiff = (lMax-lMin);
  //Iterative procedure to invert a double gaussian given a PVal
  //  int lId = 0; int lN1 = 4;  int lN2 = 10; 
  int lId = 0; int lN1 = 10;  int lN2 = 100; 
  for(int i0 = 0; i0 < lN1; i0++) { 
    if(i0 != 0) lMin = lMin + (lId-1)*lDiff/lN2;
    if(i0 != 0) lDiff/=lN2;
    for(int i1 = 0; i1 < lN2; i1++) { 
      double pVal = lMin + lDiff/lN2*i1;
      pVal = diGausPVal(pVal,iFrac,iSigma1,iSigma2);
      if(pVal > iPVal) {lId = i1; break;}
      //      if(pVal < iPVal && lDiff < 0 ) {lId = i1; break;}
    }
  }
  //  cout << "-- Final Val "  <<  (lMin + (lId-0.5)*lDiff/lN2) << " -- " << lId << endl;
  return (lMin + (lId-0.5)*lDiff/lN2);
}

double RecoilCorrector::oneGausPInverse(double iPVal,double iFrac,double iSigma1,double iSigma2) {

  double lVal = TMath::ErfInverse(iPVal);
  double pVal = lVal *iSigma1;
  //  double sigmaD = diGausPVal(pVal,iFrac,iSigma1,iSigma2);
  return pVal;

}

void RecoilCorrector::metDistributionType2(double &iMet,double &iMPhi,double iGenPt,double iGenPhi,
					   double iLepPt,double iLepPhi,
					   TF1 *iU1Default,
					   TF1 *iU1RZDatFit,  TF1 *iU1RZMCFit,
					   TF1 *iU1MSZDatFit, TF1 *iU1MSZMCFit, 
					   TF1 *iU1S1ZDatFit, TF1 *iU1S1ZMCFit, 
					   TF1 *iU1S2ZDatFit, TF1 *iU1S2ZMCFit, 
					   TF1 *iU2MSZDatFit, TF1 *iU2MSZMCFit,
					   TF1 *iU2S1ZDatFit, TF1 *iU2S1ZMCFit,  		   		   
					   TF1 *iU2S2ZDatFit, TF1 *iU2S2ZMCFit,  		   		   
					   //					   TF1 *iU1U2ZDatCorr,TF1 *iU1U2ZMCCorr, // MARIA comment for now
					   double &iU1,double &iU2,double iFlucU2, double iFlucU1, double iScale,
					   bool doSingleGauss) {
  //  cout << "inside metType2 " << endl;

  double pDefU1    = iU1Default->Eval(iGenPt);
  double lRescale  = sqrt((TMath::Pi())/2.);		     
  double pDU1       = iU1RZDatFit ->Eval(iGenPt);
  //double pDU2       = 0; sPM
  double pDFrac1    = iU1MSZDatFit->Eval(iGenPt)*lRescale;
  double pDSigma1_1 = iU1S1ZDatFit->Eval(iGenPt)*pDFrac1;
  double pDSigma1_2 = iU1S2ZDatFit->Eval(iGenPt)*pDFrac1;
  double pDFrac2    = iU2MSZDatFit->Eval(iGenPt)*lRescale;
  double pDSigma2_1 = iU2S1ZDatFit->Eval(iGenPt)*pDFrac2;
  double pDSigma2_2 = iU2S2ZDatFit->Eval(iGenPt)*pDFrac2;
  //double pDMean1    = pDFrac1;
  //double pDMean2    = pDFrac2;
 
  double pMU1       = iU1RZMCFit  ->Eval(iGenPt);
  //  double pMU2       = 0; 
  double pMFrac1    = iU1MSZMCFit ->Eval(iGenPt)*lRescale;
  double pMSigma1_1 = iU1S1ZMCFit ->Eval(iGenPt)*pMFrac1;
  double pMSigma1_2 = iU1S2ZMCFit ->Eval(iGenPt)*pMFrac1;
  double pMFrac2    = iU2MSZMCFit ->Eval(iGenPt)*lRescale;
  double pMSigma2_1 = iU2S1ZMCFit ->Eval(iGenPt)*pMFrac2;
  double pMSigma2_2 = iU2S2ZMCFit ->Eval(iGenPt)*pMFrac2;
  //double pMMean1    = pMFrac1;
  //double pMMean2    = pMFrac2;
  //Uncertainty propagation

  if(iFlucU1 != 0 || iFlucU2 != 0 || iScale != 0) { 

    double lEUR1    = getError(iGenPt,iU1Default  ,PFU1);
    //    double lEUS1_1  = getError(iGenPt,iU1S1ZDatFit,PFS1U1);
    //    double lEUS1_2  = getError(iGenPt,iU1S2ZDatFit,PFS2U1);
    double lEU1Frac = getError(iGenPt,iU1MSZDatFit,PFMSU1);
    //    double lEUS2_1  = getError(iGenPt,iU2S1ZDatFit,PFS1U2);
    //    double lEUS2_2  = getError(iGenPt,iU2S2ZDatFit,PFS2U2);
    double lEU2Frac = getError(iGenPt,iU2MSZDatFit,PFMSU2);

    //cout << "Err u1    : " << lEU1Frac << " -- " << iFluc << " -- " << pDFrac1 << " -- " << iU1MSZDatFit->GetParError(0) << endl;
    //cout << "Err u2    : " << lEU2Frac << " -- " << iFluc << " -- " << pDFrac2 << endl;
    //cout << "Err u1 s1 : " << lEUS1_1 << endl;
    //cout << "Err u1 s2 : " << lEUS1_2 << endl;
    //cout << "Err u2 s1 : " << lEUS2_1 << endl;
    //cout << "Err u2 s2 : " << lEUS2_2 << endl;
  
    //Modify all the different parameters the choice of signs makes it maximal
    pDU1       = pDU1       + iScale*lEUR1;             //Recoil

    pDFrac1    = pDFrac1    + iFlucU1*(lEU1Frac);        //Mean RMS 
    pDSigma1_1 = pDSigma1_1 + iFlucU1*lEU1Frac;//lEUS1_1*pDFrac1;    //Sigma 1 smalles sigma
    pDSigma1_2 = pDSigma1_2 + iFlucU1*lEU1Frac;//lEUS1_2*pDFrac1;    //Sigma 2 (Maximal when oppsite sigma 1)

    pDFrac2    = pDFrac2    + iFlucU2*(lEU2Frac);        //Mean RMS for U2
    pDSigma2_1 = pDSigma2_1 + iFlucU2*lEU2Frac;//lEUS2_1*pDFrac2;    //Sigma 1 U2
    pDSigma2_2 = pDSigma2_2 + iFlucU2*lEU2Frac;//(lEUS2_2)*pDFrac2;

  }

  pDFrac1     = (pDFrac1-pDSigma1_2)/(pDSigma1_1-pDSigma1_2);
  pDFrac2     = (pDFrac2-pDSigma2_2)/(pDSigma2_1-pDSigma2_2);
  pMFrac1     = (pMFrac1-pMSigma1_2)/(pMSigma1_1-pMSigma1_2);
  pMFrac2     = (pMFrac2-pMSigma2_2)/(pMSigma2_1-pMSigma2_2);

  double pUX   = iMet*cos(iMPhi) + iLepPt*cos(iLepPhi);
  double pUY   = iMet*sin(iMPhi) + iLepPt*sin(iLepPhi);
  double pU    = sqrt(pUX*pUX+pUY*pUY);

  double pCos  = - (pUX*cos(iGenPhi) + pUY*sin(iGenPhi))/pU;
  double pSin  =   (pUX*sin(iGenPhi) - pUY*cos(iGenPhi))/pU;

  /////
  
  double pU1   = pU*pCos;
  double pU2   = pU*pSin;
  double pU1Diff  = pU1-pDefU1;
  double pU2Diff  = pU2;

  double p1Charge        = pU1Diff/fabs(pU1Diff);
  double p2Charge        = pU2Diff/fabs(pU2Diff);

  /*
  double pTU1Diff        = pU1Diff;
  // double lMU1U2  = iU1U2ZMCCorr->Eval(iGenPt);
  // pU1Diff                = deCorrelate(pMMean1,lMU1U2,0.,0.,pU1Diff/pMMean1,pU2Diff/pMMean1 ,0.,0.);
  //pU2Diff                = deCorrelate(pMMean2,lMU1U2,0.,0.,pU2Diff/pMMean2,pTU1Diff/pMMean2,0.,0.);
  double pU1ValM         = diGausPVal(fabs(pU1Diff),pMFrac1,pMSigma1_1,pMSigma1_2);
  double pU2ValM         = diGausPVal(fabs(pU2Diff),pMFrac2,pMSigma2_1,pMSigma2_2);
  double pU1ValD         = diGausPInverse(pU1ValM  ,pDFrac1,pDSigma1_1,pDSigma1_2);
  double pU2ValD         = diGausPInverse(pU2ValM  ,pDFrac2,pDSigma2_1,pDSigma2_2);
  */

  double pU1ValM = 0;
  double pU2ValM = 0 ;
  double pU1ValD = 0 ;
  double pU2ValD = 0;

  if(doSingleGauss) {

    pU1ValM         = diGausPVal(fabs(pU1Diff),1,iU1MSZMCFit ->Eval(iGenPt)*lRescale,0); // when is singleGauss pMFrac1=1 pMSigma1_1=fullRMS pMSigma1_2=0                                   
    pU2ValM         = diGausPVal(fabs(pU2Diff),1,iU2MSZMCFit ->Eval(iGenPt)*lRescale,0);
    pU1ValD         = oneGausPInverse(pU1ValM  ,1,iU1MSZDatFit->Eval(iGenPt)*lRescale,0);
    pU2ValD         = oneGausPInverse(pU2ValM  ,1,iU2MSZDatFit->Eval(iGenPt)*lRescale,0);

  } else {

    pU1ValM         = diGausPVal(fabs(pU1Diff),pMFrac1,pMSigma1_1,pMSigma1_2); // when is singleGauss pMFrac1=1 pMSigma1_1=fullRMS pMSigma1_2=0                                             
    pU2ValM         = diGausPVal(fabs(pU2Diff),pMFrac2,pMSigma2_1,pMSigma2_2);
    pU1ValD         = diGausPInverse(pU1ValM  ,pDFrac1,pDSigma1_1,pDSigma1_2);
    pU2ValD         = diGausPInverse(pU2ValM  ,pDFrac2,pDSigma2_1,pDSigma2_2);

  }

  if(pU1ValM==fabs(pU1Diff)) pU1ValD=fabs(pU1Diff); // in those cases do nothing  since Erf is zero                                                                                       
  if(pU2ValM==fabs(pU2Diff)) pU2ValD=fabs(pU2Diff); // in those cases do nothing  since Erf is zero                                                                                       


  pU1ValD*=p1Charge;
  pU2ValD*=p2Charge;
  pDefU1 *= (pDU1/pMU1);


  pU1   = pDefU1             + pU1ValD;
  pU2   =                      pU2ValD;
  iMet  = calculate(0,iLepPt,iLepPhi,iGenPhi,pU1,pU2);
  iMPhi = calculate(1,iLepPt,iLepPhi,iGenPhi,pU1,pU2);

  iU1   = pU1;
  iU2   = pU2;
  return;

  /*
  //Not Used Current
  iU1U2ZMCCorr ->Eval(iGenPt);
  iU1U2ZDatCorr->Eval(iGenPt);
  */

}

void RecoilCorrector::metDistribution(double &iPFMet,double &iPFMPhi,double &iTKMet,double &iTKMPhi,
				      double iGenPt,double iGenPhi,
		                      double iLepPt,double iLepPhi,TRandom3 *iRand,
		                      TF1 *iU1RPFFit,   TF1 *iU1RTKFit,
		                      TF1 *iU1MSPFFit,  TF1 *iU1MSTKFit, 
		                      TF1 *iU1S1PFFit,  TF1 *iU1S1TKFit,
		                      TF1 *iU1S2PFFit,  TF1 *iU1S2TKFit,
		                      TF1 *iU2MSPFFit,  TF1 *iU2MSTKFit, 		      
		                      TF1 *iU2S1PFFit,  TF1 *iU2S1TKFit, 
		                      TF1 *iU2S2PFFit,  TF1 *iU2S2TKFit, 
		                      TF1 *iPFU1U2Corr, TF1 *iTKU1U2Corr,
		                      TF1 *iPFTKU1Corr, TF1 *iPFTKU2Corr,
		                      TF1 *iPFTKU1MCorr,TF1 *iPFTKU2MCorr,
				      double       &iU1, double &iU2,
				      double iFluc,double iScale) {
  //Important constants re-scaling of sigma on left and mean wpt of W resbos on right
  double lRescale  = sqrt((TMath::Pi())/2.); //double lPtMean = 16.3; //==> tuned for W bosons
  ///
  double pPFU1       = CorrVal(iGenPt,iU1RPFFit->Eval(iGenPt),PFU1);
  double pPFU2       = 0;
  double pPFSigma1_1 = CorrVal(iGenPt,iU1S1PFFit->Eval(iGenPt),PFS1U1)*CorrVal(iGenPt,iU1MSPFFit->Eval(iGenPt),PFMSU1)*lRescale;
  double pPFSigma1_2 = CorrVal(iGenPt,iU1S2PFFit->Eval(iGenPt),PFS2U1)*CorrVal(iGenPt,iU1MSPFFit->Eval(iGenPt),PFMSU1)*lRescale;
  double pPFFrac1    = CorrVal(iGenPt,iU1MSPFFit->Eval(iGenPt),PFMSU1)                         *lRescale;
  double pPFSigma2_1 = CorrVal(iGenPt,iU2S1PFFit->Eval(iGenPt),PFS1U2)*CorrVal(iGenPt,iU2MSPFFit->Eval(iGenPt),PFMSU2)*lRescale;
  double pPFSigma2_2 = CorrVal(iGenPt,iU2S2PFFit->Eval(iGenPt),PFS2U2)*CorrVal(iGenPt,iU2MSPFFit->Eval(iGenPt),PFMSU2)*lRescale;
  double pPFFrac2    = CorrVal(iGenPt,iU2MSPFFit->Eval(iGenPt),PFMSU2)                         *lRescale;
  if(pPFSigma1_1 > pPFSigma1_2) {double pT = pPFSigma1_2; pPFSigma1_2 = pPFSigma1_1; pPFSigma1_1 = pT;}
  if(pPFSigma2_1 > pPFSigma2_2) {double pT = pPFSigma2_2; pPFSigma2_2 = pPFSigma2_1; pPFSigma2_2 = pT;}
  
  double pTKU1       = CorrVal(iGenPt,iU1RTKFit->Eval(iGenPt),TKU1);
  double pTKU2       = 0;
  double pTKSigma1_1 = CorrVal(iGenPt,iU1S1TKFit->Eval(iGenPt),TKS1U1)*CorrVal(iGenPt,iU1MSTKFit->Eval(iGenPt),TKMSU1)*lRescale;
  double pTKSigma1_2 = CorrVal(iGenPt,iU1S2TKFit->Eval(iGenPt),TKS2U1)*CorrVal(iGenPt,iU1MSTKFit->Eval(iGenPt),TKMSU1)*lRescale;
  double pTKFrac1    = CorrVal(iGenPt,iU1MSTKFit->Eval(iGenPt),TKMSU1)                         *lRescale;
  double pTKSigma2_1 = CorrVal(iGenPt,iU2S1TKFit->Eval(iGenPt),TKS1U2)*CorrVal(iGenPt,iU2MSTKFit->Eval(iGenPt),TKMSU2)*lRescale;
  double pTKSigma2_2 = CorrVal(iGenPt,iU2S2TKFit->Eval(iGenPt),TKS2U2)*CorrVal(iGenPt,iU2MSTKFit->Eval(iGenPt),TKMSU2)*lRescale;
  double pTKFrac2    = CorrVal(iGenPt,iU2MSTKFit->Eval(iGenPt),TKMSU2)                         *lRescale;
  if(pTKSigma1_1 > pTKSigma1_2) {double pT = pTKSigma1_2; pTKSigma1_2 = pTKSigma1_1; pTKSigma1_1 = pT;}
  if(pTKSigma2_1 > pTKSigma2_2) {double pT = pTKSigma2_2; pTKSigma2_2 = pTKSigma2_1; pTKSigma2_2 = pT;}

  //Uncertainty propagation
  if(iFluc != 0 || iScale != 0) { 
    double lEUR1    = getError(iGenPt ,iU1RPFFit , PFU1);
    double lEUS1_1  = getError(iGenPt ,iU1S1PFFit, PFS1U1);
    double lEUS1_2  = getError(iGenPt ,iU1S2PFFit, PFS2U1);
    double lEU1Frac = getError(iGenPt ,iU1MSPFFit, PFMSU1)*lRescale;
    double lEUS2_1  = getError(iGenPt ,iU2S1PFFit, PFS1U2);
    double lEUS2_2  = getError(iGenPt ,iU2S2PFFit, PFS2U2);
    double lEU2Frac = getError(iGenPt ,iU2MSPFFit, PFMSU2)*lRescale;
    //cout << "===> " << pPFSigma1_1 << " -- " << iU1S2PFFit->GetParError(0) << " -- " << lEUS1_1 << endl;
    //Modify all the different parameters the choice of signs makes it maximal
    if(iU1S1PFFit->Eval(iGenPt) > 1) {double pPF = lEUS1_1; lEUS1_1 = lEUS1_2; lEUS1_1 = pPF;}
    if(iU2S1PFFit->Eval(iGenPt) > 1) {double pPF = lEUS2_1; lEUS2_1 = lEUS2_2; lEUS2_1 = pPF;}

    pPFU1       = pPFU1       + iScale*lEUR1;              //Recoil
    pPFFrac1    = pPFFrac1    + iFluc*(lEU1Frac);        //Mean RMS 
    pPFSigma1_1 = pPFSigma1_1 + iFluc*lEUS1_1*pPFFrac1;    //Sigma 1 smalles sigma
    pPFSigma1_2 = pPFSigma1_2 - iFluc*lEUS1_2*pPFFrac1;    //Sigma 2 (Maximal when oppsite sigma 1)
    pPFFrac2    = pPFFrac2    + iFluc*(lEU2Frac);        //Mean RMS for U2
    pPFSigma2_1 = pPFSigma2_1 + iFluc*lEUS2_1*pPFFrac2  ;    //Sigma 1 U2
    pPFSigma2_2 = pPFSigma2_2 - iFluc*(lEUS2_2)*pPFFrac2;
    
    lEUR1    = getError(iGenPt,iU1RTKFit ,TKU1);
    lEUS1_1  = getError(iGenPt,iU1S1TKFit,TKS1U1);
    lEUS1_2  = getError(iGenPt,iU1S2TKFit,TKS2U1);
    lEU1Frac = getError(iGenPt,iU1MSTKFit,TKMSU1)*lRescale;
    lEUS2_1  = getError(iGenPt,iU2S1TKFit,TKS1U2);
    lEUS2_2  = getError(iGenPt,iU2S2TKFit,TKS2U2);
    lEU2Frac = getError(iGenPt,iU2MSTKFit,TKMSU2)*lRescale;
    if(iU1S1TKFit->Eval(iGenPt) > 1) {double pPF = lEUS1_1; lEUS1_1 = lEUS1_2; lEUS1_1 = pPF;}
    if(iU2S1TKFit->Eval(iGenPt) > 1) {double pPF = lEUS2_1; lEUS2_1 = lEUS2_2; lEUS2_1 = pPF;}
    //Modify all the different parameters the choice of signs makes it maximal
    pTKU1       = pTKU1        + iScale*lEUR1;              //Recoil
    pTKFrac1    = pTKFrac1     + iFluc*(lEU1Frac);        //Mean RMS 
    pTKSigma1_1 = pTKSigma1_1  +  iFluc*lEUS1_1 *pTKFrac1;    //Sigma 1 smalles sigma
    pTKSigma1_2 = pTKSigma1_2  -  iFluc*lEUS1_2 *pTKFrac1;    //Sigma 2 (Maximal when oppsite sigma 1)
    pTKFrac2    = pTKFrac2     + iFluc*(lEU2Frac);        //Mean RMS for U2
    pTKSigma2_1 = pTKSigma2_1  + iFluc*lEUS2_1  *pTKFrac2;    //Sigma 1 U2
    pTKSigma2_2 = pTKSigma2_2  - iFluc*(lEUS2_2)*pTKFrac2;
  }
  //double lMeanU1 = pPFFrac1;
  //double lMeanU2 = pPFFrac2;
    
  //Caculat the proper fraction
  pPFFrac1 = (pPFFrac1-pPFSigma1_2)/(pPFSigma1_1-pPFSigma1_2);
  pPFFrac2 = (pPFFrac2-pPFSigma2_2)/(pPFSigma2_1-pPFSigma2_2);

  pTKFrac1 = (pTKFrac1-pTKSigma1_2)/(pTKSigma1_1-pTKSigma1_2);
  pTKFrac2 = (pTKFrac2-pTKSigma2_2)/(pTKSigma2_1-pTKSigma2_2);
  //if(iGenPt > 60) cout << "===> " << pFrac1 << " -- " <<pFrac2 << endl;
  double pPFVal0 = iRand->Uniform(0,1);
  double pPFVal1 = iRand->Uniform(0,1);

  double pTKVal0 = iRand->Uniform(0,1);
  double pTKVal1 = iRand->Uniform(0,1);

  double pPFCorr1     = iRand->Gaus(0,1);     double pPFCorr2     = iRand->Gaus(0,1);  
  //double pPFCorrT1    = iRand->Gaus(0,1);     double pPFCorrT2    = iRand->Gaus(0,1);  

  double pTKCorr1     = iRand->Gaus(0,1);     double pTKCorr2     = iRand->Gaus(0,1);  
  //double pTKCorrT1    = iRand->Gaus(0,1);     double pTKCorrT2    = iRand->Gaus(0,1);  

  double lPFU1U2  = TMath::Max(iPFU1U2Corr->Eval(iGenPt),0.);//iPFU1U2Corr->Eval(iGenPt) ,0.);
  double lTKU1U2  = TMath::Max(iTKU1U2Corr->Eval(iGenPt),0.);//iTKU1U2Corr->Eval(iGenPt) ,0.);
  double lPFTKU1  = iPFTKU1Corr ->Eval(iGenPt); //TMath::Max(iPFTKU1Corr->Eval(iGenPt) ,0.);
  double lPFTKU2  = iPFTKU2Corr ->Eval(iGenPt);//TMath::Max(iPFTKU2Corr->Eval(iGenPt) ,0.);
  double lPFTKU1M = iPFTKU1MCorr->Eval(iGenPt);
  double lPFTKU2M = iPFTKU2MCorr->Eval(iGenPt);

  //Needs some more work
  double pScale = 1.; //pPFFrac1 = (pPFFrac1+pPFFrac2)/2.; //pPFFrac2 = pPFFrac1;//+pPFFrac2)/2.;
  pPFSigma1_1 = ((pPFVal0 < pPFFrac1)*(pPFSigma1_1)+(pPFVal0 > pPFFrac1)*(pPFSigma1_2)); pPFVal0 = -1;
  pPFSigma2_1 = ((pPFVal1 < pPFFrac2)*(pPFSigma2_1)+(pPFVal1 > pPFFrac2)*(pPFSigma2_2)); pPFVal1 = -1;
  pTKSigma1_1 = (pTKVal0 < pTKFrac1)*(pTKSigma1_1)+(pTKVal0 > pTKFrac1)*(pTKSigma1_2);   pTKVal0 = -1;
  pTKSigma2_1 = (pTKVal1 < pTKFrac2)*(pTKSigma2_1)+(pTKVal1 > pTKFrac2)*(pTKSigma2_2);   pTKVal1 = -1;
  
  //double pPFSigma1_1_old = pPFSigma1_1*(lMeanU2/lMeanU1);
  //double pPFVal1_1 = correlatedSeed(pPFSigma1_1*pScale,lPFU1U2,lPFTKU1,lPFTKU1M,pPFSigma1_1,pPFSigma2_1*(lMeanU1/lMeanU2),pTKCorr1,pTKCorr2);
  //double pPFVal2_1 = correlatedSeed(pPFSigma2_1*pScale,lPFU1U2,lPFTKU2,lPFTKU2M,pPFSigma2_1,pPFSigma1_1_old,pTKCorr2,pTKCorr1);

  double pPFVal1_1 = correlatedSeed(pPFSigma1_1*pScale,lPFU1U2,lPFTKU1,lPFTKU1M,pPFCorr1,pPFCorr2,pTKCorr1,pTKCorr2);
  double pPFVal2_1 = correlatedSeed(pPFSigma2_1*pScale,lPFU1U2,lPFTKU2,lPFTKU2M,pPFCorr2,pPFCorr1,pTKCorr2,pTKCorr1);
  double pTKVal1_1 = correlatedSeed(pTKSigma1_1*pScale,lTKU1U2,lPFTKU1,lPFTKU2M,pTKCorr1,pTKCorr2,pPFCorr1,pPFCorr2);
  double pTKVal2_1 = correlatedSeed(pTKSigma2_1*pScale,lTKU1U2,lPFTKU2,lPFTKU1M,pTKCorr2,pTKCorr1,pPFCorr2,pPFCorr1);
   
  //double pPFVal1_2 = correlatedSeed(pPFSigma1_2*pScale,lPFU1U2,lPFTKU1,lPFTKU1M,pPFCorrT1,pPFCorrT2,pTKCorrT1,pTKCorrT2);
  //double pPFVal2_2 = correlatedSeed(pPFSigma2_2*pScale,lPFU1U2,lPFTKU2,lPFTKU2M,pPFCorrT2,pPFCorrT1,pTKCorrT2,pTKCorrT1);
  //double pTKVal1_2 = correlatedSeed(pTKSigma1_2*pScale,lTKU1U2,lPFTKU1,lPFTKU2M,pTKCorrT1,pTKCorrT2,pPFCorrT1,pPFCorrT2);
  //double pTKVal2_2 = correlatedSeed(pTKSigma2_2*pScale,lTKU1U2,lPFTKU2,lPFTKU1M,pTKCorrT2,pTKCorrT1,pPFCorrT2,pPFCorrT1);

  pPFU1   = (pPFVal0 < pPFFrac1)*(pPFVal1_1+pPFU1);//+(pPFVal0 > pPFFrac1)*(pPFVal1_2+pPFU1);
  pPFU2   = (pPFVal1 < pPFFrac2)*(pPFVal2_1+pPFU2);//+(pPFVal1 > pPFFrac2)*(pPFVal2_2+pPFU2);
  pTKU1   = (pTKVal0 < pTKFrac1)*(pTKVal1_1+pTKU1);//+(pTKVal0 > pTKFrac1)*(pTKVal1_2+pTKU1);
  pTKU2   = (pTKVal1 < pTKFrac2)*(pTKVal2_1+pTKU2);//+(pTKVal1 > pTKFrac2)*(pTKVal2_2+pTKU2);

  iPFMet  = calculate(0,iLepPt,iLepPhi,iGenPhi,pPFU1,pPFU2);
  iPFMPhi = calculate(1,iLepPt,iLepPhi,iGenPhi,pPFU1,pPFU2);
  iTKMet  = calculate(0,iLepPt,iLepPhi,iGenPhi,pTKU1,pTKU2);
  iTKMPhi = calculate(1,iLepPt,iLepPhi,iGenPhi,pTKU1,pTKU2);
  iU1 = pTKU1;
  iU2 = pTKU2;
  return;
  //Not used right now
  //iPFTKU2Corr->Eval(0);
  //iPFTKU1MCorr.Eval(0);
  //iPFTKU2MCorr.Eval(0);
}

//-----------------------------------------------------------------------------------------------------------------------------------------
double RecoilCorrector::calculate(int iMet,double iEPt,double iEPhi,double iWPhi,double iU1,double iU2) { 
  double lMX = -iEPt*cos(iEPhi) - iU1*cos(iWPhi) + iU2*sin(iWPhi);
  double lMY = -iEPt*sin(iEPhi) - iU1*sin(iWPhi) - iU2*cos(iWPhi);

  if(iMet == 0) return sqrt(lMX*lMX + lMY*lMY);
  if(iMet == 1) {if(lMX > 0) {return atan(lMY/lMX);} return (fabs(lMY)/lMY)*3.14159265 + atan(lMY/lMX); } 
  if(iMet == 2) return lMX;
  if(iMet == 3) return lMY;
  return lMY;

}

//-----------------------------------------------------------------------------------------------------------------------------------------
double RecoilCorrector::getCorError2(double iVal,TF1 *iFit) { 
  double lE = sqrt(iFit->GetParError(0))  + iVal*sqrt(iFit->GetParError(2));
  if(fabs(iFit->GetParError(4)) > 0) lE += iVal*iVal*sqrt(iFit->GetParError(4));
  return lE*lE;
}

//-----------------------------------------------------------------------------------------------------------------------------------------
double RecoilCorrector::getError2(double iVal,TF1 *iFit) { 
  //  return iFit->GetParError(0);
  double lE2 = iFit->GetParError(0) + iVal*iFit->GetParError(1) + iVal*iVal*iFit->GetParError(2);
  if(iFit->GetParError(3) != 0 ) lE2 += iVal*iVal*iVal*  iFit->GetParError(3);
  if(iFit->GetParError(4) != 0 ) lE2 += iVal*iVal*iVal*iVal*  iFit->GetParError(4);
  if(iFit->GetParError(5) != 0 ) lE2 += iVal*iVal*iVal*iVal*iVal*  iFit->GetParError(5);
  if(iFit->GetParError(6) != 0 ) lE2 += iVal*iVal*iVal*iVal*iVal*iVal*  iFit->GetParError(6);

  if(lE2<0) {
    cout << "negative error: this will cause nan for iVal=" << iVal << endl;
    cout << "iFit->GetParError(0) " << iFit->GetParError(0) << endl;
    cout << "iFit->GetParError(1) " << iFit->GetParError(1) << " xE1 " << iVal*iFit->GetParError(1) << endl;
    cout << "iFit->GetParError(2) " << iFit->GetParError(2) << " x2E2 " << iVal*iVal*iFit->GetParError(2) << endl;
    cout << "iFit->GetParError(3) " << iFit->GetParError(3) << " x3E3 " << iVal*iVal*iVal*iFit->GetParError(3) <<  endl;
    cout << "iFit->GetParError(4) " << iFit->GetParError(4) << " x4E4 " << iVal*iVal*iVal*iVal*iFit->GetParError(4) <<  endl;
    cout << "iFit->GetParError(5) " << iFit->GetParError(5) << " x5E5 " << iVal*iVal*iVal*iVal*iVal*iFit->GetParError(5) <<  endl;
    cout << "iFit->GetParError(6) " << iFit->GetParError(6) << " x6E6 " << iVal*iVal*iVal*iVal*iVal*iVal*iFit->GetParError(6) <<  endl;
    cout << "THIS SHOULD NEVER HAPPEN, MEANS THE ERRORS ARE NOT PROPERLY STORED : so do not trust them" << endl;
  }

  return lE2;
}

//-----------------------------------------------------------------------------------------------------------------------------------------
double RecoilCorrector::getError(double iVal,TF1 *iFit,Recoil iType) {

  //  return sqrt(getError2(iVal,iFit)); 
  // MARIA: this should be good for the scale for type2 (for type1 need only the lER part)                                                                                  

  double lEW2  = getError2(iVal,iFit);
  double lEZD2 = getError2(iVal,getFunc(true ,iType));
  double lEZM2 = getError2(iVal,getFunc(false,iType));
  double lZDat = getFunc(true ,iType)->Eval(iVal);
  double lZMC  = getFunc(false,iType)->Eval(iVal);
  double lWMC  = iFit                ->Eval(iVal);
  double lR    = lZDat/lZMC;
  double lER   = lR*lR/lZDat/lZDat*lEZD2 + lR*lR/lZMC/lZMC*lEZM2;
  double lVal  = lR*lR*lEW2 + lWMC*lWMC*lER;

  //cout << "====> Error Data : "<<  lEZD2 << " MC : " << lEZM2 << " -- Rat " << lR << " -- DatV " << lZDat << " -- MCV " << lZMC << " -- " << lWMC << " -- Total " << lER  << " -- W : " << lEW2 << " -- All : " << lVal << endl;                                                                                                                      
  return sqrt(lVal);

}

//-----------------------------------------------------------------------------------------------------------------------------------------
double RecoilCorrector::mag(double iV0,double iV1,double iV2,double iV3) { 
  return sqrt(iV0+iV1*iV1+2*iV1*0.88 + iV2*iV2+2.*iV2*0.88+ iV3*iV3+2.*iV3*0.88);//
  //return sqrt(iV0*iV0 + iV1*iV1 + iV2*iV2 + iV3*iV3);
}

//-----------------------------------------------------------------------------------------------------------------------------------------
double RecoilCorrector::correlatedSeed(double iVal, double iCorr1,double iCorr2,double iCorr3,double iSeed0,double iSeed1,double iSeed2,double iSeed3) { 
  double lMag = mag(1.,iCorr1,iCorr2,iCorr3); 
  //double lVal = ((1./lMag)*iSeed0 + (iCorr1/lMag)*iSeed1 + (iCorr2/lMag)*iSeed2 + (iCorr3/lMag)*iSeed3)*iVal;
  double lVal = ((1./lMag) + (iCorr1/lMag)*(iSeed1) + (iCorr2/lMag)*fabs(iSeed2) + (iCorr3/lMag)*fabs(iSeed3))*iVal;
  lVal*=iSeed0;
  return lVal;
}
double RecoilCorrector::deCorrelate(double iVal, double iCorr1,double iCorr2,double iCorr3,double iSeed0,double iSeed1,double iSeed2,double iSeed3) { 
  double lMag = mag(1.,iCorr1,iCorr2,iCorr3); 
  double lVal =  (1 - iCorr1*fabs(iSeed1) - iCorr2*fabs(iSeed2) - iCorr3*fabs(iSeed3))*lMag;
  return lVal*iVal*iSeed0;
}

//a = 1/m + 1/m b' 
//b = 1/m + 1/m a'
