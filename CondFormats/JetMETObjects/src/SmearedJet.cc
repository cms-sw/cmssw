#include "CondFormats/JetMETObjects/interface/SmearedJet.h"
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectorParameters.h"
#include <vector>

////////////////////////////////////////////////////////////////////
//------- default constructor -----------
SmearedJet::SmearedJet() 
{
  mRnd = new TRandom();
  mRnd->SetSeed(0);
  //mFormula.push_back("[0]-[1]/(pow(log10(x),[2])+[3])+[4]/x");
  mFormula.push_back("[0]-[1]*pow(x,-[2])+[3]/x");
  //mFormula.push_back("sqrt([0]*[0]+[1]*[1]/x+[2]*[2]/(x*x))");
  mFormula.push_back("sqrt([0]/(x*x)+[1]*pow(x,-[2]))");
  mFormula.push_back("sqrt([0]*[0]+[1]*[1]/x+[2]*[2]/(x*x))");
  mFormula.push_back("sqrt([0]*[0]+[1]*[1]/x+[2]*[2]/(x*x))");
  mFormula.push_back("0.5+0.5*TMath::Erf([0]*(x-[1]))");
  mName.push_back("Response");
  mName.push_back("PtResolution");
  mName.push_back("EtaResolution");
  mName.push_back("PhiResolution");
  mName.push_back("RecoEfficiency");
  for(int i=0;i<5;i++)
    {
      mIsSet.push_back(false);
      TF1 tmp(mName[i].c_str(),mFormula[i].c_str(),1.,5000.);
      mFunction.push_back(tmp);
      SimpleJetCorrectorParameters tmp1;
      mParameters.push_back(tmp1);
    }
}

////////////////////////////////////////////////////////////////////
//------- parameter setter -------
void SmearedJet::setParameters(const std::string& fDataFile, ParType fOption)
{
  checkOption(fOption);
  mIsSet[fOption]      = true;
  mParameters[fOption] = SimpleJetCorrectorParameters(fDataFile);  
} 

////////////////////////////////////////////////////////////////////
//------- destructor -----------
SmearedJet::~SmearedJet() 
{
  delete mRnd;
}

////////////////////////////////////////////////////////////////////
//------- returns the Pt,Eta,Phi resolutions or the reco efficiency -------
double SmearedJet::getValue(double fPt, double fEta, ParType fOption)
{
  double result(0);
  checkOption(fOption);
  if (!mIsSet[fOption])
    {
      std::cout << "The "<<mName[fOption]<<" parameters must be set first." << std::endl;
      abort();
    }
  else
    {   
      int band = mParameters[fOption].bandIndex(fEta);
      if (band<0) 
        {
          if (fEta < mParameters[fOption].record(0).etaMin())
            band = 0;
          else if (fEta > mParameters[fOption].record(mParameters[fOption].size()-1).etaMax())  
            band = mParameters[fOption].size()-1;
          else
            result = 1.; 
        }
      else
        { 
          const std::vector<float>& p = mParameters[fOption].record(band).parameters();
          mFunction[fOption].SetRange(p[0],p[1]);
          for(unsigned int i=0; i<p.size()-2; i++)
            mFunction[fOption].SetParameter(i,p[2+i]);
          result = mFunction[fOption].Eval((fPt < p[0]) ? p[0] : (fPt > p[1]) ? p[1] : fPt);  
        }
    }
  return result;
}

////////////////////////////////////////////////////////////////////
//------- returns a smeared value -------
double SmearedJet::getSmeared(double fPt, double fEta, double fPhi, VarType fOption)
{
  double m(0),s(0.);
  if (fOption == vRawPt)
    {
      m = getValue(fPt,fEta,pResp)*fPt;
      s = getValue(fPt,fEta,pPtResol)*fPt; // pt resolution is relative
    }
  else if (fOption == vPt)
    {
      m = fPt;
      s = getValue(fPt,fEta,pPtResol)*fPt; // pt resolution is relative
    }
  else if (fOption == vEta)
    {
      m = fEta; 
      s = getValue(fPt,fEta,pEtaResol);// eta resolution is absolute
    }
  else if (fOption == vPhi)
    {
      m = fPhi;
      s = getValue(fPt,fEta,pPhiResol);// phi resolution is absolute
    }
  else
    {
      std::cout << "Unknown option: " << fOption <<std::endl;
      abort();
    } 
  return mRnd->Gaus(m,s);
}

////////////////////////////////////////////////////////////////////
//------- returns a smeared vector -------
PtEtaPhiELorentzVectorD SmearedJet::getSmeared(PtEtaPhiELorentzVectorD fP4, VarType fOption)
{
  PtEtaPhiELorentzVectorD result;
  if (fOption==vRawPt || fOption==vPt)
    {
      double pt  = getSmeared(fP4.Pt(),fP4.Eta(),fP4.Phi(),fOption);
      double eta = getSmeared(fP4.Pt(),fP4.Eta(),fP4.Phi(),vEta);
      double phi = getSmeared(fP4.Pt(),fP4.Eta(),fP4.Phi(),vPhi);
      double e   = (pt/fP4.pt())*fP4.E();
      result.SetPt(pt);
      result.SetEta(eta);
      result.SetPhi(phi);
      result.SetE(e);
    }
  else
    {
      std::cout << "Unknown option: " << fOption <<std::endl;
      abort();
    } 
  return result;
}

////////////////////////////////////////////////////////////////////
void SmearedJet::checkOption(ParType fOption)
{
  if (fOption!=pResp && fOption!=pPtResol && fOption!=pEtaResol && fOption!=pPhiResol && fOption!=pRecoEff)
    {
      std::cout << "Unknown option: " << fOption <<std::endl;
      abort();
    }
}

////////////////////////////////////////////////////////////////////
void SmearedJet::checkOption(VarType fOption)
{
  if (fOption!=vRawPt && fOption!=vPt && fOption!=vEta && fOption!=vPhi)
    {
      std::cout << "Unknown option: " << fOption <<std::endl;
      abort();
    } 
}

////////////////////////////////////////////////////////////////////
bool SmearedJet::isReconstructed(double fPt, double fEta)
{
  bool result(false);
  double eff = getValue(fPt,fEta,pRecoEff); 
  double r   = mRnd->Uniform(0,1);
  if (r < eff)
    result = true;
  return result;
}

////////////////////////////////////////////////////////////////////
bool SmearedJet::isReconstructed(PtEtaPhiELorentzVectorD fP4)
{
  return isReconstructed(fP4.Pt(),fP4.Eta());
}








