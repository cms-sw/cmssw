#include "CondFormats/JetMETObjects/interface/FWLiteJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL2RelativeCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL3AbsoluteCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL3PFAbsoluteCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL4EMFCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL5FlavorCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL7PartonCorrector.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <vector>
#include <string>

using namespace std;
using namespace edm;

////////////////////////////////////////////////////////////////////////////////
FWLiteJetCorrector::FWLiteJetCorrector()
{
  mLevels.push_back("No Correction");
  mDataFiles.push_back("");
  mFlavorOption = "";
  mPartonOption = "";
}
////////////////////////////////////////////////////////////////////////////////
FWLiteJetCorrector::FWLiteJetCorrector(std::vector<std::string> Levels, std::vector<std::string> CorrectionTags)
{
  if (Levels.size() != CorrectionTags.size())
    {
      throw cms::Exception ("FWLiteJetCorrector") 
        << "number of correction levels: " << Levels.size() << " doesn't match the number of data file tags: " << CorrectionTags.size();
    }  
  for(unsigned int i=0;i<Levels.size();i++)
    mLevels.push_back(Levels[i]);
  for(unsigned int i=0;i<CorrectionTags.size();i++)
    {
      string tmp = "CondFormats/JetMETObjects/data/"+CorrectionTags[i]+".txt";
      edm::FileInPath f1(tmp);  
      mDataFiles.push_back(f1.fullPath());
    }
  mFlavorOption = "";
  mPartonOption = "";      
}
////////////////////////////////////////////////////////////////////////////////
FWLiteJetCorrector::FWLiteJetCorrector(vector<string> Levels, vector<string> CorrectionTags, string FlavorOption, string PartonOption)
{
  if (Levels.size() != CorrectionTags.size())
    {
      throw cms::Exception ("FWLiteJetCorrector") 
        << "number of correction levels: " << Levels.size() << " doesn't match the number of data file tags: " << CorrectionTags.size();
    }
  for(unsigned int i=0;i<Levels.size();i++)
    mLevels.push_back(Levels[i]);
  for(unsigned int i=0;i<CorrectionTags.size();i++)
    {
      string tmp = "CondFormats/JetMETObjects/data/"+CorrectionTags[i]+".txt";
      edm::FileInPath f1(tmp);  
      mDataFiles.push_back(f1.fullPath());
    }
  mFlavorOption = FlavorOption;
  mPartonOption = PartonOption;      
}
////////////////////////////////////////////////////////////////////////////////
FWLiteJetCorrector::~FWLiteJetCorrector()
{
  
}
////////////////////////////////////////////////////////////////////////////////
double FWLiteJetCorrector::getCorrection(double pt, double eta)
{
  double tmpPt,corPt,scale,factor;
  corPt = pt;
  factor = 1.;
  for(unsigned int i=0;i<mLevels.size();i++)
    { 
      tmpPt = corPt;
      if (mLevels[i]== "L2")
        {
          SimpleL2RelativeCorrector L2Corrector(mDataFiles[i]);
          scale = L2Corrector.correctionPtEta(tmpPt,eta);
        }
      else if (mLevels[i] == "L3")
        {
          SimpleL3AbsoluteCorrector L3Corrector(mDataFiles[i]);  
          scale = L3Corrector.correctionPtEta(tmpPt,eta);
        } 
      else if (mLevels[i] == "L3PF")
        {
          SimpleL3PFAbsoluteCorrector L3Corrector(mDataFiles[i]);  
          scale = L3Corrector.correctionPtEta(tmpPt,eta);
        }	
      else if (mLevels[i] == "L4")
        scale = 1.;  
      else if (mLevels[i] == "L5")
        {
	  if (mFlavorOption=="")
	    scale = 1.;
	  else
	    {    
              SimpleL5FlavorCorrector L5Corrector(mDataFiles[i],mFlavorOption);  
              scale = L5Corrector.correctionPtEta(tmpPt,eta);
	    }  
        }
      else if (mLevels[i]=="L6")
        scale = 1.; 
      else if (mLevels[i]=="L1")
        scale = 1.;	  	
      else if (mLevels[i] == "L7")
        {
	  if (mPartonOption=="")
	    scale = 1.; 
	  else
	    {  
              SimpleL7PartonCorrector L7Corrector(mDataFiles[i],mPartonOption);  
              scale = L7Corrector.correctionPtEta(tmpPt,eta);
	    }  
        }
      else
	scale = 1.; 	
      factor*=scale; 	
      corPt*=scale;
    }
  return factor; 
}
////////////////////////////////////////////////////////////////////////////////
double FWLiteJetCorrector::getCorrection(double pt, double eta, double emf)
{
  double tmpPt,corPt,scale,factor;
  corPt = pt;
  factor = 1.;
  for(unsigned int i=0;i<mLevels.size();i++)
    { 
      tmpPt = corPt;
      if (mLevels[i]== "L2")
        {
          SimpleL2RelativeCorrector L2Corrector(mDataFiles[i]);
          scale = L2Corrector.correctionPtEta(tmpPt,eta);
        }
      else if (mLevels[i] == "L3")
        {
          SimpleL3AbsoluteCorrector L3Corrector(mDataFiles[i]);  
          scale = L3Corrector.correctionPtEta(tmpPt,eta);
        } 
      else if (mLevels[i] == "L3PF")
        {
          SimpleL3PFAbsoluteCorrector L3Corrector(mDataFiles[i]);  
          scale = L3Corrector.correctionPtEta(tmpPt,eta);
        }	
      else if (mLevels[i] == "L4")
        {
          SimpleL4EMFCorrector L4Corrector(mDataFiles[i]);  
          scale = L4Corrector.correctionPtEtaEmfraction(tmpPt,eta,emf);
        }  
      else if (mLevels[i] == "L5")
        {
	  if (mFlavorOption=="")
	    scale = 1.;
	  else
	    {    
              SimpleL5FlavorCorrector L5Corrector(mDataFiles[i],mFlavorOption);  
              scale = L5Corrector.correctionPtEta(tmpPt,eta);
	    }  
        }
      else if (mLevels[i]=="L6")
        scale = 1.; 
      else if (mLevels[i]=="L1")
        scale = 1.;	  	
      else if (mLevels[i] == "L7")
        {
	  if (mPartonOption=="")
	    scale = 1.; 
	  else
	    {  
              SimpleL7PartonCorrector L7Corrector(mDataFiles[i],mPartonOption);  
              scale = L7Corrector.correctionPtEta(tmpPt,eta);
	    }  
        }
      else
	scale = 1.; 	
      factor*=scale; 	
      corPt*=scale;
    }
  return factor; 
}
////////////////////////////////////////////////////////////////////////////////
vector<double> FWLiteJetCorrector::getSubCorrections(double pt, double eta)
{
  double tmpPt,corPt,scale;
  vector<double> factors;
  corPt = pt;
  for(unsigned int i=0;i<mLevels.size();i++)
    { 
      tmpPt = corPt;
      if (mLevels[i]== "L2")
        {
          SimpleL2RelativeCorrector L2Corrector(mDataFiles[i]);
          scale = L2Corrector.correctionPtEta(tmpPt,eta);
        }
      else if (mLevels[i] == "L3")
        {
          SimpleL3AbsoluteCorrector L3Corrector(mDataFiles[i]);  
          scale = L3Corrector.correctionPtEta(tmpPt,eta);
        } 
      else if (mLevels[i] == "L3PF")
        {
          SimpleL3PFAbsoluteCorrector L3Corrector(mDataFiles[i]);  
          scale = L3Corrector.correctionPtEta(tmpPt,eta);
        }	
      else if (mLevels[i] == "L4")
        scale = 1.;
      else if (mLevels[i] == "L5")
        {
	  if (mFlavorOption=="")
	    scale = 1.;
	  else
	    {    
              SimpleL5FlavorCorrector L5Corrector(mDataFiles[i],mFlavorOption);  
              scale = L5Corrector.correctionPtEta(tmpPt,eta);
	    }  
        }
      else if (mLevels[i]=="L6")
        scale = 1.;
      else if (mLevels[i]=="L1")
        scale = 1.;
      else if (mLevels[i] == "L7")
        {
	  if (mPartonOption=="")
	    scale = 1.;
	  else
	    {  
              SimpleL7PartonCorrector L7Corrector(mDataFiles[i],mPartonOption);  
              scale = L7Corrector.correctionPtEta(tmpPt,eta);
	    }  
        }
      else
        scale = 1.;
      factors.push_back(scale); 	
      corPt*=scale;
    }
  return factors; 
}
////////////////////////////////////////////////////////////////////////////////
vector<double> FWLiteJetCorrector::getSubCorrections(double pt, double eta, double emf)
{
  double tmpPt,corPt,scale;
  vector<double> factors;
  corPt = pt;
  for(unsigned int i=0;i<mLevels.size();i++)
    { 
      tmpPt = corPt;
      if (mLevels[i]== "L2")
        {
          SimpleL2RelativeCorrector L2Corrector(mDataFiles[i]);
          scale = L2Corrector.correctionPtEta(tmpPt,eta);
        }
      else if (mLevels[i] == "L3")
        {
          SimpleL3AbsoluteCorrector L3Corrector(mDataFiles[i]);  
          scale = L3Corrector.correctionPtEta(tmpPt,eta);
        } 
      else if (mLevels[i] == "L3PF")
        {
          SimpleL3PFAbsoluteCorrector L3Corrector(mDataFiles[i]);  
          scale = L3Corrector.correctionPtEta(tmpPt,eta);
        }	
      else if (mLevels[i] == "L4")
        {
          SimpleL4EMFCorrector L4Corrector(mDataFiles[i]);  
          scale = L4Corrector.correctionPtEtaEmfraction(tmpPt,eta,emf);
        }
      else if (mLevels[i] == "L5")
        {
	  if (mFlavorOption=="")
	    scale = 1.;
	  else
	    {    
              SimpleL5FlavorCorrector L5Corrector(mDataFiles[i],mFlavorOption);  
              scale = L5Corrector.correctionPtEta(tmpPt,eta);
	    }  
        }
      else if (mLevels[i]=="L6")
        scale = 1.;
      else if (mLevels[i]=="L1")
        scale = 1.;
      else if (mLevels[i] == "L7")
        {
	  if (mPartonOption=="")
	    scale = 1.;
	  else
	    {  
              SimpleL7PartonCorrector L7Corrector(mDataFiles[i],mPartonOption);  
              scale = L7Corrector.correctionPtEta(tmpPt,eta);
	    }  
        }
      else
        scale = 1.;
      factors.push_back(scale); 	
      corPt*=scale;
    }
  return factors; 
}
////////////////////////////////////////////////////////////////////////////////
