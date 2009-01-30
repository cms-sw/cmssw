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
  
}
////////////////////////////////////////////////////////////////////////////////
FWLiteJetCorrector::FWLiteJetCorrector(std::string CorrectionLevels, std::string CorrectionTags)
{
  initCorrectors(CorrectionLevels, CorrectionTags); 
}
////////////////////////////////////////////////////////////////////////////////
FWLiteJetCorrector::FWLiteJetCorrector(std::string CorrectionLevels, std::string CorrectionTags, std::string Options)
{
  initCorrectors(CorrectionLevels, CorrectionTags, Options);       
}
////////////////////////////////////////////////////////////////////////////////
void FWLiteJetCorrector::initCorrectors(std::string CorrectionLevels, std::string CorrectionTags)
{
  mLevels = parseLevels(CorrectionLevels);
  vector<string> Tags = parseLevels(CorrectionTags);
  vector<string> DataFiles;
  if (mLevels.size() != Tags.size())
    {
      throw cms::Exception ("FWLiteJetCorrector") 
        << "number of correction levels: " << mLevels.size() << " doesn't match the number of data file tags: " << Tags.size();
    }  
  for(unsigned int i=0;i<Tags.size();i++)
    {
      string tmp = "CondFormats/JetMETObjects/data/"+Tags[i]+".txt";
      edm::FileInPath f1(tmp);  
      DataFiles.push_back(f1.fullPath());
    }
  for(unsigned int i=0;i<mLevels.size();i++)
    {
      if (mLevels[i]=="L2")
        mL2Corrector = new SimpleL2RelativeCorrector(DataFiles[i]);
      else if (mLevels[i]=="L3" && ((int)Tags[i].find("Calo")>=0 || (int)Tags[i].find("JPT")>=0))
        {
          mL3Option = "Calo";
          mL3Corrector = new SimpleL3AbsoluteCorrector(DataFiles[i]);
        }
      else if (mLevels[i]=="L3" && (int)Tags[i].find("PF")>=0)
        {
          mL3Option = "PF";  
          mL3PFCorrector = new SimpleL3PFAbsoluteCorrector(DataFiles[i]);
        }
      else if (mLevels[i]=="L4")
        mL4Corrector = new SimpleL4EMFCorrector(DataFiles[i]);
      else if (mLevels[i]=="L5")
        {
          throw cms::Exception ("FWLiteJetCorrector") 
            << "asking L5Flavor correction without specifying flavor option";
        }
      else if (mLevels[i]=="L7")
        {
          throw cms::Exception ("FWLiteJetCorrector") 
            << "asking L7Parton correction without specifying parton option";
        }
      else
        {
          throw cms::Exception ("FWLiteJetCorrector") 
            << "unknown correction level: " << mLevels[i];
        }
    }
        
}
////////////////////////////////////////////////////////////////////////////////
void FWLiteJetCorrector::initCorrectors(std::string CorrectionLevels, std::string CorrectionTags, std::string Options)
{
  mLevels = parseLevels(CorrectionLevels);
  std::vector<std::string> Tags = parseLevels(CorrectionTags);
  std::vector<std::string> DataFiles;
  std::string FlavorOption = parseOption(Options,"Flavor");
  std::string PartonOption = parseOption(Options,"Parton");
  if (mLevels.size() != Tags.size())
    {
      throw cms::Exception ("FWLiteJetCorrector") 
        << "number of correction levels: " << mLevels.size() << " doesn't match the number of data file tags: " << Tags.size();
    }  
  for(unsigned int i=0;i<Tags.size();i++)
    {
      string tmp = "CondFormats/JetMETObjects/data/"+Tags[i]+".txt";
      edm::FileInPath f1(tmp);  
      DataFiles.push_back(f1.fullPath());
    }
  for(unsigned int i=0;i<mLevels.size();i++)
    {
      if (mLevels[i]=="L2")
        mL2Corrector = new SimpleL2RelativeCorrector(DataFiles[i]);
      else if (mLevels[i]=="L3" && ((int)Tags[i].find("Calo")>=0 || (int)Tags[i].find("JPT")>=0))
        {
          mL3Option = "Calo";
          mL3Corrector = new SimpleL3AbsoluteCorrector(DataFiles[i]);
        }
      else if (mLevels[i]=="L3" && (int)Tags[i].find("PF")>=0)
        {
          mL3Option = "PF";  
          mL3PFCorrector = new SimpleL3PFAbsoluteCorrector(DataFiles[i]);
        }
      else if (mLevels[i]=="L4")
        mL4Corrector = new SimpleL4EMFCorrector(DataFiles[i]);
      else if (mLevels[i]=="L5" && FlavorOption.length()==0)
        {
          throw cms::Exception ("FWLiteJetCorrector") 
            << "asking L5Flavor correction without specifying flavor option";
        }
      else if (mLevels[i]=="L5" && FlavorOption.length()>0)
        mL5Corrector = new SimpleL5FlavorCorrector(DataFiles[i],FlavorOption);
      else if (mLevels[i]=="L7" && PartonOption.length()==0)
        {
          throw cms::Exception ("FWLiteJetCorrector") 
            << "asking L7Parton correction without specifying parton option";
        }
      else if (mLevels[i]=="L7" && PartonOption.length()>0)
        mL7Corrector = new SimpleL7PartonCorrector(DataFiles[i],PartonOption);
      else
        {
          throw cms::Exception ("FWLiteJetCorrector") 
            << "unknown correction level: " << mLevels[i];
        }
    }
        
}
////////////////////////////////////////////////////////////////////////////////
FWLiteJetCorrector::~FWLiteJetCorrector()
{
  delete mL2Corrector;
  delete mL3Corrector;
  delete mL3PFCorrector; 
  delete mL4Corrector;
  delete mL5Corrector;
  delete mL7Corrector;
}
////////////////////////////////////////////////////////////////////////////////
vector<string> FWLiteJetCorrector::parseLevels(string ss)
{
  vector<string> result;
  unsigned int pos(0),j,newPos;
  int i;
  string tmp;
  while (pos<ss.length())
    {
      tmp = "";
      i = ss.find("," , pos);
      if (i<0 && pos==0)
        {
          result.push_back(ss);
          pos = ss.length();
        }
      else if (i<0 && pos>0)
        {
          for(j=pos;j<ss.length();j++)
            tmp+=ss[j];
          result.push_back(tmp);
          pos = ss.length();
        }  
      else
        {
          newPos = i;
          for(j=pos;j<newPos;j++)
            tmp+=ss[j];
          result.push_back(tmp);
          pos = newPos+1;     
        }
    }
  return result;
}
////////////////////////////////////////////////////////////////////////////////
string FWLiteJetCorrector::parseOption(string ss, string type)
{
  string result;
  int pos1(-1),pos2(-1);
  pos1 = ss.find(type+":");
  if (pos1<0)
    result = "";
  else
    {
      pos2 = ss.find(",",pos1+type.length()+1); 
      if (pos2<0)
        result = ss.substr(pos1+type.length()+1,ss.length()-pos1-type.length()-1);
      else
        result = ss.substr(pos1+type.length()+1,pos2-pos1-type.length()-1);
    }
  return result;
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
        scale = mL2Corrector->correctionPtEta(tmpPt,eta);
      else if (mLevels[i] == "L3" && mL3Option == "Calo")
        scale = mL3Corrector->correctionPtEta(tmpPt,eta); 
      else if (mLevels[i] == "L3" && mL3Option == "PF")
        scale = mL3PFCorrector->correctionPtEta(tmpPt,eta);
      else if (mLevels[i] == "L4")
        scale = 1.;  
      else if (mLevels[i] == "L5")
        scale = mL5Corrector->correctionPtEta(tmpPt,eta);
      else if (mLevels[i] == "L7")
        scale = mL7Corrector->correctionPtEta(tmpPt,eta);
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
        scale = mL2Corrector->correctionPtEta(tmpPt,eta);
      else if (mLevels[i] == "L3" && mL3Option == "Calo")
        scale = mL3Corrector->correctionPtEta(tmpPt,eta); 
      else if (mLevels[i] == "L3" && mL3Option == "PF")
        scale = mL3PFCorrector->correctionPtEta(tmpPt,eta);
      else if (mLevels[i] == "L4")
        scale = mL4Corrector->correctionPtEtaEmfraction(tmpPt,eta,emf);   
      else if (mLevels[i] == "L5")
        scale = mL5Corrector->correctionPtEta(tmpPt,eta);
      else if (mLevels[i] == "L7")
        scale = mL7Corrector->correctionPtEta(tmpPt,eta);
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
        scale = mL2Corrector->correctionPtEta(tmpPt,eta);
      else if (mLevels[i] == "L3" && mL3Option == "Calo")
        scale = mL3Corrector->correctionPtEta(tmpPt,eta); 
      else if (mLevels[i] == "L3" && mL3Option == "PF")
        scale = mL3PFCorrector->correctionPtEta(tmpPt,eta);
      else if (mLevels[i] == "L4")
        scale = 1.;  
      else if (mLevels[i] == "L5")
        scale = mL5Corrector->correctionPtEta(tmpPt,eta);
      else if (mLevels[i] == "L7")
        scale = mL7Corrector->correctionPtEta(tmpPt,eta);
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
        scale = mL2Corrector->correctionPtEta(tmpPt,eta);
      else if (mLevels[i] == "L3" && mL3Option == "Calo")
        scale = mL3Corrector->correctionPtEta(tmpPt,eta); 
      else if (mLevels[i] == "L3" && mL3Option == "PF")
        scale = mL3PFCorrector->correctionPtEta(tmpPt,eta);
      else if (mLevels[i] == "L4")
        scale = mL4Corrector->correctionPtEtaEmfraction(tmpPt,eta,emf);   
      else if (mLevels[i] == "L5")
        scale = mL5Corrector->correctionPtEta(tmpPt,eta);
      else if (mLevels[i] == "L7")
        scale = mL7Corrector->correctionPtEta(tmpPt,eta);
      else
	scale = 1.;
      factors.push_back(scale); 	
      corPt*=scale;
    }
  return factors; 
}
////////////////////////////////////////////////////////////////////////////////
