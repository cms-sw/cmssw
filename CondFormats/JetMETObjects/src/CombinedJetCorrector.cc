// This is the file "CombinedJetCorrector.cc". 
// This is the implementation of the class CombinedJetCorrector.
// Author: Konstantinos Kousouris, 
// Email:  kkousour@fnal.gov

#include "CondFormats/JetMETObjects/interface/CombinedJetCorrector.h"
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
CombinedJetCorrector::CombinedJetCorrector()
{
  mL2Corrector   = new SimpleL2RelativeCorrector();
  mL3Corrector   = new SimpleL3AbsoluteCorrector();
  mL3PFCorrector = new SimpleL3PFAbsoluteCorrector(); 
  mL4Corrector   = new SimpleL4EMFCorrector();
  mL5Corrector   = new SimpleL5FlavorCorrector();
  mL7Corrector   = new SimpleL7PartonCorrector();
  mL3Option      = "";
  mLevels.push_back("");
}
////////////////////////////////////////////////////////////////////////////////
CombinedJetCorrector::CombinedJetCorrector(std::string CorrectionLevels, std::string CorrectionTags)
{
  initCorrectors(CorrectionLevels, CorrectionTags,""); 
}
////////////////////////////////////////////////////////////////////////////////
CombinedJetCorrector::CombinedJetCorrector(std::string CorrectionLevels, std::string CorrectionTags, std::string Options)
{
  initCorrectors(CorrectionLevels, CorrectionTags, Options);       
}
////////////////////////////////////////////////////////////////////////////////
void CombinedJetCorrector::initCorrectors(std::string CorrectionLevels, std::string CorrectionTags, std::string Options)
{
  //---- Read the CorrectionLevels string and parse the requested sub-correction levels.
  mLevels = parseLevels(removeSpaces(CorrectionLevels));
  //---- Read the CorrectionTags string and parse the requested sub-correction tags.
  std::vector<std::string> Tags = parseLevels(removeSpaces(CorrectionTags));
  std::vector<std::string> DataFiles;
  //---- Read the Options string and define the FlavorOption and PartonOption.
  std::string FlavorOption = parseOption(removeSpaces(Options),"Flavor");
  std::string PartonOption = parseOption(removeSpaces(Options),"Parton");

  //---- Check the consistency between tags and requested sub-corrections. 
  checkConsistency(mLevels,Tags);  
  
  //---- Construct the full path correction parameters filenames.
  for(unsigned int i=0;i<Tags.size();i++)
    {
      string tmp = "CondFormats/JetMETObjects/data/"+Tags[i]+".txt";
      edm::FileInPath f1(tmp);  
      DataFiles.push_back(f1.fullPath());
    }
  //---- Create instances of the requested sub-correctors.
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
          throw cms::Exception ("CombinedJetCorrector") 
            << "asking L5Flavor correction without specifying flavor option";
        }
      else if (mLevels[i]=="L5" && FlavorOption.length()>0)
        mL5Corrector = new SimpleL5FlavorCorrector(DataFiles[i],FlavorOption);
      else if (mLevels[i]=="L7" && PartonOption.length()==0)
        {
          throw cms::Exception ("CombinedJetCorrector") 
            << "asking L7Parton correction without specifying parton option";
        }
      else if (mLevels[i]=="L7" && PartonOption.length()>0)
        mL7Corrector = new SimpleL7PartonCorrector(DataFiles[i],PartonOption);
      else
        {
          throw cms::Exception ("CombinedJetCorrector") 
            << "unknown correction level: " << mLevels[i];
        }
    } 
}
////////////////////////////////////////////////////////////////////////////////
void CombinedJetCorrector::checkConsistency(std::vector<std::string> Levels, std::vector<std::string> Tags)
{
  //---- First check: the number of tags must be equal to the number of sub-corrections.
  if (Levels.size() != Tags.size())
    {
      throw cms::Exception ("CombinedJetCorrector") 
        << "number of correction levels: " << Levels.size() << " doesn't match the number of data file tags: " << Tags.size();
    }
  //---- Second check: each tag must contain the corresponding sub-correction level.
  for(unsigned int i=0;i<Tags.size();i++)
    if ((int)Tags[i].find(Levels[i])<0)
      {
        throw cms::Exception ("CombinedJetCorrector") 
          << "inconsistent tag: " << Tags[i] << " for the requested correction: " << Levels[i];
      }
}
////////////////////////////////////////////////////////////////////////////////
CombinedJetCorrector::~CombinedJetCorrector()
{
  delete mL2Corrector;
  delete mL3Corrector;
  delete mL3PFCorrector; 
  delete mL4Corrector;
  delete mL5Corrector;
  delete mL7Corrector;
}
////////////////////////////////////////////////////////////////////////////////
vector<string> CombinedJetCorrector::parseLevels(string ss)
{
  vector<string> result;
  unsigned int pos(0),j,newPos;
  int i;
  string tmp;
  //---- The ss string must be of the form: "LX:LY:...:LZ"
  while (pos<ss.length())
    {
      tmp = "";
      i = ss.find(":" , pos);
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
string CombinedJetCorrector::parseOption(string ss, string type)
{
  string result;
  int pos1(-1),pos2(-1);
  //---- The ss string must be of the form: "type1:option1&type2:option2&..."
  pos1 = ss.find(type+":");
  if (pos1<0)
    result = "";
  else
    {
      pos2 = ss.find("&",pos1+type.length()+1); 
      if (pos2<0)
        result = ss.substr(pos1+type.length()+1,ss.length()-pos1-type.length()-1);
      else
        result = ss.substr(pos1+type.length()+1,pos2-pos1-type.length()-1);
    }
  return result;
}
////////////////////////////////////////////////////////////////////////////////
string CombinedJetCorrector::removeSpaces(string ss)
{
  string result("");
  string aChar;
  for(unsigned int i=0;i<ss.length();i++)
    {
      aChar = ss.substr(i,1);
      if (aChar != " ")
        result+=aChar;
    }
  return result; 
}
////////////////////////////////////////////////////////////////////////////////
double CombinedJetCorrector::getCorrection(double pt, double eta)
{
  double corPt,scale,factor;
  corPt = pt;
  factor = 1.;
  for(unsigned int i=0;i<mLevels.size();i++)
    { 
      if (mLevels[i]== "L2")
        scale = mL2Corrector->correctionPtEta(corPt,eta);
      else if (mLevels[i] == "L3" && mL3Option == "Calo")
        scale = mL3Corrector->correctionPtEta(corPt,eta); 
      else if (mLevels[i] == "L3" && mL3Option == "PF")
        scale = mL3PFCorrector->correctionPtEta(corPt,eta);
      else if (mLevels[i] == "L4")
        scale = 1.;  
      else if (mLevels[i] == "L5")
        scale = mL5Corrector->correctionPtEta(corPt,eta);
      else if (mLevels[i] == "L7")
        scale = mL7Corrector->correctionPtEta(corPt,eta);
      else
	scale = 1.; 	
      factor*=scale; 	
      corPt*=scale;
    }
  return factor; 
}
////////////////////////////////////////////////////////////////////////////////
double CombinedJetCorrector::getCorrection(double pt, double eta, double emf)
{
  double corPt,scale,factor;
  corPt = pt;
  factor = 1.;
  for(unsigned int i=0;i<mLevels.size();i++)
    { 
      if (mLevels[i]== "L2")
        scale = mL2Corrector->correctionPtEta(corPt,eta);
      else if (mLevels[i] == "L3" && mL3Option == "Calo")
        scale = mL3Corrector->correctionPtEta(corPt,eta); 
      else if (mLevels[i] == "L3" && mL3Option == "PF")
        scale = mL3PFCorrector->correctionPtEta(corPt,eta);
      else if (mLevels[i] == "L4")
        scale = mL4Corrector->correctionPtEtaEmfraction(corPt,eta,emf);   
      else if (mLevels[i] == "L5")
        scale = mL5Corrector->correctionPtEta(corPt,eta);
      else if (mLevels[i] == "L7")
        scale = mL7Corrector->correctionPtEta(corPt,eta);
      else
	scale = 1.; 	
      factor*=scale; 	
      corPt*=scale;
    }
  return factor; 
}
////////////////////////////////////////////////////////////////////////////////
//returns a vector with the correction factors UP to the given level/////
vector<double> CombinedJetCorrector::getSubCorrections(double pt, double eta)
{
  double corPt,scale,factor;
  vector<double> factors;
  corPt = pt;
  factor = 1;
  for(unsigned int i=0;i<mLevels.size();i++)
    { 
      if (mLevels[i]== "L2")
        scale = mL2Corrector->correctionPtEta(corPt,eta);
      else if (mLevels[i] == "L3" && mL3Option == "Calo")
        scale = mL3Corrector->correctionPtEta(corPt,eta); 
      else if (mLevels[i] == "L3" && mL3Option == "PF")
        scale = mL3PFCorrector->correctionPtEta(corPt,eta);
      else if (mLevels[i] == "L4")
        scale = 1.;  
      else if (mLevels[i] == "L5")
        scale = mL5Corrector->correctionPtEta(corPt,eta);
      else if (mLevels[i] == "L7")
        scale = mL7Corrector->correctionPtEta(corPt,eta);
      else
	scale = 1.; 
      factor*=scale;	
      factors.push_back(factor); 	
      corPt*=scale;
    }
  return factors; 
}
////////////////////////////////////////////////////////////////////////////////
//returns a vector with the correction factors UP to the given level/////
vector<double> CombinedJetCorrector::getSubCorrections(double pt, double eta, double emf)
{
  double corPt,scale,factor;
  vector<double> factors;
  corPt = pt;
  factor = 1;
  for(unsigned int i=0;i<mLevels.size();i++)
    { 
      if (mLevels[i]== "L2")
        scale = mL2Corrector->correctionPtEta(corPt,eta);
      else if (mLevels[i] == "L3" && mL3Option == "Calo")
        scale = mL3Corrector->correctionPtEta(corPt,eta); 
      else if (mLevels[i] == "L3" && mL3Option == "PF")
        scale = mL3PFCorrector->correctionPtEta(corPt,eta);
      else if (mLevels[i] == "L4")
        scale = mL4Corrector->correctionPtEtaEmfraction(corPt,eta,emf);   
      else if (mLevels[i] == "L5")
        scale = mL5Corrector->correctionPtEta(corPt,eta);
      else if (mLevels[i] == "L7")
        scale = mL7Corrector->correctionPtEta(corPt,eta);
      else
	scale = 1.;
      factor*=scale;	
      factors.push_back(factor); 	
      corPt*=scale;
    }
  return factors; 
}
////////////////////////////////////////////////////////////////////////////////
