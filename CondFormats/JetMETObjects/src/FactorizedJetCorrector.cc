// This is the file "FactorizedJetCorrector.cc". 
// This is the implementation of the class FactorizedJetCorrector.
// Author: Konstantinos Kousouris, Philipp Schieferdecker
// Email:  kkousour@fnal.gov, philipp.schieferdecker@cern.ch

#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "Math/PtEtaPhiE4D.h"
#include "Math/Vector3D.h"
#include "Math/LorentzVector.h"
#include <vector>
#include <string>

#ifdef STANDALONE
#include <sstream>
#include <stdexcept>
#else
#include "FWCore/Utilities/interface/Exception.h"
#endif


//------------------------------------------------------------------------ 
//--- Default FactorizedJetCorrector constructor -------------------------
//------------------------------------------------------------------------
FactorizedJetCorrector::FactorizedJetCorrector()
{
  mJetEta = -9999;
  mJetPt  = -9999;
  mJetPhi = -9999;
  mJetE   = -9999;
  mJetEMF = -9999;
  mLepPx  = -9999;
  mLepPy  = -9999;
  mLepPz  = -9999;
  mIsJetEset   = false;
  mIsJetPtset  = false;
  mIsJetPhiset = false;
  mIsJetEtaset = false;
  mIsJetEMFset = false;
  mIsLepPxset  = false;
  mIsLepPyset  = false;
  mIsLepPzset  = false;
  mAddLepToJet = false;
}
//------------------------------------------------------------------------ 
//--- FactorizedJetCorrector constructor ---------------------------------
//------------------------------------------------------------------------
FactorizedJetCorrector::FactorizedJetCorrector(const std::string& fLevels, const std::string& fFiles, const std::string& fOptions)
{
  mJetEta = -9999;
  mJetPt  = -9999;
  mJetPhi = -9999;
  mJetE   = -9999;
  mJetEMF = -9999;
  mLepPx  = -9999;
  mLepPy  = -9999;
  mLepPz  = -9999;
  mIsJetEset   = false;
  mIsJetPtset  = false;
  mIsJetPhiset = false;
  mIsJetEtaset = false;
  mIsJetEMFset = false;
  mIsLepPxset  = false;
  mIsLepPyset  = false;
  mIsLepPzset  = false;
  mAddLepToJet = false;
  initCorrectors(fLevels, fFiles, fOptions);       
}
//------------------------------------------------------------------------ 
//--- FactorizedJetCorrector destructor ----------------------------------
//------------------------------------------------------------------------
FactorizedJetCorrector::~FactorizedJetCorrector()
{
  for(unsigned i=0;i<mCorrectors.size();i++)
    delete mCorrectors[i];
}
//------------------------------------------------------------------------ 
//--- initialises the correctors -----------------------------------------
//------------------------------------------------------------------------
void FactorizedJetCorrector::initCorrectors(const std::string& fLevels, const std::string& fFiles, const std::string& fOptions)
{
  //---- Read the CorrectionLevels string and parse the requested sub-correction levels.
  std::vector<std::string> tmp = parseLevels(removeSpaces(fLevels));
  for(unsigned i=0;i<tmp.size();i++)
    {
      if (tmp[i] == "L1")
        mLevels.push_back(kL1);
      else if (tmp[i] == "L2")
        mLevels.push_back(kL2);
      else if (tmp[i] == "L3")
        mLevels.push_back(kL3);
      else if (tmp[i] == "L4")
        mLevels.push_back(kL4);
      else if (tmp[i] == "L5")
        mLevels.push_back(kL5);
      else if (tmp[i] == "L6")
        mLevels.push_back(kL6);
      else if (tmp[i] == "L7")
        mLevels.push_back(kL7);
      else
        {
	  #ifdef STANDALONE
	    std::stringstream sserr; 
            sserr<<"FactorizedJetCorrector ERROR: unknown correction level "<<tmp[i];
	    throw std::runtime_error(sserr.str());
          #else
            throw cms::Exception("FactorizedJetCorrector")<<" unknown correction level "<<tmp[i];
          #endif
	}							
    }   	 
  //---- Read the parameter filenames string and parse the requested sub-correction tags.
  std::vector<std::string> Files = parseLevels(removeSpaces(fFiles));
  //---- Read the Options string and define the FlavorOption and PartonOption.
  std::string FlavorOption = parseOption(removeSpaces(fOptions),"Flavor");
  std::string PartonOption = parseOption(removeSpaces(fOptions),"Parton");
  //---- Check the consistency between tags and requested sub-corrections. 
  checkConsistency(tmp,Files);  
  //---- Create instances of the requested sub-correctors.
  for(unsigned i=0;i<mLevels.size();i++)
    {     	    
      if (mLevels[i]==kL1 || mLevels[i]==kL2 || mLevels[i]==kL3 || mLevels[i]==kL4 || mLevels[i]==kL6)
        mCorrectors.push_back(new SimpleJetCorrector(Files[i])); 
      else if (mLevels[i]==kL5 && FlavorOption.length()==0) 
        {
          #ifdef STANDALONE
	    std::stringstream sserr; 
            sserr<<"FactorizedJetCorrector ERROR: must specify flavor option when requesting L5Flavor correction!";
	    throw std::runtime_error(sserr.str());
          #else
            throw cms::Exception("FactorizedJetCorrector")
	    <<" asking L5Flavor correction without specifying flavor option";
          #endif
        }
      else if (mLevels[i]==kL5 && FlavorOption.length()>0)
        mCorrectors.push_back(new SimpleJetCorrector(Files[i],FlavorOption));
      else if (mLevels[i]==kL7 && PartonOption.length()==0) 
        {
          #ifdef STANDALONE
	    std::stringstream sserr; 
            sserr<<"FactorizedJetCorrector ERROR: must specify parton option when requesting L7Parton correction!";
	    throw std::runtime_error(sserr.str());
          #else
            throw cms::Exception("FactorizedJetCorrector")
	    <<" asking L7Parton correction without specifying parton option";
          #endif
        }
      else if (mLevels[i]==kL7 && PartonOption.length()>0)
        mCorrectors.push_back(new SimpleJetCorrector(Files[i],PartonOption));
      else 
        {
          #ifdef STANDALONE
	     std::stringstream sserr; 
             sserr<<"FactorizedJetCorrector ERROR: "<<"unknown correction level: "<<tmp[i];
	     throw std::runtime_error(sserr.str());
          #else
             throw cms::Exception("FactorizedJetCorrector")<<" unknown correction level: "<<tmp[i];
          #endif
        }
      mBinTypes.push_back(mapping(mCorrectors[i]->parameters().definitions().binVar())); 
      mParTypes.push_back(mapping(mCorrectors[i]->parameters().definitions().parVar()));	
    } 
}
//------------------------------------------------------------------------ 
//--- Mapping between variable names and variable types ------------------
//------------------------------------------------------------------------
std::vector<FactorizedJetCorrector::VarTypes> FactorizedJetCorrector::mapping(const std::vector<std::string>& fNames)
{
  std::vector<VarTypes> result;
  for(unsigned i=0;i<fNames.size();i++)
    {
      std::string ss = fNames[i]; 
      if (ss=="JetPt")
	 result.push_back(kJetPt);
      else if (ss=="JetEta")
         result.push_back(kJetEta); 
      else if (ss=="JetPhi")
	 result.push_back(kJetPhi);
      else if (ss=="JetE")
	 result.push_back(kJetE);
      else if (ss=="JetEMF")
	 result.push_back(kJetEMF);
      else if (ss=="LepPx")
	 result.push_back(kLepPx);
      else if (ss=="LepPy")
	 result.push_back(kLepPy);           
      else if (ss=="LepPz")
	 result.push_back(kLepPz);
      else
         {
          #ifdef STANDALONE
	     std::stringstream sserr; 
             sserr<<"FactorizedJetCorrector ERROR: "<<"unknown parameter name: "<<ss;
	     throw std::runtime_error(sserr.str());
          #else
             throw cms::Exception("FactorizedJetCorrector")<<" unknown parameter name: "<<ss;
          #endif
        }
    }
  return result;  
}
//------------------------------------------------------------------------ 
//--- Consistency checker ------------------------------------------------
//------------------------------------------------------------------------
void FactorizedJetCorrector::checkConsistency(const std::vector<std::string>& fLevels, const std::vector<std::string>& fTags)
{
  //---- First check: the number of tags must be equal to the number of sub-corrections.
  if (fLevels.size() != fTags.size()) 
    {
      #ifdef STANDALONE
         std::stringstream sserr; 
         sserr<<"FactorizedJetCorrector ERROR: "
	      <<"number of correction levels: "<<fLevels.size()
	      <<" doesn't match # of tags: "<<fTags.size();
         throw std::runtime_error(sserr.str());
      #else
         throw cms::Exception("FactorizedJetCorrector")
         <<" number of correction levels: "<<fLevels.size()
         <<" doesn't match the number of data file tags: "<<fTags.size();
      #endif
    }
  //---- Second check: each tag must contain the corresponding sub-correction level.
  for(unsigned int i=0;i<fTags.size();i++) 
    {
      if ((int)fTags[i].find(fLevels[i])<0) 
        {
          #ifdef STANDALONE
             std::stringstream sserr; 
             sserr<<"FactorizedJetCorrector ERROR: "
		  <<"inconsistent tag: "<<fTags[i]<<" for "
		  <<"the requested correction: "<<fLevels[i];
             throw std::runtime_error(sserr.str());
          #else
             throw cms::Exception("FactorizedJetCorrector")
             <<" inconsistent tag: "<<fTags[i]<<" for the requested correction: "<<fLevels[i];
          #endif
        }
    }
}
//------------------------------------------------------------------------ 
//--- String parser ------------------------------------------------------
//------------------------------------------------------------------------
std::vector<std::string> FactorizedJetCorrector::parseLevels(const std::string& ss)
{
  std::vector<std::string> result;
  unsigned int pos(0),j,newPos;
  int i;
  std::string tmp;
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
//------------------------------------------------------------------------ 
//--- String parser ------------------------------------------------------
//------------------------------------------------------------------------
std::string FactorizedJetCorrector::parseOption(const std::string& ss, const std::string& type)
{
  std::string result;
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
//------------------------------------------------------------------------ 
//--- String manipulator -------------------------------------------------
//------------------------------------------------------------------------
std::string FactorizedJetCorrector::removeSpaces(const std::string& ss)
{
  std::string result("");
  std::string aChar;
  for(unsigned int i=0;i<ss.length();i++)
    {
      aChar = ss.substr(i,1);
      if (aChar != " ")
        result+=aChar;
    }
  return result; 
}
//------------------------------------------------------------------------ 
//--- Returns the correction ---------------------------------------------
//------------------------------------------------------------------------
float FactorizedJetCorrector::getCorrection()
{
  std::vector<float> vv = getSubCorrections();
  return vv[vv.size()-1];
}
//------------------------------------------------------------------------ 
//--- Returns the vector of subcorrections, up to a given level ----------
//------------------------------------------------------------------------
std::vector<float> FactorizedJetCorrector::getSubCorrections()
{
  float scale,factor;
  std::vector<float> factors;
  std::vector<float> vx,vy;
  factor = 1;
  for(unsigned int i=0;i<mLevels.size();i++)
    { 
      vx = fillVector(mBinTypes[i]);
      vy = fillVector(mParTypes[i]);
      if (mLevels[i]==kL2)
        mCorrectors[i]->setInterpolation(true); 
      scale = mCorrectors[i]->correction(vx,vy); 	
      factor*=scale; 
      factors.push_back(factor);	
      mJetE *=scale;
      mJetPt*=scale;
    }
  mIsJetEset   = false;
  mIsJetPtset  = false;
  mIsJetPhiset = false;
  mIsJetEtaset = false;
  mIsJetEMFset = false;
  mIsLepPxset  = false;
  mIsLepPyset  = false;
  mIsLepPzset  = false;
  mAddLepToJet = false;
  return factors; 
}
//------------------------------------------------------------------------ 
//--- Reads the parameter names and fills a vector of floats -------------
//------------------------------------------------------------------------
std::vector<float> FactorizedJetCorrector::fillVector(std::vector<VarTypes> fVarTypes)
{
  std::vector<float> result;
  for(unsigned i=0;i<fVarTypes.size();i++) 
    {
      if (fVarTypes[i] == kJetEta)
        {
          if (!mIsJetEtaset) 
            {
              #ifdef STANDALONE
	         std::stringstream sserr; 
                 sserr<<"FactorizedJetCorrector ERROR: jet eta is not set";
                 throw std::runtime_error(sserr.str());
              #else
	         throw cms::Exception("FactorizedJetCorrector")<<" jet eta is not set";
              #endif
            }
          result.push_back(mJetEta);
        }
      else if (fVarTypes[i] == kJetPt) 
        {
          if (!mIsJetPtset)
            {
              #ifdef STANDALONE
	        std::stringstream sserr; 
                sserr<<"FactorizedJetCorrector ERROR: jet pt is not set";
                throw std::runtime_error(sserr.str());
              #else
	        throw cms::Exception("FactorizedJetCorrector")<<" jet pt is not set";
              #endif
            }
          result.push_back(mJetPt);
        }
      else if (fVarTypes[i] == kJetPhi) 
        {
          if (!mIsJetPhiset) 
            {
              #ifdef STANDALONE
	         std::stringstream sserr; 
                 sserr<<"FactorizedJetCorrector ERROR: jet phi is not set";  
                 throw std::runtime_error(sserr.str());
              #else
	         throw cms::Exception("FactorizedJetCorrector")<<" jet phi is not set";
              #endif
            }
          result.push_back(mJetPt);
        }
      else if (fVarTypes[i] == kJetE) 
        {
          if (!mIsJetEset) 
            {
              #ifdef STANDALONE
	         std::stringstream sserr; 
                 sserr<<"FactorizedJetCorrector ERROR: jet energy is not set";
                 throw std::runtime_error(sserr.str());
              #else
	         throw cms::Exception("FactorizedJetCorrector")<<" jet energy is not set";
              #endif
            }
          result.push_back(mJetE);
        }
      else if (fVarTypes[i] == kJetEMF) 
        {
          if (!mIsJetEMFset) 
            {
              #ifdef STANDALONE
	        std::stringstream sserr; 
                sserr<<"FactorizedJetCorrector ERROR: jet emf is not set";
                throw std::runtime_error(sserr.str());
              #else
	        throw cms::Exception("FactorizedJetCorrector")<<" jet emf is not set";
              #endif
            }
          result.push_back(mJetEMF);
        } 
      else if (fVarTypes[i] == kLepPx) 
        {
          if (!mIsLepPxset) 
            { 
              #ifdef STANDALONE
	        std::stringstream sserr; 
                sserr<<"FactorizedJetCorrector ERROR: lepton px is not set"; 
                throw std::runtime_error(sserr.str());
              #else
	        throw cms::Exception("FactorizedJetCorrector")<<" lepton px is not set";
              #endif
            }
          result.push_back(mLepPx);
        }
      else if (fVarTypes[i] == kLepPy) 
        {
          if (!mIsLepPyset) 
            {
              #ifdef STANDALONE
	        std::stringstream sserr; 
                sserr<<"FactorizedJetCorrector ERROR: lepton py is not set";
                throw std::runtime_error(sserr.str());
              #else
	        throw cms::Exception("FactorizedJetCorrector")<<" lepton py is not set";
              #endif
            }
          result.push_back(mLepPy);
        }
      else if (fVarTypes[i] == kLepPz) 
        {
          if (!mIsLepPzset) 
            {
              #ifdef STANDALONE
	        std::stringstream sserr; 
                sserr<<"FactorizedJetCorrector ERROR: lepton pz is not set";
                throw std::runtime_error(sserr.str()); 
              #else
	        throw cms::Exception("FactorizedJetCorrector")<<" lepton pz is not set";
              #endif
            }
          result.push_back(mLepPz);
        }
      else 
        {
          #ifdef STANDALONE
            std::stringstream sserr; 
            sserr<<"FactorizedJetCorrector ERROR: unknown parameter "<<fVarTypes[i];
            throw std::runtime_error(sserr.str());
          #else
            throw cms::Exception("FactorizedJetCorrector")<<" unknown parameter "<<fVarTypes[i];
          #endif
        }
    }
  return result;      
}
//------------------------------------------------------------------------ 
//--- Calculate the PtRel (needed for the SLB) ---------------------------
//------------------------------------------------------------------------
float FactorizedJetCorrector::getPtRel()
{
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> > PtEtaPhiELorentzVector;
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> > XYZVector;
  PtEtaPhiELorentzVector jet;
  XYZVector lep;
  jet.SetPt(mJetPt);
  jet.SetEta(mJetEta);
  jet.SetPhi(mJetPhi);
  jet.SetE(mJetE);
  lep.SetXYZ(mLepPx,mLepPy,mLepPz);
  float lj_x = (mAddLepToJet) ? lep.X()+jet.Px() : jet.Px();
  float lj_y = (mAddLepToJet) ? lep.Y()+jet.Py() : jet.Py();
  float lj_z = (mAddLepToJet) ? lep.Z()+jet.Pz() : jet.Pz();
  // absolute values squared
  float lj2  = lj_x*lj_x+lj_y*lj_y+lj_z*lj_z;
  if (lj2<=0) 
    {
      #ifdef STANDALONE
        std::stringstream sserr; 
        sserr<<"FactorizedJetCorrector ERROR: "<<"lepton+jet momentum sq is not positive: "<<lj2;
        throw std::runtime_error(sserr.str());
      #else
        throw cms::Exception("FactorizedJetCorrector")<<" not positive lepton-jet momentum: "<<lj2;
      #endif
    }
  float lep2 = lep.X()*lep.X()+lep.Y()*lep.Y()+lep.Z()*lep.Z();
  // projection vec(mu) to lepjet axis
  float lepXlj = lep.X()*lj_x+lep.Y()*lj_y+lep.Z()*lj_z;
  // absolute value squared and normalized
  float pLrel2 = lepXlj*lepXlj/lj2;
  // lep2 = pTrel2 + pLrel2
  float pTrel2 = lep2-pLrel2;
  return (pTrel2 > 0) ? std::sqrt(pTrel2) : 0.0;
}
//------------------------------------------------------------------------ 
//--- Setters ------------------------------------------------------------
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJetEta(float fEta)
{
  mJetEta = fEta;
  mIsJetEtaset = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJetPt(float fPt)
{
  mJetPt = fPt;
  mIsJetPtset  = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJetPhi(float fPhi)
{
  mJetPhi = fPhi;
  mIsJetPhiset  = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJetE(float fE)
{
  mJetE = fE;
  mIsJetEset   = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJetEMF(float fEMF)
{
  mJetEMF = fEMF;
  mIsJetEMFset = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setLepPx(float fPx)
{
  mLepPx = fPx;
  mIsLepPxset  = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setLepPy(float fPy)
{
  mLepPy = fPy;
  mIsLepPyset  = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setLepPz(float fPz)
{
  mLepPz = fPz;
  mIsLepPzset  = true;
}
//------------------------------------------------------------------------
