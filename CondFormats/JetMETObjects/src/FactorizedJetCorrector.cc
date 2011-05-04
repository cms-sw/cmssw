// This is the file "FactorizedJetCorrector.cc". 
// This is the implementation of the class FactorizedJetCorrector.
// Author: Konstantinos Kousouris, Philipp Schieferdecker
// Email:  kkousour@fnal.gov, philipp.schieferdecker@cern.ch

#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/src/Utilities.cc"
#include "Math/PtEtaPhiE4D.h"
#include "Math/Vector3D.h"
#include "Math/LorentzVector.h"
#include <vector>
#include <string>
#include <sstream>

//------------------------------------------------------------------------ 
//--- Default FactorizedJetCorrector constructor -------------------------
//------------------------------------------------------------------------
FactorizedJetCorrector::FactorizedJetCorrector()
{
  mJetEta  = -9999;
  mJetPt   = -9999;
  mJetPhi  = -9999;
  mJetE    = -9999;
  mJetEMF  = -9999;
  mJetA    = -9999;
  mRho     = -9999;
  mLepPx   = -9999;
  mLepPy   = -9999;
  mLepPz   = -9999;
  mNPV     = -9999;
  mJPTrawE = -9999;
  mJPTrawEt = -9999;
  mJPTrawPt = -9999;
  mJPTrawEta = -9999;
  mJPTrawOff = -9999;
  mAddLepToJet      = false;
  mIsNPVset         = false;
  mIsJetEset        = false;
  mIsJetPtset       = false;
  mIsJetPhiset      = false;
  mIsJetEtaset      = false;
  mIsJetEMFset      = false;
  mIsJetAset        = false;
  mIsRhoset         = false;
  mIsJPTrawP4set    = false;
  mIsJPTrawOFFset   = false;
  mIsLepPxset       = false;
  mIsLepPyset       = false;
  mIsLepPzset       = false;
  mIsAddLepToJetset = false;
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
  mJetA   = -9999;
  mRho    = -9999;
  mLepPx  = -9999;
  mLepPy  = -9999;
  mLepPz  = -9999;
  mNPV    = -9999;
  mJPTrawE = -9999;
  mJPTrawEt = -9999;
  mJPTrawPt = -9999;
  mJPTrawEta = -9999;
  mJPTrawOff = -9999;
  mAddLepToJet      = false;
  mIsNPVset         = false;
  mIsJetEset        = false;
  mIsJetPtset       = false;
  mIsJetPhiset      = false;
  mIsJetEtaset      = false;
  mIsJetEMFset      = false;
  mIsJetAset        = false;
  mIsRhoset         = false;
  mIsJPTrawP4set    = false;
  mIsJPTrawOFFset   = false;
  mIsLepPxset       = false;
  mIsLepPyset       = false;
  mIsLepPzset       = false;
  mIsAddLepToJetset = false;
  initCorrectors(fLevels, fFiles, fOptions);       
}
//------------------------------------------------------------------------
//--- FactorizedJetCorrector constructor ---------------------------------
//------------------------------------------------------------------------
FactorizedJetCorrector::FactorizedJetCorrector(const std::vector<JetCorrectorParameters>& fParameters)
{
  mJetEta = -9999;
  mJetPt  = -9999;
  mJetPhi = -9999;
  mJetE   = -9999;
  mJetEMF = -9999;
  mJetA   = -9999;
  mRho    = -9999;
  mLepPx  = -9999;
  mLepPy  = -9999;
  mLepPz  = -9999;
  mNPV    = -9999;
  mJPTrawE = -9999;
  mJPTrawEt = -9999;
  mJPTrawPt = -9999;
  mJPTrawEta = -9999;
  mJPTrawOff = -9999;
  mAddLepToJet      = false;
  mIsNPVset         = false;
  mIsJetEset        = false;
  mIsJetPtset       = false;
  mIsJetPhiset      = false;
  mIsJetEtaset      = false;
  mIsJetEMFset      = false;
  mIsJetAset        = false;
  mIsRhoset         = false;
  mIsJPTrawP4set    = false;
  mIsJPTrawOFFset   = false;
  mIsLepPxset       = false;
  mIsLepPyset       = false;
  mIsLepPzset       = false;
  mIsAddLepToJetset = false;
  for(unsigned i=0;i<fParameters.size();i++) {
    std::string ss = fParameters[i].definitions().level();
    if (ss == "L1Offset")
      mLevels.push_back(kL1);
    else if (ss == "L1JPTOffset")
      mLevels.push_back(kL1JPT);
    else if (ss == "L2Relative")
      mLevels.push_back(kL2);
    else if (ss == "L3Absolute")
      mLevels.push_back(kL3);
    else if (ss == "L4EMF")
      mLevels.push_back(kL4);
    else if (ss == "L5Flavor")
      mLevels.push_back(kL5);
    else if (ss == "L6SLB")
      mLevels.push_back(kL6);
    else if (ss == "L7Parton")
      mLevels.push_back(kL7);
    else if (ss == "L1FastJet")
      mLevels.push_back(kL1fj);
    mCorrectors.push_back(new SimpleJetCorrector(fParameters[i]));
    mBinTypes.push_back(mapping(mCorrectors[i]->parameters().definitions().binVar()));
    mParTypes.push_back(mapping(mCorrectors[i]->parameters().definitions().parVar()));
  }  
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
  for(unsigned i=0;i<tmp.size();i++) {
    if (tmp[i] == "L1Offset")
      mLevels.push_back(kL1);
    else if (tmp[i] == "L1JPTOffset")
      mLevels.push_back(kL1JPT);
    else if (tmp[i] == "L2Relative")
      mLevels.push_back(kL2);
    else if (tmp[i] == "L3Absolute")
      mLevels.push_back(kL3);
    else if (tmp[i] == "L4EMF")
      mLevels.push_back(kL4);
    else if (tmp[i] == "L5Flavor")
      mLevels.push_back(kL5);
    else if (tmp[i] == "L6SLB")
      mLevels.push_back(kL6);
    else if (tmp[i] == "L7Parton")
      mLevels.push_back(kL7);
    else if (tmp[i] == "L1FastJet")
      mLevels.push_back(kL1fj);
    else {
      std::stringstream sserr; 
      sserr<<"unknown correction level "<<tmp[i];
      handleError("FactorizedJetCorrector",sserr.str());
    }							
  }   	 
  //---- Read the parameter filenames string and parse the requested sub-correction tags.
  std::vector<std::string> Files = parseLevels(removeSpaces(fFiles));
  //---- Read the Options string and define the FlavorOption and PartonOption.
  std::string FlavorOption = parseOption(removeSpaces(fOptions),"L5Flavor");
  std::string PartonOption = parseOption(removeSpaces(fOptions),"L7Parton");
  //---- Check the consistency between tags and requested sub-corrections. 
  checkConsistency(tmp,Files);  
  //---- Create instances of the requested sub-correctors.
  for(unsigned i=0;i<mLevels.size();i++) {     	    
    if (mLevels[i]==kL1||mLevels[i]==kL1JPT||mLevels[i]==kL2||mLevels[i]==kL3||mLevels[i]==kL4||mLevels[i]==kL6||mLevels[i]==kL1fj)
      mCorrectors.push_back(new SimpleJetCorrector(Files[i])); 
    else if (mLevels[i]==kL5 && FlavorOption.length()==0) 
      handleError("FactorizedJetCorrector","must specify flavor option when requesting L5Flavor correction!");
    else if (mLevels[i]==kL5 && FlavorOption.length()>0)
      mCorrectors.push_back(new SimpleJetCorrector(Files[i],FlavorOption));
    else if (mLevels[i]==kL7 && PartonOption.length()==0) 
      handleError("FactorizedJetCorrector","must specify parton option when requesting L7Parton correction!");
    else if (mLevels[i]==kL7 && PartonOption.length()>0)
      mCorrectors.push_back(new SimpleJetCorrector(Files[i],PartonOption));
    else {
      std::stringstream sserr; 
      sserr<<"unknown correction level "<<tmp[i];
      handleError("FactorizedJetCorrector",sserr.str());
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
  for(unsigned i=0;i<fNames.size();i++) {
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
    else if (ss=="RelLepPt")
      result.push_back(kRelLepPt);
    else if (ss=="PtRel")
      result.push_back(kPtRel);
    else if (ss=="NPV")
      result.push_back(kNPV);
    else if (ss=="JetA")
      result.push_back(kJetA);
    else if (ss=="Rho")
      result.push_back(kRho);
    else if (ss=="JPTrawE")
      result.push_back(kJPTrawE);
    else if (ss=="JPTrawEt")
      result.push_back(kJPTrawEt);
    else if (ss=="JPTrawPt")
      result.push_back(kJPTrawPt);
    else if (ss=="JPTrawEta")
      result.push_back(kJPTrawEta);
    else if (ss=="JPTrawOff")
      result.push_back(kJPTrawOff);
    else {
      std::stringstream sserr; 
      sserr<<"unknown parameter name: "<<ss;
      handleError("FactorizedJetCorrector",sserr.str());
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
  if (fLevels.size() != fTags.size()) {
    std::stringstream sserr; 
    sserr<<"number of correction levels: "<<fLevels.size()<<" doesn't match # of tags: "<<fTags.size();
    handleError("FactorizedJetCorrector",sserr.str());
  }
  //---- Second check: each tag must contain the corresponding sub-correction level.
  for(unsigned int i=0;i<fTags.size();i++) {
    if ((int)fTags[i].find(fLevels[i])<0) {
      std::stringstream sserr; 
      sserr<<"inconsistent tag: "<<fTags[i]<<" for "<<"the requested correction: "<<fLevels[i];
      handleError("FactorizedJetCorrector",sserr.str());
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
  while (pos<ss.length()) {
    tmp = "";
    i = ss.find(":" , pos);
    if (i<0 && pos==0) {
      result.push_back(ss);
      pos = ss.length();
    }
    else if (i<0 && pos>0) {
      for(j=pos;j<ss.length();j++)
        tmp+=ss[j];
      result.push_back(tmp);
      pos = ss.length();
    }  
    else {
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
  else {
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
  for(unsigned int i=0;i<ss.length();i++) {
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
  for(unsigned int i=0;i<mLevels.size();i++) { 
    vx = fillVector(mBinTypes[i]);
    vy = fillVector(mParTypes[i]);
    //if (mLevels[i]==kL2 || mLevels[i]==kL6)
      //mCorrectors[i]->setInterpolation(true); 
    scale = mCorrectors[i]->correction(vx,vy); 
    //----- For JPT jets, the offset is stored in order to be used later by the the L1JPTOffset
    if ((mLevels[i]==kL1 || mLevels[i]==kL1fj) && mIsJPTrawP4set && !mIsJPTrawOFFset) { 
      mJPTrawOff = scale;
      mIsJPTrawOFFset = true;
    }	
    else if (mLevels[i]==kL6 && mAddLepToJet) { 
      scale  *= 1.0 + getLepPt() / mJetPt;
      mJetE  *= scale;
      mJetPt *= scale;
      factor *= scale;
    }
    else {
      mJetE  *= scale;
      mJetPt *= scale;
      factor *= scale;  
    }
    factors.push_back(factor);	
  }
  mIsNPVset       = false;
  mIsJetEset      = false;
  mIsJetPtset     = false;
  mIsJetPhiset    = false;
  mIsJetEtaset    = false;
  mIsJetEMFset    = false;
  mIsJetAset      = false;
  mIsRhoset       = false;
  mIsJPTrawP4set  = false;
  mIsJPTrawOFFset = false;
  mIsLepPxset     = false;
  mIsLepPyset     = false;
  mIsLepPzset     = false;
  mAddLepToJet    = false;
  return factors; 
}
//------------------------------------------------------------------------ 
//--- Reads the parameter names and fills a vector of floats -------------
//------------------------------------------------------------------------
std::vector<float> FactorizedJetCorrector::fillVector(std::vector<VarTypes> fVarTypes)
{
  std::vector<float> result;
  for(unsigned i=0;i<fVarTypes.size();i++) {
    if (fVarTypes[i] == kJetEta) {
      if (!mIsJetEtaset) 
        handleError("FactorizedJetCorrector","jet eta is not set");
      result.push_back(mJetEta);
    }
    else if (fVarTypes[i] == kNPV) {
      if (!mIsNPVset)
        handleError("FactorizedJetCorrector","number of primary vertices is not set");
      result.push_back(mNPV);
    }
    else if (fVarTypes[i] == kJetPt) {
      if (!mIsJetPtset)
        handleError("FactorizedJetCorrector","jet pt is not set");
      result.push_back(mJetPt);
    }
    else if (fVarTypes[i] == kJetPhi) {
      if (!mIsJetPhiset) 
        handleError("FactorizedJetCorrector","jet phi is not set");
      result.push_back(mJetPhi);
    }
    else if (fVarTypes[i] == kJetE) {
      if (!mIsJetEset) 
        handleError("FactorizedJetCorrector","jet E is not set");
      result.push_back(mJetE);
    }
    else if (fVarTypes[i] == kJetEMF) {
      if (!mIsJetEMFset) 
        handleError("FactorizedJetCorrector","jet EMF is not set");
      result.push_back(mJetEMF);
    } 
    else if (fVarTypes[i] == kJetA) {
      if (!mIsJetAset) 
        handleError("FactorizedJetCorrector","jet area is not set");
      result.push_back(mJetA);
    }
    else if (fVarTypes[i] == kRho) {
      if (!mIsRhoset) 
        handleError("FactorizedJetCorrector","fastjet density Rho is not set");
      result.push_back(mRho);
    }
    else if (fVarTypes[i] == kJPTrawE) {
      if (!mIsJPTrawP4set) 
        handleError("FactorizedJetCorrector","raw CaloJet P4 for JPT is not set");
      result.push_back(mJPTrawE);
    }
    else if (fVarTypes[i] == kJPTrawEt) {
      if (!mIsJPTrawP4set) 
        handleError("FactorizedJetCorrector","raw CaloJet P4 for JPT is not set");
      result.push_back(mJPTrawEt);
    }
    else if (fVarTypes[i] == kJPTrawPt) {
      if (!mIsJPTrawP4set)
        handleError("FactorizedJetCorrector","raw CaloJet P4 for JPT is not set");
      result.push_back(mJPTrawPt);
    }
    else if (fVarTypes[i] == kJPTrawEta) {
      if (!mIsJPTrawP4set) 
        handleError("FactorizedJetCorrector","raw CaloJet P4 for JPT is not set");
      result.push_back(mJPTrawEta);
    }
    else if (fVarTypes[i] == kJPTrawOff) {
      if (!mIsJPTrawOFFset) 
        handleError("FactorizedJetCorrector","Offset correction for JPT is not set");
      result.push_back(mJPTrawOff);
    }
    else if (fVarTypes[i] == kRelLepPt) {
      if (!mIsJetPtset||!mIsAddLepToJetset||!mIsLepPxset||!mIsLepPyset) 
        handleError("FactorizedJetCorrector","can't calculate rel lepton pt");
      result.push_back(getRelLepPt());
    }
    else if (fVarTypes[i] == kPtRel) {
      if (!mIsJetPtset||!mIsJetEtaset||!mIsJetPhiset||!mIsJetEset||!mIsAddLepToJetset||!mIsLepPxset||!mIsLepPyset||!mIsLepPzset) 
        handleError("FactorizedJetCorrector","can't calculate ptrel");
      result.push_back(getPtRel());
    }
    else {
      std::stringstream sserr; 
      sserr<<"unknown parameter "<<fVarTypes[i];
      handleError("FactorizedJetCorrector",sserr.str());
    }
  }
  return result;      
}
//------------------------------------------------------------------------ 
//--- Calculate the lepPt (needed for the SLB) ---------------------------
//------------------------------------------------------------------------
float FactorizedJetCorrector::getLepPt() const
{
  return std::sqrt(mLepPx*mLepPx + mLepPy*mLepPy);
}
//------------------------------------------------------------------------ 
//--- Calculate the relLepPt (needed for the SLB) ---------------------------
//------------------------------------------------------------------------
float FactorizedJetCorrector::getRelLepPt() const
{
  float lepPt = getLepPt();
  return (mAddLepToJet) ? lepPt/(mJetPt + lepPt) : lepPt/mJetPt;
}
//------------------------------------------------------------------------ 
//--- Calculate the PtRel (needed for the SLB) ---------------------------
//------------------------------------------------------------------------
float FactorizedJetCorrector::getPtRel() const
{
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> >
    PtEtaPhiELorentzVector;
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> >
    XYZVector;
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
  if (lj2<=0) {
    std::stringstream sserr; 
    sserr<<"lepton+jet momentum sq is not positive: "<<lj2;
    handleError("FactorizedJetCorrector",sserr.str());
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
void FactorizedJetCorrector::setNPV(int fNPV)
{
  mNPV = fNPV;
  mIsNPVset = true;
}
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
void FactorizedJetCorrector::setJetA(float fA)
{
  mJetA = fA;
  mIsJetAset = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setRho(float fRho)
{
  mRho = fRho;
  mIsRhoset = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJPTrawP4(TLorentzVector fJPTrawP4)
{
  mJPTrawE   = fJPTrawP4.Energy();
  mJPTrawEt  = fJPTrawP4.Et();
  mJPTrawPt  = fJPTrawP4.Pt();
  mJPTrawEta = fJPTrawP4.Eta();
  mIsJPTrawP4set = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrector::setJPTrawOff(float fJPTrawOff)
{
  mJPTrawOff = fJPTrawOff;
  mIsJPTrawOFFset = true;
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
void FactorizedJetCorrector::setAddLepToJet(bool fAddLepToJet)
{
  mAddLepToJet = fAddLepToJet;
  mIsAddLepToJetset = true;
}
