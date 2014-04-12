// This is the file "FactorizedJetCorrectorCalculator.cc". 
// This is the implementation of the class FactorizedJetCorrectorCalculator.
// Author: Konstantinos Kousouris, Philipp Schieferdecker
// Email:  kkousour@fnal.gov, philipp.schieferdecker@cern.ch

#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"
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
//--- Default FactorizedJetCorrectorCalculator constructor -------------------------
//------------------------------------------------------------------------
FactorizedJetCorrectorCalculator::FactorizedJetCorrectorCalculator()
{
}
//------------------------------------------------------------------------ 
//--- FactorizedJetCorrectorCalculator constructor ---------------------------------
//------------------------------------------------------------------------
FactorizedJetCorrectorCalculator::FactorizedJetCorrectorCalculator(const std::string& fLevels, const std::string& fFiles, const std::string& fOptions)
{
  initCorrectors(fLevels, fFiles, fOptions);
}
//------------------------------------------------------------------------
//--- FactorizedJetCorrectorCalculator constructor ---------------------------------
//------------------------------------------------------------------------
FactorizedJetCorrectorCalculator::FactorizedJetCorrectorCalculator(const std::vector<JetCorrectorParameters>& fParameters)
{
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
//--- FactorizedJetCorrectorCalculator destructor ----------------------------------
//------------------------------------------------------------------------
FactorizedJetCorrectorCalculator::~FactorizedJetCorrectorCalculator()
{
  for(unsigned i=0;i<mCorrectors.size();i++)
    delete mCorrectors[i];
}
//------------------------------------------------------------------------ 
//--- initialises the correctors -----------------------------------------
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::initCorrectors(const std::string& fLevels, const std::string& fFiles, const std::string& fOptions)
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
      handleError("FactorizedJetCorrectorCalculator",sserr.str());
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
      handleError("FactorizedJetCorrectorCalculator","must specify flavor option when requesting L5Flavor correction!");
    else if (mLevels[i]==kL5 && FlavorOption.length()>0)
      mCorrectors.push_back(new SimpleJetCorrector(Files[i],FlavorOption));
    else if (mLevels[i]==kL7 && PartonOption.length()==0) 
      handleError("FactorizedJetCorrectorCalculator","must specify parton option when requesting L7Parton correction!");
    else if (mLevels[i]==kL7 && PartonOption.length()>0)
      mCorrectors.push_back(new SimpleJetCorrector(Files[i],PartonOption));
    else {
      std::stringstream sserr; 
      sserr<<"unknown correction level "<<tmp[i];
      handleError("FactorizedJetCorrectorCalculator",sserr.str());
    }
    mBinTypes.push_back(mapping(mCorrectors[i]->parameters().definitions().binVar())); 
    mParTypes.push_back(mapping(mCorrectors[i]->parameters().definitions().parVar()));	
  } 
}
//------------------------------------------------------------------------ 
//--- Mapping between variable names and variable types ------------------
//------------------------------------------------------------------------
std::vector<FactorizedJetCorrectorCalculator::VarTypes> FactorizedJetCorrectorCalculator::mapping(const std::vector<std::string>& fNames) const
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
      handleError("FactorizedJetCorrectorCalculator",sserr.str());
    }
  }
  return result;  
}
//------------------------------------------------------------------------ 
//--- Consistency checker ------------------------------------------------
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::checkConsistency(const std::vector<std::string>& fLevels, const std::vector<std::string>& fTags)
{
  //---- First check: the number of tags must be equal to the number of sub-corrections.
  if (fLevels.size() != fTags.size()) {
    std::stringstream sserr; 
    sserr<<"number of correction levels: "<<fLevels.size()<<" doesn't match # of tags: "<<fTags.size();
    handleError("FactorizedJetCorrectorCalculator",sserr.str());
  }
  //---- Second check: each tag must contain the corresponding sub-correction level.
  for(unsigned int i=0;i<fTags.size();i++) {
    if ((int)fTags[i].find(fLevels[i])<0) {
      std::stringstream sserr; 
      sserr<<"inconsistent tag: "<<fTags[i]<<" for "<<"the requested correction: "<<fLevels[i];
      handleError("FactorizedJetCorrectorCalculator",sserr.str());
    }
  }
}
//------------------------------------------------------------------------ 
//--- String parser ------------------------------------------------------
//------------------------------------------------------------------------
std::vector<std::string> FactorizedJetCorrectorCalculator::parseLevels(const std::string& ss) const
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
std::string FactorizedJetCorrectorCalculator::parseOption(const std::string& ss, const std::string& type) const
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
std::string FactorizedJetCorrectorCalculator::removeSpaces(const std::string& ss) const
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
float FactorizedJetCorrectorCalculator::getCorrection(FactorizedJetCorrectorCalculator::VariableValues& iValues) const
{
  std::vector<float> vv = getSubCorrections(iValues);
  return vv[vv.size()-1];
}
//------------------------------------------------------------------------ 
//--- Returns the vector of subcorrections, up to a given level ----------
//------------------------------------------------------------------------
std::vector<float> FactorizedJetCorrectorCalculator::getSubCorrections( FactorizedJetCorrectorCalculator::VariableValues& iValues) const
{
  float scale,factor;
  std::vector<float> factors;
  std::vector<float> vx,vy;
  factor = 1;
  for(unsigned int i=0;i<mLevels.size();i++) { 
    vx = fillVector(mBinTypes[i],iValues);
    vy = fillVector(mParTypes[i],iValues);
    //if (mLevels[i]==kL2 || mLevels[i]==kL6)
      //mCorrectors[i]->setInterpolation(true); 
    scale = mCorrectors[i]->correction(vx,vy); 
    //----- For JPT jets, the offset is stored in order to be used later by the the L1JPTOffset
    if ((mLevels[i]==kL1 || mLevels[i]==kL1fj) && iValues.mIsJPTrawP4set && !iValues.mIsJPTrawOFFset) { 
      iValues.setJPTrawOff(scale);
    }	
    else if (mLevels[i]==kL6 && iValues.mAddLepToJet) { 
      scale  *= 1.0 + getLepPt(iValues) / iValues.mJetPt;
      iValues.mJetE  *= scale;
      iValues.mJetPt *= scale;
      factor *= scale;
    }
    else {
      iValues.mJetE  *= scale;
      iValues.mJetPt *= scale;
      factor *= scale;  
    }
    factors.push_back(factor);	
  }
  iValues.reset();
  return factors; 
}
//------------------------------------------------------------------------ 
//--- Reads the parameter names and fills a vector of floats -------------
//------------------------------------------------------------------------
std::vector<float> FactorizedJetCorrectorCalculator::fillVector(const std::vector<VarTypes>& fVarTypes, 
								const FactorizedJetCorrectorCalculator::VariableValues& iValues) const
{
//  std::vector<VarTypes> fVarTypes = _fVarTypes;
  std::vector<float> result;
  for(unsigned i=0;i<fVarTypes.size();i++) {
    if (fVarTypes[i] == kJetEta) {
      if (!iValues.mIsJetEtaset) 
        handleError("FactorizedJetCorrectorCalculator","jet eta is not set");
      result.push_back(iValues.mJetEta);
    }
    else if (fVarTypes[i] == kNPV) {
      if (!iValues.mIsNPVset)
        handleError("FactorizedJetCorrectorCalculator","number of primary vertices is not set");
      result.push_back(iValues.mNPV);
    }
    else if (fVarTypes[i] == kJetPt) {
      if (!iValues.mIsJetPtset)
        handleError("FactorizedJetCorrectorCalculator","jet pt is not set");
      result.push_back(iValues.mJetPt);
    }
    else if (fVarTypes[i] == kJetPhi) {
      if (!iValues.mIsJetPhiset) 
        handleError("FactorizedJetCorrectorCalculator","jet phi is not set");
      result.push_back(iValues.mJetPhi);
    }
    else if (fVarTypes[i] == kJetE) {
      if (!iValues.mIsJetEset) 
        handleError("FactorizedJetCorrectorCalculator","jet E is not set");
      result.push_back(iValues.mJetE);
    }
    else if (fVarTypes[i] == kJetEMF) {
      if (!iValues.mIsJetEMFset) 
        handleError("FactorizedJetCorrectorCalculator","jet EMF is not set");
      result.push_back(iValues.mJetEMF);
    } 
    else if (fVarTypes[i] == kJetA) {
      if (!iValues.mIsJetAset) 
        handleError("FactorizedJetCorrectorCalculator","jet area is not set");
      result.push_back(iValues.mJetA);
    }
    else if (fVarTypes[i] == kRho) {
      if (!iValues.mIsRhoset) 
        handleError("FactorizedJetCorrectorCalculator","fastjet density Rho is not set");
      result.push_back(iValues.mRho);
    }
    else if (fVarTypes[i] == kJPTrawE) {
      if (!iValues.mIsJPTrawP4set) 
        handleError("FactorizedJetCorrectorCalculator","raw CaloJet P4 for JPT is not set");
      result.push_back(iValues.mJPTrawE);
    }
    else if (fVarTypes[i] == kJPTrawEt) {
      if (!iValues.mIsJPTrawP4set) 
        handleError("FactorizedJetCorrectorCalculator","raw CaloJet P4 for JPT is not set");
      result.push_back(iValues.mJPTrawEt);
    }
    else if (fVarTypes[i] == kJPTrawPt) {
      if (!iValues.mIsJPTrawP4set)
        handleError("FactorizedJetCorrectorCalculator","raw CaloJet P4 for JPT is not set");
      result.push_back(iValues.mJPTrawPt);
    }
    else if (fVarTypes[i] == kJPTrawEta) {
      if (!iValues.mIsJPTrawP4set) 
        handleError("FactorizedJetCorrectorCalculator","raw CaloJet P4 for JPT is not set");
      result.push_back(iValues.mJPTrawEta);
    }
    else if (fVarTypes[i] == kJPTrawOff) {
      if (!iValues.mIsJPTrawOFFset) 
        handleError("FactorizedJetCorrectorCalculator","Offset correction for JPT is not set");
      result.push_back(iValues.mJPTrawOff);
    }
    else if (fVarTypes[i] == kRelLepPt) {
      if (!iValues.mIsJetPtset||!iValues.mIsAddLepToJetset||!iValues.mIsLepPxset||!iValues.mIsLepPyset) 
        handleError("FactorizedJetCorrectorCalculator","can't calculate rel lepton pt");
      result.push_back(getRelLepPt(iValues));
    }
    else if (fVarTypes[i] == kPtRel) {
      if (!iValues.mIsJetPtset||!iValues.mIsJetEtaset||!iValues.mIsJetPhiset||!iValues.mIsJetEset||!iValues.mIsAddLepToJetset||!iValues.mIsLepPxset||!iValues.mIsLepPyset||!iValues.mIsLepPzset) 
        handleError("FactorizedJetCorrectorCalculator","can't calculate ptrel");
      result.push_back(getPtRel(iValues));
    }
    else {
      std::stringstream sserr; 
      sserr<<"unknown parameter "<<fVarTypes[i];
      handleError("FactorizedJetCorrectorCalculator",sserr.str());
    }
  }
  return result;      
}
//------------------------------------------------------------------------ 
//--- Calculate the lepPt (needed for the SLB) ---------------------------
//------------------------------------------------------------------------
float FactorizedJetCorrectorCalculator::getLepPt(const FactorizedJetCorrectorCalculator::VariableValues& iValues) const
{
  return std::sqrt(iValues.mLepPx*iValues.mLepPx + iValues.mLepPy*iValues.mLepPy);
}
//------------------------------------------------------------------------ 
//--- Calculate the relLepPt (needed for the SLB) ---------------------------
//------------------------------------------------------------------------
float FactorizedJetCorrectorCalculator::getRelLepPt(const FactorizedJetCorrectorCalculator::VariableValues& iValues) const
{
  float lepPt = getLepPt(iValues);
  return (iValues.mAddLepToJet) ? lepPt/(iValues.mJetPt + lepPt) : lepPt/iValues.mJetPt;
}
//------------------------------------------------------------------------ 
//--- Calculate the PtRel (needed for the SLB) ---------------------------
//------------------------------------------------------------------------
float FactorizedJetCorrectorCalculator::getPtRel(const FactorizedJetCorrectorCalculator::VariableValues& iValues) const
{
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> >
    PtEtaPhiELorentzVector;
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> >
    XYZVector;
  PtEtaPhiELorentzVector jet;
  XYZVector lep;
  jet.SetPt(iValues.mJetPt);
  jet.SetEta(iValues.mJetEta);
  jet.SetPhi(iValues.mJetPhi);
  jet.SetE(iValues.mJetE);
  lep.SetXYZ(iValues.mLepPx,iValues.mLepPy,iValues.mLepPz);
  float lj_x = (iValues.mAddLepToJet) ? lep.X()+jet.Px() : jet.Px();
  float lj_y = (iValues.mAddLepToJet) ? lep.Y()+jet.Py() : jet.Py();
  float lj_z = (iValues.mAddLepToJet) ? lep.Z()+jet.Pz() : jet.Pz();
  // absolute values squared
  float lj2  = lj_x*lj_x+lj_y*lj_y+lj_z*lj_z;
  if (lj2<=0) {
    std::stringstream sserr; 
    sserr<<"lepton+jet momentum sq is not positive: "<<lj2;
    handleError("FactorizedJetCorrectorCalculator",sserr.str());
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
//--- Default FactorizedJetCorrectorCalculator::VariableValues constructor -------------------------
//------------------------------------------------------------------------
FactorizedJetCorrectorCalculator::VariableValues::VariableValues()
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
//--- Setters ------------------------------------------------------------
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setNPV(int fNPV)
{
  mNPV = fNPV;
  mIsNPVset = true;
}
void FactorizedJetCorrectorCalculator::VariableValues::setJetEta(float fEta)
{
  mJetEta = fEta;
  mIsJetEtaset = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setJetPt(float fPt)
{
  mJetPt = fPt;
  mIsJetPtset  = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setJetPhi(float fPhi)
{
  mJetPhi = fPhi;
  mIsJetPhiset  = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setJetE(float fE)
{
  mJetE = fE;
  mIsJetEset   = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setJetEMF(float fEMF)
{
  mJetEMF = fEMF;
  mIsJetEMFset = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setJetA(float fA)
{
  mJetA = fA;
  mIsJetAset = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setRho(float fRho)
{
  mRho = fRho;
  mIsRhoset = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setJPTrawP4(const TLorentzVector& fJPTrawP4)
{
  mJPTrawE   = fJPTrawP4.Energy();
  mJPTrawEt  = fJPTrawP4.Et();
  mJPTrawPt  = fJPTrawP4.Pt();
  mJPTrawEta = fJPTrawP4.Eta();
  mIsJPTrawP4set = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setJPTrawOff(float fJPTrawOff)
{
  mJPTrawOff = fJPTrawOff;
  mIsJPTrawOFFset = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setLepPx(float fPx)
{
  mLepPx = fPx;
  mIsLepPxset  = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setLepPy(float fPy)
{
  mLepPy = fPy;
  mIsLepPyset  = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setLepPz(float fPz)
{
  mLepPz = fPz;
  mIsLepPzset  = true;
}
//------------------------------------------------------------------------
void FactorizedJetCorrectorCalculator::VariableValues::setAddLepToJet(bool fAddLepToJet)
{
  mAddLepToJet = fAddLepToJet;
  mIsAddLepToJetset = true;
}

void FactorizedJetCorrectorCalculator::VariableValues::reset()
{
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
}
