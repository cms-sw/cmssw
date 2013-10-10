// This is the header file "FactorizedJetCorrector.h". This is the interface for the 
// class FactorizedJetCorrector.
// Author: Konstantinos Kousouris, Philipp Schieferdecker
// Email:  kkousour@fnal.gov, philipp.schieferdecker@cern.ch

#ifndef FACTORIZED_JET_CORRECTOR_H
#define FACTORIZED_JET_CORRECTOR_H

#include "CondFormats/Common/interface/Serializable.h"

#include <vector>
#include <string>
#include "TLorentzVector.h"

class SimpleJetCorrector;
class JetCorrectorParameters;

class FactorizedJetCorrector
{
  public:
    enum VarTypes   {kJetPt,kJetEta,kJetPhi,kJetE,kJetEMF,kRelLepPt,kPtRel,kNPV,kJetA,kRho,kJPTrawE,kJPTrawEt,kJPTrawPt,kJPTrawEta,kJPTrawOff};
    enum LevelTypes {kL1,kL2,kL3,kL4,kL5,kL6,kL7,kL1fj,kL1JPT};
    FactorizedJetCorrector();
    FactorizedJetCorrector(const std::string& fLevels, const std::string& fTags, const std::string& fOptions="");
    FactorizedJetCorrector(const std::vector<JetCorrectorParameters>& fParameters);
    ~FactorizedJetCorrector();
    void setNPV		(int   fNPV);
    void setJetEta      (float fEta);
    void setJetPt       (float fPt); 
    void setJetE        (float fE);
    void setJetPhi      (float fE);
    void setJetEMF      (float fEMF); 
    void setJetA        (float fA);
    void setRho         (float fRho);
    void setJPTrawP4    (const TLorentzVector& fJPTrawP4);
    void setJPTrawOff   (float fJPTrawOff);
    void setLepPx       (float fLepPx);
    void setLepPy       (float fLepPy);
    void setLepPz       (float fLepPz);
    void setAddLepToJet (bool fAddLepToJet);
    float getCorrection();
    std::vector<float> getSubCorrections();
    
       
  private:
  //---- Member Functions ----  
    FactorizedJetCorrector(const FactorizedJetCorrector&);
    FactorizedJetCorrector& operator= (const FactorizedJetCorrector&);
    float getLepPt()    const;
    float getRelLepPt() const;
    float getPtRel()    const;
    std::string parseOption(const std::string& ss, const std::string& type);
    std::string removeSpaces(const std::string& ss);
    std::vector<std::string> parseLevels(const std::string& ss);
    void initCorrectors(const std::string& fLevels, const std::string& fFiles, const std::string& fOptions);
    void checkConsistency(const std::vector<std::string>& fLevels, const std::vector<std::string>& fTags);
    std::vector<float> fillVector(const std::vector<VarTypes>& fVarTypes);
    std::vector<VarTypes> mapping(const std::vector<std::string>& fNames);
    //---- Member Data ---------
    int   mNPV;
    float mJetE;
    float mJetEta;
    float mJetPt;
    float mJetPhi;
    float mJetEMF; 
    float mJetA;
    float mRho;
    float mJPTrawE;
    float mJPTrawEt;
    float mJPTrawPt;
    float mJPTrawEta; 
    float mJPTrawOff;
    float mLepPx;
    float mLepPy;
    float mLepPz;
    bool  mAddLepToJet;
    bool  mIsNPVset;
    bool  mIsJetEset;
    bool  mIsJetPtset;
    bool  mIsJetPhiset;
    bool  mIsJetEtaset;
    bool  mIsJetEMFset; 
    bool  mIsJetAset;
    bool  mIsRhoset;
    bool  mIsJPTrawP4set;
    bool  mIsJPTrawOFFset;
    bool  mIsLepPxset;
    bool  mIsLepPyset;
    bool  mIsLepPzset;
    bool  mIsAddLepToJetset;
    std::vector<LevelTypes> mLevels;
    std::vector<std::vector<VarTypes> > mParTypes,mBinTypes; 
    std::vector<SimpleJetCorrector*> mCorrectors;
};
#endif
