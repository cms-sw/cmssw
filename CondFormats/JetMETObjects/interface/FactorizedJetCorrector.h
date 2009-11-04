// This is the header file "FactorizedJetCorrector.h". This is the interface for the 
// class FactorizedJetCorrector.
// Author: Konstantinos Kousouris, Philipp Schieferdecker
// Email:  kkousour@fnal.gov, philipp.schieferdecker@cern.ch

#ifndef FACTORIZED_JET_CORRECTOR_H
#define FACTORIZED_JET_CORRECTOR_H

#include <vector>
#include <string>

class SimpleJetCorrector;

class FactorizedJetCorrector
{
  public:
    FactorizedJetCorrector();
    FactorizedJetCorrector(const std::string& fLevels, const std::string& fTags);
    FactorizedJetCorrector(const std::string& fLevels, const std::string& fTags, const std::string& fOptions);
    ~FactorizedJetCorrector();
    void setJetEta      (float fEta);
    void setJetPt       (float fPt); 
    void setJetE        (float fE);
    void setJetPhi      (float fE);
    void setJetEMF      (float fEMF); 
    void setLepPx       (float fLepPx);
    void setLepPy       (float fLepPy);
    void setLepPz       (float fLepPz);
    void setAddLepToJet (bool fAddLepToJet) {mAddLepToJet = fAddLepToJet;}
    float getCorrection();
    std::vector<float> getSubCorrections();
       
  private:
    //---- Member Functions ----  
    FactorizedJetCorrector(const FactorizedJetCorrector&);
    FactorizedJetCorrector& operator= (const FactorizedJetCorrector&);
    float getPtRel();
    std::string parseOption(const std::string& ss, const std::string& type);
    std::string removeSpaces(const std::string& ss);
    std::vector<std::string> parseLevels(const std::string& ss);
    void initCorrectors(const std::string& fLevels, const std::string& fTags, const std::string& fOptions);
    void checkConsistency(const std::vector<std::string>& fLevels, const std::vector<std::string>& fTags);
    std::vector<float> fillVector(const std::vector<std::string>& fNames);
    //---- Member Data ---------
    float mJetE;
    float mJetEta;
    float mJetPt;
    float mJetPhi;
    float mJetEMF; 
    float mLepPx;
    float mLepPy;
    float mLepPz;
    bool  mAddLepToJet;
    bool  mIsJetEset;
    bool  mIsJetPtset;
    bool  mIsJetPhiset;
    bool  mIsJetEtaset;
    bool  mIsJetEMFset; 
    bool  mIsLepPxset;
    bool  mIsLepPyset;
    bool  mIsLepPzset;
    std::vector<std::string> mLevels; 
    std::vector<SimpleJetCorrector*> mCorrectors;
};
#endif
