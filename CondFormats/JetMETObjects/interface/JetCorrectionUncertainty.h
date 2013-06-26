#ifndef JetCorrectionUncertainty_h
#define JetCorrectionUncertainty_h

#include <string>
#include <vector>
class SimpleJetCorrectionUncertainty;
class JetCorrectorParameters;

class JetCorrectionUncertainty 
{
  public:
    JetCorrectionUncertainty();
    JetCorrectionUncertainty(const std::string& fDataFile);
    JetCorrectionUncertainty(const JetCorrectorParameters& fParameters);
    ~JetCorrectionUncertainty();

    void setParameters  (const std::string& fDataFile);
    void setJetEta      (float fEta);
    void setJetPt       (float fPt); 
    void setJetE        (float fE);
    void setJetPhi      (float fE);
    void setJetEMF      (float fEMF); 
    void setLepPx       (float fLepPx);
    void setLepPy       (float fLepPy);
    void setLepPz       (float fLepPz);
    void setAddLepToJet (bool fAddLepToJet) {mAddLepToJet = fAddLepToJet;}
    float getUncertainty(bool fDirection);

 private:
  JetCorrectionUncertainty(const JetCorrectionUncertainty&);
  JetCorrectionUncertainty& operator= (const JetCorrectionUncertainty&);
  std::vector<float> fillVector(const std::vector<std::string>& fNames);
  float getPtRel();
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
  SimpleJetCorrectionUncertainty* mUncertainty;
};

#endif

