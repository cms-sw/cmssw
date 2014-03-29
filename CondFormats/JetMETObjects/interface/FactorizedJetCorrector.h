// This is the header file "FactorizedJetCorrector.h". This is the interface for the 
// class FactorizedJetCorrector.
// Author: Konstantinos Kousouris, Philipp Schieferdecker
// Email:  kkousour@fnal.gov, philipp.schieferdecker@cern.ch

#ifndef FACTORIZED_JET_CORRECTOR_H
#define FACTORIZED_JET_CORRECTOR_H

#include <vector>
#include <string>
#include "TLorentzVector.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"

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
    //---- Member Data ---------
    FactorizedJetCorrectorCalculator::VariableValues mValues;
    FactorizedJetCorrectorCalculator mCalc;
};
#endif
