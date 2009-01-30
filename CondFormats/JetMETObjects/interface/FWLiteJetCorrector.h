#ifndef FWLITEJETCORRECTOR_H
#define FWLITEJETCORRECTOR_H

#include "CondFormats/JetMETObjects/interface/FWLiteJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL2RelativeCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL3AbsoluteCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL3PFAbsoluteCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL4EMFCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL5FlavorCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL7PartonCorrector.h"

#include <vector>
#include <string>

class SimpleL2RelativeCorrector;
class SimpleL3AbsoluteCorrector;
class SimpleL3PFAbsoluteCorrector;
class SimpleL4EMFCorrector;
class SimpleL5FlavorCorrector;
class SimpleL7PartonCorrector;
class FWLiteJetCorrector
{
  public:
    FWLiteJetCorrector();
    FWLiteJetCorrector(std::string CorrectionLevels, std::string CorrectionTags);
    FWLiteJetCorrector(std::string CorrectionLevels, std::string CorrectionTags, std::string Options);
    double getCorrection(double pt, double eta);
    double getCorrection(double pt, double eta, double emf);
    std::vector<double> getSubCorrections(double pt, double eta);
    std::vector<double> getSubCorrections(double pt, double eta, double emf);
    ~FWLiteJetCorrector();
       
  private:
    SimpleL2RelativeCorrector*   mL2Corrector;
    SimpleL3AbsoluteCorrector*   mL3Corrector;
    SimpleL3PFAbsoluteCorrector* mL3PFCorrector; 
    SimpleL4EMFCorrector*        mL4Corrector;
    SimpleL5FlavorCorrector*     mL5Corrector;
    SimpleL7PartonCorrector*     mL7Corrector;
    std::string mL3Option;
    std::vector<std::string> mLevels; 
    std::vector<std::string> parseLevels(std::string ss); 
    std::string parseOption(std::string ss, std::string type);
    void initCorrectors(std::string CorrectionLevels, std::string CorrectionTags);
    void initCorrectors(std::string CorrectionLevels, std::string CorrectionTags, std::string Options);
};
#endif
