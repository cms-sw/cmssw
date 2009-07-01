// This is the header file "CombinedJetCorrector.h". This is the interface for the 
// class CombinedJetCorrector.
// Author: Konstantinos Kousouris, 
// Email:  kkousour@fnal.gov

#ifndef COMBINEDJETCORRECTOR_H
#define COMBINEDJETCORRECTOR_H

#include <vector>
#include <string>

class SimpleL2RelativeCorrector;
class SimpleL3AbsoluteCorrector;
class SimpleL3PFAbsoluteCorrector;
class SimpleL4EMFCorrector;
class SimpleL5FlavorCorrector;
class SimpleL7PartonCorrector;

class CombinedJetCorrector
{
  public:
    CombinedJetCorrector();
    CombinedJetCorrector(std::string CorrectionLevels, std::string CorrectionTags);
    CombinedJetCorrector(std::string CorrectionLevels, std::string CorrectionTags, std::string Options);
    double getCorrection(double pt, double eta);
    double getCorrection(double pt, double eta, double emf);
    std::vector<double> getSubCorrections(double pt, double eta);
    std::vector<double> getSubCorrections(double pt, double eta, double emf);
    ~CombinedJetCorrector();
       
  private:
    //---- Member Functions ----  
    CombinedJetCorrector(const CombinedJetCorrector&);
    CombinedJetCorrector& operator= (const CombinedJetCorrector&);

    std::vector<std::string> parseLevels(std::string ss); 
    std::string parseOption(std::string ss, std::string type);
    std::string removeSpaces(std::string ss);
    void initCorrectors(std::string CorrectionLevels, std::string CorrectionTags, std::string Options);
    void checkConsistency(std::vector<std::string> Levels, std::vector<std::string> Tags);
    //---- Member Data ---------
    std::string mL3Option;
    std::vector<std::string> mLevels; 
    
    SimpleL2RelativeCorrector*   mL2Corrector;
    SimpleL3AbsoluteCorrector*   mL3Corrector;
    SimpleL3PFAbsoluteCorrector* mL3PFCorrector; 
    SimpleL4EMFCorrector*        mL4Corrector;
    SimpleL5FlavorCorrector*     mL5Corrector;
    SimpleL7PartonCorrector*     mL7Corrector;
};
#endif
