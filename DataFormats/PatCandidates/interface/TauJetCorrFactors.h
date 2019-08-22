#ifndef DataFormats_PatCandidates_TauJetCorrFactors_h
#define DataFormats_PatCandidates_TauJetCorrFactors_h

/**
   \class    pat::TauCorrFactors TauCorrFactors.h "DataFormats/PatCandidates/interface/TauCorrFactors.h"
   \brief    Class for the storage of tau-jet energy correction factors
   
   Class for the storage of tau-jet energy correction factors that have been calculated during pat tuple production. 
   The class is created to deal with a flexible number and order of the JES correction factors, which are 
   expected to be nested. I.e. each correction level implies that all previous correction have been applied 
   in advance. This scheme corresponds to the jet energy correction scheme propagated by the JetMET PAG.
   In dividual levels of JEC are safed as CorrectionFactor, which is a 

   std::pair<std::string, float>. 

   The std::string contains a human readable label indicating the corection level, the float
   contains the JEC factor.
   
   To move from one correction 
   level to another correction level the initial correction level of the jet need to be uncorrected before
   applying the final correction factor. The class is expected to be used from within the pat::Tau only, 
   this is taken care of automatically. 
*/

#include <vector>
#include <string>
#include <cmath>

namespace pat {

  class TauJetCorrFactors {
  public:
    // tau-jet energy correction factor.
    // the std::string indicates the correction level according to jetMET definitions.
    typedef std::pair<std::string, float> CorrectionFactor;

  public:
    // default Constructor
    TauJetCorrFactors(){};
    // constructor by value
    TauJetCorrFactors(const std::string& label, const std::vector<CorrectionFactor>& jec);

    // instance label of the jet energy corrections set
    std::string jecSet() const { return label_; }
    // correction level from unsigned int
    std::string jecLevel(const unsigned int& level) const { return jec_.at(level).first; };
    // correction level from std::string
    int jecLevel(const std::string& level) const;

    // correction factor up to a given level
    float correction(unsigned int level) const;
    // a list of the labels of all correction levels according to jetMET definitions, separated by '\n'
    std::string correctionLabelString() const;
    // a vector of the labels of all correction levels according to jetMET definitions
    std::vector<std::string> correctionLabels() const;
    // label of a specific correction factor according to jetMET definitions; for overflow a string ERROR is returned
    std::string correctionLabel(unsigned int level) const {
      return (level < jec_.size() ? jec_.at(level).first : std::string("ERROR"));
    };
    // number of available correction factors
    unsigned int numberOfCorrectionLevels() const { return jec_.size(); };
    // print function for debugging
    void print() const;

  private:
    // instance label of jet energy correction factors
    std::string label_;
    // vector of CorrectionFactors. NOTE: the correction factors are expected to appear
    // nested; they may appear in arbitary number and order according to the configuration
    // of the jetCorrFactors module.
    std::vector<CorrectionFactor> jec_;
  };
}  // namespace pat

#endif
