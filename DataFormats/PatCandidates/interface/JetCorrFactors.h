#ifndef DataFormats_PatCandidates_JetCorrFactors_h
#define DataFormats_PatCandidates_JetCorrFactors_h

/**
   \class    pat::JetCorrFactors JetCorrFactors.h "DataFormats/PatCandidates/interface/JetCorrFactors.h"
   \brief    Class for the storage of jet correction factors
   
   Class for the storage of jet correction factors that have been calculated during pat tuple production. 
   The class is created to deal with a flexible number and order of the JES correction factors, which are 
   expected to be nested. I.e. each correction level implies that all previous correction have been applied 
   in advance. This scheme corresponds to the jet energy correction scheme propagated by the JetMET PAG.
   In dividual levels of JEC are safed as CorrectionFactor, which is a 

   std::pair<std::string, std::vector<float> >. 

   The std::string contains a human readable label indicating the corection level, the std::vector<float> 
   contains the JEC factors, which are expected to have a length of 1 or 5. In this scheme the vector of 
   length 1 is reserved for flavor independent CorrectionFactors, while the vector of length 5 corresponds 
   to flavor dependent CorrectionFactors. The individual positions within the vector are expected to be 
   distributed according to the Flavor enumerator of the class as: 

   GLUON, UDS, CHARM, BOTTOM, NONE

   The structure is checked in the constructor of the class. The function _correction_ returns potentially 
   flavor dependent correction factor of the JES relative to an uncorrected jet. To move from one correction 
   level to another correction level the initial correction level of the jet need to be uncorrected before
   applying the final correction factor. The class is expected to be used from within the pat::Jet only, 
   this is taken care of automatically. 
*/

#include <vector>
#include <string>
#include <cmath>

namespace pat {

  class JetCorrFactors {

  public:
    // jet energy correction factor. For flavor independent jet energy corrections the
    // std::vector<float> holds just a single entry. From the first flavor dependent entry
    // in the chain on it holds five floats corresponding to the flavors: none, gluon, uds, 
    // charm, bottom; in this case the entry for none will be set to -1; the std::string 
    // indicates the correction level according to jetMET definitions.
    typedef std::pair<std::string, std::vector<float> > CorrectionFactor;
    // order of flavor dependent CorrectionFactors
    enum Flavor { GLUON, UDS, CHARM, BOTTOM, NONE };
    // number of maximally available flavor types
    static const unsigned int MAX_FLAVORS = 4;

  public:
    // default Constructor
    JetCorrFactors() {};
    // constructor by value
    JetCorrFactors(const std::string& label, const std::vector<CorrectionFactor>& jec);

    // instance label of the jet energy corrections set 
    std::string jecSet() const { return label_; }
    // correction level from unsigned int
    std::string jecLevel(const unsigned int& level) const { return jec_.at(level).first; };
    // correction level from std::string
    int jecLevel(const std::string& level) const;
    // jet energy correction flavor from enum
    std::string jecFlavor(const Flavor& flavor) const;
    // jet energy correction flavor from std::string
    Flavor jecFlavor(std::string flavor) const;

    // correction factor up to a given level and flavor (per default the flavor is NONE)
    float correction(unsigned int level, Flavor flavor=NONE) const;
    // a list of the labels of all correction levels according to jetMET definitions, separated by '\n'
    std::string correctionLabelString() const;
    // a vector of the labels of all correction levels according to jetMET definitions
    std::vector<std::string> correctionLabels() const;
    // label of a specific correction factor according to jetMET definitions; for overflow a string ERROR is returned
    std::string correctionLabel(unsigned int level) const { return (level<jec_.size() ? jec_.at(level).first : std::string("ERROR")); };
    // check whether CorrectionFactor is flavor independent or not
    bool flavorDependent(unsigned int level) const { return (level<jec_.size() ? jec_.at(level).second.size()==MAX_FLAVORS : false); };
    // number of available correction factors
    unsigned int numberOfCorrectionLevels() const { return jec_.size(); };
    // print function for debugging
    void print() const;

  private:
    // check consistency of input vector
    bool flavorDependent(const CorrectionFactor& jec) const { return (jec.second.size()==MAX_FLAVORS); }
    // check consistency of input vector
    bool flavorIndependent(const CorrectionFactor& jec) const { return (jec.second.size()==1); }
    // check consistency of input vector
    bool isValid(const CorrectionFactor& jec) const { return (flavorDependent(jec) || flavorIndependent(jec)); }    
    
  private:
    // instance label of jet energy correction factors
    std::string label_;
    // vector of CorrectionFactors. NOTE: the correction factors are expected to appear 
    // nested; they may appear in arbitary number and order according to the configuration 
    // of the jetCorrFactors module. CorrectionFactors appear in two versions: as a single 
    // float (for flavor independent corrections) or as a std::vector of four floats (for 
    // flavor dependent corrections). Due to the nested structure of the CorrectionFactors 
    // from the first flavor dependent CorrectionFactor in the chain on each correction is 
    // flavor dependent. 
    std::vector<CorrectionFactor> jec_;
  };
}

#endif
