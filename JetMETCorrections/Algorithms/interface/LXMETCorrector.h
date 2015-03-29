// Generic LX MET corrector class. Inherits from (should be METCorrector) JetCorrector.h
#ifndef LXMETCorrector_h
#define LXMETCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/METCorrectorParameters.h"

//----- classes declaration -----------------------------------
namespace edm 
{
  class ParameterSet;
}
//class FactorizedJetCorrector;
//----- LXXXCorrector interface -------------------------------
class LXMETCorrector : public JetCorrector 
//class METCorrector
{
  public:
    //----- constructors---------------------------------------
    //LXMETCorrector(const METCorrectorParameters& fConfig, const edm::ParameterSet& fParameters);
    LXMETCorrector(const JetCorrectorParameters& fConfig, const edm::ParameterSet& fParameters);   

    //----- destructor ----------------------------------------
    virtual ~LXMETCorrector();

    //----- apply correction using Jet information only -------
    virtual double correction(const LorentzVector& fJet) const;

    //----- apply correction using Jet information only -------
    virtual double correction(const reco::Jet& fJet) const;

    //----- if correction needs event information -------------
    virtual bool eventRequired() const {return false;} 

    //----- if correction needs a jet reference -------------
    virtual bool refRequired() const { return false; }

  private:
    //----- member data ---------------------------------------
    unsigned mLevel;
    FactorizedJetCorrector* mCorrector;
};

#endif
