// L1Offset jet corrector class. Inherits from JetCorrector.h
#ifndef L1JPTOffsetCorrector_h
#define L1JPTOffsetCorrector_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

//----- classes declaration -----------------------------------
namespace edm 
{
  class ParameterSet;
}
class FactorizedJetCorrectorCalculator;
//----- LXXXCorrector interface -------------------------------
class L1JPTOffsetCorrector : public JetCorrector 
{
  public:
    //----- constructors---------------------------------------
    L1JPTOffsetCorrector(const JetCorrectorParameters& fConfig, const edm::ParameterSet& fParameters);   

    //----- destructor ----------------------------------------
    virtual ~L1JPTOffsetCorrector();

    //----- apply correction using Jet information only -------
    virtual double correction(const LorentzVector& fJet) const;

    //----- apply correction using Jet information only -------
    virtual double correction(const reco::Jet& fJet) const;

    //----- apply correction using all event information
    virtual double correction(const reco::Jet& fJet, 
                              const edm::Event& fEvent, 
                              const edm::EventSetup& fSetup) const;
    //----- if correction needs event information -------------
    virtual bool eventRequired() const {return true;} 
    virtual bool refRequired() const {return false;}

  private:
    //----- member data ---------------------------------------
    std::string mOffsetService;
    bool mIsOffsetSet;
    FactorizedJetCorrectorCalculator* mCorrector;
};

#endif
