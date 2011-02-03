// L1Offset jet corrector class. Inherits from JetCorrector.h
#ifndef L1OffsetCorrector_h
#define L1OffsetCorrector_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

//----- classes declaration -----------------------------------
namespace edm 
{
  class ParameterSet;
}
class FactorizedJetCorrector;
//----- LXXXCorrector interface -------------------------------
class L1OffsetCorrector : public JetCorrector 
{
  public:
    //----- constructors---------------------------------------
    L1OffsetCorrector(const JetCorrectorParameters& fConfig, const edm::ParameterSet& fParameters);   

    //----- destructor ----------------------------------------
    virtual ~L1OffsetCorrector();

    //----- apply correction using Jet information only -------
    virtual double correction(const LorentzVector& fJet) const;

    //----- apply correction using Jet information only -------
    virtual double correction(const reco::Jet& fJet) const;

    //----- apply correction using all event information
    virtual double correction(const reco::Jet& fJet, 
                              const edm::RefToBase<reco::Jet>& fJetRef,
                              const edm::Event& fEvent, 
                              const edm::EventSetup& fSetup) const;

    //----- if correction needs event information -------------
    virtual bool eventRequired() const {return true;} 

  private:
    //----- member data ---------------------------------------
    std::string mVertexCollName;
    int mMinVtxNdof;
    FactorizedJetCorrector* mCorrector;

};

#endif
