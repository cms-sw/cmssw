// L1Offset jet corrector class. Inherits from JetCorrector.h
#ifndef L1OffsetCorrector_h
#define L1OffsetCorrector_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

//----- classes declaration -----------------------------------
namespace edm {
  class ParameterSet;
}
class FactorizedJetCorrectorCalculator;
//----- LXXXCorrector interface -------------------------------
class L1OffsetCorrector : public JetCorrector {
public:
  //----- constructors---------------------------------------
  L1OffsetCorrector(const JetCorrectorParameters& fConfig, const edm::ParameterSet& fParameters);

  //----- destructor ----------------------------------------
  ~L1OffsetCorrector() override;

  //----- apply correction using Jet information only -------
  double correction(const LorentzVector& fJet) const override;

  //----- apply correction using Jet information only -------
  double correction(const reco::Jet& fJet) const override;

  //----- apply correction using all event information
  double correction(const reco::Jet& fJet, const edm::Event& fEvent, const edm::EventSetup& fSetup) const override;
  //----- if correction needs event information -------------
  bool eventRequired() const override { return true; }

  //----- if correction needs a jet reference -------------
  bool refRequired() const override { return false; }

private:
  //----- member data ---------------------------------------
  std::string mVertexCollName;
  int mMinVtxNdof;
  FactorizedJetCorrectorCalculator* mCorrector;
};

#endif
