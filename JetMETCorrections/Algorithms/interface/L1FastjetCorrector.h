//
// L1FastjetCorrector
// ------------------
//
// 08/09/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
//
#ifndef L1FastjetCorrector_h
#define L1FastjetCorrector_h 1

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"

class L1FastjetCorrector : public JetCorrector
{
public:
  // construction / destruction
  L1FastjetCorrector(const JetCorrectorParameters& fParam, const edm::ParameterSet& fConfig);
  virtual ~L1FastjetCorrector();
  
public:
  //member functions
  
  /// apply correction using Jet information only
  virtual double correction(const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction(const reco::Jet& fJet) const;
  /// apply correction using all event information
  virtual double correction(const reco::Jet& fJet,
			    const edm::Event& fEvent,
			    const edm::EventSetup& fSetup) const;
  /// if correction needs event information
  virtual bool eventRequired() const { return true; }

  //----- if correction needs a jet reference -------------
  virtual bool refRequired() const { return false; }

private:
  // member data
  edm::InputTag srcRho_;
  FactorizedJetCorrectorCalculator const* mCorrector;
};

#endif
