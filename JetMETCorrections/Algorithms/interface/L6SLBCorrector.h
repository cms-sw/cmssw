////////////////////////////////////////////////////////////////////////////////
//
// Level6 SLB (Semileptonic BJet) Corrector
// ----------------------------------------
//
//           25/10/2009  Hauke Held             <hauke.held@cern.ch>
//                       Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////
#ifndef L6SLBCorrector_h
#define L6SLBCorrector_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

class L6SLBCorrector : public JetCorrector {
  //
  // construction / destruction
  //
public:
  L6SLBCorrector(const JetCorrectorParameters& fParam, const edm::ParameterSet& fConfig);
  ~L6SLBCorrector() override;

  //
  // member functions
  //
public:
  /// apply correction using Jet information only
  double correction(const LorentzVector& fJet) const override;
  /// apply correction using Jet information only
  double correction(const reco::Jet& fJet) const override;
  /// apply correction using all event information
  double correction(const reco::Jet& fJet,
                    const edm::RefToBase<reco::Jet>& refToRawJet,
                    const edm::Event& fEvent,
                    const edm::EventSetup& fSetup) const override;

  /// if correction needs event information
  bool eventRequired() const override { return true; }

  //----- if correction needs a jet reference -------------
  bool refRequired() const override { return true; }

  //
  // private member functions
  //
private:
  int getBTagInfoIndex(const edm::RefToBase<reco::Jet>& refToRawJet,
                       const std::vector<reco::SoftLeptonTagInfo>& tags) const;

  //
  // member data
  //
private:
  std::string tagName_;
  bool addMuonToJet_;
  edm::InputTag srcBTagInfoElec_;
  edm::InputTag srcBTagInfoMuon_;
  FactorizedJetCorrectorCalculator* corrector_;
};

#endif
