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


class L6SLBCorrector : public JetCorrector
{
  //
  // construction / destruction
  //
public:
  L6SLBCorrector (const JetCorrectorParameters& fParam, const edm::ParameterSet& fConfig);
  virtual ~L6SLBCorrector ();
  

  //
  // member functions
  //
public:
  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;
  /// apply correction using all event information
  virtual double correction (const reco::Jet& fJet,
			     const edm::RefToBase<reco::Jet>& refToRawJet,
			     const edm::Event& fEvent, 
			     const edm::EventSetup& fSetup) const;
  
  /// if correction needs event information
  virtual bool eventRequired () const {return true;} 

  //----- if correction needs a jet reference -------------
  virtual bool refRequired () const {return true;} 
  
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
  std::string             tagName_;
  bool                    addMuonToJet_;
  edm::InputTag           srcBTagInfoElec_;
  edm::InputTag           srcBTagInfoMuon_;
  FactorizedJetCorrectorCalculator* corrector_;
  
};

#endif
