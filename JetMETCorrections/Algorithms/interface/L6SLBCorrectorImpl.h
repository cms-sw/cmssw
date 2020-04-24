////////////////////////////////////////////////////////////////////////////////
//
// Level6 SLB (Semileptonic BJet) Corrector
// ----------------------------------------
//
//           25/10/2009  Hauke Held             <hauke.held@cern.ch>
//                       Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////
#ifndef L6SLBCorrectorImpl_h
#define L6SLBCorrectorImpl_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrectorImpl.h"
#include "JetMETCorrections/Algorithms/interface/JetCorrectorImplMakerBase.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

namespace edm 
{
  class ParameterSet;
  class Event;
  class EventSetup;
  class ConsumesCollector;
  class ConfigurationDescriptions;
}

class L6SLBCorrectorImplMaker : public JetCorrectorImplMakerBase {
 public:
  L6SLBCorrectorImplMaker(edm::ParameterSet const&, edm::ConsumesCollector);
  std::unique_ptr<reco::JetCorrectorImpl> make(edm::Event const&, edm::EventSetup const&);

  static void fillDescriptions(edm::ConfigurationDescriptions& iDescriptions);
 private:
  edm::EDGetTokenT<std::vector<reco::SoftLeptonTagInfo>> elecToken_;
  edm::EDGetTokenT<std::vector<reco::SoftLeptonTagInfo>> muonToken_;
  bool                    addMuonToJet_;
};

class L6SLBCorrectorImpl : public reco::JetCorrectorImpl
{
  //
  // construction / destruction
  //
public:
  typedef L6SLBCorrectorImplMaker Maker;

  L6SLBCorrectorImpl (  std::shared_ptr<FactorizedJetCorrectorCalculator const> corrector,
			edm::RefProd<std::vector<reco::SoftLeptonTagInfo>> const& bTagInfoMuon,
			edm::RefProd<std::vector<reco::SoftLeptonTagInfo>> const& bTagInfoElec,
			bool addMuonToJet);

  //
  // member functions
  //
public:
  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const override;
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const override;
  /// apply correction using all event information
  virtual double correction (const reco::Jet& fJet,
			     const edm::RefToBase<reco::Jet>& refToRawJet) const override;
  
  //----- if correction needs a jet reference -------------
  virtual bool refRequired () const override {return true;} 
  
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
  //edm::InputTag           srcBTagInfoElec_;
  //edm::InputTag           srcBTagInfoMuon_;
  std::shared_ptr<FactorizedJetCorrectorCalculator const> corrector_;
  edm::RefProd<std::vector<reco::SoftLeptonTagInfo>> bTagInfoMuon_;
  edm::RefProd<std::vector<reco::SoftLeptonTagInfo>> bTagInfoElec_;
  bool                    addMuonToJet_;

};

#endif
