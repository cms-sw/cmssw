#include "HLTrigger/special/interface/HLTHcalIsolatedBunchFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <cmath>

HLTHcalIsolatedBunchFilter::HLTHcalIsolatedBunchFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {

  hltJetSeedLabel_     = iConfig.getParameter<edm::InputTag> ("L1JetInputTag");
  hltTauSeedLabel_     = iConfig.getParameter<edm::InputTag> ("L1TauInputTag");
  minEta_              = iConfig.getParameter<double> ("MinEta");
  maxEta_              = iConfig.getParameter<double> ("MaxEta");
  minPhi_              = iConfig.getParameter<double> ("MinPhi");
  maxPhi_              = iConfig.getParameter<double> ("MaxPhi");
  minPt_               = iConfig.getParameter<double> ("MinPt"); 

  hltJetToken_ = consumes<l1t::JetBxCollection>(hltJetSeedLabel_);
  hltTauToken_ = consumes<l1t::TauBxCollection>(hltTauSeedLabel_);
  std::cout << "Input Parameters:: minPt = " << minPt_ << " Eta " 
	    << minEta_ << ":" << maxEta_ << " Phi " << minPhi_ 
	    << ":" << maxPhi_ << " GT Seed " << hltJetSeedLabel_
	    << " and " << hltTauSeedLabel_ << std::endl;
}

HLTHcalIsolatedBunchFilter::~HLTHcalIsolatedBunchFilter()= default;


void
HLTHcalIsolatedBunchFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("L1JetInputTag",edm::InputTag("hltCaloStage2Digis:Jet"));
  desc.add<edm::InputTag>("L1TauInputTag",edm::InputTag("hltCaloStage2Digis:Tau"));
  desc.add<double>("MinEta",1.305);
  desc.add<double>("MaxEta",3.000);
  desc.add<double>("MinPhi",5.4105);
  desc.add<double>("MaxPhi",5.5796);
  desc.add<double>("MinPt", 20.0);
  descriptions.add("hltHcalIsolatedBunchFilter",desc);
}

bool HLTHcalIsolatedBunchFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {

  // the filter object
  if (saveTags()) {
    // jet 
    filterproduct.addCollectionTag(hltJetSeedLabel_);
    // tau 
    filterproduct.addCollectionTag(hltTauSeedLabel_);
  }

  bool accept(false);
  const int ibx(0);

  if (!accept) {
    edm::Handle<l1t::TauBxCollection> taus;
    iEvent.getByToken(hltTauToken_, taus);
    if (!taus.isValid()) { 
      edm::LogWarning("HLTHcalIsolatedBunch")
	<< "L1TauBxCollection with input tag " << hltTauSeedLabel_
	<< "\nrequested in configuration, but not found in the event."
	<< std::endl;
    } else {
      std::cout << "L1TauBxCollection has " << taus->size() << " BX's and "
		<< taus->size(ibx) << " candidates in BX " << ibx << std::endl;
      for (l1t::TauBxCollection::const_iterator p=taus->begin(ibx);
	   p!=taus->end(ibx); ++p) {
	if (p->pt() > minPt_) {
	  double eta = p->eta();
	  if (eta > minEta_ && eta < maxEta_) {
	    double phi = p->phi();
	    if (phi < 0) phi += 2*M_PI;
	    if (phi > minPhi_ && phi < maxPhi_) {
	      accept = true;
	      break;
	    }
	  }
	}
      }
    }
  }
      
  if (!accept) {
    edm::Handle<l1t::JetBxCollection> jets;
    iEvent.getByToken(hltJetToken_, jets);
    if (!jets.isValid()) { 
      edm::LogWarning("HLTHcalIsolatedBunch")
	<< "L1JetBxCollection with input tag " << hltJetSeedLabel_
	<< "\nrequested in configuration, but not found in the event."
	<< "\nNo jets added to filterproduct." << std::endl;
    } else {
      std::cout << "L1JetBxCollection has " << jets->size() << " BX's and "
		<< jets->size(ibx) << " candidates in BX " << ibx << std::endl;
      for (l1t::JetBxCollection::const_iterator p=jets->begin(ibx);
	   p!=jets->end(ibx); ++p) {
	if (p->pt() > minPt_) {
	  double eta = p->eta();
	  if (eta > minEta_ && eta < maxEta_) {
	    double phi = p->phi();
	    if (phi < 0) phi += 2*M_PI;
	    if (phi > minPhi_ && phi < maxPhi_) {
	      accept = true;
	      break;
	    }
	  }
	}
      }
    }
  }

  std::cout << "Selection flag " << accept << std::endl;
  return accept;
}
