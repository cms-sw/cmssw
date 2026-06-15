////////////////////////////////////////////////////////////////////////////////
//
// L1FastjetCorrector
// ------------------
//
//            08/09/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include <memory>

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "JetMETCorrections/Algorithms/interface/L1FastjetCorrectorImpl.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/Common/interface/Handle.h"

using namespace std;

L1FastjetCorrectorImplMaker::L1FastjetCorrectorImplMaker(edm::ParameterSet const& fConfig,
                                                         edm::ConsumesCollector fCollector)
    : JetCorrectorImplMakerBase(fConfig, fCollector),
      rhoToken_(fCollector.consumes<double>(fConfig.getParameter<edm::InputTag>("srcRho"))),
      skipMissingProduct_(fConfig.getParameter<bool>("skipMissingProduct")) {}

std::unique_ptr<reco::JetCorrectorImpl> L1FastjetCorrectorImplMaker::make(edm::Event const& fEvent,
                                                                          edm::EventSetup const& fSetup) {
  auto corrector = getCalculator(fSetup, [](const std::string& level) {
    if (level != "L1FastJet") {
      throw cms::Exception("L1FastjetCorrector") << " correction level: " << level << " is not L1FastJet";
    }
  });

  edm::Handle<double> hRho;
  fEvent.getByToken(rhoToken_, hRho);

  if (hRho.isValid()) {
    return std::make_unique<L1FastjetCorrectorImpl>(corrector, *hRho);
  }
  // Handle missing product
  else if (skipMissingProduct_) {
    edm::LogWarning("L1FastjetCorrector")
        << "Rho product is missing, but skipMissingProduct is set to true. Returning corrector with rho = 0.";
    return std::make_unique<L1FastjetCorrectorImpl>(corrector, 0.0);
  } else {
    throw cms::Exception("L1FastjetCorrector") << "Rho product is missing and skipMissingProduct is set to false.";
  }
}

void L1FastjetCorrectorImplMaker::fillDescriptions(edm::ConfigurationDescriptions& iDescriptions) {
  edm::ParameterSetDescription desc;
  addToDescription(desc);
  desc.add<edm::InputTag>("srcRho", edm::InputTag(""));
  desc.add<bool>("skipMissingProduct", false);
  iDescriptions.addWithDefaultLabel(desc);
}

//______________________________________________________________________________
////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
double L1FastjetCorrectorImpl::correction(const LorentzVector& fJet) const {
  throw cms::Exception("EventRequired") << "Wrong interface correction(LorentzVector), event required!";
  return 1.0;
}

//______________________________________________________________________________
double L1FastjetCorrectorImpl::correction(const reco::Jet& fJet) const {
  FactorizedJetCorrectorCalculator::VariableValues values;
  values.setJetEta(fJet.eta());
  values.setJetPt(fJet.pt());
  values.setJetE(fJet.energy());
  values.setJetA(fJet.jetArea());
  values.setRho(rho_);
  values.setJetPhi(fJet.phi());
  return corrector_->getCorrection(values);
}
