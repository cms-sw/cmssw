////////////////////////////////////////////////////////////////////////////////
//
// L1FastjetCorrector
// ------------------
//
//            08/09/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "JetMETCorrections/Algorithms/interface/L1FastjetCorrectorImpl.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
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
							 edm::ConsumesCollector fCollector):
  JetCorrectorImplMakerBase(fConfig),
  rhoToken_(fCollector.consumes<double>(fConfig.getParameter<edm::InputTag>("srcRho")))
{
}

std::unique_ptr<reco::JetCorrectorImpl>
L1FastjetCorrectorImplMaker::make(edm::Event const& fEvent, edm::EventSetup const& fSetup) {
  auto corrector = getCalculator(fSetup, [](const std::string& level) {
      if(level != "L1FastJet") {
      throw cms::Exception("L1FastjetCorrector")<<" correction level: "<<level<<" is not L1FastJet";
      }
    });

  edm::Handle<double> hRho;
  fEvent.getByToken(rhoToken_,hRho);
  return std::unique_ptr<L1FastjetCorrectorImpl>(new L1FastjetCorrectorImpl(corrector, *hRho) );
}

void 
L1FastjetCorrectorImplMaker::fillDescriptions(edm::ConfigurationDescriptions& iDescriptions)
{
  edm::ParameterSetDescription desc;
  addToDescription(desc);
  desc.add<edm::InputTag>("srcRho");
  iDescriptions.addDefault(desc);
}


//______________________________________________________________________________
////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
double L1FastjetCorrectorImpl::correction (const LorentzVector& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(LorentzVector), event required!";
  return 1.0;
}


//______________________________________________________________________________
double L1FastjetCorrectorImpl::correction (const reco::Jet& fJet) const
{
  FactorizedJetCorrectorCalculator::VariableValues values;
  values.setJetEta(fJet.eta());
  values.setJetPt(fJet.pt());
  values.setJetE(fJet.energy());
  values.setJetA(fJet.jetArea());
  values.setRho(rho_);
  return corrector_->getCorrection(values);  
}




