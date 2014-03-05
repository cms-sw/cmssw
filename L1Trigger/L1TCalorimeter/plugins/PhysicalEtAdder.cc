#include "L1Trigger/L1TCalorimeter/plugins/PhysicalEtAdder.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include <vector>

//#include <stdio.h>

l1t::PhysicalEtAdder::PhysicalEtAdder(const edm::ParameterSet& ps) {

  produces<l1t::L1TEGammaCollection>();
  produces<l1t::L1TTauCollection>();
  produces<l1t::L1TJetCollection>();
  produces<l1t::L1TEtSumCollection>();

  EGammaToken_ = consumes<L1TEGammaCollection>(iConfig.getParameter<InputTag>("InputCollection"));
  TauToken_ = consumes<L1TTauCollection>(iConfig.getParameter<InputTag>("InputCollection"));
  JetToken_ = consumes<L1TJetCollection>(iConfig.getParameter<InputTag>("InputCollection"));
  EtSumToken_ = consumes<L1TEtSumCollection>(iConfig.getParameter<InputTag>("InputCollection"));
}

l1t::PhysicalEtAdder::~PhysicalEtAdder() {

}

// ------------ method called to produce the data  ------------
void
l1t::PhysicalEtAdder::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // store new collections which include physical quantities
  std::auto_ptr<l1t::L1TEGammaCollection> new_egammas (new l1t::L1TEGammaCollection);
  std::auto_ptr<l1t::L1TTauCollection> new_taus (new l1t::L1TTauCollection);
  std::auto_ptr<l1t::L1TJetCollection> new_jets (new l1t::L1TJetCollection);
  std::auto_ptr<l1t::L1TEtSumCollection> new_etsums (new l1t::L1TEtSumCollection);

  edm::Handle<l1t::L1TEGammaCollection> old_egammas;
  edm::Handle<l1t::L1TTauCollection> old_taus;
  edm::Handle<l1t::L1TJetCollection> old_jets;
  edm::Handle<l1t::L1TEtSumCollection> old_etsums;

  iEvent.getByToken(EGammaToken_, old_egammas);
  iEvent.getByToken(TauToken_, old_taus);
  iEvent.getByToken(JetToken_, old_jets);
  iEvent.getByToken(EtSumToken_, old_etsums);





  iEvent.put(new_egammas);
  iEvent.put(new_taus);
  iEvent.put(new_jets);
  iEvent.put(new_etsums);

}

// ------------ method called once each job just before starting event loop  ------------
void
l1t::PhysicalEtAdder::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
l1t::PhysicalEtAdder::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
  void
  l1t::PhysicalEtAdder::beginRun(edm::Run const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when ending the processing of a run  ------------
/*
  void
  l1t::PhysicalEtAdder::endRun(edm::Run const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
  void
  l1t::PhysicalEtAdder::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup cons
  t&)
  {
  }
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
  void
  l1t::PhysicalEtAdder::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&
  )
  {
  }
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
l1t::PhysicalEtAdder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::PhysicalEtAdder);
