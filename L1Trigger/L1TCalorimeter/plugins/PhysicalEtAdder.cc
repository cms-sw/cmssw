#include "L1Trigger/L1TCalorimeter/plugins/PhysicalEtAdder.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include <vector>

#include "math.h"

double getPhysicalEta(int etaIndex);
double getPhysicalPhi(int phiIndex);

l1t::PhysicalEtAdder::PhysicalEtAdder(const edm::ParameterSet& ps) {

  produces<L1TEGammaCollection>();
  produces<L1TTauCollection>();
  produces<L1TJetCollection>();
  produces<L1TEtSumCollection>();

  EGammaToken_ = consumes<L1TEGammaCollection>(ps.getParameter<edm::InputTag>("InputCollection"));
  TauToken_ = consumes<L1TTauCollection>(ps.getParameter<edm::InputTag>("InputCollection"));
  JetToken_ = consumes<L1TJetCollection>(ps.getParameter<edm::InputTag>("InputCollection"));
  EtSumToken_ = consumes<L1TEtSumCollection>(ps.getParameter<edm::InputTag>("InputCollection"));
}

l1t::PhysicalEtAdder::~PhysicalEtAdder() {

}

// ------------ method called to produce the data  ------------
void
l1t::PhysicalEtAdder::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // store new collections which include physical quantities
  std::auto_ptr<L1TEGammaCollection> new_egammas (new L1TEGammaCollection);
  std::auto_ptr<L1TTauCollection> new_taus (new L1TTauCollection);
  std::auto_ptr<L1TJetCollection> new_jets (new L1TJetCollection);
  std::auto_ptr<L1TEtSumCollection> new_etsums (new L1TEtSumCollection);

  edm::Handle<L1TEGammaCollection> old_egammas;
  edm::Handle<L1TTauCollection> old_taus;
  edm::Handle<L1TJetCollection> old_jets;
  edm::Handle<L1TEtSumCollection> old_etsums;

  iEvent.getByToken(EGammaToken_, old_egammas);
  iEvent.getByToken(TauToken_, old_taus);
  iEvent.getByToken(JetToken_, old_jets);
  iEvent.getByToken(EtSumToken_, old_etsums);


  //get the proper scales for conversion to physical et
  edm::ESHandle< L1CaloEtScale > emScale ;
  iSetup.get< L1EmEtScaleRcd >().get( emScale ) ;

  edm::ESHandle< L1CaloEtScale > jetScale ;
  iSetup.get< L1JetEtScaleRcd >().get( jetScale ) ;

  edm::ESHandle< L1CaloEtScale > hwForJetScale ;
  iSetup.get< L1JetEtScaleRcd >().get( hwForJetScale ) ;

  edm::ESHandle< L1CaloEtScale > htMissScale ;
  std::vector< bool > htMissMatched ;
  iSetup.get< L1HtMissScaleRcd >().get( htMissScale ) ;

  int firstBX = old_egammas->getFirstBX();
  int lastBX = old_egammas->getLastBX();

  new_egammas->setBXRange(firstBX, lastBX);
  new_taus->setBXRange(firstBX, lastBX);
  new_jets->setBXRange(firstBX, lastBX);
  new_taus->setBXRange(firstBX, lastBX);

  for(int bx = firstBX; bx <= lastBX; ++bx)
  {
    for(L1TEGammaCollection::const_iterator itEGamma = old_egammas->begin(bx);
	itEGamma != old_egammas->end(bx); ++itEGamma)
    {
      const double pt = itEGamma->hwPt() * emScale->linearLsb();
      const double eta = getPhysicalEta(itEGamma->hwEta());
      const double phi = getPhysicalPhi(itEGamma->hwPhi());
      //const double eta = itEGamma->hwEta();
      //const double phi = itEGamma->hwPhi();

      const double px = pt*cos(phi);
      const double py = pt*sin(phi);
      const double pz = pt*sinh(eta);
      const double e = sqrt(px*px + py*py + pz*pz);
      math::XYZTLorentzVector *p4 = new math::XYZTLorentzVector(px, py, pz, e);

      l1t::EGamma *eg = new l1t::EGamma(*p4, itEGamma->hwPt(),
				       itEGamma->hwEta(), itEGamma->hwPhi(),
				       itEGamma->hwQual(), itEGamma->hwIso());
      new_egammas->push_back(bx, *eg);


    }

    for(L1TTauCollection::const_iterator itTau = old_taus->begin(bx);
	itTau != old_taus->end(bx); ++itTau)
    {
    }

    for(L1TJetCollection::const_iterator itJet = old_jets->begin(bx);
	itJet != old_jets->end(bx); ++itJet)
    {
      const double pt = itJet->hwPt() * jetScale->linearLsb();
      const double eta = getPhysicalEta(itJet->hwEta());
      const double phi = getPhysicalPhi(itJet->hwPhi());
      //const double eta = itJet->hwEta();
      //const double phi = itJet->hwPhi();

      const double px = pt*cos(phi);
      const double py = pt*sin(phi);
      const double pz = pt*sinh(eta);
      const double e = sqrt(px*px + py*py + pz*pz);
      math::XYZTLorentzVector *p4 = new math::XYZTLorentzVector(px, py, pz, e);

      l1t::Jet *jet = new l1t::Jet(*p4, itJet->hwPt(),
				   itJet->hwEta(), itJet->hwPhi(),
				   itJet->hwQual());
      new_jets->push_back(bx, *jet);

    }

    for(L1TEtSumCollection::const_iterator itEtSum = old_etsums->begin(bx);
	itEtSum != old_etsums->end(bx); ++itEtSum)
    {
    }
  }

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

double getPhysicalEta(int etaIndex)
{
  const double rgnEtaValues[11] = {
     0.174, // HB and inner HE bins are 0.348 wide
     0.522,
     0.870,
     1.218,
     1.566,
     1.956, // Last two HE bins are 0.432 and 0.828 wide
     2.586,
     3.250, // HF bins are 0.5 wide
     3.750,
     4.250,
     4.750
  };
  if(etaIndex < 11) {
    return -rgnEtaValues[-(etaIndex - 10)]; // 0-10 are negative eta values
  }
  else if (etaIndex < 22) {
    return rgnEtaValues[etaIndex - 11]; // 11-21 are positive eta values
  }
  return -9;
}

double getPhysicalPhi(int phiIndex)
{
  if (phiIndex < 10)
    return 2. * M_PI * phiIndex / 18.;
  if (phiIndex < 18)
    return -M_PI + 2. * M_PI * (phiIndex - 9) / 18.;
  return -9;
}
