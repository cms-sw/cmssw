#include "L1Trigger/L1TCalorimeter/plugins/PhysicalEtAdder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include <vector>

#include <stdio.h>

double getPhysicalEta(int etaIndex, bool forward = false);
double getPhysicalPhi(int phiIndex);

l1t::PhysicalEtAdder::PhysicalEtAdder(const edm::ParameterSet& ps) {

  produces<l1t::EGammaBxCollection>();
  produces<l1t::TauBxCollection>();
  produces<l1t::JetBxCollection>();
  produces<l1t::EtSumBxCollection>();

  EGammaToken_ = consumes<l1t::EGammaBxCollection>(ps.getParameter<edm::InputTag>("InputCollection"));
  TauToken_ = consumes<l1t::TauBxCollection>(ps.getParameter<edm::InputTag>("InputCollection"));
  JetToken_ = consumes<l1t::JetBxCollection>(ps.getParameter<edm::InputTag>("InputCollection"));
  EtSumToken_ = consumes<l1t::EtSumBxCollection>(ps.getParameter<edm::InputTag>("InputCollection"));
}

l1t::PhysicalEtAdder::~PhysicalEtAdder() {

}

// ------------ method called to produce the data  ------------
void
l1t::PhysicalEtAdder::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // store new collections which include physical quantities
  std::auto_ptr<l1t::EGammaBxCollection> new_egammas (new l1t::EGammaBxCollection);
  std::auto_ptr<l1t::TauBxCollection> new_taus (new l1t::TauBxCollection);
  std::auto_ptr<l1t::JetBxCollection> new_jets (new l1t::JetBxCollection);
  std::auto_ptr<l1t::EtSumBxCollection> new_etsums (new l1t::EtSumBxCollection);

  edm::Handle<l1t::EGammaBxCollection> old_egammas;
  edm::Handle<l1t::TauBxCollection> old_taus;
  edm::Handle<l1t::JetBxCollection> old_jets;
  edm::Handle<l1t::EtSumBxCollection> old_etsums;

  iEvent.getByToken(EGammaToken_, old_egammas);
  iEvent.getByToken(TauToken_, old_taus);
  iEvent.getByToken(JetToken_, old_jets);
  iEvent.getByToken(EtSumToken_, old_etsums);


  //get the proper scales for conversion to physical et
  edm::ESHandle< L1CaloEtScale > emScale ;
  iSetup.get< L1EmEtScaleRcd >().get( emScale ) ;

  edm::ESHandle< L1CaloEtScale > jetScale ;
  iSetup.get< L1JetEtScaleRcd >().get( jetScale ) ;

  edm::ESHandle< L1CaloEtScale > htMissScale ;
  iSetup.get< L1HtMissScaleRcd >().get( htMissScale ) ;

  int firstBX = old_egammas->getFirstBX();
  int lastBX = old_egammas->getLastBX();

  new_egammas->setBXRange(firstBX, lastBX);
  new_taus->setBXRange(firstBX, lastBX);
  new_jets->setBXRange(firstBX, lastBX);
  new_taus->setBXRange(firstBX, lastBX);

  for(int bx = firstBX; bx <= lastBX; ++bx)
  {
    for(l1t::EGammaBxCollection::const_iterator itEGamma = old_egammas->begin(bx);
	itEGamma != old_egammas->end(bx); ++itEGamma)
    {
      //const double pt = itEGamma->hwPt() * emScale->linearLsb();
      const double et = emScale->et( itEGamma->hwPt() );
      const double eta = getPhysicalEta(itEGamma->hwEta());
      const double phi = getPhysicalPhi(itEGamma->hwPhi());
      math::PtEtaPhiMLorentzVector p4(et, eta, phi, 0);

      l1t::EGamma eg(*&p4, itEGamma->hwPt(),
		     itEGamma->hwEta(), itEGamma->hwPhi(),
		     itEGamma->hwQual(), itEGamma->hwIso());
      new_egammas->push_back(bx, *&eg);


    }

    for(l1t::TauBxCollection::const_iterator itTau = old_taus->begin(bx);
	itTau != old_taus->end(bx); ++itTau)
    {
      // use the full-circle conversion to match l1extra, accounts for linearLsb and max value automatically
      //const uint16_t rankPt = jetScale->rank((uint16_t)itTau->hwPt());
      //const double et = jetScale->et( rankPt ) ;

      // or use the emScale to get finer-grained et
      //const double et = itTau->hwPt() * emScale->linearLsb();

      // we are now already in the rankPt
      const double et = jetScale->et( itTau->hwPt() );

      const double eta = getPhysicalEta(itTau->hwEta());
      const double phi = getPhysicalPhi(itTau->hwPhi());
      math::PtEtaPhiMLorentzVector p4(et, eta, phi, 0);

      l1t::Tau tau(*&p4, itTau->hwPt(),
		   itTau->hwEta(), itTau->hwPhi(),
		   itTau->hwQual(), itTau->hwIso());
      new_taus->push_back(bx, *&tau);

    }

    for(l1t::JetBxCollection::const_iterator itJet = old_jets->begin(bx);
	itJet != old_jets->end(bx); ++itJet)
    {
      // use the full-circle conversion to match l1extra, accounts for linearLsb and max value automatically
      //const uint16_t rankPt = jetScale->rank((uint16_t)itJet->hwPt());
      //const double et = jetScale->et( rankPt ) ;

      // or use the emScale to get finer-grained et
      //const double et = itJet->hwPt() * emScale->linearLsb();

      // we are now already in the rankPt
      const double et = jetScale->et( itJet->hwPt() );

      const bool forward = ((itJet->hwQual() & 0x2) != 0);
      const double eta = getPhysicalEta(itJet->hwEta(), forward);
      const double phi = getPhysicalPhi(itJet->hwPhi());
      math::PtEtaPhiMLorentzVector p4(et, eta, phi, 0);

      l1t::Jet jet(*&p4, itJet->hwPt(),
		   itJet->hwEta(), itJet->hwPhi(),
		   itJet->hwQual());
      new_jets->push_back(bx, *&jet);

    }

    for(l1t::EtSumBxCollection::const_iterator itEtSum = old_etsums->begin(bx);
	itEtSum != old_etsums->end(bx); ++itEtSum)
    {
      const double et = itEtSum->hwPt() * emScale->linearLsb();
      //hack while we figure out the right scales
      //double et = emScale->et( itEtSum->hwPt() );
      const l1t::EtSum::EtSumType sumType = itEtSum->getType();
      //if(sumType == EtSum::EtSumType::kMissingHt)
      //et = htMissScale->et( itEtSum->hwPt() );

      const double eta = getPhysicalEta(itEtSum->hwEta());
      const double phi = getPhysicalPhi(itEtSum->hwPhi());

      math::PtEtaPhiMLorentzVector p4(et, eta, phi, 0);

      l1t::EtSum eg(*&p4, sumType, itEtSum->hwPt(),
		    itEtSum->hwEta(), itEtSum->hwPhi(),
		    itEtSum->hwQual());
      new_etsums->push_back(bx, *&eg);
      

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

int getRegionEta(int gtEta, bool forward)
{
  // backwards conversion is
  // unsigned rctEta = (iEta<11 ? 10-iEta : iEta-11);
  // return (((rctEta % 7) & 0x7) | (iEta<11 ? 0x8 : 0));
  int centralGtEta[] = {11, 12, 13, 14, 15, 16, 17, -100, 10, 9, 8, 7, 6, 5, 4};
  int forwardGtEta[] = {18, 19, 20, 21, -100, -100, -100, -100, 3, 2, 1, 0};

  //printf("%i, %i\n",gtEta,forward);

  int regionEta;

  if(!forward)
  {
    regionEta = centralGtEta[gtEta];
  } else
    regionEta = forwardGtEta[gtEta];

  if(regionEta == -100)
    edm::LogError("EtaIndexError")
      << "Bad eta index passed to PhysicalEtAdder::getRegionEta, " << gtEta << std::endl;

  return regionEta;
}

// adapted these from the UCT2015 codebase.
double getPhysicalEta(int gtEta, bool forward)
{
  int etaIndex = getRegionEta(gtEta, forward);

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
