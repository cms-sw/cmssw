#include "L1Trigger/L1TCalorimeter/plugins/L1TPhysicalEtAdder.h"
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

using namespace l1t;

L1TPhysicalEtAdder::L1TPhysicalEtAdder(const edm::ParameterSet& ps) {

  produces<EGammaBxCollection>();
  produces<TauBxCollection>("rlxTaus");
  produces<TauBxCollection>("isoTaus");
  produces<JetBxCollection>();
  produces<JetBxCollection>("preGtJets");
  produces<EtSumBxCollection>();
  produces<CaloSpareBxCollection>("HFRingSums");
  produces<CaloSpareBxCollection>("HFBitCounts");

  EGammaToken_ = consumes<EGammaBxCollection>(ps.getParameter<edm::InputTag>("InputCollection"));
  RlxTauToken_ = consumes<TauBxCollection>(ps.getParameter<edm::InputTag>("InputRlxTauCollection"));
  IsoTauToken_ = consumes<TauBxCollection>(ps.getParameter<edm::InputTag>("InputIsoTauCollection"));
  JetToken_ = consumes<JetBxCollection>(ps.getParameter<edm::InputTag>("InputCollection"));
  preGtJetToken_ = consumes<JetBxCollection>(ps.getParameter<edm::InputTag>("InputPreGtJetCollection"));
  EtSumToken_ = consumes<EtSumBxCollection>(ps.getParameter<edm::InputTag>("InputCollection"));
  HfSumsToken_ = consumes<CaloSpareBxCollection>(ps.getParameter<edm::InputTag>("InputHFSumsCollection"));
  HfCountsToken_ = consumes<CaloSpareBxCollection>(ps.getParameter<edm::InputTag>("InputHFCountsCollection"));
}

L1TPhysicalEtAdder::~L1TPhysicalEtAdder() {

}

// ------------ method called to produce the data  ------------
void
L1TPhysicalEtAdder::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // store new collections which include physical quantities
  std::auto_ptr<EGammaBxCollection> new_egammas (new EGammaBxCollection);
  std::auto_ptr<TauBxCollection> new_rlxtaus (new TauBxCollection);
  std::auto_ptr<TauBxCollection> new_isotaus (new TauBxCollection);
  std::auto_ptr<JetBxCollection> new_jets (new JetBxCollection);
  std::auto_ptr<JetBxCollection> new_preGtJets (new JetBxCollection);
  std::auto_ptr<EtSumBxCollection> new_etsums (new EtSumBxCollection);
  std::auto_ptr<CaloSpareBxCollection> new_hfsums (new CaloSpareBxCollection);
  std::auto_ptr<CaloSpareBxCollection> new_hfcounts (new CaloSpareBxCollection);

  edm::Handle<EGammaBxCollection> old_egammas;
  edm::Handle<TauBxCollection> old_rlxtaus;
  edm::Handle<TauBxCollection> old_isotaus;
  edm::Handle<JetBxCollection> old_jets;
  edm::Handle<JetBxCollection> old_preGtJets;
  edm::Handle<EtSumBxCollection> old_etsums;
  edm::Handle<CaloSpareBxCollection> old_hfsums;
  edm::Handle<CaloSpareBxCollection> old_hfcounts;

  iEvent.getByToken(EGammaToken_, old_egammas);
  iEvent.getByToken(RlxTauToken_, old_rlxtaus);
  iEvent.getByToken(IsoTauToken_, old_isotaus);
  iEvent.getByToken(JetToken_, old_jets);
  iEvent.getByToken(preGtJetToken_, old_preGtJets);
  iEvent.getByToken(EtSumToken_, old_etsums);
  iEvent.getByToken(HfSumsToken_, old_hfsums);
  iEvent.getByToken(HfCountsToken_, old_hfcounts);

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
  new_rlxtaus->setBXRange(firstBX, lastBX);
  new_isotaus->setBXRange(firstBX, lastBX);
  new_jets->setBXRange(firstBX, lastBX);
  new_preGtJets->setBXRange(firstBX, lastBX);
  new_etsums->setBXRange(firstBX, lastBX);
  new_hfsums->setBXRange(firstBX, lastBX);
  new_hfcounts->setBXRange(firstBX, lastBX);

  for(int bx = firstBX; bx <= lastBX; ++bx)
  {
    for(EGammaBxCollection::const_iterator itEGamma = old_egammas->begin(bx);
	itEGamma != old_egammas->end(bx); ++itEGamma)
    {
      //const double pt = itEGamma->hwPt() * emScale->linearLsb();
      const double et = emScale->et( itEGamma->hwPt() );
      const double eta = getPhysicalEta(itEGamma->hwEta());
      const double phi = getPhysicalPhi(itEGamma->hwPhi());
      math::PtEtaPhiMLorentzVector p4(et, eta, phi, 0);

      EGamma eg(*&p4, itEGamma->hwPt(),
		     itEGamma->hwEta(), itEGamma->hwPhi(),
		     itEGamma->hwQual(), itEGamma->hwIso());
      new_egammas->push_back(bx, *&eg);


    }

    for(TauBxCollection::const_iterator itTau = old_rlxtaus->begin(bx);
	itTau != old_rlxtaus->end(bx); ++itTau)
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

      Tau tau(*&p4, itTau->hwPt(),
		   itTau->hwEta(), itTau->hwPhi(),
		   itTau->hwQual(), itTau->hwIso());
      new_rlxtaus->push_back(bx, *&tau);

    }

    for(TauBxCollection::const_iterator itTau = old_isotaus->begin(bx);
	itTau != old_isotaus->end(bx); ++itTau)
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

      Tau tau(*&p4, itTau->hwPt(),
		   itTau->hwEta(), itTau->hwPhi(),
		   itTau->hwQual(), itTau->hwIso());
      new_isotaus->push_back(bx, *&tau);

    }

    for(JetBxCollection::const_iterator itJet = old_jets->begin(bx);
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

      Jet jet(*&p4, itJet->hwPt(),
		   itJet->hwEta(), itJet->hwPhi(),
		   itJet->hwQual());
      new_jets->push_back(bx, *&jet);

    }

    for(JetBxCollection::const_iterator itJet = old_preGtJets->begin(bx);
	itJet != old_preGtJets->end(bx); ++itJet)
    {
      // use the full-circle conversion to match l1extra, accounts for linearLsb and max value automatically
      //const uint16_t rankPt = jetScale->rank((uint16_t)itJet->hwPt());
      //const double et = jetScale->et( rankPt ) ;

      // or use the emScale to get finer-grained et
      const double et = itJet->hwPt() * emScale->linearLsb();

      // we are now already in the rankPt
      //const double et = jetScale->et( itJet->hwPt() );

      const bool forward = ((itJet->hwQual() & 0x2) != 0);
      const double eta = getPhysicalEta(itJet->hwEta(), forward);
      const double phi = getPhysicalPhi(itJet->hwPhi());
      math::PtEtaPhiMLorentzVector p4(et, eta, phi, 0);

      Jet jet(*&p4, itJet->hwPt(),
		   itJet->hwEta(), itJet->hwPhi(),
		   itJet->hwQual());
      new_preGtJets->push_back(bx, *&jet);

    }


    for(EtSumBxCollection::const_iterator itEtSum = old_etsums->begin(bx);
	itEtSum != old_etsums->end(bx); ++itEtSum)
    {
      double et = itEtSum->hwPt() * emScale->linearLsb();
      //hack while we figure out the right scales
      //double et = emScale->et( itEtSum->hwPt() );
      const EtSum::EtSumType sumType = itEtSum->getType();

      const double eta = getPhysicalEta(itEtSum->hwEta());
      double phi = getPhysicalPhi(itEtSum->hwPhi());
      if(sumType == EtSum::EtSumType::kMissingHt){
	et = htMissScale->et( itEtSum->hwPt() );
	double regionPhiWidth=2. * 3.1415927 / L1CaloRegionDetId::N_PHI;
	phi=phi+(regionPhiWidth/2.); // add the region half-width to match L1Extra MHT phi
      }


      math::PtEtaPhiMLorentzVector p4(et, eta, phi, 0);

      EtSum eg(*&p4, sumType, itEtSum->hwPt(),
		    itEtSum->hwEta(), itEtSum->hwPhi(),
		    itEtSum->hwQual());
      new_etsums->push_back(bx, *&eg);


    }

    for(CaloSpareBxCollection::const_iterator itCaloSpare = old_hfsums->begin(bx);
	itCaloSpare != old_hfsums->end(bx); ++itCaloSpare)
    {
      //just pass through for now
      //a different scale is needed depending on the type
      new_hfsums->push_back(bx, *itCaloSpare);
    }

    for(CaloSpareBxCollection::const_iterator itCaloSpare = old_hfcounts->begin(bx);
	itCaloSpare != old_hfcounts->end(bx); ++itCaloSpare)
    {
      //just pass through for now
      //a different scale is needed depending on the type
      new_hfcounts->push_back(bx, *itCaloSpare);
    }

  }

  iEvent.put(new_egammas);
  iEvent.put(new_rlxtaus,"rlxTaus");
  iEvent.put(new_isotaus,"isoTaus");
  iEvent.put(new_jets);
  iEvent.put(new_preGtJets,"preGtJets");
  iEvent.put(new_etsums);
  iEvent.put(new_hfsums,"HFRingSums");
  iEvent.put(new_hfcounts,"HFBitCounts");
}

// ------------ method called once each job just before starting event loop  ------------
void
L1TPhysicalEtAdder::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TPhysicalEtAdder::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
  void
  L1TPhysicalEtAdder::beginRun(edm::Run const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when ending the processing of a run  ------------
/*
  void
  L1TPhysicalEtAdder::endRun(edm::Run const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
  void
  L1TPhysicalEtAdder::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup cons
  t&)
  {
  }
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
  void
  L1TPhysicalEtAdder::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&
  )
  {
  }
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TPhysicalEtAdder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TPhysicalEtAdder);

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
      << "Bad eta index passed to L1TPhysicalEtAdder::getRegionEta, " << gtEta << std::endl;

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
