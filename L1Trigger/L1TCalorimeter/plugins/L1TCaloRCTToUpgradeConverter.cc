#include "L1Trigger/L1TCalorimeter/plugins/L1TCaloRCTToUpgradeConverter.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include <vector>

//#include <stdio.h>

l1t::L1TCaloRCTToUpgradeConverter::L1TCaloRCTToUpgradeConverter(const edm::ParameterSet& ps) {

  produces<l1t::CaloRegionBxCollection>();
  produces<l1t::CaloEmCandBxCollection>();

  rgnToken_ = consumes<L1CaloRegionCollection>(ps.getParameter<edm::InputTag>("regionTag"));
  emToken_ = consumes<L1CaloEmCollection>(ps.getParameter<edm::InputTag>("emTag"));

  // firstBx_ = -ps.getParameter<unsigned>("preSamples");
  // lastBx_  =  ps.getParameter<unsigned>("postSamples");

}

l1t::L1TCaloRCTToUpgradeConverter::~L1TCaloRCTToUpgradeConverter() {

}

// ------------ method called to produce the data  ------------
void
l1t::L1TCaloRCTToUpgradeConverter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // check status of RCT conditions & renew if needed


  // store new formats
  std::auto_ptr<BXVector<l1t::CaloEmCand> > emcands (new l1t::CaloEmCandBxCollection);
  std::auto_ptr<BXVector<l1t::CaloRegion> > regions (new l1t::CaloRegionBxCollection);

  // get old formats
  edm::Handle<L1CaloEmCollection> ems;
  edm::Handle<L1CaloRegionCollection> rgns;

  iEvent.getByToken(emToken_, ems);
  iEvent.getByToken(rgnToken_, rgns);

  // get the firstBx_ and lastBx_ from the input datatypes (assume bx for em same as rgn)
  int firstBx = 0;
  int lastBx = 0;
  for (std::vector<L1CaloEmCand>::const_iterator em=ems->begin(); em!=ems->end(); ++em) {
    int bx = em->bx();
    if (bx < firstBx) firstBx = bx;
    if (bx > lastBx) lastBx = bx;
  }

  emcands->setBXRange(firstBx, lastBx);
  regions->setBXRange(firstBx, lastBx);

  // loop over EM
  for (std::vector<L1CaloEmCand>::const_iterator em=ems->begin(); em!=ems->end(); ++em) {

    // get physical units
    // double pt = 0.;
    // double eta = 0.;
    // double phi = 0.;
    //math::PtEtaPhiMLorentzVector p4( pt+1.e-6, eta, phi, 0. );
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > p4(0,0,0,0);

    //l1t::CaloStage1Cluster cluster;
    l1t::CaloEmCand EmCand(*&p4,
			   (int) em->rank(),
			   (int) em->regionId().ieta(),
			   (int) em->regionId().iphi(),
			   0);

    // create new format
    emcands->push_back( em->bx(), EmCand );

  }

  // loop over regions
  for (std::vector<L1CaloRegion>::const_iterator rgn=rgns->begin(); rgn!=rgns->end(); ++rgn) {

    // get physical units
    // double pt = 0.;
    // double eta = 0.;
    // double phi = 0.;
    //math::PtEtaPhiMLorentzVector p4( pt+1.e-6, eta, phi, 0 );

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > p4(0,0,0,0);


    // create new format
    //l1t::CaloRegion region;
    l1t::CaloRegion region(*&p4,
			   0.,
			   0.,
			   (int) rgn->et(),
			   (int) rgn->id().ieta(),
			   (int) rgn->id().iphi(),
			   0,
			   0,
			   0);

    // add to output
    regions->push_back( rgn->bx(), region );

  }

  iEvent.put(emcands);
  iEvent.put(regions);

}

// ------------ method called once each job just before starting event loop  ------------
void
l1t::L1TCaloRCTToUpgradeConverter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
l1t::L1TCaloRCTToUpgradeConverter::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
  void
  l1t::L1TCaloRCTToUpgradeConverter::beginRun(edm::Run const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when ending the processing of a run  ------------
/*
  void
  l1t::L1TCaloRCTToUpgradeConverter::endRun(edm::Run const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
  void
  l1t::L1TCaloRCTToUpgradeConverter::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup cons
  t&)
  {
  }
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
  void
  l1t::L1TCaloRCTToUpgradeConverter::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&
  )
  {
  }
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
l1t::L1TCaloRCTToUpgradeConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1TCaloRCTToUpgradeConverter);
