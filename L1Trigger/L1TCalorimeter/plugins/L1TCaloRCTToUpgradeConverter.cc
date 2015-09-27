#include "L1Trigger/L1TCalorimeter/plugins/L1TCaloRCTToUpgradeConverter.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include <vector>


using namespace l1t;

L1TCaloRCTToUpgradeConverter::L1TCaloRCTToUpgradeConverter(const edm::ParameterSet& ps) {

  produces<CaloRegionBxCollection>();
  produces<CaloEmCandBxCollection>();

  rgnToken_ = consumes<L1CaloRegionCollection>(ps.getParameter<edm::InputTag>("regionTag"));
  emToken_ = consumes<L1CaloEmCollection>(ps.getParameter<edm::InputTag>("emTag"));
}

L1TCaloRCTToUpgradeConverter::~L1TCaloRCTToUpgradeConverter() {

}

// ------------ method called to produce the data  ------------
void
L1TCaloRCTToUpgradeConverter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // check status of RCT conditions & renew if needed


  // store new formats
  std::auto_ptr<BXVector<CaloEmCand> > emcands (new CaloEmCandBxCollection);
  std::auto_ptr<BXVector<CaloRegion> > regions (new CaloRegionBxCollection);

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

    //CaloStage1Cluster cluster;
    CaloEmCand EmCand(*&p4,
			   (int) em->rank(),
			   (int) em->regionId().ieta(),
			   (int) em->regionId().iphi(),
			   (int) em->index());

    EmCand.setHwIso((int) em->isolated());
    //std::cout<<"ISO:    "<<EmCand.hwIso()<<"    "<<em->isolated()<<std::endl;

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

    bool tauVeto = rgn->fineGrain(); //equivalent to tauVeto for HB/HE, includes extra info for HF
    int hwQual = (int) tauVeto;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > p4(0,0,0,0);


    // create new format
    // several values here are stage 2 only, leave empty
    CaloRegion region(*&p4,           //  LorentzVector& p4,
      0.,                          //  etEm,
      0.,                          //  etHad,
      (int) rgn->et(),             //  pt,
      (int) rgn->id().ieta(),      //  eta,
      (int) rgn->id().iphi(),      //  phi,
      hwQual,                      //  qual,
      0,                           //  hwEtEm,
      0);                          //  hwEtHad

    // add to output
    regions->push_back( rgn->bx(), region );

  }

  iEvent.put(emcands);
  iEvent.put(regions);

}

// ------------ method called once each job just before starting event loop  ------------
void
L1TCaloRCTToUpgradeConverter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TCaloRCTToUpgradeConverter::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
  void
  L1TCaloRCTToUpgradeConverter::beginRun(edm::Run const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when ending the processing of a run  ------------
/*
  void
  L1TCaloRCTToUpgradeConverter::endRun(edm::Run const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
  void
  L1TCaloRCTToUpgradeConverter::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup cons
  t&)
  {
  }
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
  void
  L1TCaloRCTToUpgradeConverter::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&
  )
  {
  }
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TCaloRCTToUpgradeConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloRCTToUpgradeConverter);
