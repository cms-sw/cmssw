#include "L1Trigger/L1TCalorimeter/plugins/L1TCaloTowerProducer.h"

#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"


l1t::L1TCaloTowerProducer::L1TCaloTowerProducer(const edm::ParameterSet& ps) :
  bxFirst_(0),
  bxLast_(0),
  ecalToken_(bxLast_+1-bxFirst_),
  hcalToken_(bxLast_+1-bxFirst_),
  ietaMin_(-32),
  ietaMax_(32),
  iphiMin_(1),
  iphiMax_(72)
{

  // register what you produce
  produces< BXVector<l1t::CaloTower> > ();
  
  // register what you consume and keep token for later access:
  for (int ibx=0; ibx<bxLast_+1-bxFirst_; ibx++) {
    ecalToken_[ibx] = consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalToken"));
    hcalToken_[ibx] = consumes<HcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("hcalToken"));
  }

}

l1t::L1TCaloTowerProducer::~L1TCaloTowerProducer() {

}



// ------------ method called to produce the data  ------------
void
l1t::L1TCaloTowerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::auto_ptr<l1t::CaloTowerBxCollection> towersColl (new l1t::CaloTowerBxCollection);


  // do event setup
  // using RCT input scale objects
  edm::ESHandle<L1CaloEcalScale> ecalScale;
  iSetup.get<L1CaloEcalScaleRcd>().get(ecalScale);
  //  const L1CaloEcalScale* e = ecalScale.product();
  
  edm::ESHandle<L1CaloHcalScale> hcalScale;
  iSetup.get<L1CaloHcalScaleRcd>().get(hcalScale);
  //  const L1CaloHcalScale* h = hcalScale.product();


  // loop over crossings
  for (int bx = bxFirst_; bx < bxLast_+1; bx++) {
   
    int ibx = bx-bxFirst_;
 
    edm::Handle<EcalTrigPrimDigiCollection> ecalTPs;
    edm::Handle<HcalTrigPrimDigiCollection> hcalTPs;
    
    iEvent.getByToken(hcalToken_[ibx], hcalTPs);
    iEvent.getByToken(ecalToken_[ibx], ecalTPs);

    // create output vector
    int nTow = (iphiMax_-iphiMin_) * (ietaMax_-ietaMin_);  // leave a gap at ieta=0 for now?
    std::vector< l1t::CaloTower > towers(nTow);

    // loop over ECAL TPs
    EcalTrigPrimDigiCollection::const_iterator ecalItr;
    for (ecalItr=ecalTPs->begin(); ecalItr!=ecalTPs->end(); ++ecalItr) {
    
      int ieta = ecalItr->id().ieta(); 
      int iphi = ecalItr->id().iphi();

      int iet = ecalItr->compressedEt();
      int ifg = ecalItr->fineGrain();

      int itow = (ieta-1)*72+iphi-1;
      towers.at(itow).setHwEtEm(iet);
      towers.at(itow).setHwFGEm(ifg);

    }

    // loop over HCAL TPs
    HcalTrigPrimDigiCollection::const_iterator hcalItr;
    for (hcalItr=hcalTPs->begin(); hcalItr!=hcalTPs->end(); ++hcalItr) {
    
      int ieta = hcalItr->id().ieta(); 
      int iphi = hcalItr->id().iphi();

      int iet = hcalItr->SOI_compressedEt();
      //int ifg = hcalItr->SOI_fineGrain();

      int itow = (ieta-1)*72+iphi-1;
      towers.at(itow).setHwEtHad(iet);
      //      towers.at(itow).setHwFGHad(ifg);

    }

    // now calculate remaining tower quantities
    for (int ieta=ietaMin_; ieta<ietaMax_+1; ieta++) {

      if (ieta==0) continue;

      for (int iphi=iphiMin_; iphi<iphiMax_+1; iphi++) {

	int itow = (ieta-1)*72+iphi-1;

	// get ECAL/HCAL raw numbers
	int ietEcal = towers.at(itow).hwEtEm();
	int ietHcal = towers.at(itow).hwEtHad();
	
	//	const LorentzVector& p4;
	int iet = ietEcal + ietHcal;   // this is nonsense, temp solution!

	towers.at(itow).setHwPt(iet);
	towers.at(itow).setHwEta(ieta);
	towers.at(itow).setHwPhi(iphi);

      }
    }
   
    // copy towers to BXVector
    // could do this more quickly with improved BXVector interface
    std::vector<l1t::CaloTower>::const_iterator towItr;
    for (towItr=towers.begin(); towItr!=towers.end(); ++towItr) {
      towersColl->push_back(bx, (*towItr) );
    }

  }

  //  iEvent.put(towers);
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
l1t::L1TCaloTowerProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
l1t::L1TCaloTowerProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
l1t::L1TCaloTowerProducer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
l1t::L1TCaloTowerProducer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
l1t::L1TCaloTowerProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup cons
t&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
l1t::L1TCaloTowerProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&
)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
l1t::L1TCaloTowerProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1TCaloTowerProducer);
