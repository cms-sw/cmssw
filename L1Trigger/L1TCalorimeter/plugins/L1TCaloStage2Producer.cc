#include "L1Trigger/L1TCalorimeter/plugins/L1TCaloStage2Producer.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"


l1t::L1TCaloStage2Producer::L1TCaloStage2Producer(const edm::ParameterSet& ps) {

}

l1t::L1TCaloStage2Producer::~L1TCaloStage2Producer() {

}

// ------------ method called to produce the data  ------------
void
l1t::L1TCaloStage2Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   LogDebug("l1t|stage 2") << "L1TCaloStage2Producer::produce function called..." << std::endl;
   
   // check parameters !
   
   
   //inputs
   Handle< BXVector<l1t::CaloTower> > towers;
   iEvent.getByToken(m_towerToken,towers);

   int bxFirst = towers->getFirstBX();
   int bxLast = towers->getLastBX();
   
   //outputs
   std::auto_ptr<l1t::EGammaBxCollection> egammas (new l1t::EGammaBxCollection(0, bxFirst, bxLast));
   std::auto_ptr<l1t::TauBxCollection> taus (new l1t::TauBxCollection(0, bxFirst, bxLast));
   std::auto_ptr<l1t::JetBxCollection> jets (new l1t::JetBxCollection(0, bxFirst, bxLast));
   std::auto_ptr<l1t::EtSumBxCollection> etsums (new l1t::EtSumBxCollection(0, bxFirst, bxLast));
   
   // loop over BX
   for(int i = towers->getFirstBX(); i < towers->getLastBX(); ++i) {
     std::auto_ptr< std::vector<l1t::CaloTower> > localTowers (new std::vector<l1t::CaloTower>);
     std::auto_ptr< std::vector<l1t::EGamma> > localEGammas (new std::vector<l1t::EGamma>);
     std::auto_ptr< std::vector<l1t::Tau> > localTaus (new std::vector<l1t::Tau>);
     std::auto_ptr< std::vector<l1t::Jet> > localJets (new std::vector<l1t::Jet>);
     std::auto_ptr< std::vector<l1t::EtSum> > localEtSums (new std::vector<l1t::EtSum>);
     
     for(std::vector<l1t::CaloTower>::const_iterator tower = towers->begin(i);
	 tower != towers->end(i);
	 ++tower) {
       localTowers->push_back(*tower);
     }
     
     m_processor->processEvent(*localTowers,
			       *localEGammas, *localTaus, *localJets, *localEtSums);
     
     for(std::vector<l1t::EGamma>::const_iterator eg = localEGammas->begin(); eg != localEGammas->end(); ++eg) egammas->push_back(i, *eg);
     for(std::vector<l1t::Tau>::const_iterator tau = localTaus->begin(); tau != localTaus->end(); ++tau) taus->push_back(i, *tau);
     for(std::vector<l1t::Jet>::const_iterator jet = localJets->begin(); jet != localJets->end(); ++jet) jets->push_back(i, *jet);
     for(std::vector<l1t::EtSum>::const_iterator etsum = localEtSums->begin(); etsum != localEtSums->end(); ++etsum) etsums->push_back(i, *etsum);
   }
   
   iEvent.put(egammas);
   iEvent.put(taus);
   iEvent.put(jets);
   iEvent.put(etsums);
   
}

// ------------ method called once each job just before starting event loop  ------------
void 
l1t::L1TCaloStage2Producer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
l1t::L1TCaloStage2Producer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
l1t::L1TCaloStage2Producer::beginRun(edm::Run const&, edm::EventSetup const&)
{

    m_fwv = boost::shared_ptr<FirmwareVersion>();
    m_fwv->setFirmwareVersion(1); //hardcode for now

    // Set the current algorithm version based on DB pars from database:
    m_processor = m_factory.create(*m_fwv);

    if (! m_processor) {
      // we complain here once per run
      edm::LogError("l1t|stage 2") << "L1TCaloStage2Producer: firmware could not be configured.\n";
    }

}
 
// ------------ method called when ending the processing of a run  ------------
void
l1t::L1TCaloStage2Producer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
l1t::L1TCaloStage2Producer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup cons
t&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
l1t::L1TCaloStage2Producer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&
)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
l1t::L1TCaloStage2Producer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1TCaloStage2Producer);
