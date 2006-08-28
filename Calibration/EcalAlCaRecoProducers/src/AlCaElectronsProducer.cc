#include "Calibration/EcalAlCaRecoProducers/interface/AlCaElectronsProducer.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


AlCaElectronsProducer::AlCaElectronsProducer(const edm::ParameterSet& iConfig)
{
electronsProducer_ = iConfig.getParameter< edm::InputTag > ("electronsProducer");
ptCut_ = iConfig.getParameter< double > ("ptCut");

LogDebug("") << "producer: " << electronsProducer_.encode() ;

   //register your products
   produces<reco::ElectronCollection>();
}


AlCaElectronsProducer::~AlCaElectronsProducer()
{
 

}


// ------------ method called to produce the data  ------------
void
AlCaElectronsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   Handle<reco::ElectronCollection> pElectrons;

   try {
     iEvent.getByLabel(electronsProducer_,pElectrons);
   } catch ( std::exception& ex ) {
     LogDebug("") << "AlCaElectronsProducer: Error! can't get product!" << std::endl;
   }

  //Create empty output collections

    std::auto_ptr<reco::ElectronCollection> miniElectronCollection(new reco::ElectronCollection);

  //Select interesting electrons
    reco::ElectronCollection::const_iterator eleIt;

    for (eleIt=pElectrons->begin(); eleIt!=pElectrons->end(); eleIt++) {
      if (eleIt->pt() >= ptCut_) {
        miniElectronCollection->push_back(*eleIt);
      }
    }


  //Put selected information in the event
  iEvent.put( miniElectronCollection, "ElectronCollection");
  
  
}
