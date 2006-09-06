#include "Calibration/EcalAlCaRecoProducers/interface/AlCaElectronsProducer.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


AlCaElectronsProducer::AlCaElectronsProducer(const edm::ParameterSet& iConfig)
{

pixelMatchElectronProducer_ = iConfig.getParameter< std::string > ("pixelMatchElectronProducer");
siStripElectronProducer_ = iConfig.getParameter< std::string > ("siStripElectronProducer");

pixelMatchElectronCollection_ = iConfig.getParameter<std::string>("pixelMatchElectronCollection");
siStripElectronCollection_ = iConfig.getParameter<std::string>("siStripElectronCollection");

alcaPixelMatchElectronCollection_ = iConfig.getParameter<std::string>("alcaPixelMatchElectronCollection");
alcaSiStripElectronCollection_ = iConfig.getParameter<std::string>("alcaSiStripElectronCollection");

ptCut_ = iConfig.getParameter< double > ("ptCut");


   //register your products
   produces<reco::SiStripElectronCollection>(alcaSiStripElectronCollection_);
   produces<reco::ElectronCollection>(alcaPixelMatchElectronCollection_);

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


   Handle<reco::SiStripElectronCollection> pSiStripElectrons;
   Handle<reco::ElectronCollection> pPixelMatchElectrons;

//   std::cout << "pixelMatchElectronProducer: "<< pixelMatchElectronProducer_<<std::endl;
//   std::cout << "pixelMatchElectronCollection: "<< pixelMatchElectronCollection_<<std::endl;
//   std::cout << "siStripElectronProducer: "<< siStripElectronProducer_<<std::endl;
//   std::cout << "siStripElectronCollection: "<< siStripElectronCollection_<<std::endl;


   try {
     iEvent.getByLabel(pixelMatchElectronProducer_, pixelMatchElectronCollection_, pPixelMatchElectrons);
     iEvent.getByLabel(siStripElectronProducer_, siStripElectronCollection_, pSiStripElectrons);
   } catch ( std::exception& ex ) {
     LogDebug("") << "AlCaElectronsProducer: Error! can't get product!" << std::endl;
//     std::cout <<  "AlCaElectronsProducer: Error! can't get product!" << std::endl;
   }

  //Create empty output collections

    std::auto_ptr<reco::SiStripElectronCollection> miniSiStripElectronCollection(new reco::SiStripElectronCollection);
    std::auto_ptr<reco::ElectronCollection> miniPixelMatchElectronCollection(new reco::ElectronCollection);

  //Select interesting electrons
    reco::SiStripElectronCollection::const_iterator siStripEleIt;

    for (siStripEleIt=pSiStripElectrons->begin(); siStripEleIt!=pSiStripElectrons->end(); siStripEleIt++) {
      if (siStripEleIt->pt() >= ptCut_) {
        miniSiStripElectronCollection->push_back(*siStripEleIt);
      }
    }

    reco::ElectronCollection::const_iterator pixelMatchEleIt;

    for (pixelMatchEleIt=pPixelMatchElectrons->begin(); pixelMatchEleIt!=pPixelMatchElectrons->end(); pixelMatchEleIt++) {
      if (pixelMatchEleIt->pt() >= ptCut_) {
        miniPixelMatchElectronCollection->push_back(*pixelMatchEleIt);
      }
    }

  //Put selected information in the event
  iEvent.put( miniPixelMatchElectronCollection,alcaPixelMatchElectronCollection_ );
  iEvent.put( miniSiStripElectronCollection,alcaSiStripElectronCollection_ );
  
 
}
