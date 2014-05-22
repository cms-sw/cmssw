#include "CaloSimhitToRechitProducerShashlik.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloTopology/interface/ShashlikGeometry.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

using namespace std;
using namespace edm;

//#define debugPrintout

CaloSimhitToRechitProducerShashlik::CaloSimhitToRechitProducerShashlik (const edm::ParameterSet& iConfig)
  : mSource        (iConfig.getParameter<edm::InputTag>("src")),
    mEnergyScale (iConfig.getParameter<double>("energyScale"))
{
  produces<EcalRecHitCollection>("EKSimRecoHits");
}

CaloSimhitToRechitProducerShashlik::~CaloSimhitToRechitProducerShashlik ()
{
} 


void CaloSimhitToRechitProducerShashlik::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  edm::ESHandle<ShashlikGeometry> hGeometry;
  iSetup.get<ShashlikGeometryRecord>().get( hGeometry );
  const ShashlikGeometry* geometry = hGeometry.product();

  EcalRecHitCollection* recHits = new EcalRecHitCollection ();
  edm::Handle<PCaloHitContainer> simCaloHits;
  iEvent.getByLabel(mSource, simCaloHits);
  
  for (PCaloHitContainer::const_iterator iSimHit = simCaloHits->begin();
       iSimHit != simCaloHits->end(); ++iSimHit) {
    double energy = iSimHit->energy();
//     double time = iSimHit->time();
//     int timeOffset = 0;
    DetId id = iSimHit->id();
    if (id.det() == DetId::Ecal && id.subdetId() == EcalShashlik) {
      const CaloCellGeometry * cell = geometry->getGeometry(id);
      if (!cell) {
	std::cout << "CaloSimhitToRechitProducerShashlik-> can not find geometry for cell " << EKDetId (id) << std::endl;
	continue;
      }
      double scaledEnergy  = scaleEnergy (id, energy, *cell);
      
      DetId geoid = EKDetId(id).geometryCell ();

      EcalRecHitCollection::iterator ihit = recHits->find (geoid);
      if (ihit == recHits->end()) {
	EcalRecHit newHit (geoid, 0, 0); // energy=0, time=0
	recHits->push_back (newHit);
	ihit = recHits->find (geoid);
// #ifdef debugPrintout
// 	cout << "CaloSimhitToRechitProducerShashlik-> new cell " << EKDetId (geoid) << ", original " << EKDetId (id) << endl; 
// #endif
      }
      ihit->setEnergy (ihit->energy() + scaledEnergy);
// #ifdef debugPrintout
//       cout << "CaloSimhitToRechitProducerShashlik-> add energy: cell " << EKDetId (geoid) 
// 	   << ", de/energy: " << scaledEnergy << '/' << ihit->energy() << endl; 
// #endif
    }
  }

#ifdef debugPrintout

  double totalEnergy = 0;

  cout << "CaloSimhitToRechitProducerShashlik-> collected cells " << endl;
  for (size_t i = 0; i < recHits->size(); ++i) {
    //    cout << i << "   " << EKDetId((*recHits)[i].id()) << " energy: " << (*recHits)[i].energy() << endl;
    totalEnergy += (*recHits)[i].energy();
  }
  cout << "totalEnergy: " << totalEnergy << endl;
#endif
  iEvent.put(std::auto_ptr<EcalRecHitCollection>(recHits),"EKSimRecoHits");
}

double CaloSimhitToRechitProducerShashlik::scaleEnergy (DetId id, double energy, const CaloCellGeometry& cell) const {
  return energy * mEnergyScale;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( CaloSimhitToRechitProducerShashlik );
