#include "RecoMuon/MuonIdentification/plugins/InterestingEcalDetIdProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"


InterestingEcalDetIdProducer::InterestingEcalDetIdProducer(const edm::ParameterSet& iConfig) 
{
  inputCollection_ = iConfig.getParameter< edm::InputTag >("inputCollection");
  produces< DetIdCollection >() ;
  muonToken_ = consumes<reco::MuonCollection>(inputCollection_);
}


InterestingEcalDetIdProducer::~InterestingEcalDetIdProducer()
{}

void InterestingEcalDetIdProducer::beginRun (const edm::Run & run, const edm::EventSetup & iSetup)  
{
   edm::ESHandle<CaloTopology> theCaloTopology;
   iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
   caloTopology_ = &(*theCaloTopology); 
}

void
InterestingEcalDetIdProducer::produce (edm::Event& iEvent, 
				       const edm::EventSetup& iSetup)
{
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(muonToken_,muons);
  
  std::auto_ptr< DetIdCollection > interestingDetIdCollection( new DetIdCollection() ) ;

  for(reco::MuonCollection::const_iterator muon = muons->begin(); muon != muons->end(); ++muon){
    if (! muon->isEnergyValid() ) continue;
    if ( muon->calEnergy().ecal_id.rawId()==0 ) continue;
    const CaloSubdetectorTopology* topology = caloTopology_->getSubdetectorTopology(DetId::Ecal,muon->calEnergy().ecal_id.subdetId());
    const std::vector<DetId>& ids = topology->getWindow(muon->calEnergy().ecal_id, 5, 5); 
    for ( std::vector<DetId>::const_iterator id = ids.begin(); id != ids.end(); ++id )
      if(std::find(interestingDetIdCollection->begin(), interestingDetIdCollection->end(), *id) 
	 == interestingDetIdCollection->end()) 
	interestingDetIdCollection->push_back(*id);
  }
  iEvent.put(interestingDetIdCollection);
}
