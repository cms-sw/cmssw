#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducerPS.h"

#include <memory>

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"



using namespace std;
using namespace edm;

PFRecHitProducerPS::PFRecHitProducerPS(const edm::ParameterSet& iConfig)
 : PFRecHitProducer(iConfig) {



  // access to the collections of rechits
  
  inputTagEcalRecHitsES_ = 
    iConfig.getParameter<InputTag>("ecalRecHitsES");
}



PFRecHitProducerPS::~PFRecHitProducerPS() {}



void PFRecHitProducerPS::createRecHits(vector<reco::PFRecHit>& rechits,
				       vector<reco::PFRecHit>& rechitsCleaned,
				       edm::Event& iEvent, 
				       const edm::EventSetup& iSetup) {

  map<unsigned, unsigned > idSortedRecHits;
  typedef map<unsigned, unsigned >::iterator IDH;


  // get the ps geometry
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);
    
  const CaloSubdetectorGeometry *psGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
    

  // ShR 28 Jul 2008: check if geometry is NULL. If so we are using
  // partial CMS gemoetry in Pilot1/2 scenarios which do not include the preshower
  if(!psGeometry) {
    LogDebug("PFRecHitProducerPS") << "No EcalPreshower geometry available. putting empty PS rechits collection in event";
    return;
  }


  // get the ps topology
  EcalPreshowerTopology psTopology(geoHandle);

  // process rechits
  Handle< EcalRecHitCollection >   pRecHits;



  bool found = iEvent.getByLabel(inputTagEcalRecHitsES_,
				 pRecHits);

  if(!found) {
    ostringstream err;
    err<<"could not find rechits "<<inputTagEcalRecHitsES_;
    LogError("PFRecHitProducerPS")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }
  else {
    assert( pRecHits.isValid() );

    const EcalRecHitCollection& psrechits = *( pRecHits.product() );
    typedef EcalRecHitCollection::const_iterator IT;
 
    for(IT i=psrechits.begin(); i!=psrechits.end(); i++) {
      const EcalRecHit& hit = *i;
      
      double energy = hit.energy();
      if( energy < thresh_Endcap_ ) continue; 
            
      const ESDetId& detid = hit.detid();
      const CaloCellGeometry *thisCell = psGeometry->getGeometry(detid);
     
      if(!thisCell) {
	LogError("PFRecHitProducerPS")<<"warning detid "<<detid.rawId()
				     <<" not found in preshower geometry"
				     <<endl;
	return;
      }
      
      const GlobalPoint& position = thisCell->getPosition();
     
      PFLayer::Layer layer = PFLayer::NONE;
            
      switch( detid.plane() ) {
      case 1:
	layer = PFLayer::PS1;
	break;
      case 2:
	layer = PFLayer::PS2;
	break;
      default:
	LogError("PFRecHitProducerPS")
	  <<"incorrect preshower plane !! plane number "
	  <<detid.plane()<<endl;
	assert(0);
      }
 
      reco::PFRecHit *pfrh
	= new reco::PFRecHit( detid.rawId(), layer, energy, 
			      position.x(), position.y(), position.z(), 
			      0,0,0 );
      
      const CaloCellGeometry::CornersVec& corners = thisCell->getCorners();
      assert( corners.size() == 8 );
      
      pfrh->setNECorner( corners[0].x(), corners[0].y(),  corners[0].z() );
      pfrh->setSECorner( corners[1].x(), corners[1].y(),  corners[1].z() );
      pfrh->setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z() );
      pfrh->setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z() );

      
      // if( !pfrh ) continue; // problem with this rechit. skip it

      rechits.push_back( *pfrh );
      delete pfrh;
      idSortedRecHits.insert( make_pair(detid.rawId(), rechits.size()-1 ) );   
    }
  }

  // do navigation
  for(unsigned i=0; i<rechits.size(); i++ ) {
    
    findRecHitNeighbours( rechits[i], idSortedRecHits, 
			  psTopology, 
			  *psGeometry, 
			  psTopology,
			  *psGeometry);
  }
}



void 
PFRecHitProducerPS::findRecHitNeighbours
( reco::PFRecHit& rh, 
  const map<unsigned,unsigned >& sortedHits, 
  const CaloSubdetectorTopology& barrelTopology, 
  const CaloSubdetectorGeometry& barrelGeometry, 
  const CaloSubdetectorTopology& endcapTopology, 
  const CaloSubdetectorGeometry& endcapGeometry ) {
  
  DetId detid( rh.detId() );

  const CaloSubdetectorTopology* topology = 0;
  const CaloSubdetectorGeometry* geometry = 0;
  
  switch( rh.layer() ) {
  case PFLayer::ECAL_ENDCAP: 
    topology = &endcapTopology;
    geometry = &endcapGeometry;
    break;
  case PFLayer::ECAL_BARREL: 
    topology = &barrelTopology;
    geometry = &barrelGeometry;
    break;
  case PFLayer::HCAL_ENDCAP:
    topology = &endcapTopology;
    geometry = &endcapGeometry;
    break;
  case PFLayer::HCAL_BARREL1:
    topology = &barrelTopology;
    geometry = &barrelGeometry;
    break;
  case PFLayer::PS1:
  case PFLayer::PS2:
    topology = &barrelTopology;
    geometry = &barrelGeometry;
    break;
  default:
    assert(0);
  }
  
  assert( topology && geometry );

  CaloNavigator<DetId> navigator(detid, topology);

  DetId north = navigator.north();  
  
  DetId northeast(0);
  if( north != DetId(0) ) {
    northeast = navigator.east();  
  }
  navigator.home();


  DetId south = navigator.south();

  

  DetId southwest(0); 
  if( south != DetId(0) ) {
    southwest = navigator.west();
  }
  navigator.home();


  DetId east = navigator.east();
  DetId southeast;
  if( east != DetId(0) ) {
    southeast = navigator.south(); 
  }
  navigator.home();
  DetId west = navigator.west();
  DetId northwest;
  if( west != DetId(0) ) {   
    northwest = navigator.north();  
  }
  navigator.home();
    
  IDH i = sortedHits.find( north.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
  
  i = sortedHits.find( northeast.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
  
  i = sortedHits.find( south.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( southwest.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( east.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( southeast.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( west.rawId() );
  if(i != sortedHits.end() ) 
     rh.add4Neighbour( i->second );
   
  i = sortedHits.find( northwest.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    

}

