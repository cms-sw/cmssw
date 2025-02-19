#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducerECAL.h"

#include <memory>

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"


using namespace std;
using namespace edm;

PFRecHitProducerECAL::PFRecHitProducerECAL(const edm::ParameterSet& iConfig)
 : PFRecHitProducer(iConfig) {

  // access to the collections of rechits

  
  inputTagEcalRecHitsEB_ = 
    iConfig.getParameter<InputTag>("ecalRecHitsEB");

  inputTagEcalRecHitsEE_ = 
    iConfig.getParameter<InputTag>("ecalRecHitsEE");
  
  
  crossBarrelEndcapBorder_ =
    iConfig.getParameter<bool>("crossBarrelEndcapBorder");

  timingCleaning_ = 
    iConfig.getParameter<bool>("timing_Cleaning");

  topologicalCleaning_ = 
    iConfig.getParameter<bool>("topological_Cleaning");

  threshCleaningEB_ = 
    iConfig.getParameter<double>("thresh_Cleaning_EB");

  threshCleaningEE_ = 
    iConfig.getParameter<double>("thresh_Cleaning_EE");

  neighbourmapcalculated_ = false;
}



PFRecHitProducerECAL::~PFRecHitProducerECAL() {}






void 
PFRecHitProducerECAL::createRecHits(vector<reco::PFRecHit>& rechits,
				    vector<reco::PFRecHit>& rechitsCleaned,
				    edm::Event& iEvent, 
				    const edm::EventSetup& iSetup ) {



  // this map is necessary to find the rechit neighbours efficiently
  //C but I should think about using Florian's hashed index to do this.
  //C in which case the map might not be necessary anymore
  // 
  // the key of this map is detId. 
  // the value is the index in the rechits vector
  map<unsigned, unsigned > idSortedRecHits;
//   typedef map<unsigned, unsigned >::iterator IDH;

  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
  // get the ecalBarrel geometry
  const CaloSubdetectorGeometry *ebtmp = 
    geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  
  const EcalBarrelGeometry* ecalBarrelGeometry = 
    dynamic_cast< const EcalBarrelGeometry* > (ebtmp);
  assert( ecalBarrelGeometry );

  // get the ecalBarrel topology
  EcalBarrelTopology ecalBarrelTopology(geoHandle);

  // get the endcap geometry
  const CaloSubdetectorGeometry *eetmp = 
    geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

  const EcalEndcapGeometry* ecalEndcapGeometry = 
    dynamic_cast< const EcalEndcapGeometry* > (eetmp);
  assert( ecalEndcapGeometry );
  

  // get the endcap topology
  EcalEndcapTopology ecalEndcapTopology(geoHandle);

    
  if(!neighbourmapcalculated_)
    ecalNeighbArray( *ecalBarrelGeometry,
		     ecalBarrelTopology,
		     *ecalEndcapGeometry,
		     ecalEndcapTopology );

         
  // get the ecalBarrel rechits

  edm::Handle<EcalRecHitCollection> rhcHandle;


  bool found = iEvent.getByLabel(inputTagEcalRecHitsEB_, 
				 rhcHandle);
  
  if(!found) {

    ostringstream err;
    err<<"could not find rechits "<<inputTagEcalRecHitsEB_;
    LogError("PFRecHitProducerECAL")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }
  else {
    assert( rhcHandle.isValid() );
    
    // process ecal ecalBarrel rechits
    for(unsigned i=0; i<rhcHandle->size(); i++) {
      
      const EcalRecHit& erh = (*rhcHandle)[i];
      const DetId& detid = erh.detid();
      double energy = erh.energy();
      // uint32_t flag = erh.recoFlag();
      double time = erh.time();

      EcalSubdetector esd=(EcalSubdetector)detid.subdetId();
      if (esd != 1) continue;

      if(energy < thresh_Barrel_ ) continue;
          
      // Check and skip the TT recovered rechits
      //if ( flag == EcalRecHit::kTowerRecovered ) { 
      if ( erh.checkFlag(EcalRecHit::kTowerRecovered) ) { 
	// std::cout << "Rechit was recovered with energy " << energy << std::endl;
	continue;
      }

      // Just clean ECAL Barrel rechits out of time by more than 5 sigma.
      // if ( timingCleaning_ && energy > threshCleaning_ && flag == EcalRecHit::kOutOfTime ) { 
      if ( ( timingCleaning_ && energy > threshCleaningEB_ && 
	     erh.checkFlag(EcalRecHit::kOutOfTime) ) ||
	   ( topologicalCleaning_ && 
	     ( erh.checkFlag(EcalRecHit::kWeird) || 
	       erh.checkFlag(EcalRecHit::kDiWeird) ) ) ) { 
	reco::PFRecHit *pfrhCleaned = createEcalRecHit(detid, energy,  
						       PFLayer::ECAL_BARREL,
						       ecalBarrelGeometry);
	if( !pfrhCleaned ) continue; // problem with this rechit. skip it      
	pfrhCleaned->setRescale(time);
	rechitsCleaned.push_back( *pfrhCleaned );
	delete pfrhCleaned;
	continue;
      } 

      
      reco::PFRecHit *pfrh = createEcalRecHit(detid, energy,  
					      PFLayer::ECAL_BARREL,
					      ecalBarrelGeometry);
      
      if( !pfrh ) continue; // problem with this rechit. skip it
      pfrh->setRescale(time);
      
      rechits.push_back( *pfrh );
      delete pfrh;
      idSortedRecHits.insert( make_pair(detid.rawId(), rechits.size()-1 ) ); 
    }      
  }



  //C proceed as for the barrel
  // process ecal endcap rechits

  found = iEvent.getByLabel(inputTagEcalRecHitsEE_,
			    rhcHandle);
  
  if(!found) {
    ostringstream err;
    err<<"could not find rechits "<<inputTagEcalRecHitsEE_;
    LogError("PFRecHitProducerECAL")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }
  else {
    assert( rhcHandle.isValid() );
    
    for(unsigned i=0; i<rhcHandle->size(); i++) {
      
      const EcalRecHit& erh = (*rhcHandle)[i];
      const DetId& detid = erh.detid();
      double energy = erh.energy();
      //uint32_t flag = erh.recoFlag();
      double time = erh.time();
      EcalSubdetector esd=(EcalSubdetector)detid.subdetId();
      if (esd != 2) continue;
      if(energy < thresh_Endcap_ ) continue;

      // Check and skip the TT recovered rechits
      if ( erh.checkFlag(EcalRecHit::kTowerRecovered) ) {
      // if ( flag == EcalRecHit::kTowerRecovered ) { 
	// std::cout << "Rechit was recovered with energy " << energy << std::endl;
	continue;
      }
      
      
      // EE cleaning
    
      if ( ( timingCleaning_ && energy > threshCleaningEE_ && 
	     erh.checkFlag(EcalRecHit::kOutOfTime) ) ||
	   ( topologicalCleaning_ && 
	     ( erh.checkFlag(EcalRecHit::kWeird) ) ) ) { 
	reco::PFRecHit *pfrhCleaned = createEcalRecHit(detid, energy,  
						       PFLayer::ECAL_ENDCAP,
						       ecalEndcapGeometry);
	if( !pfrhCleaned ) continue; // problem with this rechit. skip it      
	pfrhCleaned->setRescale(time);
	rechitsCleaned.push_back( *pfrhCleaned );
	delete pfrhCleaned;
	continue;
      } 
      



      reco::PFRecHit *pfrh = createEcalRecHit(detid, energy,
					      PFLayer::ECAL_ENDCAP,
					      ecalEndcapGeometry);
      if( !pfrh ) continue; // problem with this rechit. skip it
      pfrh->setRescale(time);

      rechits.push_back( *pfrh );
      delete pfrh;
      idSortedRecHits.insert( make_pair(detid.rawId(), rechits.size()-1 ) ); 
    }
  }


  // do navigation
  for(unsigned i=0; i<rechits.size(); i++ ) {
    
//     findRecHitNeighbours( rechits[i], idSortedRecHits, 
// 			  ecalBarrelTopology, 
// 			  *ecalBarrelGeometry, 
// 			  ecalEndcapTopology,
// 			  *ecalEndcapGeometry);
    findRecHitNeighboursECAL( rechits[i], idSortedRecHits ); 
			      
  }
} 

  


reco::PFRecHit* 
PFRecHitProducerECAL::createEcalRecHit( const DetId& detid,
					double energy,
					PFLayer::Layer layer,
					const CaloSubdetectorGeometry* geom ) {

  math::XYZVector position;
  math::XYZVector axis;

  const CaloCellGeometry *thisCell 
    = geom->getGeometry(detid);
  
  // find rechit geometry
  if(!thisCell) {
    LogError("PFRecHitProducerECAL")
      <<"warning detid "<<detid.rawId()
      <<" not found in geometry"<<endl;
    return 0;
  }
  
  position.SetCoordinates ( thisCell->getPosition().x(),
			    thisCell->getPosition().y(),
			    thisCell->getPosition().z() );

  
  
  // the axis vector is the difference 
  const TruncatedPyramid* pyr 
    = dynamic_cast< const TruncatedPyramid* > (thisCell);    
  if( pyr ) {
    axis.SetCoordinates( pyr->getPosition(1).x(), 
			 pyr->getPosition(1).y(), 
			 pyr->getPosition(1).z() ); 
    
    math::XYZVector axis0( pyr->getPosition(0).x(), 
			   pyr->getPosition(0).y(), 
			   pyr->getPosition(0).z() );
    
    axis -= axis0;    
  }
  else return 0;

//   if( !geomfound ) {
//     LogError("PFRecHitProducerECAL")<<"cannor find geometry for detid "
// 				 <<detid.rawId()<<" in layer "<<layer<<endl;
//     return 0; // geometry not found, skip rechit
//   }
  
  
  reco::PFRecHit *rh 
    = new reco::PFRecHit( detid.rawId(), layer, 
			  energy, 
			  position.x(), position.y(), position.z(), 
			  axis.x(), axis.y(), axis.z() ); 


  const CaloCellGeometry::CornersVec& corners = thisCell->getCorners();
  assert( corners.size() == 8 );

  rh->setNECorner( corners[0].x(), corners[0].y(),  corners[0].z() );
  rh->setSECorner( corners[1].x(), corners[1].y(),  corners[1].z() );
  rh->setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z() );
  rh->setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z() );

  return rh;
}




bool
PFRecHitProducerECAL::findEcalRecHitGeometry(const DetId& detid, 
					  const CaloSubdetectorGeometry* geom,
					  math::XYZVector& position, 
					  math::XYZVector& axis ) {
  

  const CaloCellGeometry *thisCell 
    = geom->getGeometry(detid);
  
  // find rechit geometry
  if(!thisCell) {
    LogError("PFRecHitProducerECAL")
      <<"warning detid "<<detid.rawId()
      <<" not found in geometry"<<endl;
    return false;
  }
  
  position.SetCoordinates ( thisCell->getPosition().x(),
			    thisCell->getPosition().y(),
			    thisCell->getPosition().z() );

  
  
  // the axis vector is the difference 
  const TruncatedPyramid* pyr 
    = dynamic_cast< const TruncatedPyramid* > (thisCell);    
  if( pyr ) {
    axis.SetCoordinates( pyr->getPosition(1).x(), 
			 pyr->getPosition(1).y(), 
			 pyr->getPosition(1).z() ); 
    
    math::XYZVector axis0( pyr->getPosition(0).x(), 
			   pyr->getPosition(0).y(), 
			   pyr->getPosition(0).z() );
    
    axis -= axis0;

    
    return true;
  }
  else return false;
}



void 
PFRecHitProducerECAL::findRecHitNeighboursECAL
( reco::PFRecHit& rh, 
  const map<unsigned,unsigned >& sortedHits ) {
  
  DetId center( rh.detId() );


  DetId north = move( center, NORTH );
  DetId northeast = move( center, NORTHEAST );
  DetId northwest = move( center, NORTHWEST ); 
  DetId south = move( center, SOUTH );  
  DetId southeast = move( center, SOUTHEAST );  
  DetId southwest = move( center, SOUTHWEST );  
  DetId east  = move( center, EAST );  
  DetId west  = move( center, WEST );  
    
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




// Build the array of (max)8 neighbors
void 
PFRecHitProducerECAL::ecalNeighbArray(
			      const EcalBarrelGeometry& barrelGeom,
			      const CaloSubdetectorTopology& barrelTopo,
			      const EcalEndcapGeometry& endcapGeom,
			      const CaloSubdetectorTopology& endcapTopo ){
  

  static const CaloDirection orderedDir[8]={SOUTHWEST,
					    SOUTH,
					    SOUTHEAST,
					    WEST,
					    EAST,
					    NORTHWEST,
					    NORTH,
                                            NORTHEAST};

  const unsigned nbarrel = 62000;
  // Barrel first. The hashed index runs from 0 to 61199
  neighboursEB_.resize(nbarrel);
  
  //std::cout << " Building the array of neighbours (barrel) " ;

  const std::vector<DetId>& vec(barrelGeom.getValidDetIds(DetId::Ecal,
							  EcalBarrel));
  unsigned size=vec.size();    
  for(unsigned ic=0; ic<size; ++ic) 
    {
      // We get the 9 cells in a square. 
      std::vector<DetId> neighbours(barrelTopo.getWindow(vec[ic],3,3));
      //      std::cout << " Cell " << EBDetId(vec[ic]) << std::endl;
      unsigned nneighbours=neighbours.size();

      unsigned hashedindex=EBDetId(vec[ic]).hashedIndex();
      if(hashedindex>=nbarrel)
        {
          LogDebug("CaloGeometryTools")  << " Array overflow " << std::endl;
        }


      // If there are 9 cells, it is easy, and this order is know:
//      6  7  8
//      3  4  5 
//      0  1  2   (0 = SOUTHWEST)

      if(nneighbours==9)
        {
          neighboursEB_[hashedindex].reserve(8);
          for(unsigned in=0;in<nneighbours;++in)
            {
              // remove the centre
              if(neighbours[in]!=vec[ic]) 
                {
                  neighboursEB_[hashedindex].push_back(neighbours[in]);
                  //          std::cout << " Neighbour " << in << " " << EBDetId(neighbours[in]) << std::endl;
                }
            }
        }
      else
        {
          DetId central(vec[ic]);
          neighboursEB_[hashedindex].resize(8,DetId(0));
          for(unsigned idir=0;idir<8;++idir)
            {
              DetId testid=central;
              bool status=stdmove(testid,orderedDir[idir],
				  barrelTopo, endcapTopo,
				  barrelGeom, endcapGeom);
              if(status) neighboursEB_[hashedindex][idir]=testid;
            }

        }
    }

  // Moved to the endcap

  //  std::cout << " done " << size << std::endl;
  //  std::cout << " Building the array of neighbours (endcap) " ;

//  vec.clear();
  const std::vector<DetId>& vecee=endcapGeom.getValidDetIds(DetId::Ecal,EcalEndcap);
  size=vecee.size();    
  // There are some holes in the hashedIndex for the EE. Hence the array is bigger than the number
  // of crystals
  const unsigned nendcap=19960;

  neighboursEE_.resize(nendcap);
  for(unsigned ic=0; ic<size; ++ic) 
    {
      // We get the 9 cells in a square. 
      std::vector<DetId> neighbours(endcapTopo.getWindow(vecee[ic],3,3));
      unsigned nneighbours=neighbours.size();
      // remove the centre
      unsigned hashedindex=EEDetId(vecee[ic]).hashedIndex();
      
      if(hashedindex>=nendcap)
        {
          LogDebug("CaloGeometryTools")  << " Array overflow " << std::endl;
        }

      if(nneighbours==9)
        {
          neighboursEE_[hashedindex].reserve(8);
          for(unsigned in=0;in<nneighbours;++in)
            {     
              // remove the centre
              if(neighbours[in]!=vecee[ic]) 
                {
                  neighboursEE_[hashedindex].push_back(neighbours[in]);
                }
            }
        }
      else
        {
          DetId central(vecee[ic]);
          neighboursEE_[hashedindex].resize(8,DetId(0));
          for(unsigned idir=0;idir<8;++idir)
            {
              DetId testid=central;
              bool status=stdmove(testid,orderedDir[idir],
				  barrelTopo, endcapTopo,
				  barrelGeom, endcapGeom);

              if(status) neighboursEE_[hashedindex][idir]=testid;
            }

        }
    }
  //  std::cout << " done " << size <<std::endl;
  neighbourmapcalculated_ = true;
}



bool 
PFRecHitProducerECAL::stdsimplemove(DetId& cell, 
				    const CaloDirection& dir,
				    const CaloSubdetectorTopology& barrelTopo,
				    const CaloSubdetectorTopology& endcapTopo,
				    const EcalBarrelGeometry& barrelGeom,
				    const EcalEndcapGeometry& endcapGeom ) 
  const {

  std::vector<DetId> neighbours;

  // BARREL CASE 
  if(cell.subdetId()==EcalBarrel) {
    EBDetId ebDetId = cell;

    neighbours = barrelTopo.getNeighbours(ebDetId,dir);

    // first try to move according to the standard navigation
    if(neighbours.size()>0 && !neighbours[0].null()) {
      cell = neighbours[0];
      return true;
    }

    // failed.

    if(crossBarrelEndcapBorder_) {
      // are we on the outer ring ?
      const int ietaAbs ( ebDetId.ietaAbs() ) ; // abs value of ieta
      if( EBDetId::MAX_IETA == ietaAbs ) {
	// get ee nbrs for for end of barrel crystals  
	
	// yes we are
	const EcalBarrelGeometry::OrderedListOfEEDetId& 
	  ol( * barrelGeom.getClosestEndcapCells( ebDetId ) ) ;
	
	// take closest neighbour on the other side, that is in the barrel.
	cell = *(ol.begin() );
	return true;
      }   
    }
  }

  // ENDCAP CASE 
  else if(cell.subdetId()==EcalEndcap) {

    EEDetId eeDetId = cell;

    neighbours= endcapTopo.getNeighbours(eeDetId,dir);

    if(neighbours.size()>0 && !neighbours[0].null()) {
      cell = neighbours[0];
      return true;
    }

    // failed.

    if(crossBarrelEndcapBorder_) {
      // are we on the outer ring ?
      const int iphi ( eeDetId.iPhiOuterRing() ) ;    
      if( iphi!= 0) {
	// yes we are
	const EcalEndcapGeometry::OrderedListOfEBDetId& 
	  ol( * endcapGeom.getClosestBarrelCells( eeDetId ) ) ;
	
	// take closest neighbour on the other side, that is in the barrel.
	cell = *(ol.begin() );
	return true;
      }   
    }
  } 

  // everything failed 
  cell = DetId(0);
  return false;
}



bool 
PFRecHitProducerECAL::stdmove(DetId& cell, 
			      const CaloDirection& dir,
			      const CaloSubdetectorTopology& barrelTopo,
			      const CaloSubdetectorTopology& endcapTopo,
			      const EcalBarrelGeometry& barrelGeom,
			      const EcalEndcapGeometry& endcapGeom  ) 
  
  const {


  bool result; 

  if(dir==NORTH) {
    result = stdsimplemove(cell,NORTH, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
    return result;
  }
  else if(dir==SOUTH) {
    result = stdsimplemove(cell,SOUTH, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
    return result;
  }
  else if(dir==EAST) {
    result = stdsimplemove(cell,EAST, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
    return result;
  }
  else if(dir==WEST) {
    result = stdsimplemove(cell,WEST, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
    return result;
  }


  // One has to try both paths
  else if(dir==NORTHEAST)
    {
      result = stdsimplemove(cell,NORTH, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
      if(result)
        return stdsimplemove(cell,EAST, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
      else
        {
          result = stdsimplemove(cell,EAST, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
          if(result)
            return stdsimplemove(cell,NORTH, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
          else
            return false; 
        }
    }
  else if(dir==NORTHWEST)
    {
      result = stdsimplemove(cell,NORTH, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
      if(result)
        return stdsimplemove(cell,WEST, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
      else
        {
          result = stdsimplemove(cell,WEST, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
          if(result)
            return stdsimplemove(cell,NORTH, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
          else
            return false; 
        }
    }
  else if(dir == SOUTHEAST)
    {
      result = stdsimplemove(cell,SOUTH, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
      if(result)
        return stdsimplemove(cell,EAST, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
      else
        {
          result = stdsimplemove(cell,EAST, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
          if(result)
            return stdsimplemove(cell,SOUTH, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
          else
            return false; 
        }
    }
  else if(dir == SOUTHWEST)
    {
      result = stdsimplemove(cell,SOUTH, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
      if(result)
        return stdsimplemove(cell,WEST, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
      else
        {
          result = stdsimplemove(cell,SOUTH, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
          if(result)
            return stdsimplemove(cell,WEST, barrelTopo, endcapTopo, barrelGeom, endcapGeom );
          else
            return false; 
        }
    }
  cell = DetId(0);
  return false;
}



DetId PFRecHitProducerECAL::move(DetId cell, 
			      const CaloDirection&dir ) const
{  
  DetId originalcell = cell; 
  if(dir==NONE || cell==DetId(0)) return false;

  // Conversion CaloDirection and index in the table
  // CaloDirection :NONE,SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST, NORTHEAST,NORTHWEST,NORTH
  // Table : SOUTHWEST,SOUTH,SOUTHEAST,WEST,EAST,NORTHWEST,NORTH, NORTHEAST
  static const int calodirections[9]={-1,1,2,0,4,3,7,5,6};
    
  assert(neighbourmapcalculated_);

  DetId result = (originalcell.subdetId()==EcalBarrel) ? 
    neighboursEB_[EBDetId(originalcell).hashedIndex()][calodirections[dir]]:
    neighboursEE_[EEDetId(originalcell).hashedIndex()][calodirections[dir]];
  return result; 
}

