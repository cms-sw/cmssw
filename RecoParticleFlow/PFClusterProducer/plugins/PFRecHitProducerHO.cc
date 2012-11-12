#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducerHO.h"

#include <memory>

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

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

#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloTowerNavigator.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

// Necessary includes for identify severity of flagged problems in HO rechits
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

using namespace std;
using namespace edm;

PFRecHitProducerHO::PFRecHitProducerHO(const edm::ParameterSet& iConfig)
  : PFRecHitProducer(iConfig)
{
  
  // access to the collections of rechits
  inputTagHORecHits_ = 
    iConfig.getParameter<InputTag>("recHitsHO");
  
  HOMaxAllowedSev_ = iConfig.getParameter<int>("HOMaxAllowedSev");
  neighbourmapcalculated_ = false;
}



PFRecHitProducerHO::~PFRecHitProducerHO() {}

void 
PFRecHitProducerHO::createRecHits(vector<reco::PFRecHit>& rechits,
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
  
  // get the HO geometry
  const CaloSubdetectorGeometry *hcalBarrelGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalOuter);
  
  // get the HO topology
  edm::ESHandle<HcalTopology> hcalBarrelTopology;
  iSetup.get<IdealGeometryRecord>().get(hcalBarrelTopology);
  
  if(!neighbourmapcalculated_)
    hoNeighbArray( *hcalBarrelGeometry,
		   *hcalBarrelTopology);

  // Get Hcal Severity Level Computer, so that the severity of each rechit flag/status may be determined
  edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComputerHndl;
  iSetup.get<HcalSeverityLevelComputerRcd>().get(hcalSevLvlComputerHndl);
  const HcalSeverityLevelComputer* hcalSevLvlComputer = hcalSevLvlComputerHndl.product();

  
  // get the HO rechits
  
  edm::Handle<HORecHitCollection> rhcHandle;
  
  
  bool found = iEvent.getByLabel(inputTagHORecHits_, 
				 rhcHandle);
  
  if(!found) {
    
    ostringstream err;
    err<<"could not find rechits "<<inputTagHORecHits_;
    LogError("PFRecHitProducerHO")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }
  else {
    assert( rhcHandle.isValid() );
    
    // process HO rechits
    for(unsigned i=0; i<rhcHandle->size(); i++) {
      
      const HORecHit& erh = (*rhcHandle)[i];
      const HcalDetId& detid = (HcalDetId)erh.detid();
      
      double energy = erh.energy();
      // uint32_t flag = erh.recoFlag();
      double time = erh.time();
      
      HcalSubdetector esd=(HcalSubdetector)detid.subdetId();
      
      if (esd != 3) continue;
      
      int hoeta=detid.ieta();
      if ( (abs(hoeta)<=4 && energy < thresh_Barrel_) || 
	   (abs(hoeta)> 4 && energy < thresh_Endcap_) ) continue;
      

      // Get Channel Quality information for the given detID
      const HcalChannelStatus* theStatus = theHcalChStatus->getValues(detid);
      unsigned theStatusValue = theStatus->getValue();
      // Now get severity of problems for the given detID, based on the rechit flag word and the channel quality status value
      int hitSeverity=hcalSevLvlComputer->getSeverityLevel(detid, erh.flags(),theStatusValue);
    
      // Skip hits whose problems are more severe than max accept level.  In the future, allow for cleaning of such hits?
      // Note:  As of April 2012, by default, all HO hits in rings +/-1, +/-2 should be identified as either "remove from calotowers" or "remove from rechit collections" in the channel quality database, and thus should be rejected by this conditional statement.
      if (hitSeverity>HOMaxAllowedSev_) 
	{
	  //std::cout <<"Rejecting HO hit HO("<<hoeta<<", "<<detid.iphi()<<", "<<detid.depth()<<std::endl;
	  continue;
	}  


      
      reco::PFRecHit *pfrh = createHORecHit(detid, energy,  
					    PFLayer::HCAL_BARREL2, // HO,
					    hcalBarrelGeometry);
      
      if( !pfrh ) continue; // problem with this rechit. skip it
      
      pfrh->setRescale(time);
      
      rechits.push_back( *pfrh );
      delete pfrh;
      idSortedRecHits.insert( make_pair(detid.rawId(), rechits.size()-1 ) ); 
    }      
  }
  
  // do navigation
  for(unsigned i=0; i<rechits.size(); i++ ) {
    findRecHitNeighboursHO( rechits[i], *hcalBarrelTopology, idSortedRecHits ); 
  }
  
} 


reco::PFRecHit* 
PFRecHitProducerHO::createHORecHit( const DetId& detid,
				    double energy,
				    PFLayer::Layer layer,
				    const CaloSubdetectorGeometry* geom ) {
  
  math::XYZVector position;
  math::XYZVector axis;
  
  const CaloCellGeometry *thisCell 
    = geom->getGeometry(detid);
  
  // find rechit geometry
  if(!thisCell) {
    LogError("PFRecHitProducerHO")
      <<"warning detid "<<detid.rawId()
      <<" not found in geometry"<<endl;
    return 0;
  }
  
  double sclel0l1r=0.946; //384.8/406.6
  
  if (abs(thisCell->getPosition().z())>130) {
    position.SetCoordinates ( sclel0l1r*thisCell->getPosition().x(),
			      sclel0l1r*thisCell->getPosition().y(),
			      sclel0l1r*thisCell->getPosition().z() );   
    
  } else {
    position.SetCoordinates ( thisCell->getPosition().x(),
			      thisCell->getPosition().y(),
			      thisCell->getPosition().z() );
  }
  
  // the axis vector is the difference 
  
  //   const TruncatedPyramid* pyr 
  //     = dynamic_cast< const TruncatedPyramid* > (thisCell);    
  //   if( pyr ) {
  //     axis.SetCoordinates( pyr->getPosition(1).x(), 
  // 			 pyr->getPosition(1).y(), 
  // 			 pyr->getPosition(1).z() ); 
  
  //     math::XYZVector axis0( pyr->getPosition(0).x(), 
  // 			   pyr->getPosition(0).y(), 
  // 			   pyr->getPosition(0).z() );
  
  //     axis -= axis0;    
  //   }
  //   else return 0;
  
  axis = math::XYZVector(0,0,0);
  
  //   if( !geomfound ) {
  //     LogError("PFRecHitProducerHO")<<"cannor find geometry for detid "
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
  
  if (abs(corners[0].z())>130.0) {
    rh->setNECorner( sclel0l1r*corners[0].x(), sclel0l1r*corners[0].y(),  sclel0l1r*corners[0].z() );
    rh->setSECorner( sclel0l1r*corners[1].x(), sclel0l1r*corners[1].y(),  sclel0l1r*corners[1].z() );
    rh->setSWCorner( sclel0l1r*corners[2].x(), sclel0l1r*corners[2].y(),  sclel0l1r*corners[2].z() );
    rh->setNWCorner( sclel0l1r*corners[3].x(), sclel0l1r*corners[3].y(),  sclel0l1r*corners[3].z() );
    
  } else {
    rh->setNECorner( corners[0].x(), corners[0].y(),  corners[0].z() );
    rh->setSECorner( corners[1].x(), corners[1].y(),  corners[1].z() );
    rh->setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z() );
    rh->setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z() );
  }
  
  return rh;
}


bool
PFRecHitProducerHO::findHORecHitGeometry(const DetId& detid, 
					 const CaloSubdetectorGeometry* geom,
					 math::XYZVector& position, 
					 math::XYZVector& axis ) {
  
  
  const CaloCellGeometry *thisCell 
    = geom->getGeometry(detid);
  
  // find rechit geometry
  if(!thisCell) {
    LogError("PFRecHitProducerHO")
      <<"warning detid "<<detid.rawId()
      <<" not found in geometry"<<endl;
    return false;
  }
  
  position.SetCoordinates ( thisCell->getPosition().x(),
			    thisCell->getPosition().y(),
			    thisCell->getPosition().z() );
  
  
  
  // the axis vector is the difference 
  //   const TruncatedPyramid* pyr 
  //     = dynamic_cast< const TruncatedPyramid* > (thisCell);    
  //   if( pyr ) {
  //     axis.SetCoordinates( pyr->getPosition(1).x(), 
  // 			 pyr->getPosition(1).y(), 
  // 			 pyr->getPosition(1).z() ); 
  
  //     math::XYZVector axis0( pyr->getPosition(0).x(), 
  // 			   pyr->getPosition(0).y(), 
  // 			   pyr->getPosition(0).z() );
  
  //     axis -= axis0;
  
  
  //     return true;
  //   }
  
  axis = math::XYZVector(0,0,0);
  return true;
  
  //  else return false;
}



void 
PFRecHitProducerHO::findRecHitNeighboursHO
( reco::PFRecHit& rh, 
  const HcalTopology& topo, 
  const map<unsigned,unsigned >& sortedHits ) {
  
  DetId center( rh.detId() );
  
  
  DetId north = move( center, topo, NORTH );
  DetId northeast = move( center, topo, NORTHEAST );
  DetId northwest = move( center, topo, NORTHWEST ); 
  DetId south = move( center, topo, SOUTH );  
  DetId southeast = move( center, topo, SOUTHEAST );  
  DetId southwest = move( center, topo, SOUTHWEST );  
  DetId east  = move( center, topo, EAST );  
  DetId west  = move( center, topo, WEST );  
  
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
PFRecHitProducerHO::hoNeighbArray(
				  const CaloSubdetectorGeometry& barrelGeom,
				  const HcalTopology& barrelTopo){
  
  static const CaloDirection orderedDir[8]={SOUTHWEST,
					    SOUTH,
					    SOUTHEAST,
					    WEST,
					    EAST,
					    NORTHWEST,
					    NORTH,
                                            NORTHEAST};
  
  const unsigned nbarrel = 2160; //62000;
  // Barrel first. The hashed index runs from 0 to 2199 61199
  neighboursHO_.resize(barrelTopo.getHOSize());
  
  //std::cout << " Building the array of neighbours (barrel) " ;
  
  const std::vector<DetId>& vec(barrelGeom.getValidDetIds(DetId::Hcal,
							  HcalOuter));
  unsigned size=vec.size();    
  for(unsigned ic=0; ic<size; ++ic) 
    {
      // We get the 9 cells in a square. 
      std::vector<DetId> neighbours(barrelTopo.getWindow(vec[ic],3,3));
      unsigned nneighbours=neighbours.size();
      
      unsigned hashedindex=barrelTopo.detId2denseIdHO(vec[ic]);
      //           std::cout << " Cell " << ic<<" "<<vec[ic].rawId()<<" "<<HcalDetId(vec[ic]) <<" "<<hashedindex<<" "<<nneighbours<< std::endl;
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
          neighboursHO_[hashedindex].reserve(8);
          for(unsigned in=0;in<nneighbours;++in)
            {
              // remove the centre
              //    cout<<"ic "<<ic<<" "<<in<<" "<<neighbours[in].rawId()<<" "<<vec[ic].rawId()<<" "<<hashedindex<<endl; 
              if(neighbours[in]!=vec[ic]) 
                {
                  neighboursHO_[hashedindex].push_back(neighbours[in]);
		  //            std::cout << " Neighbour " << ic<<" "<<size<<" "<<in << " " <<hashedindex<<" "<< HcalDetId(neighbours[in]) << std::endl;
                }
            }
        }
      else
        {
          DetId central(vec[ic]);
          neighboursHO_[hashedindex].resize(8,DetId(0));
          for(unsigned idir=0;idir<8;++idir)
            {
              DetId testid=central;
              bool status=stdmove(testid,orderedDir[idir],
				  barrelTopo, barrelGeom);
              if(status) neighboursHO_[hashedindex][idir]=testid;
            }
	  
        }
    }
  
  //    std::cout << " done " << size <<std::endl;
  neighbourmapcalculated_ = true;
}

bool 
PFRecHitProducerHO::stdsimplemove(DetId& cell, 
				  const CaloDirection& dir,
				  const CaloSubdetectorTopology& barrelTopo,
				  const CaloSubdetectorGeometry& barrelGeom)
  const {
  
  std::vector<DetId> neighbours;
  
  // BARREL CASE 
  if(cell.subdetId()==HcalOuter) {
    HcalDetId hoDetId = cell;
    
    neighbours = barrelTopo.getNeighbours(hoDetId,dir);
    
    // first try to move according to the standard navigation
    if(neighbours.size()>0 && !neighbours[0].null()) {
      cell = neighbours[0];
      return true;
    }
    
    // failed.
    
    
  }
  
  // everything failed 
  cell = DetId(0);
  return false;
}



bool 
PFRecHitProducerHO::stdmove(DetId& cell, 
			    const CaloDirection& dir,
			    const CaloSubdetectorTopology& barrelTopo,
			    const CaloSubdetectorGeometry& barrelGeom)
  
  const {
  
  
  bool result; 
  
  if(dir==NORTH) {
    result = stdsimplemove(cell,NORTH, barrelTopo, barrelGeom);
    return result;
  }
  else if(dir==SOUTH) {
    result = stdsimplemove(cell,SOUTH, barrelTopo, barrelGeom);
    return result;
  }
  else if(dir==EAST) {
    result = stdsimplemove(cell,EAST, barrelTopo, barrelGeom);
    return result;
  }
  else if(dir==WEST) {
    result = stdsimplemove(cell,WEST, barrelTopo, barrelGeom);
    return result;
  }
  
  
  // One has to try both paths
  else if(dir==NORTHEAST)
    {
      result = stdsimplemove(cell,NORTH, barrelTopo, barrelGeom);
      if(result)
        return stdsimplemove(cell,EAST, barrelTopo, barrelGeom);
      else
        {
          result = stdsimplemove(cell,EAST, barrelTopo, barrelGeom);
          if(result)
            return stdsimplemove(cell,NORTH, barrelTopo, barrelGeom);
          else
            return false; 
        }
    }
  else if(dir==NORTHWEST)
    {
      result = stdsimplemove(cell,NORTH, barrelTopo, barrelGeom );
      if(result)
        return stdsimplemove(cell,WEST, barrelTopo, barrelGeom );
      else
        {
          result = stdsimplemove(cell,WEST, barrelTopo, barrelGeom );
          if(result)
            return stdsimplemove(cell,NORTH, barrelTopo, barrelGeom );
          else
            return false; 
        }
    }
  else if(dir == SOUTHEAST)
    {
      result = stdsimplemove(cell,SOUTH, barrelTopo, barrelGeom );
      if(result)
        return stdsimplemove(cell,EAST, barrelTopo, barrelGeom );
      else
        {
          result = stdsimplemove(cell,EAST, barrelTopo, barrelGeom );
          if(result)
            return stdsimplemove(cell,SOUTH, barrelTopo, barrelGeom );
          else
            return false; 
        }
    }
  else if(dir == SOUTHWEST)
    {
      result = stdsimplemove(cell,SOUTH, barrelTopo, barrelGeom );
      if(result)
        return stdsimplemove(cell,WEST, barrelTopo, barrelGeom );
      else
        {
          result = stdsimplemove(cell,SOUTH, barrelTopo, barrelGeom );
          if(result)
            return stdsimplemove(cell,WEST, barrelTopo, barrelGeom );
          else
            return false; 
        }
    }
  cell = DetId(0);
  return false;
}

DetId PFRecHitProducerHO::move(DetId cell, 
			       const HcalTopology&topo,
			       const CaloDirection&dir ) const
{  
  DetId originalcell = cell; 
  if(dir==NONE || cell==DetId(0)) return false;
  
  // Conversion CaloDirection and index in the table
  // CaloDirection :NONE,SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST, NORTHEAST,NORTHWEST,NORTH
  // Table : SOUTHWEST,SOUTH,SOUTHEAST,WEST,EAST,NORTHWEST,NORTH, NORTHEAST
  static const int calodirections[9]={-1,1,2,0,4,3,7,5,6};
  
  assert(neighbourmapcalculated_);
  
  DetId result = neighboursHO_[topo.detId2denseIdHO(originalcell)][calodirections[dir]];
  return result; 
}

