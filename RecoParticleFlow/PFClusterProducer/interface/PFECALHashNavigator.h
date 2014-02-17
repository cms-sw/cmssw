#ifndef RecoParticleFlow_PFClusterProducer_PFECALHashNavigator_h
#define RecoParticleFlow_PFClusterProducer_PFECALHashNavigator_h


#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"

  static const CaloDirection orderedDir[8]={SOUTHWEST,
					    SOUTH,
					    SOUTHEAST,
					    WEST,
					    EAST,
					    NORTHWEST,
					    NORTH,
					    NORTHEAST};


class PFECALHashNavigator : public PFRecHitNavigatorBase {
 public:



  PFECALHashNavigator() {

  }



  PFECALHashNavigator(const edm::ParameterSet& iConfig):
    PFRecHitNavigatorBase(iConfig){

  crossBarrelEndcapBorder_ =
    iConfig.getParameter<bool>("crossBarrelEndcapBorder");

  neighbourmapcalculated_ = false;

  }

  void beginEvent(const edm::EventSetup& iSetup) {
      edm::ESHandle<CaloGeometry> geoHandle;
      iSetup.get<CaloGeometryRecord>().get(geoHandle);
      
      const CaloSubdetectorGeometry *ebTmp = 
	geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
      const CaloSubdetectorGeometry *eeTmp = 
	geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

      barrelGeometry_  = dynamic_cast< const EcalBarrelGeometry* > (ebTmp);
      endcapGeometry_  = dynamic_cast < const EcalEndcapGeometry* > (eeTmp);

      // get the ecalBarrel topology
      barrelTopology_ = new  EcalBarrelTopology(geoHandle);
      endcapTopology_ = new  EcalEndcapTopology(geoHandle);

      ecalNeighbArray(*barrelGeometry_,*barrelTopology_,*endcapGeometry_,*endcapTopology_);

  }

  void associateNeighbours(reco::PFRecHit& rh,std::auto_ptr<reco::PFRecHitCollection>& hits,edm::RefProd<reco::PFRecHitCollection>& refprod) {



  DetId center( rh.detId() );


  DetId north = move( center, NORTH );
  DetId northeast = move( center, NORTHEAST );
  DetId northwest = move( center, NORTHWEST ); 
  DetId south = move( center, SOUTH );  
  DetId southeast = move( center, SOUTHEAST );  
  DetId southwest = move( center, SOUTHWEST );  
  DetId east  = move( center, EAST );  
  DetId west  = move( center, WEST );  


  associateNeighbour(north,rh,hits,refprod,0,1,0);
  associateNeighbour(northeast,rh,hits,refprod,1,1,0);
  associateNeighbour(south,rh,hits,refprod,0,-1,0);
  associateNeighbour(southwest,rh,hits,refprod,-1,-1,0);
  associateNeighbour(east,rh,hits,refprod,1,0,0);
  associateNeighbour(southeast,rh,hits,refprod,1,-1,0);
  associateNeighbour(west,rh,hits,refprod,-1,0,0);
  associateNeighbour(northwest,rh,hits,refprod,-1,1,0);

  }








 protected:


  void ecalNeighbArray(const EcalBarrelGeometry& barrelGeom,
		       const CaloSubdetectorTopology& barrelTopo,
		       const EcalEndcapGeometry& endcapGeom,
		       const CaloSubdetectorTopology& endcapTopo ){
  


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



bool stdsimplemove(DetId& cell, 
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



bool stdmove(DetId& cell, 
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



DetId move(DetId cell, 
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


  EcalEndcapTopology *endcapTopology_;
  EcalBarrelTopology *barrelTopology_;

  const EcalEndcapGeometry *endcapGeometry_;
  const EcalBarrelGeometry *barrelGeometry_;

  /// for each ecal barrel rechit, keep track of the neighbours
  std::vector<std::vector<DetId> >  neighboursEB_;

  /// for each ecal endcap rechit, keep track of the neighbours
  std::vector<std::vector<DetId> >  neighboursEE_;
  
  /// set to true in ecalNeighbArray
  bool  neighbourmapcalculated_;

  /// if true, navigation will cross the barrel-endcap border
  bool  crossBarrelEndcapBorder_;


};





#endif


