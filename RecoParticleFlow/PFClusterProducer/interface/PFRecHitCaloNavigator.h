#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitCaloNavigator_h
#define RecoParticleFlow_PFClusterProducer_PFRecHitCaloNavigator_h


#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
/* #include "DataFormats/EcalDetId/interface/EBDetId.h" */
/* #include "DataFormats/EcalDetId/interface/EEDetId.h" */
/* #include "DataFormats/EcalDetId/interface/ESDetId.h" */
/* #include "DataFormats/HcalDetId/interface/HcalDetId.h" */

#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"


template <typename DET,typename TOPO,bool ownsTopo=true,unsigned DIM=2>
class PFRecHitCaloNavigator : public PFRecHitNavigatorBase {
 public:
 static constexpr unsigned _2D=2, _3D=3;

 struct CellInfo {
   bool visited;
   DetId cell;
 CellInfo() : visited(false), cell(0) { }
   
 CellInfo(bool a_visited, const DetId &a_cell) :
   visited(a_visited),
     cell(a_cell) { }
 };
 
 typedef std::tuple<short,short,short> Coordinate;
 typedef std::vector<std::pair<DetId,Coordinate> > NeighbourInfo;

 virtual ~PFRecHitCaloNavigator() { if(!ownsTopo) { topology_.release(); } }

 void associateNeighbours(reco::PFRecHit& hit, 
			  std::auto_ptr<reco::PFRecHitCollection>& hits,
			  const DetIdToHitIdx& hitmap,
			  edm::RefProd<reco::PFRecHitCollection>& refProd) override {
   auto visited_cells = std::move( getWindow( hit ) );
   for (unsigned int i=0; i<visited_cells.size(); i++) {
     if (!visited_cells[i].first.cell.null()) {
       const DetId& id = visited_cells[i].first.cell;
       const Coordinate& coord = visited_cells[i].second;
       if( std::get<0>(coord) != 0 ||
	   std::get<1>(coord) != 0 ||
	   std::get<2>(coord) != 0 ) {
	 associateNeighbour( id,hit,hits,hitmap,refProd,
			     std::get<0>(coord),
			     std::get<1>(coord),
			     std::get<2>(coord) );
       }
     }
   } 
 }

 void associateNeighbours(reco::PFRecHit& hit,
			  std::auto_ptr<reco::PFRecHitCollection>& hits,
			  edm::RefProd<reco::PFRecHitCollection>& refProd) {
   auto visited_cells = std::move( getWindow( hit ) );
   for (unsigned int i=0; i<visited_cells.size(); i++) {
     if (!visited_cells[i].first.cell.null()) {
       const DetId& id = visited_cells[i].first.cell;
       const Coordinate& coord = visited_cells[i].second;
       if( std::get<0>(coord) != 0 ||
	   std::get<1>(coord) != 0 ||
	   std::get<2>(coord) != 0 ) {
	 associateNeighbour( id,hit,hits,refProd,
			     std::get<0>(coord),
			     std::get<1>(coord),
			     std::get<2>(coord) );
       }
     }
   }
 }
 

 std::vector<std::pair<CellInfo,Coordinate> > getWindow(reco::PFRecHit& hit) {
   constexpr unsigned short halfDim = 1; // range in each dimension
   constexpr unsigned short dimSize = 3;
   constexpr CaloDirection directions[6] = { NORTH, SOUTH, EAST, WEST, UP, DOWN }; // max available directions
   
   DET detid( hit.detId() );
   if( detid.null() ) return std::vector<std::pair<CellInfo,Coordinate> >();
   // this is a ripoff of getWindow() from CaloSubDetTopology
   NeighbourInfo cellsInWindow;
   DET tmpId(detid);
   NeighbourInfo fringe;
   fringe.emplace_back(tmpId,std::make_tuple(0,0,0));
   
   std::vector<std::pair<CellInfo,Coordinate> > visited_cells;
   visited_cells.resize(std::pow(dimSize,DIM));
   
   while (fringe.size() > 0) {
     NeighbourInfo::value_type cur = fringe.back();
     fringe.pop_back();
     // check all 2*DIM neighbours (in case of 2D only check NSEW)
     for (unsigned dirnum = 0; dirnum < 2*DIM; ++dirnum) {
       Coordinate neighbour = getNeighbourIndex(cur.second,directions[dirnum]);
       //If outside the window range
       if ( std::get<0>(neighbour) < -halfDim ||
	    std::get<0>(neighbour) > halfDim ||
	    std::get<1>(neighbour) < -halfDim ||
	    std::get<1>(neighbour) > halfDim ||
	    std::get<2>(neighbour) < -halfDim ||
	    std::get<2>(neighbour) > halfDim )
	 continue;
       
       //Found integer index in the matrix
       unsigned int_index = ( std::get<0>(neighbour) + halfDim +
			      dimSize * (std::get<1>(neighbour) + halfDim ) +
			      (DIM==3)*dimSize*dimSize * ( std::get<2>(neighbour) + halfDim ) );	
       assert(int_index < visited_cells.size());
       
       // check whether we have seen this neighbour already
       if (visited_cells[int_index].first.visited)
	 // we have seen this one already
	 continue;
       
       // a new cell, get the DetId of the neighbour, mark it
       // as visited and add it to the fringe
       visited_cells[int_index].first.visited = true;
       std::vector<DetId> neighbourCells = getNeighbours(cur.first,directions[dirnum]);
       
       if ( neighbourCells.size() == 1 ) {
	 visited_cells[int_index].first.cell = neighbourCells[0];
	 visited_cells[int_index].second = neighbour;
       } else if ( neighbourCells.size() == 0 ) {
	 visited_cells[int_index].first.cell = DetId(0);
	 visited_cells[int_index].second = neighbour;
       } else {
	 throw cms::Exception("getWindowError") << "Not supported subdetector for getWindow method";
       }
       
       if (!visited_cells[int_index].first.cell.null())
	 fringe.emplace_back(visited_cells[int_index].first.cell,neighbour);
       
     } // loop over all possible directions
   } // while some cells are left on the fringe
   
   return visited_cells;
 }

 protected:
  std::unique_ptr<const TOPO> topology_;

 std::vector<DetId> getNeighbours(const DetId& id, const CaloDirection& dir) const {
   switch(dir) {
   case NONE:
     return std::vector<DetId>();
     break;
   case SOUTH:
     return std::vector<DetId>(1,topology_->goSouth(id));
     break;
   case NORTH:
     return std::vector<DetId>(1,topology_->goNorth(id));
     break;
   case EAST:
     return std::vector<DetId>(1,topology_->goEast(id));
     break;
   case WEST:
     return std::vector<DetId>(1,topology_->goWest(id));
     break;
   case UP:
     if( DIM != 3 ) throw cms::Exception("getNeighboursError") << "Unsopported direction";
     return std::vector<DetId>(1,topology_->goUp(id));
     break;
   case DOWN:
     if( DIM != 3 ) throw cms::Exception("getNeighboursError") << "Unsupported direction";
     return std::vector<DetId>(1,topology_->goDown(id));
     break;
   default:
     throw cms::Exception("getNeighboursError") << "Unsupported direction";
   }
   return std::vector<DetId>();
 }
 
 Coordinate getNeighbourIndex(const Coordinate &coord, const CaloDirection& dir) const {
   switch (dir) {
   case NORTH: return std::make_tuple(std::get<0>(coord),std::get<1>(coord)+1,std::get<2>(coord)); break;
   case SOUTH: return std::make_tuple(std::get<0>(coord),std::get<1>(coord)-1,std::get<2>(coord)); break;
   
   case EAST: return std::make_tuple(std::get<0>(coord)+1,std::get<1>(coord),std::get<2>(coord)); break;
   case WEST: return std::make_tuple(std::get<0>(coord)-1,std::get<1>(coord),std::get<2>(coord)); break;
   
   case UP:
   if( DIM != 3 ) cms::Exception("getNeighourIndexError") << "Unsupported direction";
   return std::make_tuple(std::get<0>(coord),std::get<1>(coord),std::get<2>(coord)+1);
   break;
   case DOWN:
   if( DIM != 3 ) cms::Exception("getNeighourIndexError") << "Unsupported direction";
   return std::make_tuple(std::get<0>(coord),std::get<1>(coord),std::get<2>(coord)-1);
   break;
   
   default:
   throw cms::Exception("getNeighourIndexError") << "Unsupported direction";
   }
 }
 
};

#endif


