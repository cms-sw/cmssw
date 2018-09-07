#ifndef RecoPixelVertexing_PixelTriplets_src_CACell_h
#define RecoPixelVertexing_PixelTriplets_src_CACell_h

#include <array>
#include <cmath>

#include "DataFormats/Math/interface/deltaPhi.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

class CACellStatus {

public:
  
  unsigned char getCAState() const {
    return theCAState;
  }
  
  // if there is at least one left neighbor with the same state (friend), the state has to be increased by 1.
  void updateState() {
    theCAState += hasSameStateNeighbors;
  }
  
  bool isRootCell(const unsigned int minimumCAState) const {
    return (theCAState >= minimumCAState);
  }
  
 public:
  unsigned char theCAState=0;
  unsigned char hasSameStateNeighbors=0;
  
};

class CACell {
public:
  using Hit = RecHitsSortedInPhi::Hit;
  using CAntuple = std::vector<unsigned int>;
  using CAntuplet = std::vector<unsigned int>;
  using CAColl = std::vector<CACell>;
  using CAStatusColl = std::vector<CACellStatus>;
  
  
  CACell(const HitDoublets* doublets, int doubletId, const int innerHitId, const int outerHitId) :
    theDoublets(doublets), theDoubletId(doubletId)
    ,theInnerR(doublets->rv(doubletId, HitDoublets::inner)) 
    ,theInnerZ(doublets->z(doubletId, HitDoublets::inner))
  {}

   
  
  Hit const & getInnerHit() const {
    return theDoublets->hit(theDoubletId, HitDoublets::inner);
  }
  
  Hit const & getOuterHit() const {
    return theDoublets->hit(theDoubletId, HitDoublets::outer);
  }
  
  
  float getInnerX() const {
    return theDoublets->x(theDoubletId, HitDoublets::inner);
  }
  
  float getOuterX() const {
    return theDoublets->x(theDoubletId, HitDoublets::outer);
  }
  
  float getInnerY() const {
    return theDoublets->y(theDoubletId, HitDoublets::inner);
  }
  
  float getOuterY() const {
    return theDoublets->y(theDoubletId, HitDoublets::outer);
  }
  
  float getInnerZ() const {
    return theInnerZ;
  }
  
  float getOuterZ() const {
    return theDoublets->z(theDoubletId, HitDoublets::outer);
  }
  
  float getInnerR() const {
    return theInnerR;
  }
  
  float getOuterR() const {
    return theDoublets->rv(theDoubletId, HitDoublets::outer);
  }
  
  float getInnerPhi() const {
    return theDoublets->phi(theDoubletId, HitDoublets::inner);
  }
  
  float getOuterPhi() const {
    return theDoublets->phi(theDoubletId, HitDoublets::outer);
  }
  
  void evolve(unsigned int me, CAStatusColl& allStatus) {
    
    allStatus[me].hasSameStateNeighbors = 0;
    auto mystate = allStatus[me].theCAState;
    
    for (auto oc : theOuterNeighbors ) {
      
      if (allStatus[oc].getCAState() == mystate) {
	
	allStatus[me].hasSameStateNeighbors = 1;
	
	break;
      }
    }
    
  }
  

  void checkAlignmentAndAct(CAColl& allCells, CAntuple & innerCells, const float ptmin, const float region_origin_x,
			    const float region_origin_y, const float region_origin_radius, const float thetaCut,
			    const float phiCut, const float hardPtCut, std::vector<CACell::CAntuplet> * foundTriplets) {
    int ncells = innerCells.size();
    int constexpr VSIZE = 16;
    int ok[VSIZE];
    float r1[VSIZE];
    float z1[VSIZE];
    auto ro = getOuterR();
    auto zo = getOuterZ();
    unsigned int cellId = this - &allCells.front();
    auto loop = [&](int i, int vs) {
      for (int j=0;j<vs; ++j) {
	auto koc = innerCells[i+j];
	auto & oc =  allCells[koc];
	r1[j] = oc.getInnerR();
	z1[j] = oc.getInnerZ();
      }
      // this vectorize!
      for (int j=0;j<vs; ++j) ok[j] = areAlignedRZ(r1[j], z1[j], ro, zo, ptmin, thetaCut);
      for (int j=0;j<vs; ++j) {
	auto koc = innerCells[i+j];
	auto & oc =  allCells[koc]; 
	if (ok[j]&&haveSimilarCurvature(oc,ptmin, region_origin_x, region_origin_y,
					region_origin_radius, phiCut, hardPtCut)) {
	  if (foundTriplets) foundTriplets->emplace_back(CACell::CAntuplet{koc,cellId});
	  else {
	    oc.tagAsOuterNeighbor(cellId);
	  }
	}
      }
    };
    auto lim = VSIZE*(ncells/VSIZE);
    for (int i=0; i<lim; i+=VSIZE) loop(i, VSIZE); 
    loop(lim, ncells-lim);
    
  }
  
  void checkAlignmentAndTag(CAColl& allCells, CAntuple & innerCells, const float ptmin, const float region_origin_x,
			    const float region_origin_y, const float region_origin_radius, const float thetaCut,
			    const float phiCut, const float hardPtCut) {
    checkAlignmentAndAct(allCells, innerCells, ptmin, region_origin_x, region_origin_y, region_origin_radius, thetaCut,
			 phiCut, hardPtCut, nullptr);
    
  }
  void checkAlignmentAndPushTriplet(CAColl& allCells, CAntuple & innerCells, std::vector<CACell::CAntuplet>& foundTriplets,
				    const float ptmin, const float region_origin_x, const float region_origin_y,
				    const float region_origin_radius, const float thetaCut, const float phiCut,
				    const float hardPtCut) {
    checkAlignmentAndAct(allCells, innerCells, ptmin, region_origin_x, region_origin_y, region_origin_radius, thetaCut,
			 phiCut, hardPtCut, &foundTriplets);
  }
  
  
  int areAlignedRZ(float r1, float z1, float ro, float zo, const float ptmin, const float thetaCut) const
  {
    float radius_diff = std::abs(r1 - ro);
    float distance_13_squared = radius_diff*radius_diff + (z1 - zo)*(z1 - zo);
    
    float pMin = ptmin*std::sqrt(distance_13_squared); //this needs to be divided by radius_diff later
    
    float tan_12_13_half_mul_distance_13_squared = fabs(z1 * (getInnerR() - ro) + getInnerZ() * (ro - r1) + zo * (r1 - getInnerR())) ;
    return tan_12_13_half_mul_distance_13_squared * pMin <= thetaCut * distance_13_squared * radius_diff;
  }
  
  
  void tagAsOuterNeighbor(unsigned int otherCell)
  {
    theOuterNeighbors.push_back(otherCell);
  }
  
  
  bool haveSimilarCurvature(const CACell & otherCell, const float ptmin,
			    const float region_origin_x, const float region_origin_y, const float region_origin_radius, const float phiCut, const float hardPtCut) const
  {
    
    
    auto x1 = otherCell.getInnerX();
    auto y1 = otherCell.getInnerY();
    
    auto x2 = getInnerX();
    auto y2 = getInnerY();
    
    auto x3 = getOuterX();
    auto y3 = getOuterY();
    
    float distance_13_squared = (x1 - x3)*(x1 - x3) + (y1 - y3)*(y1 - y3);
    float tan_12_13_half_mul_distance_13_squared = std::abs(y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)) ;
    // high pt : just straight
    if(tan_12_13_half_mul_distance_13_squared * ptmin <= 1.0e-4f*distance_13_squared)
      {
	
	float distance_3_beamspot_squared = (x3-region_origin_x) * (x3-region_origin_x) + (y3-region_origin_y) * (y3-region_origin_y);
	
	float dot_bs3_13 = ((x1 - x3)*( region_origin_x - x3) + (y1 - y3) * (region_origin_y-y3));
	float proj_bs3_on_13_squared = dot_bs3_13*dot_bs3_13/distance_13_squared;
	
	float distance_13_beamspot_squared  = distance_3_beamspot_squared -  proj_bs3_on_13_squared;
	
	return distance_13_beamspot_squared < (region_origin_radius+phiCut)*(region_origin_radius+phiCut);
      }
    
    //87 cm/GeV = 1/(3.8T * 0.3)
    
    //take less than radius given by the hardPtCut and reject everything below
    float minRadius = hardPtCut*87.f;  // FIXME move out and use real MagField
    
    auto det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2);
    
    
    auto offset = x2 * x2 + y2*y2;
    
    auto bc = (x1 * x1 + y1 * y1 - offset)*0.5f;
    
    auto cd = (offset - x3 * x3 - y3 * y3)*0.5f;
    
    
    
    auto idet = 1.f / det;
    
    auto x_center = (bc * (y2 - y3) - cd * (y1 - y2)) * idet;
    auto y_center = (cd * (x1 - x2) - bc * (x2 - x3)) * idet;
    
    auto radius = std::sqrt((x2 - x_center)*(x2 - x_center) + (y2 - y_center)*(y2 - y_center));
    
    if(radius < minRadius)  return false;  // hard cut on pt
    
    auto centers_distance_squared = (x_center - region_origin_x)*(x_center - region_origin_x) + (y_center - region_origin_y)*(y_center - region_origin_y);
    auto region_origin_radius_plus_tolerance = region_origin_radius + phiCut;
    auto minimumOfIntersectionRange = (radius - region_origin_radius_plus_tolerance)*(radius - region_origin_radius_plus_tolerance);
    
    if (centers_distance_squared >= minimumOfIntersectionRange) {
      auto maximumOfIntersectionRange = (radius + region_origin_radius_plus_tolerance)*(radius + region_origin_radius_plus_tolerance);
      return centers_distance_squared <= maximumOfIntersectionRange;
    } 
    
    return false;
    
  }
  
  
  // trying to free the track building process from hardcoded layers, leaving the visit of the graph
  // based on the neighborhood connections between cells.
  
  void findNtuplets(CAColl& allCells, std::vector<CAntuplet>& foundNtuplets, CAntuplet& tmpNtuplet, const unsigned int minHitsPerNtuplet) const {
    
    // the building process for a track ends if:
    // it has no outer neighbor
    // it has no compatible neighbor
    // the ntuplets is then saved if the number of hits it contains is greater than a threshold
    
    if (tmpNtuplet.size() == minHitsPerNtuplet - 1)
      {
	foundNtuplets.push_back(tmpNtuplet);
      }
    else
      {
	unsigned int numberOfOuterNeighbors = theOuterNeighbors.size();
	for (unsigned int i = 0; i < numberOfOuterNeighbors; ++i) {
	  tmpNtuplet.push_back((theOuterNeighbors[i]));
	  allCells[theOuterNeighbors[i]].findNtuplets(allCells,foundNtuplets, tmpNtuplet, minHitsPerNtuplet);
	  tmpNtuplet.pop_back();
	}
      }
    
  }
  
  
private:
  
  CAntuple theOuterNeighbors;
  
  const HitDoublets* theDoublets;  
  const int theDoubletId;
  
  const float theInnerR;
  const float theInnerZ;
  
};


#endif // RecoPixelVertexing_PixelTriplets_src_CACell_h
