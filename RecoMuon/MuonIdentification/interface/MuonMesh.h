#ifndef _MuonIdentification_MuonMesh_h_
#define _MuonIdentification_MuonMesh_h_
//
// Creates a mesh of muons connected by overlapping segments
// Original author: Lindsey Gray
//

#include <vector>
#include <utility>
#include <map>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"

class CSCGeometry;

class MuonMesh {

  typedef std::map<reco::Muon*,
                   std::vector<std::pair<reco::Muon*,
                                         std::pair<reco::MuonChamberMatch*,
                                                   reco::MuonSegmentMatch*
                                                  >
                                        >
                              >
                  > MeshType;

  typedef std::vector<std::pair<reco::Muon*,
                                std::pair<reco::MuonChamberMatch*,
                                          reco::MuonSegmentMatch*
                                         >
                               >
                     > AssociationType;

 public:
  
  MuonMesh(const edm::ParameterSet&);
  
  void runMesh(std::vector<reco::Muon>* p) {fillMesh(p); pruneMesh();}

  void clearMesh() { mesh_.clear(); }

  void setCSCGeometry(const CSCGeometry* pg) { geometry_ = pg; } 

  bool isDuplicateOf(const CSCSegmentRef& lhs, const CSCSegmentRef& rhs) const;
  bool isDuplicateOf(const std::pair<CSCDetId,CSCSegmentRef>& rhs, 
		     const std::pair<CSCDetId,CSCSegmentRef>& lhs) const;
  bool isClusteredWith(const std::pair<CSCDetId,CSCSegmentRef>& lhs, 
		       const std::pair<CSCDetId,CSCSegmentRef>& rhs) const;

 private:

  void fillMesh(std::vector<reco::Muon>*);

  void pruneMesh();

  
  // implement to remove cases where two segments in the same
  // chamber overlap within 2 sigma of ALL of their errors
  bool withinTwoSigma(const std::pair<CSCDetId,CSCSegmentRef>& rhs, 
		      const std::pair<CSCDetId,CSCSegmentRef>& lhs) const { return false; }
  
  
  
  

  MeshType mesh_;

  // geometry cache for segment arbitration
   const CSCGeometry* geometry_;
   
   // do various cleanings?
   const bool doME1a, doOverlaps, doClustering;
   // overlap and clustering parameters
   const double OverlapDPhi, OverlapDTheta, ClusterDPhi, ClusterDTheta;
};

#endif
