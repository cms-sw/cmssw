#ifndef KDTreeLinkerTools_h
#define KDTreeLinkerTools_h

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include <map>
#include <set>

using BlockEltSet = std::set<reco::PFBlockElement *>;
using RecHitSet = std::set<const reco::PFRecHit *>;

using RecHit2BlockEltMap = std::map<const reco::PFRecHit *, BlockEltSet>;
using BlockElt2BlockEltMap = std::map<reco::PFBlockElement *, BlockEltSet>;

// Box structure used to define 2D field.
// It's used in KDTree building step to divide the detector
// space (ECAL, HCAL...) and in searching step to create a bounding
// box around the demanded point (Track collision point, PS projection...).
struct KDTreeBox {
  double dim1min = 0.0;
  double dim1max = 0.0;
  double dim2min = 0.0;
  double dim2max = 0.0;
};

// Data stored in each KDTree node.
// The dim1/dim2 fields are usually the duplication of some PFRecHit values
// (eta/phi or x/y). But in some situations, phi field is shifted by +-2.Pi
struct KDTreeNodeInfo {
  const reco::PFRecHit *ptr = nullptr;
  double dim1;
  double dim2;
};

// KDTree node.
struct KDTreeNode {
  KDTreeNodeInfo rh;            // data
  KDTreeNode *left = nullptr;   // left son
  KDTreeNode *right = nullptr;  // right son
  KDTreeBox region;             // Region bounding box
};

#endif
