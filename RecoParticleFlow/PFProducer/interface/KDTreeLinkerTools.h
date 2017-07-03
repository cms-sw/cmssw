#ifndef KDTreeLinkerTools_h
#define KDTreeLinkerTools_h

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include <map>
#include <set>

typedef std::set<reco::PFBlockElement*>				BlockEltSet; 
typedef std::set<const reco::PFRecHit*>				RecHitSet;

typedef std::map<const reco::PFRecHit*, BlockEltSet>		RecHit2BlockEltMap;
typedef std::map<reco::PFBlockElement*, BlockEltSet>		BlockElt2BlockEltMap;


// Box structure used to define 2D field.
// It's used in KDTree building step to divide the detector
// space (ECAL, HCAL...) and in searching step to create a bounding
// box around the demanded point (Track collision point, PS projection...).
struct KDTreeBox
{
  double dim1min, dim1max;
  double dim2min, dim2max;
  
  public:

  KDTreeBox(double d1min, double d1max, 
	    double d2min, double d2max)
    : dim1min (d1min), dim1max(d1max)
    , dim2min (d2min), dim2max(d2max)
  {}

  KDTreeBox()
    : dim1min (0), dim1max(0)
    , dim2min (0), dim2max(0)
  {}
};

  
// Data stored in each KDTree node.
// The dim1/dim2 fields are usually the duplication of some PFRecHit values
// (eta/phi or x/y). But in some situations, phi field is shifted by +-2.Pi
struct KDTreeNodeInfo 
{
  const reco::PFRecHit *ptr;
  double dim1;
  double dim2;
  
  public:
  KDTreeNodeInfo()
    : ptr(nullptr)
  {}
  
  KDTreeNodeInfo(const reco::PFRecHit	*rhptr,
		 double			d1,
		 double			d2)
    : ptr(rhptr), dim1(d1), dim2(d2)
  {}  
};


// KDTree node.
struct KDTreeNode
{
  // Data
  KDTreeNodeInfo rh;
  
  // Right/left sons.
  KDTreeNode *left, *right;
  
  // Region bounding box.
  KDTreeBox region;
  
  public:
  KDTreeNode()
    : left(nullptr), right(nullptr)
  {}
  
  void setAttributs(const KDTreeBox&		regionBox,
		    const KDTreeNodeInfo&	rhinfo) 
  {
    rh = rhinfo;
    region = regionBox;
  }
  
  void setAttributs(const KDTreeBox&   regionBox) 
  {
    region = regionBox;
  }
};



#endif
