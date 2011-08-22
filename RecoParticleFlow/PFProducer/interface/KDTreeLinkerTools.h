#ifndef KDTreeLinkerTools_h
#define KDTreeLinkerTools_h

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include <map>
#include <set>

namespace KDTreeLinker
{
  typedef std::set<reco::PFBlockElement*>			BlockEltSet; 
  typedef std::set<const reco::PFBlockElement*>			BlockEltSet_const; 

  typedef std::set<const reco::PFRecHit*>			RecHitSet;

  typedef std::map<const reco::PFRecHit*, BlockEltSet_const>	RecHitClusterMap;
  typedef std::map<reco::PFBlockElement*, BlockEltSet_const>	BlockEltClusterMap;

  // Box structure used to define 2D field with eta/phi axis.
  // It's used in KDTree building step to divide the ECAL 
  // eta/phi space and in searching step to create a bounding
  // box around the demanded point.
  struct TBox
  {
    double etamin, etamax;
    double phimin, phimax;
    
  public:
    TBox(double emin, double emax, 
	 double pmin, double pmax)
      : etamin (emin), etamax(emax)
      , phimin (pmin), phimax(pmax)
    {}

    TBox()
      : etamin (0), etamax(0)
      , phimin (0), phimax(0)
    {}

  };

  
  // Rechit data stored in each KDTree node.
  // The eta/phi field duplicate the recHit eta/phi. But in some
  // situations, phi field is shifted by +-2.Pi
  struct RHinfo 
  {
    const reco::PFRecHit *ptr;
    double eta;
    double phi;

  public:
    RHinfo()
      : ptr(0)
    {}

    RHinfo(const reco::PFRecHit	*rhptr,
	   double		e,
	   double		p)
      : ptr(rhptr), eta(e), phi(p)
    {}

  };


  // KDTree node.
  struct TNode
  {
    // Data
    RHinfo rh;

    // Right/left sons.
    TNode *left, *right;

    // Region bounding box.
    TBox region;

  public:
    TNode()
      : left(0), right(0)
    {}

    void setAttributs(const TBox&   regionBox,
		      const RHinfo& rhinfo)
    {
      rh = rhinfo;
      region = regionBox;
    }

    void setAttributs(const TBox&   regionBox)
    {
      region = regionBox;
    }
  };
}


#endif
