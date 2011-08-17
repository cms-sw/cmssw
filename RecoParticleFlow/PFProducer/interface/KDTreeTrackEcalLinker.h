#ifndef KDTreeTrackEcalLinker_h
#define KDTreeTrackEcalLinker_h

//#include "TMath.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include <list>
#include <vector>
#include <map>
#include <set>

#include <sys/time.h>


namespace KDTreeLinker
{
  
  // struct for storing info about a rechit
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

  };

  // KDTree node
  struct TNode
  {
    const RHinfo rh;
    TNode *left, *right;
    const TBox region;

  public:
    TNode(const TBox&   regionBox,
	  const RHinfo& rhinfo)
      : rh(rhinfo), left(0), right(0), region(regionBox)
    {}

    TNode(const TBox&   regionBox)
      : left(0), right(0), region(regionBox)
    {}

  };



  struct PFBlockElement_ptr_cmp 
  {
    bool operator() (const reco::PFBlockElement *lhs, const reco::PFBlockElement *rhs) 
      const { 
      return lhs < rhs;
    }
  };

  struct PFRecHit_ptr_cmp
  {
    bool operator() (const reco::PFRecHit *lhs, const reco::PFRecHit *rhs)
      const {
      return lhs < rhs;
    }
  };

  struct twicecall_cmp 
  {
    bool operator() (const std::vector<double>& lhs, const std::vector<double>&rhs) 
      const { 
      for (size_t i = 0; i < 4; ++i)
	if (lhs[i] >= rhs[i])
	  return false;

      return true;
    }
  };


  class KDTree
  {
  public:
    KDTree();
    ~KDTree();

    void build(std::vector<RHinfo>	&eltList,
	       const TBox		&region);
    
    void search(const TBox		&trackBox,
		std::vector<RHinfo>	&recHits);

    void clear();
    
  private:
    TNode*	root_;

  private:
    void swap(RHinfo &e1, RHinfo &e2);
    
    int medianSearch(std::vector<RHinfo>	&eltList,
		     int			low,
		     int			high,
		     int			treeDepth);

    TNode *recBuild(std::vector<RHinfo> &eltList, 
		    int			low,
		    int			hight,
		    int			depth,
		    const TBox&		region);

    void recSearch(const TNode		*current,
		   const TBox		&trackBox,
		   std::vector<RHinfo>	&recHits);    


    void addSubtree(const TNode		*current, 
		    std::vector<RHinfo> &recHits);

    void clearTree(TNode *&current);

    TNode *getRoot();
  };

  typedef std::set<reco::PFBlockElement*, PFBlockElement_ptr_cmp >		BlockEltSetNC; 
  typedef std::set<const reco::PFBlockElement*, PFBlockElement_ptr_cmp >	BlockEltSet; 
  typedef std::set<const reco::PFRecHit*, PFRecHit_ptr_cmp >			RecHitSet;
  typedef std::map<const reco::PFRecHit*, BlockEltSet, PFRecHit_ptr_cmp >	RecHitClusterMap;
  typedef std::map<reco::PFBlockElement*, BlockEltSet, PFBlockElement_ptr_cmp > BlockEltClusterMap;
  typedef std::pair<const reco::PFBlockElement*, BlockEltSet>			TLink;

  class TrackEcalLinker
  {
  public:

    TrackEcalLinker(double phiOffset = 0.25,
		    double ecalDiameter = 0.04,
		    bool debug = false);
    ~TrackEcalLinker();

    void setPhiOffset(double phiOffset);
    void setEcalDiameter(double ecalDiameter);
    double getPhiOffset() const;
    double getEcalDiameter() const;

    void setDebug(bool isDebug);

    void buildTree();
    void searchLinks();
    void clear();
    
    void insertTrack(reco::PFBlockElement* track);
    void insertCluster(const reco::PFBlockElement* cluster,
		       const std::vector<reco::PFRecHitFraction> &fraction);

    bool isCorrectTrack(reco::PFBlockElement* track) const;
    bool isEcalCluster(const reco::PFBlockElement* cluster) const;
    bool isLinked(reco::PFBlockElement* track,
		  const reco::PFBlockElement* cluster) const;

    void updateTracksWithLinks();

    void printTrackLinks(reco::PFBlockElement* track);

  private:
    double		phiOffset_;
    double		ecalDiameter_;
    bool		debug_;

    BlockEltSetNC	tracksSet_;
    BlockEltSet		clustersSet_;
    RecHitSet		rechitsSet_;

    RecHitClusterMap	rhClustersLinks_;
    BlockEltClusterMap	trackClusterLinks_;
  
    KDTree		tree_;
  };


}


#endif /* !KDTreeTrackEcalLinker_h */
