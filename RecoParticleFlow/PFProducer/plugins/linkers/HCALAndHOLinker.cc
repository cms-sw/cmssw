#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class HCALAndHOLinker : public BlockElementLinkerBase {
public:
  HCALAndHOLinker(const edm::ParameterSet& conf) :
    BlockElementLinkerBase(conf),
    _useKDTree(conf.getParameter<bool>("useKDTree")),
    _debug(conf.getUntrackedParameter<bool>("debug",false)) {}
  
  double testLink 
  ( const reco::PFBlockElement*,
    const reco::PFBlockElement* ) const override;

private:
  bool _useKDTree,_debug;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  HCALAndHOLinker, 
		  "HCALAndHOLinker");

double HCALAndHOLinker::testLink
  ( const reco::PFBlockElement* elem1,
    const reco::PFBlockElement* elem2) const {  
  const reco::PFBlockElementCluster *hcalelem(nullptr), *hoelem(nullptr);
  double dist(-1.0);
  if( elem1->type() < elem2->type() ) {
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    hoelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    hoelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFClusterRef& hcalref = hcalelem->clusterRef();
  const reco::PFClusterRef& horef = hoelem->clusterRef();
  const reco::PFCluster::REPPoint& hcalreppos = hcalref->positionREP();
  if( hcalref.isNull() || horef.isNull() ) {
    throw cms::Exception("BadClusterRefs") 
      << "PFBlockElementCluster's refs are null!";
  }  
  dist = ( std::abs(hcalreppos.Eta()) < 1.5 ?
	   LinkByRecHit::computeDist( hcalreppos.Eta(),
				      hcalreppos.Phi(), 
				      horef->positionREP().Eta(), 
				      horef->positionREP().Phi() )
	   : -1.0 );
  return (dist < 0.2 ? dist : -1.0);
}
