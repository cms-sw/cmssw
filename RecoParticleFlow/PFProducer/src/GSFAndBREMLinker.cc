#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class GSFAndBremLinker : public BlockElementLinkerBase {
public:
  GSFAndBremLinker(const edm::ParameterSet& conf) :
    BlockElementLinkerBase(conf),
    _useKDTree(conf.getParameter<bool>("useKDTree")),
    _useConvertedBrems(conf.getParameter<bool>("useConvertedBrems")),
    _debug(conf.getUntrackedParameter<bool>("debug",false)) {}
  
  double operator() 
  ( const std::unique_ptr<reco::PFBlockElement>&,
    const std::unique_ptr<reco::PFBlockElement>& ) const override;

private:
  bool _useKDTree,_useConvertedBrems,_debug;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  GSFAndBremLinker, 
		  "GSFAndBremLinker");

double GSFAndBremLinker::operator()
  ( const std::unique_ptr<reco::PFBlockElement>& elem1,
    const std::unique_ptr<reco::PFBlockElement>& elem2) const {   
  double dist = -1.0;
  const reco::PFBlockElementGsfTrack * gsfelem(NULL);
  const reco::PFBlockElementBrem * bremelem(NULL);
  if( elem1->type() < elem2->type() ) {
    gsfelem = static_cast<const reco::PFBlockElementGsfTrack *>(elem1.get());
    bremelem = static_cast<const reco::PFBlockElementBrem *>(elem2.get());
  } else {
    gsfelem = static_cast<const reco::PFBlockElementGsfTrack *>(elem2.get());
    bremelem = static_cast<const reco::PFBlockElementBrem *>(elem1.get());
  }  
  const reco::GsfPFRecTrackRef& gsfref = gsfelem->GsftrackRefPF();
  const reco::GsfPFRecTrackRef& bremref = bremelem->GsftrackRefPF();
  if( gsfref.isNonnull() && bremref.isNonnull() && gsfref == bremref ) {
    dist = 0.001;
  }
  return dist;
}
