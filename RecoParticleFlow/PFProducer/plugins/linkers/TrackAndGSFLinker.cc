#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class TrackAndGSFLinker : public BlockElementLinkerBase {
public:
  TrackAndGSFLinker(const edm::ParameterSet& conf) :
    BlockElementLinkerBase(conf),
    _useKDTree(conf.getParameter<bool>("useKDTree")),
    _useConvertedBrems(conf.getParameter<bool>("useConvertedBrems")),
    _debug(conf.getUntrackedParameter<bool>("debug",false)) {}
  
  double testLink 
  ( const reco::PFBlockElement*,
    const reco::PFBlockElement* ) const override;

private:
  bool _useKDTree,_useConvertedBrems,_debug;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  TrackAndGSFLinker, 
		  "TrackAndGSFLinker");

double TrackAndGSFLinker::testLink
  ( const reco::PFBlockElement* elem1,
    const reco::PFBlockElement* elem2) const { 
  constexpr reco::PFBlockElement::TrackType T_FROM_GAMMACONV =
    reco::PFBlockElement::T_FROM_GAMMACONV;
  double dist = -1.0;
  const reco::PFBlockElementGsfTrack * gsfelem(nullptr);
  const reco::PFBlockElementTrack * tkelem(nullptr);
  if( elem1->type() < elem2->type() ) {
    tkelem = static_cast<const reco::PFBlockElementTrack *>(elem1);
    gsfelem = static_cast<const reco::PFBlockElementGsfTrack *>(elem2);
  } else {
    tkelem = static_cast<const reco::PFBlockElementTrack *>(elem2);
    gsfelem = static_cast<const reco::PFBlockElementGsfTrack *>(elem1);
  }  
  
  const reco::PFRecTrackRef& trackref = tkelem->trackRefPF();  	  
  const reco::GsfPFRecTrackRef& gsfref = gsfelem->GsftrackRefPF();
  const reco::TrackRef& kftrackref= trackref->trackRef();
  const reco::TrackBaseRef kftrackrefbase(kftrackref);
  const reco::PFRecTrackRef& refkf = gsfref->kfPFRecTrackRef();
  if(refkf.isNonnull()) {
    const reco::TrackRef& gsftrackref = refkf->trackRef();
    if ( gsftrackref.isNonnull() && kftrackref.isNonnull() &&
	 kftrackref == gsftrackref ) {      
	dist = 0.001;
    }
  }
  
  //override for converted brems
  if(_useConvertedBrems) {
    if(tkelem->isLinkedToDisplacedVertex()){
      const std::vector<reco::PFRecTrackRef>& convbrems = 
	gsfref->convBremPFRecTrackRef();
      for( const auto& convbrem : convbrems ) {
	if( tkelem->trackType(T_FROM_GAMMACONV) &&
	    kftrackref == convbrem->trackRef() ) {
	  dist = 0.001;
	} else { // check the base ref as well (for dedicated conversions?)
	  const reco::TrackBaseRef convbrembase(convbrem->trackRef());
	  if( convbrembase == kftrackrefbase ) {
	    dist = 0.001;
	  }
	}
      }
    }
  }  
  return dist;
}
