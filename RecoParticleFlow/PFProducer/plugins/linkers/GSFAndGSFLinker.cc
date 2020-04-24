#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class GSFAndGSFLinker : public BlockElementLinkerBase {
public:
  GSFAndGSFLinker(const edm::ParameterSet& conf) :
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
		  GSFAndGSFLinker, 
		  "GSFAndGSFLinker");

double GSFAndGSFLinker::testLink
  ( const reco::PFBlockElement* elem1,
    const reco::PFBlockElement* elem2) const { 
  constexpr reco::PFBlockElement::TrackType T_FROM_GAMMACONV =
    reco::PFBlockElement::T_FROM_GAMMACONV;
  double dist = -1.0;
  const reco::PFBlockElementGsfTrack * gsfelem1 =
    static_cast<const reco::PFBlockElementGsfTrack *>(elem1);
  const reco::PFBlockElementGsfTrack * gsfelem2 = 
    static_cast<const reco::PFBlockElementGsfTrack *>(elem2);
  const reco::GsfPFRecTrackRef& gsfref1 = gsfelem1->GsftrackRefPF();
  const reco::GsfPFRecTrackRef& gsfref2 = gsfelem2->GsftrackRefPF();
  if( gsfref1.isNonnull() && gsfref2.isNonnull() ) {
    if( gsfelem1->trackType(T_FROM_GAMMACONV) != // we want **one** primary GSF 
	gsfelem2->trackType(T_FROM_GAMMACONV) &&
	gsfref1->trackId() == gsfref2->trackId() ) {
      dist = 0.001;
    }	
  }
  return dist;
}
