#include "TrackFromParentImporter.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"

template<>
class ParentCollectionAdaptor<reco::GsfPFRecTrackCollection> {
public:
  static bool check_importable(const reco::GsfPFRecTrackCollection::value_type& t) {
    return true;
  }
  static const std::vector<reco::PFRecTrackRef>& 
  get_track_refs(const reco::GsfPFRecTrackCollection::value_type& t) {
    return t.convBremPFRecTrackRef();
  }
  static void set_element_info(reco::PFBlockElement* elem,
			     const edm::Ref<reco::GsfPFRecTrackCollection>& parref) {
    elem->setTrackType(reco::PFBlockElement::T_FROM_GAMMACONV, true);
  }
};

typedef TrackFromParentImporter<reco::GsfPFRecTrackCollection> 
ConvBremTrackImporter;

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  ConvBremTrackImporter, 
		  "ConvBremTrackImporter");
