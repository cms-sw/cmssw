#include "TrackFromParentImporter.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"

template<>
class ParentCollectionAdaptor<reco::PFV0Collection> {
public:
  static bool check_importable(const reco::PFV0Collection::value_type& t) {
    return true;
  }
  static const std::vector<reco::PFRecTrackRef>& 
  get_track_refs(const reco::PFV0Collection::value_type& t) {
    return t.pfTracks();
  }
  static void set_element_info(reco::PFBlockElement* elem,
			     const edm::Ref<reco::PFV0Collection>& parref) {
    elem->setV0Ref(parref->originalV0(), reco::PFBlockElement::T_FROM_V0);
  }
};

typedef TrackFromParentImporter<reco::PFV0Collection> 
V0TrackImporter;

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  V0TrackImporter, 
		  "V0TrackImporter");
