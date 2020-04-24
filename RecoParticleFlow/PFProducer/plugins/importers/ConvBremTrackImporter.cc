#include "TrackFromParentImporter.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"

namespace {
  class ConvBremAdaptor {
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
}

typedef pflow::importers::TrackFromParentImporter<reco::GsfPFRecTrackCollection,ConvBremAdaptor> ConvBremTrackImporter;

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  ConvBremTrackImporter, 
		  "ConvBremTrackImporter");
