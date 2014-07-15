#include "TrackFromParentImporter.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"

namespace {
  class ConversionAdaptor {
  public:
    static bool check_importable(const reco::PFConversionCollection::value_type& t) {
      return (t.pfTracks().size() >= 2);
    }
    static const std::vector<reco::PFRecTrackRef>& 
    get_track_refs(const reco::PFConversionCollection::value_type& t) {
      return t.pfTracks();
    }
    static void set_element_info(reco::PFBlockElement* elem,
				 const edm::Ref<reco::PFConversionCollection>& parref) {
      elem->setConversionRef(parref->originalConversion(),
			     reco::PFBlockElement::T_FROM_GAMMACONV);
    }
  };
}

typedef pflow::importers::TrackFromParentImporter<reco::PFConversionCollection,ConversionAdaptor> ConversionTrackImporter;

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  ConversionTrackImporter, 
		  "ConversionTrackImporter");
