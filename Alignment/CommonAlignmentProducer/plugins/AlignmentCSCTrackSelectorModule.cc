
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"

//the selectores used to select the tracks
#include "Alignment/CommonAlignmentProducer/interface/AlignmentCSCTrackSelector.h"
#include "Alignment/CommonAlignmentProducer/interface/AlignmentGlobalTrackSelector.h"
#include "Alignment/CommonAlignmentProducer/interface/AlignmentTwoBodyDecayTrackSelector.h"

// the following include is necessary to clone all track branches
// including recoTrackExtras and TrackingRecHitsOwned.
// if you remove it the code will compile, but the cloned
// tracks have only the recoTracks branch!
#include "CommonTools/RecoAlgos/interface/TrackSelector.h"

struct CSCTrackConfigSelector {

      typedef std::vector<const reco::Track*> container;
      typedef container::const_iterator const_iterator;
      typedef reco::TrackCollection collection;

      CSCTrackConfigSelector( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) : theBaseSelector(cfg) {}

      const_iterator begin() const { return theSelectedTracks.begin(); }
      const_iterator end() const { return theSelectedTracks.end(); }
      size_t size() const { return theSelectedTracks.size(); }

      void select( const edm::Handle<reco::TrackCollection> & c,  const edm::Event & evt, const edm::EventSetup &/*dummy*/)
      {
	 container all;
	 for( reco::TrackCollection::const_iterator i=c.product()->begin();i!=c.product()->end();++i){
	    all.push_back(& * i );
	 }
	 theSelectedTracks = theBaseSelector.select(all, evt); // might add dummy
      }

   private:
      container theSelectedTracks;

      AlignmentCSCTrackSelector theBaseSelector;
};

typedef ObjectSelectorStream<CSCTrackConfigSelector>  AlignmentCSCTrackSelectorModule;

DEFINE_FWK_MODULE( AlignmentCSCTrackSelectorModule );
