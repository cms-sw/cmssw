
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"

//the selectores used to select the tracks
#include "Calibration/TkAlCaRecoProducers/interface/CalibrationTrackSelector.h"

// the following include is necessary to clone all track branches
// including recoTrackExtras and TrackingRecHitsOwned.
// if you remove it the code will compile, but the cloned
// tracks have only the recoTracks branch!
#include "CommonTools/RecoAlgos/interface/TrackSelector.h"

struct SiStripCalTrackConfigSelector {

  typedef std::vector<const reco::Track*> container;
  typedef container::const_iterator const_iterator;
  typedef reco::TrackCollection collection;

 SiStripCalTrackConfigSelector( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) :
    theBaseSelector(cfg)
  {
    //TODO Wrap the BaseSelector into its own PSet
    theBaseSwitch =
      cfg.getParameter<bool>("applyBasicCuts") ||
      cfg.getParameter<bool>("minHitsPerSubDet") ||
      cfg.getParameter<bool>("applyNHighestPt") ||
      cfg.getParameter<bool>("applyMultiplicityFilter");

  }

  const_iterator begin() const { return theSelectedTracks.begin(); }
  const_iterator end() const { return theSelectedTracks.end(); }
  size_t size() const { return theSelectedTracks.size(); }

  void select( const edm::Handle<reco::TrackCollection> & c,  const edm::Event & evt,
               const edm::EventSetup &/*dummy*/)
  {
    theSelectedTracks.clear();
    for( reco::TrackCollection::const_iterator i=c.product()->begin();i!=c.product()->end();++i){
      theSelectedTracks.push_back(& * i );
    }
    // might add EvetSetup to the select(...) method of the Selectors
    if(theBaseSwitch)
      theSelectedTracks=theBaseSelector.select(theSelectedTracks,evt);
  }

private:
  container theSelectedTracks;

  bool theBaseSwitch;
  CalibrationTrackSelector theBaseSelector;

};

typedef ObjectSelectorStream<SiStripCalTrackConfigSelector>  CalibrationTrackSelectorModule;

DEFINE_FWK_MODULE( CalibrationTrackSelectorModule );
