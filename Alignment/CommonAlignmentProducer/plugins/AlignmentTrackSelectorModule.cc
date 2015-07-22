
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"

//the selectores used to select the tracks
#include "Alignment/CommonAlignmentProducer/interface/AlignmentTrackSelector.h"
#include "Alignment/CommonAlignmentProducer/interface/AlignmentGlobalTrackSelector.h"
#include "Alignment/CommonAlignmentProducer/interface/AlignmentTwoBodyDecayTrackSelector.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

// the following include is necessary to clone all track branches
// including recoTrackExtras and TrackingRecHitsOwned (in future also "owned clusters"?).
// if you remove it the code will compile, but the cloned
// tracks have only the recoTracks branch!
#include "CommonTools/RecoAlgos/interface/TrackSelector.h"

struct TrackConfigSelector {

  typedef std::vector<const reco::Track*> container;
  typedef container::const_iterator const_iterator;
  typedef reco::TrackCollection collection;

 TrackConfigSelector( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) :
   theBaseSelector(cfg, iC),
   theGlobalSelector(cfg.getParameter<edm::ParameterSet>("GlobalSelector"), iC),
   theTwoBodyDecaySelector(cfg.getParameter<edm::ParameterSet>("TwoBodyDecaySelector"), iC)
  {
    //TODO Wrap the BaseSelector into its own PSet
    theBaseSwitch = theBaseSelector.useThisFilter();

    theGlobalSwitch =  theGlobalSelector.useThisFilter();

    theTwoBodyDecaySwitch = theTwoBodyDecaySelector.useThisFilter();
  }

  const_iterator begin() const { return theSelectedTracks.begin(); }
  const_iterator end() const { return theSelectedTracks.end(); }
  size_t size() const { return theSelectedTracks.size(); }

  void select( const edm::Handle<reco::TrackCollection> & c,  const edm::Event & evt,
               const edm::EventSetup& eSetup)
  {
    theSelectedTracks.clear();
    for( reco::TrackCollection::const_iterator i=c.product()->begin();i!=c.product()->end();++i){
      theSelectedTracks.push_back(& * i );
    }
    // might add EvetSetup to the select(...) method of the Selectors
    if(theBaseSwitch)
      theSelectedTracks=theBaseSelector.select(theSelectedTracks,evt,eSetup);
    if(theGlobalSwitch)
      theSelectedTracks=theGlobalSelector.select(theSelectedTracks,evt,eSetup);
    if(theTwoBodyDecaySwitch)
      theSelectedTracks=theTwoBodyDecaySelector.select(theSelectedTracks,evt,eSetup);
  }

private:
  container theSelectedTracks;

  bool theBaseSwitch, theGlobalSwitch, theTwoBodyDecaySwitch;
  AlignmentTrackSelector theBaseSelector;
  AlignmentGlobalTrackSelector theGlobalSelector;
  AlignmentTwoBodyDecayTrackSelector theTwoBodyDecaySelector;

};

typedef ObjectSelectorStream<TrackConfigSelector>  AlignmentTrackSelectorModule;

DEFINE_FWK_MODULE( AlignmentTrackSelectorModule );
