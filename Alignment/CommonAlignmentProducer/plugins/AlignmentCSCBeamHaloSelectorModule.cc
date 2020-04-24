
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"
#include "Alignment/CommonAlignmentProducer/interface/AlignmentCSCBeamHaloSelector.h"

// the following include is necessary to clone all track branches
// including recoTrackExtras and TrackingRecHitsOwned.
// if you remove it the code will compile, but the cloned
// tracks have only the recoTracks branch!
#include "CommonTools/RecoAlgos/interface/TrackSelector.h"

struct CSCBeamHaloConfigSelector {

  typedef std::vector<const reco::Track*> container;
  typedef container::const_iterator const_iterator;
  typedef reco::TrackCollection collection;

  CSCBeamHaloConfigSelector( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) :
    CSCBeamHaloConfigSelector(cfg, iC) {}
  CSCBeamHaloConfigSelector( const edm::ParameterSet & cfg, edm::ConsumesCollector & iC ) :
    theSelector(cfg, iC) {}

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  size_t size() const { return selected_.size(); }

  void select( const edm::Handle<reco::TrackCollection> & c,  const edm::Event & evt,
               const edm::EventSetup &/*dummy*/)
  {
    all_.clear();
    selected_.clear();
    for (collection::const_iterator i = c.product()->begin(), iE = c.product()->end();
         i != iE; ++i){
      all_.push_back(& * i );
    }
    selected_ = theSelector.select(all_, evt); // might add dummy...
  }

private:
  container all_,selected_;
  AlignmentCSCBeamHaloSelector theSelector;
};

typedef ObjectSelectorStream<CSCBeamHaloConfigSelector>  AlignmentCSCBeamHaloSelectorModule;

DEFINE_FWK_MODULE( AlignmentCSCBeamHaloSelectorModule );
