
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//-----------------------------------------------------------------------------

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentProducer.h"

//-----------------------------------------------------------------------------

#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentTrackSelector.h"
#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"

struct TrackConfigSelector {

  typedef std::vector<const reco::Track*> container;
  typedef container::const_iterator const_iterator;
  typedef reco::TrackCollection collection; 

  TrackConfigSelector( const edm::ParameterSet & cfg ) :
    theSelector(cfg) {}

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  bool empty() const { return selected_.empty(); }

  void select( const reco::TrackCollection & c,  const edm::Event & evt) {
    all_.clear();
    selected_.clear();
    for( reco::TrackCollection::const_iterator i=c.begin();i!=c.end();++i){
      all_.push_back(& * i );
    }
    selected_=theSelector.select(all_,evt);
  }

private:
  container all_,selected_;
  AlignmentTrackSelector theSelector;
};

typedef ObjectSelector<TrackConfigSelector>  AlignmentTrackSelectorModule;

//-----------------------------------------------------------------------------

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_LOOPER( AlignmentProducer );
DEFINE_ANOTHER_FWK_MODULE( AlignmentTrackSelectorModule );
