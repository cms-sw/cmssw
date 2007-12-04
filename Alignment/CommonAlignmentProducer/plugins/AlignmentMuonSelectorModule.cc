/** \class AlignmentMuonSelectorModule
 *
 * selects a subset of a muon collection and clones
 * Track, TrackExtra parts and RecHits collection
 * for SA, GB and Tracker Only options
 * 
 * \author Javier Fernandez, IFCA
 *
 * \version $Revision: 1.1 $
 *
 * $Id: AlignmentMuonSelectorModule.cc,v 1.1 2007/05/02 11:57:00 fronga Exp $
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "Alignment/CommonAlignmentProducer/interface/AlignmentMuonSelector.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 

// the following include is necessary to clone all track branches
// including recoTrackExtras and TrackingRecHitsOwned.
// if you remove it the code will compile, but the cloned
// tracks have only the recoMuons branch!

struct MuonConfigSelector {

  typedef std::vector<const reco::Muon*> container;
  typedef container::const_iterator const_iterator;
  typedef reco::MuonCollection collection; 

  MuonConfigSelector( const edm::ParameterSet & cfg ) :
    theSelector(cfg) {}

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  size_t size() const { return selected_.size(); }

  void select( const edm::Handle<reco::MuonCollection> & c,  const edm::Event & evt) {
    all_.clear();
    selected_.clear();
    for( reco::MuonCollection::const_iterator i=c.product()->begin();i!=c.product()->end();++i){
      all_.push_back(& * i );
    }
    selected_=theSelector.select(all_,evt);
  }

private:
  container all_,selected_;
  AlignmentMuonSelector theSelector;
};

typedef ObjectSelector<MuonConfigSelector>  AlignmentMuonSelectorModule;

DEFINE_FWK_MODULE( AlignmentMuonSelectorModule );

