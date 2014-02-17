/** \class AlignmentMuonSelectorModule
 *
 * selects a subset of a muon collection and clones
 * Track, TrackExtra parts and RecHits collection
 * for SA, GB and Tracker Only options
 * 
 * \author Javier Fernandez, IFCA
 *
 * \version $Revision: 1.4 $
 *
 * $Id: AlignmentMuonSelectorModule.cc,v 1.4 2009/03/09 23:00:27 flucke Exp $
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
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

  void select( const edm::Handle<reco::MuonCollection> & c,  const edm::Event & evt, const edm::EventSetup &/* dummy*/)
  {
    all_.clear();
    selected_.clear();
    for (collection::const_iterator i = c.product()->begin(), iE = c.product()->end();
         i != iE; ++i){
      all_.push_back(& * i );
    }
    selected_ = theSelector.select(all_, evt); // might add dummy 
  }

private:
  container all_,selected_;
  AlignmentMuonSelector theSelector;
};

typedef ObjectSelector<MuonConfigSelector>  AlignmentMuonSelectorModule;

DEFINE_FWK_MODULE( AlignmentMuonSelectorModule );

