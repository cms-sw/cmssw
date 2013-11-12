#ifndef RecoSelectors_RecoTrackRefSelector_h
#define RecoSelectors_RecoTrackRefSelector_h
/* \class RecoTrackRefSelector
 *
 * \author Ian Tomalin, RAL
 *
 *  $Date: 2009/10/13 12:07:49 $
 *  $Revision: 1.1 $
 *
 */
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/RecoAlgos/interface/RecoTrackSelector.h"

class RecoTrackRefSelector : public RecoTrackSelector {

  // Would be better for RecoTrackRefSelector and RecoTrackSelector to inherit from common base class,
  // since this doesn't needed the "selected_" data member from the base class.
  // To do in future ...

 public:
  typedef reco::TrackRefVector ref_container;
  typedef ref_container::const_iterator const_ref_iterator;

  /// Constructors
  RecoTrackRefSelector() {}

  RecoTrackRefSelector ( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) : RecoTrackSelector(cfg, iC) {}

  RecoTrackRefSelector ( double ptMin, double minRapidity, double maxRapidity,
		         double tip, double lip, int minHit, int min3DHit, double maxChi2,
    		         const std::vector<std::string>& quality , const std::vector<std::string>& algorithm ) :
          RecoTrackSelector ( ptMin, minRapidity, maxRapidity,
   		              tip, lip, minHit, min3DHit, maxChi2,
                              quality , algorithm ) {}

  const_ref_iterator begin() const { return ref_selected_.begin(); }
  const_ref_iterator end() const { return ref_selected_.end(); }

  void select( const edm::Handle<collection>& c, const edm::Event & event, const edm::EventSetup&) {
    ref_selected_.clear();
    edm::Handle<reco::BeamSpot> beamSpot;
    event.getByToken(bsSrcToken_,beamSpot);
    for (unsigned int i = 0; i < c->size(); i++) {

      edm::Ref<collection> trk(c, i);

      if ( operator()(*trk,beamSpot.product()) ) {
	ref_selected_.push_back( trk );
      }
    }
  }

  size_t size() const { return ref_selected_.size(); }

 private:
  ref_container ref_selected_;
};

#endif
