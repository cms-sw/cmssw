#ifndef RecoSelectors_RecoTrackRefSelector_h
#define RecoSelectors_RecoTrackRefSelector_h
/* \class RecoTrackRefSelector
 *
 * \author Ian Tomalin, RAL
 *
 */
#include "CommonTools/RecoAlgos/interface/RecoTrackSelectorBase.h"

class RecoTrackRefSelector : public RecoTrackSelectorBase {
 public:
  typedef reco::TrackCollection collection;
  typedef reco::TrackRefVector ref_container;
  typedef ref_container::const_iterator const_ref_iterator;

  /// Constructors
  RecoTrackRefSelector() {}

  RecoTrackRefSelector ( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) : RecoTrackSelectorBase(cfg, iC) {}

  const_ref_iterator begin() const { return ref_selected_.begin(); }
  const_ref_iterator end() const { return ref_selected_.end(); }

  void select( const edm::Handle<collection>& c, const edm::Event & event, const edm::EventSetup&es) {
    init(event, es);
    ref_selected_.clear();
    for (unsigned int i = 0; i < c->size(); i++) {
      edm::Ref<collection> trk(c, i);
      if ( operator()(*trk) ) {
	ref_selected_.push_back( trk );
      }
    }
  }

  size_t size() const { return ref_selected_.size(); }

 private:
  ref_container ref_selected_;
};

#endif
