#ifndef CommonTools_RecoAlgos_RecoTrackViewRefSelector_h
#define CommonTools_RecoAlgos_RecoTrackViewRefSelector_h

#include "CommonTools/RecoAlgos/interface/RecoTrackSelectorBase.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

class RecoTrackViewRefSelector : public RecoTrackSelectorBase {
 public:
  typedef edm::View<reco::Track> collection;
  typedef edm::RefToBaseVector<reco::Track> ref_container;
  typedef ref_container::const_iterator const_ref_iterator;

  /// Constructors
  RecoTrackViewRefSelector() {}

  RecoTrackViewRefSelector ( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) : RecoTrackSelectorBase(cfg, iC) {}

  const_ref_iterator begin() const { return ref_selected_.begin(); }
  const_ref_iterator end() const { return ref_selected_.end(); }

  void select( const edm::Handle<collection>& c, const edm::Event & event, const edm::EventSetup&es) {
    init(event, es);
    ref_selected_.clear();
    for (unsigned int i = 0; i < c->size(); i++) {
      auto trk = c->refAt(i);
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
