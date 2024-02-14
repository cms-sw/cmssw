#ifndef CommonTools_UtilAlgos_CollectionCountEventSelector_h
#define CommonTools_UtilAlgos_CollectionCountEventSelector_h

/** \class CollectionCountEventSelector
 *
 * Selects an event if a collection has at least N entries
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: CollectionCountEventSelector.h,v 1.2 2010/02/20 20:55:24 wmtan Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/UtilAlgos/interface/CollectionInCollectionFilterTrait.h"
#include "CommonTools/UtilAlgos/interface/EventSelectorBase.h"

template <typename C,
          typename S = AnySelector,
          typename N = MinNumberSelector,
          typename CS = typename helper::CollectionInCollectionFilterTrait<C, S, N>::type>
class CollectionCountEventSelector : public EventSelectorBase {
public:
  /// constructor
  explicit CollectionCountEventSelector(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : srcToken_(iC.consumes<C>(cfg.template getParameter<edm::InputTag>("src"))),
        select_(reco::modules::make<S>(cfg, iC)),
        sizeSelect_(reco::modules::make<N>(cfg, iC)) {}

  static void fillPSetDescription(edm::ParameterSetDescription& desc) {
    desc.add<edm::InputTag>("src", edm::InputTag());
    reco::modules::fillPSetDescription<S>(desc);
    reco::modules::fillPSetDescription<N>(desc);
  }

  bool operator()(edm::Event& evt, const edm::EventSetup&) const override {
    edm::Handle<C> source;
    evt.getByToken(srcToken_, source);
    return CS::filter(*source, select_, sizeSelect_);
  }

private:
  /// source collection label
  edm::EDGetTokenT<C> srcToken_;

  /// object filter
  S select_;

  /// minimum number of entries in a collection
  N sizeSelect_;
};

#endif
