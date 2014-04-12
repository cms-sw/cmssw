#ifndef CommonTools_UtilAlgos_OverlapExclusionSelector_h
#define CommonTools_UtilAlgos_OverlapExclusionSelector_h
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"

namespace edm { class EventSetup; }

template<typename C, typename T, typename O>
class OverlapExclusionSelector {
public:
  OverlapExclusionSelector(const edm::ParameterSet& cfg, edm::ConsumesCollector && iC) :
    OverlapExclusionSelector(cfg, iC) {};
  OverlapExclusionSelector(const edm::ParameterSet& cfg, edm::ConsumesCollector & iC);
  void newEvent(const edm::Event&, const edm::EventSetup&) const;
  bool operator()(const T&) const;
private:
  edm::EDGetTokenT<C> srcToken_;
  mutable typename C::const_iterator begin_, end_;
  O overlap_;
};

template<typename C, typename T, typename O>
OverlapExclusionSelector<C, T, O>::OverlapExclusionSelector(const edm::ParameterSet& cfg, edm::ConsumesCollector & iC) :
  srcToken_(iC.consumes<C>(cfg.template getParameter<edm::InputTag>("overlap"))),
  overlap_(cfg) {
}

template<typename C, typename T, typename O>
       void OverlapExclusionSelector<C, T, O>::newEvent(const edm::Event& evt, const edm::EventSetup&) const {
  edm::Handle<C> h;
  evt.getByToken(srcToken_, h);
  begin_ = h->begin();
  end_ = h->end();
}

template<typename C, typename T, typename O>
bool OverlapExclusionSelector<C, T, O>::operator()(const T& t) const {
  bool noOverlap = true;
  for(typename C::const_iterator i = begin_; i != end_; ++i) {
    if(overlap_(*i, t)) {
      noOverlap = false;
      break;
    }
  }
  return noOverlap;
}

#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"

EVENTSETUP_STD_INIT_T3(OverlapExclusionSelector);

#endif
