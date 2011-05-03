#ifndef Analysis_AnalysisFilters_interface_PVSelector_h
#define Analysis_AnalysisFilters_interface_PVSelector_h

#include "FWCore/Common/interface/EventBase.h"
#include "DataFormats/Common/interface/Handle.h"

#include "PhysicsTools/SelectorUtils/interface/EventSelector.h"
#include "PhysicsTools/SelectorUtils/interface/PVObjectSelector.h"


// make a selector for this selection
class PVSelector : public Selector<edm::EventBase> {
public:

  PVSelector() {}

 PVSelector( edm::ParameterSet const & params ) :
  pvSrc_ (params.getParameter<edm::InputTag>("pvSrc") ),
  pvSel_ (params) 
  {
    retInternal_ = getBitTemplate();
  }
  
  bool operator() ( edm::EventBase const & event,  pat::strbitset & ret ) {
    event.getByLabel(pvSrc_, h_primVtx);

    // check if there is a good primary vertex

    if ( h_primVtx->size() < 1 ) return false;

    reco::Vertex const & pv = h_primVtx->at(0);

    return pvSel_( pv );
  }

  using EventSelector::operator();

  edm::Handle<std::vector<reco::Vertex> > const & vertices() const { return h_primVtx; }

private:
  edm::InputTag                           pvSrc_;
  PVObjectSelector                        pvSel_;
  edm::Handle<std::vector<reco::Vertex> > h_primVtx;
};

#endif
