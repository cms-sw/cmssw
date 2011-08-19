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
    push_back("NPV", params.getParameter<int>("NPV") );
    set("NPV");
    retInternal_ = getBitTemplate();
    indexNPV_ = index_type(&bits_, "NPV");
  }
  
  bool operator() ( edm::EventBase const & event,  pat::strbitset & ret ) {
    event.getByLabel(pvSrc_, h_primVtx);

    // check if there is a good primary vertex

    if ( h_primVtx->size() < 1 ) return false;


    int npv = 0;
    for ( std::vector<reco::Vertex>::const_iterator ibegin = h_primVtx->begin(),
	    iend = h_primVtx->end(), i = ibegin; i != iend; ++i ) {
      reco::Vertex const & pv = *i;
      if ( pvSel_( pv ) ) {
	++npv;
      }
    }

    return npv >= cut(indexNPV_, int() );
  }

  using EventSelector::operator();

  edm::Handle<std::vector<reco::Vertex> > const & vertices() const { return h_primVtx; }

private:
  edm::InputTag                           pvSrc_;
  PVObjectSelector                        pvSel_;
  edm::Handle<std::vector<reco::Vertex> > h_primVtx;
  index_type                              indexNPV_;
};

#endif
