#ifndef Analysis_AnalysisFilters_interface_PVSelector_h
#define Analysis_AnalysisFilters_interface_PVSelector_h

#include "FWCore/Common/interface/EventBase.h"
#include "DataFormats/Common/interface/Handle.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <vector>
#include <string>

// make a selector for this selection
class PVSelector : public Selector<edm::EventBase> {
public:
 PVSelector( edm::ParameterSet const & params ) :
  pvSrc_ (params.getParameter<edm::InputTag>("pvSrc") ) {
    push_back("PV NDOF", params.getParameter<double>("minNdof") );
    push_back("PV Z", params.getParameter<double>("maxPVZ") );
    set("PV NDOF");
    set("PV Z");
  }
  
  bool operator() ( edm::EventBase const & event,  std::strbitset & ret ) {
    event.getByLabel(pvSrc_, h_primVtx);

    // check if there is a good primary vertex

    if ( h_primVtx->size() < 1 ) return false;

    reco::Vertex const & pv = h_primVtx->at(0);

    if ( pv.isFake() ) return false;

    if ( pv.ndof() >= cut("PV NDOF", double() )
	 || ignoreCut("PV NDOF")    ) {
      passCut(ret, "PV NDOF" );
      if ( fabs(pv.z()) <= cut("PV Z", double()) 
	   || ignoreCut("PV Z")    ) 
	passCut(ret, "PV Z" );
    }
  
    return (bool)ret;
  }

  edm::Handle<std::vector<reco::Vertex> > const & vertices() const { return h_primVtx; }

private:
  edm::InputTag                           pvSrc_;
  edm::Handle<std::vector<reco::Vertex> > h_primVtx;
};

#endif
