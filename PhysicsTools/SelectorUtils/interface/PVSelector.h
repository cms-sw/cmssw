#ifndef Analysis_AnalysisFilters_interface_PVSelector_h
#define Analysis_AnalysisFilters_interface_PVSelector_h

#include "FWCore/Common/interface/EventBase.h"
#include "DataFormats/Common/interface/Handle.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "PhysicsTools/SelectorUtils/interface/EventSelector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <vector>
#include <string>

// make a selector for this selection
class PVSelector : public Selector<edm::EventBase> {
public:

  PVSelector() {}

 PVSelector( edm::ParameterSet const & params ) :
  pvSrc_ (params.getParameter<edm::InputTag>("pvSrc") ) {
    push_back("PV NDOF", params.getParameter<double>("minNdof") );
    push_back("PV Z", params.getParameter<double>("maxZ") );
    push_back("PV RHO", params.getParameter<double>("maxRho") );
    set("PV NDOF");
    set("PV Z");
    set("PV RHO");

    indexNDOF_ = index_type (&bits_, "PV NDOF");
    indexZ_    = index_type (&bits_, "PV Z");
    indexRho_  = index_type (&bits_, "PV RHO");
    
    if ( params.exists("cutsToIgnore") )
      setIgnoredCuts( params.getParameter<std::vector<std::string> >("cutsToIgnore") );

    retInternal_ = getBitTemplate();
  }
  
  bool operator() ( edm::EventBase const & event,  pat::strbitset & ret ) {
    event.getByLabel(pvSrc_, h_primVtx);

    // check if there is a good primary vertex

    if ( h_primVtx->size() < 1 ) return false;

    reco::Vertex const & pv = h_primVtx->at(0);

    if ( pv.isFake() ) return false;

    if ( pv.ndof() >= cut(indexNDOF_, double() )
	 || ignoreCut(indexNDOF_)    ) {
      passCut(ret, indexNDOF_ );
      if ( fabs(pv.z()) <= cut(indexZ_, double()) 
	   || ignoreCut(indexZ_)    ) {
	passCut(ret, indexZ_ );
	if ( fabs(pv.position().Rho()) <= cut(indexRho_, double() )
	     || ignoreCut(indexRho_) ) {
	  passCut( ret, indexRho_);
	}
      }
    }

    setIgnored(ret);
  
    return (bool)ret;
  }

  using EventSelector::operator();

  edm::Handle<std::vector<reco::Vertex> > const & vertices() const { return h_primVtx; }

private:
  edm::InputTag                           pvSrc_;
  edm::Handle<std::vector<reco::Vertex> > h_primVtx;

  index_type indexNDOF_;
  index_type indexZ_;
  index_type indexRho_;
};

#endif
