#ifndef Analysis_AnalysisFilters_interface_PVObjectSelector_h
#define Analysis_AnalysisFilters_interface_PVObjectSelector_h

#ifndef __GCCXML__
#include "FWCore/Framework/interface/ConsumesCollector.h"
#endif
#include "FWCore/Common/interface/EventBase.h"
#include "DataFormats/Common/interface/Handle.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <vector>
#include <string>

// make a selector for this selection
class PVObjectSelector : public Selector<reco::Vertex> {
public:

  PVObjectSelector() {}

#ifndef __GCCXML__
  PVObjectSelector( edm::ParameterSet const & params, edm::ConsumesCollector&& iC ) :
    PVObjectSelector( params )
  {}
#endif

 PVObjectSelector( edm::ParameterSet const & params )  {
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

 bool operator() ( reco::Vertex const & pv,  pat::strbitset & ret ) {
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

 using Selector<reco::Vertex>::operator();

private:
  index_type indexNDOF_;
  index_type indexZ_;
  index_type indexRho_;
};

#endif
