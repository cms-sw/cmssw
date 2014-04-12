#ifndef UtilAlgos_PdgIdSelector_h
#define UtilAlgos_PdgIdSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/PdgIdSelector.h"

namespace reco {
  namespace modules {

    template<>
    struct ParameterAdapter<PdgIdSelector> {
      static PdgIdSelector make( const edm::ParameterSet & cfg, edm::ConsumesCollector & iC ) {
	return PdgIdSelector( cfg.getParameter<std::vector<int> >( "pdgId" ) );
      }
    };

  }
}


// Introducing a simpler way to use a string object selector outside of the
// heavily-templated infrastructure above. This simply translates the cfg
// into the string that the functor expects.
namespace reco{
    class PdgIdSelectorHandler : public PdgIdSelector  {
    public:
      explicit PdgIdSelectorHandler( const edm::ParameterSet & cfg ) :
        PdgIdSelector(cfg.getParameter<std::vector<int> >("pdgId"))
      {
      }
  };
}


#endif

