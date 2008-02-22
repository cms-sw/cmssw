#ifndef UtilAlgos_FunctionMinSelector_h
#define UtilAlgos_FunctionMinSelector_h

#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/FunctionMinSelector.h"
#include "PhysicsTools/RecoAlgos/plugins/SuperClusterEt.h"

namespace reco {
   namespace modules {
     typedef FunctionMinSelector<SuperClusterEt> EtMinSuperClusterSelector;

     template<>
     struct ParameterAdapter<EtMinSuperClusterSelector> {
       static EtMinSuperClusterSelector make(
         const edm::ParameterSet & cfg ) {
	 return EtMinSuperClusterSelector( cfg.getParameter<double>( "etMin" ) );
       }
     };

   }
}

#endif
