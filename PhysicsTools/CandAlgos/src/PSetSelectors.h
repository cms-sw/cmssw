#ifndef PHYSICSTOOLS_PSETPTMINSELECTOR_H
#define PHYSICSTOOLS_PSETPTMINSELECTOR_H
// $Id: PSetSelectors.h,v 1.3 2005/10/25 08:47:05 llista Exp $#
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/CandAlgos/interface/CandSelectorBase.h"
#include "PhysicsTools/CandUtils/interface/PtMinSelector.h"
#include "PhysicsTools/CandUtils/interface/MassWindowSelector.h"

namespace candmodules {
  
  class PtMinCandSelector : public CandSelectorBase {
  public:
    explicit PtMinCandSelector( const edm::ParameterSet & cfg ) :
      CandSelectorBase( cfg.getParameter<std::string>("src"),
        boost::shared_ptr<CandSelector> (
          new PtMinSelector( cfg.getParameter<double>( "ptMin" ) ) ) ) {
    }
  };
  
  class MassWindowCandSelector : public CandSelectorBase {
  public:
    explicit MassWindowCandSelector( const edm::ParameterSet & cfg ) :
      CandSelectorBase( cfg.getParameter<std::string>("src"),
        boost::shared_ptr<CandSelector> (
          new  MassWindowSelector( cfg.getParameter<double>( "massMin" ), 
				   cfg.getParameter<double>( "massMax" ) ) ) ) {
    }
  };
  
}

#endif
