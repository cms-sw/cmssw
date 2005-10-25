#ifndef PHYSICSTOOLS_PSETPTMINSELECTOR_H
#define PHYSICSTOOLS_PSETPTMINSELECTOR_H
// $Id: PSetSelectors.h,v 1.2 2005/10/24 09:42:46 llista Exp $#
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/CandAlgos/interface/CandSelectorBase.h"
#include "PhysicsTools/CandUtils/interface/PtMinSelector.h"
#include "PhysicsTools/CandUtils/interface/MassWindowSelector.h"

namespace candmodules {
  
  class PtMinCandSelector : public CandSelectorBase {
  public:
    explicit PtMinCandSelector( const edm::ParameterSet & cfg ) :
      CandSelectorBase( cfg.getParameter<std::string>("src"),
        boost::shared_ptr<aod::Candidate::selector> (
          new PtMinSelector( cfg.getParameter<double>( "ptMin" ) ) ) ) {
    }
  };
  
  class MassWindowCandSelector : public CandSelectorBase {
  public:
    explicit MassWindowCandSelector( const edm::ParameterSet & cfg ) :
      CandSelectorBase( cfg.getParameter<std::string>("src"),
        boost::shared_ptr<aod::Candidate::selector> (
          new  MassWindowSelector( cfg.getParameter<double>( "massMin" ), 
				   cfg.getParameter<double>( "massMax" ) ) ) ) {
    }
  };
  
}

#endif
