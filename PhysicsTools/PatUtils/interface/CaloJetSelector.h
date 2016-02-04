#ifndef PhysicsTools_PatUtils_CaloJetSelector_h
#define PhysicsTools_PatUtils_CaloJetSelector_h

/**
   \class pat::CaloJetSelector CaloJetSelector.h "PhysicsTools/PatUtils/CaloJetSelector.h"
    \brief Selects good Jets
   
    The calo jet selector returns a flag (see pat::ParticleStatus) based on one of the possible
    selections. It is called by the generic JetSelector in case of "custom" selection.

    \author C. Autermann (Uni Hamburg)
    \version $Id: CaloJetSelector.h,v 1.5 2008/03/10 14:23:58 fronga Exp $
**/

#include <string>
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "PhysicsTools/PatUtils/interface/JetSelection.h"

#include "PhysicsTools/PatUtils/interface/ParticleCode.h"

namespace pat {

  class CaloJetSelector {


  public:
    CaloJetSelector( const JetSelection& config ) : config_( config ) {}
    ~CaloJetSelector() {}

    /// Returns 0 if Jet matches criteria, a flag otherwise.
    /// Criteria depend on the selector's configuration.
    const ParticleStatus
    filter( const reco::CaloJet& Jet ) const;

  private:

    JetSelection config_;
    
  }; // class
  
} // namespace

#endif
