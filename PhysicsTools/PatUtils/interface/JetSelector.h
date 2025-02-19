#ifndef PhysicsTools_PatUtils_JetSelector_h
#define PhysicsTools_PatUtils_JetSelector_h

/**
    \class pat::JetSelector JetSelector.h "PhysicsTools/PatUtils/JetSelector.h"
    \brief Selects good Jets

    The Jet selector returns a flag (see pat::ParticleStatus) based on one of the possible
    selections: either cut-based  or custom (user-defined set of cuts). 
    This is driven by the configuration parameters (see the PATJetCleaner 
    documentation for configuration details).
    
    The parameters are passed to the selector through a JetSelection struct.
    (An adapter exists for use in CMSSW: 
    reco::modules::ParameterAdapter<pat::JetSelector<JetIn>>.)
   
    \author C. Autermann (Uni Hamburg)
    \version $Id: JetSelector.h,v 1.5 2008/03/10 14:23:58 fronga Exp $
**/

#include <string>
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "PhysicsTools/PatUtils/interface/JetSelection.h"
#include "PhysicsTools/PatUtils/interface/CaloJetSelector.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "PhysicsTools/PatUtils/interface/ParticleCode.h"


namespace pat {

  typedef edm::ValueMap<double> JetValueMap;

  template<typename JetType>
  class JetSelector {


  public:
    JetSelector( const JetSelection& config );
    ~JetSelector() {}

    /// Returns 0 if Jet matches criteria, a flag otherwise.
    /// Criteria depend on the selector's configuration.
    /// Jet IDs only need to be provided if selection is based
    /// on it (cut, neural net or likelihood). Cluster shapes are for
    /// custom selection only.
    const ParticleStatus
    filter( const unsigned int&       index,
            const edm::View<JetType>& Jets,
	    const JetValueMap*        JetMap
           ) const;
    

  private:

    JetSelection config_;

    std::auto_ptr<CaloJetSelector> CaloJetSelector_;///Selects CaloJets
    //std::auto_ptr<CaloJetSelector> PFSelector_;///Selects PFJets

  }; // class

} // namespace

#endif
