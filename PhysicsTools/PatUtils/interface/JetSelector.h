#ifndef PhysicsTools_PatUtils_JetSelector_h
#define PhysicsTools_PatUtils_JetSelector_h

/**
    \class pat::JetSelector JetSelector.h "PhysicsTools/PatUtils/JetSelector.h"
    \brief Selects good Jets
   
    The Jet selector returns a flag (passed=0) based on one of the possible
    selections: either eId-based (cut, likelihood, neural net) or custom (user-defined
    set of cuts). This is driven by the configuration parameters.
      PSet configuration = {
        string type = "none | cut | likelihood | neuralnet | custom"
        [ double value = xxx  // likelihood/neuralnet cut value ]
        [ // List of custom cuts
          double ... = xxx
          double ... = xxx
          double ... = xxx 
        ]
      }
   
    This class is based upon the ElectronSelector by F. Ronga
   
    \author C. Autermann (Uni Hamburg)
    \version $Id: JetSelector.h,v 1.3 2008/02/19 18:04:17 auterman Exp $
**/

#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "PhysicsTools/PatUtils/interface/CaloJetSelector.h"
#include "DataFormats/Common/interface/ValueMap.h"


namespace pat {

  typedef edm::ValueMap<double> JetValueMap;

  template<typename Jet>
  class JetSelector {


  public:
    JetSelector( const edm::ParameterSet& config );
    ~JetSelector() {}

    /// Returns 0 if Jet matches criteria, a flag otherwise.
    /// Criteria depend on the selector's configuration.
    /// Jet IDs only need to be provided if selection is based
    /// on it (cut, neural net or likelihood). Cluster shapes are for
    /// custom selection only.
    const unsigned int 
    filter( const unsigned int&     index,
            const edm::View<Jet>&   Jets,
	    const JetValueMap * JetMap
           ) const;
    

  private:
   
    std::string       selectionType_;
    ///The selection w.r.t. i.e. the JetRejector-likelihood
    double value_;

    std::auto_ptr<CaloJetSelector> CaloJetSelector_;///Selects CaloJets
    //std::auto_ptr<CaloJetSelector> PFSelector_;///Selects PFJets

  }; // class

} // namespace

#endif
