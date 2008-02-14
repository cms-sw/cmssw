#ifndef PhysicsTools_PatUtils_JetSelector_h
#define PhysicsTools_PatUtils_JetSelector_h

/**
    \class JetSelector JetSelector.h "PhysicsTools/PatUtils/JetSelector.h"
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
    \version $Id: JetSelector.h,v 1.0 2008/02/12 15:51:01 auterman Exp $
**/

#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"


namespace pat {

  enum JetType { GOOD = 0, BAD, ELSE };

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
            const edm::View<Jet>&   Jets
           ) const;
    
    /// Returns the Jet ID object based of the given Jet.
    /// The latter is defined by an index in the vector of Jets.
    /// The ID is found in the association map.
    //const reco::JetIDRef& 
    //JetID( const unsigned int&        index,
    //            const edm::View<Jet>& Jets
    //            ) const;

  private:

    edm::ParameterSet selectionCfg_; 
    std::string       selectionType_;

    double value_; // Cut value for btag or tau //@@ copied from ElectronSel.


    /// Full-fledged selection based on SusyAnalyser
    const unsigned int  
    customSelection_( const unsigned int&        index,
                      const edm::View<Jet>& Jets ) const;
    
    // Custom selection cuts
    double EMFmin_;              
    double EMFmax_;
    double EoverPmax_;
    double Etamax_;
    double PTrackoverPJetmin_;
    int    NTracksmin_;

  }; // class
  
  typedef JetSelector<reco::Jet>     BaseJetSelector;
  typedef JetSelector<reco::PFJet>   PFJetSelector;
  typedef JetSelector<reco::CaloJet> CaloJetSelector;
} // namespace

#endif
