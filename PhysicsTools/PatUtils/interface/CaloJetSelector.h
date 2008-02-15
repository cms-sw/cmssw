#ifndef PhysicsTools_PatUtils_CaloJetSelector_h
#define PhysicsTools_PatUtils_CaloJetSelector_h

/**
    \class CaloJetSelector CaloJetSelector.h "PhysicsTools/PatUtils/CaloJetSelector.h"
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
    \version $Id: CaloJetSelector.h,v 1.1 2008/02/14 12:38:10 auterman Exp $
**/

#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"


namespace pat {

  enum JetType { GOOD = 0, BAD = 1, ELSE = 2}; //bool would be propably enough

  class CaloJetSelector {


  public:
    CaloJetSelector( const edm::ParameterSet& config );
    ~CaloJetSelector() {}

    /// Returns 0 if Jet matches criteria, a flag otherwise.
    /// Criteria depend on the selector's configuration.
    /// Jet IDs only need to be provided if selection is based
    /// on it (cut, neural net or likelihood). Cluster shapes are for
    /// custom selection only.
    const unsigned int 
    filter( const reco::CaloJet&   Jet ) const;
    
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

    double value_; // Cut value for e.g. JetMET likelihood

    /// Full-fledged selection based on SusyAnalyser, ...
    const unsigned int  
    customSelection_( const reco::CaloJet& Jet ) const;
    
    /// Custom selection cuts
    ///SUSY-Analyzer:
    double EMFmin_;                    double EMFmax_;
    //double EoverPmax_;               //necessary??
    double Etamax_;
    //double PTrackoverPJetmin_;       //not defined for CaloJets
    //int    NTracksmin_;              //not defined for CaloJets

    ///used variables JetRejectorTool:
    double PTmin_;
    double EMvsHadFmin_;               double EMvsHadFmax_;
    double HadFmin_;                   double HadFmax_;
    double N90min_;                    double N90max_;
    double NCaloTowersmin_;            double NCaloTowersmax_;
    double HighestTowerOverJetmin_;    double HighestTowerOverJetmax_;
    double RWidthmin_;                 double RWidthmax_;
    double PTjetOverArea_min_;         double PTjetOverArea_max_;
    double PTtowerOverArea_min_;       double PTtowerOverArea_max_;

  }; // class
  
} // namespace

#endif
