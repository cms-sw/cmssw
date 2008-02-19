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
    \version $Id: CaloJetSelector.h,v 1.1 2008/02/15 15:54:26 auterman Exp $
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
    const unsigned int 
    filter( const reco::CaloJet& Jet ) const;

  private:

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
