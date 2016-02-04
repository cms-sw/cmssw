#ifndef PhysicsTools_PatUtils_JetSelection_h
#define PhysicsTools_PatUtils_JetSelection_h

namespace pat {

  /// Structure defining the jet selection.
  /// Used in the jet selectors.
  struct JetSelection {
    std::string selectionType; ///< Choose selection type (see PATJetCleaner)
    
    double value;     ///< Cut value for likelihood-based selection
    
    /// @name Cuts for "custom" CaloJet selection:
    //@{ 
    // From SusyAnalyzer
    double EMFmin;                     double EMFmax;
    double Etamax;
    // From JetRejectorTool
    double Ptmin;
    double EMvsHadFmin;               double EMvsHadFmax;
    double HadFmin;                   double HadFmax;
    double N90min;                    double N90max;
    double NCaloTowersmin;            double NCaloTowersmax;
    double HighestTowerOverJetmin;    double HighestTowerOverJetmax;
    double RWidthmin;                 double RWidthmax;
    double PtJetOverArea_min;         double PtJetOverArea_max;
    double PtTowerOverArea_min;       double PtTowerOverArea_max;
    //@}
  };
}
  
#endif
