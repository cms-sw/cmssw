#ifndef PhysicsTools_PatUtils_MuonSelector_h
#define PhysicsTools_PatUtils_MuonSelector_h

/**
    \class MuonSelector MuonSelector.h "PhysicsTools/PatUtils/MuonSelector.h"
    \brief Selects good muons

    The muon selector returns a flag (passed=0) based on one of the possible
    selections: reconstruction-based (global muons) or muId based (various algorithms) 
    or custom (user-defined set of cuts). This is driven by the configuration parameters.

    PSet selection = {
      string type = "none | globalMuons | muId  | custom" // muId not implemented yet
      [ // If custom, give cut values
        double dPbyPmax = ...
        double chi2max  = ...
        int    nHitsMin = ...
      ]

    }

    \author F. Ronga (ETH Zurich)
    \version $Id: MuonSelector.h,v 1.4 2008/02/04 14:20:55 fronga Exp $
*/

#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"

namespace pat {

  enum MuonType { GOOD = 0, BAD };

  class MuonSelector {

  public:
    
    MuonSelector( const edm::ParameterSet& config );
    ~MuonSelector() {}

    /// Returns 0 if muon matches criteria, a flag otherwise.
    /// Criteria depend on the selector's configuration.
    const unsigned int 
    filter( const unsigned int& index,
            const edm::View<reco::Muon>&   muons ) const;
    
  private:

    edm::ParameterSet selectionCfg_; 
    std::string       selectionType_;

    /// Full-fledged selection based on SusyAnalyser
    const unsigned int  
    customSelection_( const unsigned int& index,
                      const edm::View<reco::Muon>& muons  ) const;
    
    // Custom selection cuts
    double dPbyPmax_, chi2max_;
    int    nHitsMin_;


  }; // class
} // namespace

#endif
