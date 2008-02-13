#ifndef PhysicsTools_PatUtils_MuonSelector_h
#define PhysicsTools_PatUtils_MuonSelector_h

/**
    \class MuonSelector MuonSelector.h "PhysicsTools/PatUtils/MuonSelector.h"
    \brief Selects good muons

    The muon selector returns a flag (see pat::ParticleStatus) based on one of the possible
    selections: reconstruction-based (global muons) or muId based (various algorithms) 
    or custom (user-defined set of cuts). 

    See the PATMuonCleaner documentation for configuration details.

    \author F.J. Ronga (ETH Zurich)
    \version $Id: MuonSelector.h,v 1.1 2008/02/07 15:48:55 fronga Exp $
*/

#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "PhysicsTools/PatUtils/interface/ParticleCode.h"

namespace pat {

  class MuonSelector {

  public:
    
    MuonSelector( const edm::ParameterSet& config );
    ~MuonSelector() {}

    /// Returns 0 if muon matches criteria, a flag otherwise.
    /// Criteria depend on the selector's configuration.
    const pat::ParticleStatus
    filter( const unsigned int& index,
            const edm::View<reco::Muon>& muons ) const;
    
  private:

    edm::ParameterSet selectionCfg_; 
    std::string       selectionType_;

    /// Full-fledged selection based on SusyAnalyser
    const pat::ParticleStatus
    customSelection_( const unsigned int& index,
                      const edm::View<reco::Muon>& muons  ) const;
    
    // Custom selection cuts
    double dPbyPmax_, chi2max_;
    int    nHitsMin_;


  }; // class
} // namespace

#endif
