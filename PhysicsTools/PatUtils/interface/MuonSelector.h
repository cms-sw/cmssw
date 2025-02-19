#ifndef PhysicsTools_PatUtils_MuonSelector_h
#define PhysicsTools_PatUtils_MuonSelector_h

/**
    \class pat::MuonSelector MuonSelector.h "PhysicsTools/PatUtils/MuonSelector.h"
    \brief Selects good muons

    The muon selector returns a flag (see pat::ParticleStatus) based on one of the possible
    selections: reconstruction-based (global muons) or muId based (various algorithms),
    or custom (user-defined set of cuts). This is driven by the configuration parameters 
    (see the PATMuonCleaner documentation for configuration details).
    
    The parameters are passed to the selector through an MuonSelection struct.
    (An adapter exists for use in CMSSW: reco::modules::ParameterAdapter< pat::MuonSelector >.)

    \author F.J. Ronga (ETH Zurich)
    \version $Id: MuonSelector.h,v 1.8 2009/04/01 18:18:53 hegner Exp $
*/

#include <string>

#include "DataFormats/MuonReco/interface/Muon.h"

#include "PhysicsTools/PatUtils/interface/ParticleCode.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

namespace pat {

  /// Structure defining the muon selection
  struct MuonSelection {
    std::string selectionType; ///< Choose selection type (see PATMuonCleaner)
    
    /// @name Cuts for "custom" selection type:
    //@{
    double dPbyPmax;
    double chi2max;
    int    nHitsMin;
    //@}
	
	/// @name Option for "muId" selection type:
    //@{
    muon::SelectionType flag;
	double minCaloCompatibility;
	double minSegmentCompatibility;
    //@}
	
  };


  class MuonSelector {

  public:
    
    MuonSelector( const MuonSelection& cfg ) : config_( cfg ) {}
    ~MuonSelector() {}

    /// Returns 0 if muon matches criteria, a flag otherwise.
    /// Criteria depend on the selector's configuration.
    const pat::ParticleStatus
    filter( const unsigned int& index,
            const edm::View<reco::Muon>& muons ) const;
    
  private:
    
    MuonSelection config_;

    /// Full-fledged selection based on SusyAnalyser
    const pat::ParticleStatus
    customSelection_( const unsigned int& index,
                      const edm::View<reco::Muon>& muons  ) const;
	const pat::ParticleStatus
    muIdSelection_( const unsigned int& index,
                      const edm::View<reco::Muon>& muons  ) const;
    

  }; // class
} // namespace

#endif
