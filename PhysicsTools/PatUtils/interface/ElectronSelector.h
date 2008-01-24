#ifndef PhysicsTools_PatUtils_ElectronSelector_h
#define PhysicsTools_PatUtils_ElectronSelector_h

///
/// \class ElectronSelector ElectronSelector.h "PhysicsTools/PatUtils/ElectronSelector.h"
/// \brief Selects good electrons
///
/// The electron selector returns a boolean decision based on one of the possible
/// selections: either eId-based (cut, likelihood, neural net) or custom (user-defined
/// set of cuts).
///
/// \author F. Ronga (ETH Zurich)
/// \version $Id$


#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"

namespace pat {

  class ElectronSelector {
  public:
    ElectronSelector( const edm::ParameterSet& config ) {}
    ~ElectronSelector() {}

    /// Returns true if electron matches criteria. 
    /// Electron IDs only need to be provided if they are used
    /// (i.e., if selection type is not "custom")
    bool filter( const reco::PixelMatchGsfElectron& electron,
                 const edm::Handle<reco::ElectronIDAssociationCollection>& electronIDs = 0 )
    { return true; }

  private:
    edm::ParameterSet selectionCfg_; 

  }; // class

} // namespace

#endif
