#ifndef PhysicsTools_PatUtils_ElectronSelector_h
#define PhysicsTools_PatUtils_ElectronSelector_h

///
/// \class ElectronSelector ElectronSelector.h "PhysicsTools/PatUtils/ElectronSelector.h"
/// \brief Selects good electrons
///
/// The electron selector returns a boolean decision based on one of the possible
/// selections: either eId-based (cut, likelihood, neural net) or custom (user-defined
/// set of cuts). This is driven by the configuration parameters:
///   PSet selection = {
///     string type = "none | cut | likelihood | neuralnet | custom"
///     [ InputTag eIDsource = ... 
///      [ double value = xxx  // likelihood/neuralnet cut value ]
///      [ // List of custom cuts
///       double ... = xxx
///       double ... = xxx
///       double ... = xxx 
///      ]
///     ]
///   }
///
/// \author F. Ronga (ETH Zurich)
/// \version $Id: ElectronSelector.h,v 1.1 2008/01/24 09:20:58 fronga Exp $

#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"

namespace pat {

  class ElectronSelector {

    typedef reco::PixelMatchGsfElectron            Electron;
    typedef reco::ElectronIDAssociationCollection  ElectronIDmap;

  public:
    ElectronSelector( const edm::ParameterSet& config );
    ~ElectronSelector() {}

    /// Returns true if electron matches criteria. 
    /// Criteria depend on the selector's configuration.
    /// Electron IDs and index only need to be provided if they are used
    /// (i.e., if selection type is not "custom").
    const bool filter( const unsigned int& index,
                       const edm::View<Electron>& electrons,
                       const ElectronIDmap& electronIDs = 0
                       ) const;

    /// Returns the electron ID object based of the given electron.
    /// The latter is defined by an index in the vector of electrons.
    /// The ID is found in the association map.
    const reco::ElectronIDRef& 
    electronID( const unsigned int& index,
                const edm::View<Electron>& electrons,
                const ElectronIDmap& electronIDs = 0
                ) const;

  private:

    /// Full-fledged selection based on SusyAnalyser
    const bool customSelection_( const unsigned int& index,
                         const edm::View<Electron>& electrons ) const;
    edm::ParameterSet selectionCfg_; 
    std::string       selectionType_;

  }; // class
} // namespace

#endif
