#ifndef PhysicsTools_PatUtils_ElectronSelector_h
#define PhysicsTools_PatUtils_ElectronSelector_h

/**
    \class pat::ElectronSelector ElectronSelector.h "PhysicsTools/PatUtils/ElectronSelector.h"
    \brief Selects good electrons
   
    The electron selector returns a flag (see pat::ParticleStatus) based on one of the possible
    selections: either eId-based (cut, likelihood, neural net) or custom (user-defined
    set of cuts). This is driven by the configuration parameters (see the PATElectronCleaner 
    documentation for configuration details).
   
    The parameters are passed to the selector through an ElectronSelection struct.
    (An adapter exists for use in CMSSW: reco::modules::ParameterAdapter< pat::ElectronSelector >.)

    \author F.J. Ronga (ETH Zurich)
    \version $Id: ElectronSelector.h,v 1.12 2008/04/09 12:05:13 llista Exp $
**/

#include <string>

#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "PhysicsTools/PatUtils/interface/ParticleCode.h"

namespace pat {

  /// Structure defining the electron selection
  struct ElectronSelection {
    std::string selectionType; ///< Choose selection type (see PATElectronCleaner)
    
    double value; ///< Cut value for likelihood or neural net
    
    /// @name Cuts for "custom" selection type:
    //@{
    double HoverEBarmax;        double HoverEEndmax;
    double SigmaEtaEtaBarmax;   double SigmaEtaEtaEndmax;
    double SigmaPhiPhiBarmax;   double SigmaPhiPhiEndmax;
    double DeltaEtaInBarmax;    double DeltaEtaInEndmax;
    double DeltaPhiInBarmax;    double DeltaPhiInEndmax;
    double DeltaPhiOutBarmax;   double DeltaPhiOutEndmax;
    double EoverPInBarmin;      double EoverPInEndmin;
    double EoverPOutBarmin;     double EoverPOutEndmin;
    double InvEMinusInvPBarmax; double InvEMinusInvPEndmax;
    double E9overE25Barmin;     double E9overE25Endmin;
    bool   doBremEoverPcomp;    ///< switch to apply cut on comparison between brem and E/P
    //@}
  };

  class ElectronSelector {

    typedef reco::GsfElectron            Electron;
    typedef reco::ElectronIDAssociationCollection  ElectronIDmap;

  public:
    ElectronSelector( const ElectronSelection& cfg ) : config_( cfg ) {}
    ~ElectronSelector() {}

    /// Returns 0 if electron matches criteria, a flag otherwise.
    /// Criteria depend on the selector's configuration.
    /// Electron IDs are only used if the selection is based
    /// on it (cut, neural net or likelihood). Cluster shapes are for
    /// custom selection only.
    const ParticleStatus
    filter( const unsigned int&          index,
            const edm::View<Electron>&   electrons,
            const ElectronIDmap&         electronIDs,
            const reco::ClusterShape*    clusterShape
            ) const;
    

  private:
    
    ElectronSelection config_;

    /// Full-fledged selection based on SusyAnalyser
    const ParticleStatus
    customSelection_( const unsigned int&        index,
                      const edm::View<Electron>& electrons,
                      const reco::ClusterShape*  clusterShape ) const;
    
    /// Returns the electron ID object based on the given electron.
    /// The latter is defined by an index in the vector of electrons.
    /// The ID is found in the association map.
    const reco::ElectronIDRef& 
    electronID_( const unsigned int&        index,
                 const edm::View<Electron>& electrons,
                 const ElectronIDmap&       electronIDs
                 ) const;


  }; // class
} // namespace

#endif
