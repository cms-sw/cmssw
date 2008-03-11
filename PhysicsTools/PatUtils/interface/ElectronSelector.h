#ifndef PhysicsTools_PatUtils_ElectronSelector_h
#define PhysicsTools_PatUtils_ElectronSelector_h

/**
    \class ElectronSelector ElectronSelector.h "PhysicsTools/PatUtils/ElectronSelector.h"
    \brief Selects good electrons
   
    The electron selector returns a flag (see pat::ParticleStatus) based on one of the possible
    selections: either eId-based (cut, likelihood, neural net) or custom (user-defined
    set of cuts). This is driven by the configuration parameters:

    See the PATElectronCleaner documentation for configuration details.
   
    \author F.J. Ronga (ETH Zurich)
    \version $Id: ElectronSelector.h,v 1.6 2008/02/13 10:30:35 fronga Exp $
**/

#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"

#include "PhysicsTools/PatUtils/interface/ParticleCode.h"

namespace pat {

  class ElectronSelector {

    typedef reco::PixelMatchGsfElectron            Electron;
    typedef reco::ElectronIDAssociationCollection  ElectronIDmap;

  public:
    ElectronSelector( const edm::ParameterSet& config );
    ~ElectronSelector() {}

    /// Returns 0 if electron matches criteria, a flag otherwise.
    /// Criteria depend on the selector's configuration.
    /// Electron IDs are only used if selection is based on it 
    /// (cut, neural net or likelihood). Cluster shapes are for
    /// custom selection only.
    const ParticleStatus
    filter( const unsigned int&          index,
            const edm::View<Electron>&   electrons,
            const ElectronIDmap&         electronID,
            const reco::ClusterShape*    clusterShape
            ) const;
    
    /// Returns the electron ID object based of the given electron.
    /// The latter is defined by an index in the vector of electrons.
    /// The ID is found in the association map.
    const reco::ElectronIDRef& 
    electronID( const unsigned int&        index,
                const edm::View<Electron>& electrons,
                const ElectronIDmap&       electronIDs
                ) const;

  private:

    edm::ParameterSet selectionCfg_; 
    std::string       selectionType_;

    double value_; // Cut value for likelihood or neural net


    /// Full-fledged selection based on SusyAnalyser
    const ParticleStatus
    customSelection_( const unsigned int&        index,
                      const edm::View<Electron>& electrons,
                      const reco::ClusterShape*  clusterShape ) const;
    
    // Custom selection cuts
    double HoverEBarmax_;        double HoverEEndmax_;
    double SigmaEtaEtaBarmax_;   double SigmaEtaEtaEndmax_;
    double SigmaPhiPhiBarmax_;   double SigmaPhiPhiEndmax_;
    double DeltaEtaInBarmax_;    double DeltaEtaInEndmax_;
    double DeltaPhiInBarmax_;    double DeltaPhiInEndmax_;
    double DeltaPhiOutBarmax_;   double DeltaPhiOutEndmax_;
    double EoverPInBarmin_;      double EoverPInEndmin_;
    double EoverPOutBarmin_;     double EoverPOutEndmin_;
    double InvEMinusInvPBarmax_; double InvEMinusInvPEndmax_;
    double E9overE25Barmin_;     double E9overE25Endmin_;

    bool   doBremEoverPcomp_; // apply cut on comparison between brem and E/P


  }; // class
} // namespace

#endif
