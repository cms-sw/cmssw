//
// $Id: PATElectronCleaner.h,v 1.2 2008/03/11 11:03:44 llista Exp $
//

#ifndef PhysicsTools_PatAlgos_PATElectronCleaner_h
#define PhysicsTools_PatAlgos_PATElectronCleaner_h

/**
  \class    pat::PATElectronCleaner PATElectronCleaner.h "PhysicsTools/PatAlgos/interface/PATElectronCleaner.h"
  \brief    Produces a clean list of electrons

   The PATElectronCleaner produces a clean list of electrons with associated 
   back-references to the original electron collection.

   The selection is based on the electron ID or on user-defined cuts.
   It is steered by the configuration parameters:

\code
 PSet selection = {
   string type = "none | cut | likelihood | neuralnet | custom"
   [ // If cut-based, give electron ID source
     InputTag eIdSource = <source>
   ]
   [ // If likelihood/neuralnet, give ID source and cut value
     InputTag eIdSource = <source>
     double value = xxx
   ]
   [ // If custom, give cluster shape sources and cut values
     InputTag clusterShapeBarrel = <source 1>
     InputTag clusterShapeEndcap = <source 2>
     double <cut> = <value>
     ...
   ]
 }
\endcode

  The actual selection is performed by the ElectronSelector.

  \author   Steven Lowette, James Lamb
  \version  $Id: PATElectronCleaner.h,v 1.2 2008/03/11 11:03:44 llista Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/plugins/CleanerHelper.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"

#include "PhysicsTools/PatUtils/interface/ElectronSelector.h"
#include "PhysicsTools/PatUtils/interface/DuplicatedElectronRemover.h"

#include <string>


namespace pat {

  class PATElectronCleaner : public edm::EDProducer {
    public:
      explicit PATElectronCleaner(const edm::ParameterSet & iConfig);
      ~PATElectronCleaner();  

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
      virtual void endJob();

    private:
      void removeDuplicates();

      edm::InputTag electronSrc_;
      bool          removeDuplicates_;

      pat::helper::CleanerHelper<reco::PixelMatchGsfElectron, 
	                         reco::PixelMatchGsfElectron,
                                 reco::PixelMatchGsfElectronCollection, 
	                         GreaterByPt<reco::PixelMatchGsfElectron> > helper_;

      pat::helper::MultiIsolator isolator_;

      pat::DuplicatedElectronRemover duplicateRemover_;  
    
      edm::ParameterSet selectionCfg_;  ///< Defines all about the selection
      std::string       selectionType_; ///< Selection type (none, custom, cut,...)
      bool              doSelection_;   ///< False if type = "none", true otherwise
      ElectronSelector  selector_;      ///< Actually performs the selection
      
      /// Returns the appropriate cluster shape.
      /// This is a copy of the Egamma code and it should disappear in the future
      /// (once cluster shapes are put directly in electron, should be in 2_0_0).
      /// See EgammaAnalysis/ElectronIDAlgos/interface/ElectronIDAlgo.h
      const reco::ClusterShapeRef& getClusterShape_( const reco::GsfElectron* electron, 
                                                     const edm::Event&        event
                                                     ) const;
  };


}

namespace reco {
  namespace modules {
    /// Helper struct to convert from ParameterSet to ElectronSelection
    template<> 
    struct ParameterAdapter<pat::ElectronSelector> { 
      static pat::ElectronSelector make(const edm::ParameterSet & cfg) {
        pat::ElectronSelection config_;
        const std::string& selectionType = cfg.getParameter<std::string>("type");
        config_.selectionType = selectionType;
        if ( selectionType == "likelihood" || selectionType == "neuralnet" )
	  config_.value = cfg.getParameter<double>("value");
        else if ( selectionType == "custom" ) {
	  config_.HoverEBarmax        = cfg.getParameter<double>("HoverEBarmax");
	  config_.SigmaEtaEtaBarmax   = cfg.getParameter<double>("SigmaEtaEtaBarmax");
	  config_.SigmaPhiPhiBarmax   = cfg.getParameter<double>("SigmaPhiPhiBarmax");
	  config_.DeltaEtaInBarmax    = cfg.getParameter<double>("DeltaEtaInBarmax");
	  config_.DeltaPhiInBarmax    = cfg.getParameter<double>("DeltaPhiInBarmax");
	  config_.DeltaPhiOutBarmax   = cfg.getParameter<double>("DeltaPhiOutBarmax");
	  config_.EoverPInBarmin      = cfg.getParameter<double>("EoverPInBarmin");
	  config_.EoverPOutBarmin     = cfg.getParameter<double>("EoverPOutBarmin");
	  config_.InvEMinusInvPBarmax = cfg.getParameter<double>("InvEMinusInvPBarmax");
	  config_.E9overE25Barmin     = cfg.getParameter<double>("E9overE25Barmin");
	  config_.HoverEEndmax        = cfg.getParameter<double>("HoverEEndmax");
	  config_.SigmaEtaEtaEndmax   = cfg.getParameter<double>("SigmaEtaEtaEndmax");
	  config_.SigmaPhiPhiEndmax   = cfg.getParameter<double>("SigmaPhiPhiEndmax");
	  config_.DeltaEtaInEndmax    = cfg.getParameter<double>("DeltaEtaInEndmax");
	  config_.DeltaPhiInEndmax    = cfg.getParameter<double>("DeltaPhiInEndmax");
	  config_.DeltaPhiOutEndmax   = cfg.getParameter<double>("DeltaPhiOutEndmax");
	  config_.EoverPInEndmin      = cfg.getParameter<double>("EoverPInEndmin");
	  config_.EoverPOutEndmin     = cfg.getParameter<double>("EoverPOutEndmin");
	  config_.InvEMinusInvPEndmax = cfg.getParameter<double>("InvEMinusInvPEndmax");
	  config_.E9overE25Endmin     = cfg.getParameter<double>("E9overE25Endmin");
	  config_.doBremEoverPcomp    = cfg.getParameter<bool>  ("doBremEoverPcomp");
	}
        return pat::ElectronSelector( config_ );
      }
    };
  }
}

#endif
