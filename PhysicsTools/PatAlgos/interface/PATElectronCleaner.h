//
// $Id: PATElectronCleaner.h,v 1.5 2008/01/25 15:36:41 fronga Exp $
//

#ifndef PhysicsTools_PatAlgos_PATElectronCleaner_h
#define PhysicsTools_PatAlgos_PATElectronCleaner_h

/**
  \class    PATElectronCleaner PATElectronCleaner.h "PhysicsTools/PatAlgos/interface/PATElectronCleaner.h"
  \brief    Produces pat::Electron's

   The PATElectronCleaner produces analysis-level pat::Electron's starting from
   a collection of objects of ElectronType. 

   Electron selection is performed based on the electron ID or on user-defined cuts. 
   The selection is steered by the configuration parameter:

   PSet selection = {
     string type = "none | cut | likelihood | neuralnet | custom"
     [ // If cut-based, give electron ID source
       InputTag eIDsource = <source>
     ]
     [ // If likelihood/neuralnet, give ID source and cut value
       InputTag eIDsource = <source>
       double value = xxx
     ]
     [ // If custom, give cluster shape sources and cut values
       InputTag barrelClusterShapeAssociation = <source 1>
       InputTag endcapClusterShapeAssociation = <source 2>
       double <cut> = <value>
       ...
     ]
   }

  \author   Steven Lowette, James Lamb
  \version  $Id: PATElectronCleaner.h,v 1.5 2008/01/25 15:36:41 fronga Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/interface/CleanerHelper.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include "PhysicsTools/PatUtils/interface/ElectronSelector.h"
#include "PhysicsTools/PatUtils/interface/DuplicatedElectronRemover.h"

#include <string>


namespace pat {
  class PATElectronCleaner : public edm::EDProducer {
    public:
      explicit PATElectronCleaner(const edm::ParameterSet & iConfig);
      ~PATElectronCleaner();  

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      void removeDuplicates();

      edm::InputTag electronSrc_;
      bool          removeDuplicates_;

      pat::helper::CleanerHelper< reco::PixelMatchGsfElectron, 
                                  reco::PixelMatchGsfElectron,
                                  reco::PixelMatchGsfElectronCollection, 
                                  GreaterByPt<reco::PixelMatchGsfElectron> > helper_;

      pat::DuplicatedElectronRemover duplicateRemover_;  
    
      edm::ParameterSet selectionCfg_;  ///< Defines all about the selection
      std::string       selectionType_; ///< Selection type (none, custom, cut,...)
      bool           doSelection_;      ///< Only false if type = "none"
      std::auto_ptr<ElectronSelector> selector_;   ///< Actually performs the selection
      
      /// Returns the appropriate cluster shape.
      /// This is a copy of the Egamma code and it should disappear in the future
      /// (once cluster shapes are put directly in electron, should be in 2_0_0).
      /// See EgammaAnalysis/ElectronIDAlgos/interface/ElectronIDAlgo.h
      const reco::ClusterShapeRef& getClusterShape_( const reco::GsfElectron* electron, 
                                                     const edm::Event&        event
                                                     ) const;

  };


}

#endif
