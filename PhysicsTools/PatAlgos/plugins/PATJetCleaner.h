//
// $Id: PATJetCleaner.h,v 1.3 2008/03/11 11:02:43 llista Exp $
//

#ifndef PhysicsTools_PatAlgos_PATJetCleaner_h
#define PhysicsTools_PatAlgos_PATJetCleaner_h

/**
  \class    pat::PATJetCleaner PATJetCleaner.h "PhysicsTools/PatAlgos/interface/PATJetCleaner.h"
  \brief    Produces pat::Jet's

   The PATJetCleaner produces analysis-level pat::Jet's starting from
   a collection of objects of JetType.

  \author   Steven Lowette, Jeremy Andrea
  \version  $Id: PATJetCleaner.h,v 1.3 2008/03/11 11:02:43 llista Exp $
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/plugins/CleanerHelper.h"
#include "PhysicsTools/PatAlgos/interface/OverlapHelper.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "PhysicsTools/Utilities/interface/EtComparator.h"
#include "PhysicsTools/PatUtils/interface/JetSelector.h"
#include "PhysicsTools/PatUtils/interface/JetSelector.icc"
#include "PhysicsTools/PatUtils/interface/JetSelection.h"

namespace pat {

  template<typename JetIn, typename JetOut>
  class PATJetCleaner : public edm::EDProducer {

    public:

      explicit PATJetCleaner(const edm::ParameterSet & iConfig);
      ~PATJetCleaner();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
      virtual void endJob();

    private:
      // configurables
      edm::InputTag            jetSrc_;
      edm::InputTag            likelihoodRej_;
      bool doLikelihoodRej_;
      // helper
      pat::helper::CleanerHelper<JetIn,
                                 JetOut,
                                 std::vector<JetOut>,
                                 GreaterByEt<JetOut> > helper_;

      pat::helper::OverlapHelper overlapHelper_;
  
       ///Jet Selection  similar to Electrons
      edm::ParameterSet selectionCfg_;  ///< Defines all about the selection
      
      typedef JetSelector<JetIn> JetSelectorType;
      std::auto_ptr<JetSelectorType> selector_;   ///< Actually performs the selection

      /// Helper method to fill configuration selection struct
      // Because I wasn't able to use the ParameterAdapter here (templates...)
      const JetSelection fillSelection_( const edm::ParameterSet & iConfig);

  };

  // now I'm typedeffing everything, but I don't think we really need all them
  typedef PATJetCleaner<reco::Jet,reco::Jet>         PATBaseJetCleaner;
  typedef PATJetCleaner<reco::PFJet,reco::PFJet>     PATPFJetCleaner;
  typedef PATJetCleaner<reco::CaloJet,reco::CaloJet> PATCaloJetCleaner;
  typedef PATJetCleaner<reco::PFJet,reco::Jet>       PATPF2BaseJetCleaner;   // I don't think we need this
  typedef PATJetCleaner<reco::CaloJet,reco::Jet>     PATCalo2BaseJetCleaner; // nor this

}


#endif
