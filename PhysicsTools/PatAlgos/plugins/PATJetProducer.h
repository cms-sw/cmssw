//
// $Id: PATJetProducer.h,v 1.1.2.1 2008/03/06 10:44:09 llista Exp $
//

#ifndef PhysicsTools_PatAlgos_PATJetProducer_h
#define PhysicsTools_PatAlgos_PATJetProducer_h

/**
  \class    PATJetProducer PATJetProducer.h "PhysicsTools/PatAlgos/interface/PATJetProducer.h"
  \brief    Produces pat::Jet's

   The PATJetProducer produces analysis-level pat::Jet's starting from
   a collection of objects of JetType.

  \author   Steven Lowette, Jeremy Andrea
  \version  $Id: PATJetProducer.h,v 1.1.2.1 2008/03/06 10:44:09 llista Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "PhysicsTools/Utilities/interface/EtComparator.h"
#include "PhysicsTools/JetCharge/interface/JetCharge.h"
#include "PhysicsTools/PatUtils/interface/SimpleJetTrackAssociator.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"


class JetFlavourIdentifier;


namespace pat {


  class ObjectResolutionCalc;


  class PATJetProducer : public edm::EDProducer {

    public:

      explicit PATJetProducer(const edm::ParameterSet & iConfig);
      ~PATJetProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:

      // configurables
      edm::InputTag            jetsSrc_;
      bool                     getJetMCFlavour_;
      edm::InputTag            jetPartonMapSource_;
      bool                     addGenPartonMatch_;
      edm::InputTag            genPartonSrc_;
      bool                     addGenJetMatch_;
      edm::InputTag            genJetSrc_;
      bool                     addPartonJetMatch_;
      edm::InputTag            partonJetSrc_;
      edm::InputTag            jetCorrFactorsSrc_;
      bool                     addResolutions_;
      bool                     useNNReso_;
      std::string              caliJetResoFile_;
      std::string              caliBJetResoFile_;

      bool                     addBTagInfo_;
      std::string              tagModuleLabelPostfix_; 
      bool                     addDiscriminators_; 
      bool                     addJetTagRefs_;
      std::vector<std::string> tagModuleLabelsToKeep_;
      bool                     addAssociatedTracks_;
      edm::ParameterSet        trackAssociationPSet_;
      bool                     addJetCharge_;
      edm::ParameterSet        jetChargePSet_;
      // tools
      ObjectResolutionCalc             * theResoCalc_;
      ObjectResolutionCalc             * theBResoCalc_;
      ::helper::SimpleJetTrackAssociator   simpleJetTrackAssociator_;
      JetCharge                        * jetCharge_;
      GreaterByEt<Jet>                   eTComparator_;

  };


}

#endif
