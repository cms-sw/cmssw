//
// $Id: PATJetProducer.h,v 1.1 2008/01/15 13:30:02 lowette Exp $
//

#ifndef PhysicsTools_PatAlgos_PATJetProducer_h
#define PhysicsTools_PatAlgos_PATJetProducer_h

/**
  \class    PATJetProducer PATJetProducer.h "PhysicsTools/PatAlgos/interface/PATJetProducer.h"
  \brief    Produces pat::Jet's

   The PATJetProducer produces analysis-level pat::Jet's starting from
   a collection of objects of JetType.

  \author   Steven Lowette, Jeremy Andrea
  \version  $Id: PATJetProducer.h,v 1.1 2008/01/15 13:30:02 lowette Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

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

      // TEMP Jet cleaning from electrons
      std::vector<Electron> selectIsolated(const std::vector<Electron> & electrons, float isoCut);
      std::vector<Muon>     selectIsolated(const std::vector<Muon> & muons,         float isoCut);
      // TEMP End

      // configurables
      edm::InputTag            jetsSrc_;
      // TEMP Jet cleaning from electrons
      bool                     doJetCleaning_;
      edm::InputTag            electronsLabel_;
      edm::InputTag            muonsLabel_;
      float                    LEPJETDR_;
      float                    ELEISOCUT_;
      float                    MUISOCUT_;
      // TEMP End
      bool                     getJetMCFlavour_;
      edm::InputTag            jetPartonMapSource_;
      bool                     addGenPartonMatch_;
      edm::InputTag            genPartonSrc_;
      bool                     addGenJetMatch_;
      edm::InputTag            genJetSrc_;
      bool                     addPartonJetMatch_;
      edm::InputTag            partonJetSrc_;
      bool                     addResolutions_;
      bool                     useNNReso_;
      std::string              caliJetResoFile_;
      std::string              caliBJetResoFile_;
      bool                     addBTagInfo_;
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
