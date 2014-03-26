//
// $Id: PATTauProducer.h,v 1.23.12.1 2013/05/31 14:54:23 veelken Exp $
//

#ifndef PhysicsTools_PatAlgos_PATTauProducer_h
#define PhysicsTools_PatAlgos_PATTauProducer_h

/**
  \class    pat::PATTauProducer PATTauProducer.h "PhysicsTools/PatAlgos/interface/PATTauProducer.h"
  \brief    Produces pat::Tau's

   The PATTauProducer produces analysis-level pat::Tau's starting from
   a collection of objects of TauType.

  \author   Steven Lowette, Christophe Delaere
  \version  $Id: PATTauProducer.h,v 1.23.12.1 2013/05/31 14:54:23 veelken Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"

#include "DataFormats/PatCandidates/interface/Tau.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataMerger.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include "DataFormats/TauReco/interface/CaloTauDiscriminator.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include <string>


namespace pat {

  class PATTauProducer : public edm::EDProducer {

    public:

      explicit PATTauProducer(const edm::ParameterSet & iConfig);
      ~PATTauProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:

      // configurables
      edm::InputTag tauSrc_;
      edm::InputTag tauTransverseImpactParameterSrc_;
      bool embedIsolationTracks_;
      bool embedLeadTrack_;
      bool embedSignalTracks_;
      bool embedLeadPFCand_; 
      bool embedLeadPFChargedHadrCand_; 
      bool embedLeadPFNeutralCand_; 
      bool embedSignalPFCands_; 
      bool embedSignalPFChargedHadrCands_; 
      bool embedSignalPFNeutralHadrCands_; 
      bool embedSignalPFGammaCands_; 
      bool embedIsolationPFCands_; 
      bool embedIsolationPFChargedHadrCands_; 
      bool embedIsolationPFNeutralHadrCands_; 
      bool embedIsolationPFGammaCands_; 

      bool          addGenMatch_;
      bool          embedGenMatch_;
      std::vector<edm::InputTag> genMatchSrc_;

      bool          addGenJetMatch_;
      bool          embedGenJetMatch_;
      edm::InputTag genJetMatchSrc_;

      bool          addTauJetCorrFactors_;
      std::vector<edm::InputTag> tauJetCorrFactorsSrc_;

      bool          addTauID_;
      typedef std::pair<std::string, edm::InputTag> NameTag;
      std::vector<NameTag> tauIDSrcs_;

      // tools
      GreaterByPt<Tau>       pTTauComparator_;

      pat::helper::MultiIsolator isolator_; 
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      std::vector<std::pair<pat::IsolationKeys,edm::InputTag> > isoDepositLabels_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;
      
      bool addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Tau>      userDataHelper_;

      template <typename TauCollectionType, typename TauDiscrType> float getTauIdDiscriminator(const edm::Handle<TauCollectionType>&, size_t, const edm::Handle<TauDiscrType>&);
  };

}

#endif
