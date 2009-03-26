//
// $Id: PATTauProducer.h,v 1.14 2009/03/26 05:02:42 hegner Exp $
//

#ifndef PhysicsTools_PatAlgos_PATTauProducer_h
#define PhysicsTools_PatAlgos_PATTauProducer_h

/**
  \class    pat::PATTauProducer PATTauProducer.h "PhysicsTools/PatAlgos/interface/PATTauProducer.h"
  \brief    Produces pat::Tau's

   The PATTauProducer produces analysis-level pat::Tau's starting from
   a collection of objects of TauType.

  \author   Steven Lowette, Christophe Delaere
  \version  $Id: PATTauProducer.h,v 1.14 2009/03/26 05:02:42 hegner Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"

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

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:

      // configurables
      edm::InputTag tauSrc_;
      bool embedIsolationTracks_;
      bool embedLeadTrack_;
      bool embedSignalTracks_;

      bool          addGenMatch_;
      bool          embedGenMatch_;
      std::vector<edm::InputTag> genMatchSrc_;

      bool          addGenJetMatch_;
      bool          embedGenJetMatch_;
      edm::InputTag genJetMatchSrc_;
      bool          addTrigMatch_;
      std::vector<edm::InputTag> trigMatchSrc_;
      bool          addResolutions_;
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

      bool useUserData_;
      pat::PATUserDataHelper<pat::Tau>      userDataHelper_;

      template <typename TauCollectionType, typename TauDiscrType> float getTauIdDiscriminator(const edm::Handle<TauCollectionType>&, size_t, const edm::Handle<TauDiscrType>&);

  };

}

#endif
