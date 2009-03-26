//
// $Id: PATMuonProducer.h,v 1.15 2008/11/28 22:05:56 lowette Exp $
//

#ifndef PhysicsTools_PatAlgos_PATMuonProducer_h
#define PhysicsTools_PatAlgos_PATMuonProducer_h

/**
  \class    pat::PATMuonProducer PATMuonProducer.h "PhysicsTools/PatAlgos/interface/PATMuonProducer.h"
  \brief    Produces pat::Muon's

   The PATMuonProducer produces analysis-level pat::Muon's starting from
   a collection of objects of reco::Muon.

  \author   Steven Lowette, Roger Wolf
  \version  $Id: PATMuonProducer.h,v 1.15 2008/11/28 22:05:56 lowette Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include "DataFormats/PatCandidates/interface/Muon.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include <string>


namespace pat {

  class LeptonLRCalc;
  class TrackerIsolationPt;
  class CaloIsolationEnergy;


  class PATMuonProducer : public edm::EDProducer {

    public:

      explicit PATMuonProducer(const edm::ParameterSet & iConfig);
      ~PATMuonProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
      typedef edm::RefToBase<reco::Muon> MuonBaseRef;

    private:

      // configurables
      edm::InputTag muonSrc_;

      bool          embedTrack_;
      bool          embedStandAloneMuon_;
      bool          embedCombinedMuon_;


      bool          embedPickyMuon_;
      bool          embedTpfmsMuon_;
      
      bool          addTeVRefits_;
      edm::InputTag pickySrc_;
      edm::InputTag tpfmsSrc_;

  
      bool          addGenMatch_;
      bool          embedGenMatch_;
      std::vector<edm::InputTag> genMatchSrc_;
      bool          addTrigMatch_;
      std::vector<edm::InputTag> trigMatchSrc_;
      bool          addResolutions_;
      bool          addLRValues_;


      // pflow specific
      bool          useParticleFlow_;
      edm::InputTag pfMuonSrc_;
      bool          embedPFCandidate_;


      typedef std::vector<edm::Handle<edm::Association<reco::GenParticleCollection> > > GenAssociations;

      typedef std::vector<edm::Handle<edm::Association<TriggerPrimitiveCollection> > > TrigAssociations;


      void fillMuon( Muon& aMuon, 
		     const MuonBaseRef& muonRef,
		     const reco::CandidateBaseRef& baseRef,
		     const GenAssociations& genMatches,
		     const TrigAssociations&  trigMatches) const;

     

      // tools
      GreaterByPt<Muon>      pTComparator_;

      pat::helper::MultiIsolator isolator_; 
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      std::vector<std::pair<pat::IsolationKeys,edm::InputTag> > isoDepositLabels_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Muon>      userDataHelper_;

  };


}

#endif
