//
// $Id: PATElectronProducer.h,v 1.18 2009/06/25 23:49:35 gpetrucc Exp $
//

#ifndef PhysicsTools_PatAlgos_PATElectronProducer_h
#define PhysicsTools_PatAlgos_PATElectronProducer_h

/**
  \class    pat::PATElectronProducer PATElectronProducer.h "PhysicsTools/PatAlgos/interface/PATElectronProducer.h"
  \brief    Produces pat::Electron's

   The PATElectronProducer produces analysis-level pat::Electron's starting from
   a collection of objects of reco::GsfElectron.

  \author   Steven Lowette, James Lamb
  \version  $Id: PATElectronProducer.h,v 1.18 2009/06/25 23:49:35 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"

#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include <string>


namespace pat {


  class TrackerIsolationPt;
  class CaloIsolationEnergy;
  class LeptonLRCalc;


  class PATElectronProducer : public edm::EDProducer {

    public:

      explicit PATElectronProducer(const edm::ParameterSet & iConfig);
      ~PATElectronProducer();  

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:

      // configurables
      edm::InputTag electronSrc_;
      bool          embedGsfTrack_;
      bool          embedSuperCluster_;
      bool          embedTrack_;
      bool          addGenMatch_;
      bool          embedGenMatch_;
      std::vector<edm::InputTag> genMatchSrc_;

      /// pflow specific
      bool          useParticleFlow_;
      edm::InputTag pfElecSrc_;
      bool          embedPFCandidate_; 

      typedef std::vector<edm::Handle<edm::Association<reco::GenParticleCollection> > > GenAssociations;

      void FillElectron(Electron& aEl,
			const edm::RefToBase<reco::GsfElectron>& elecRef,
			const reco::CandidateBaseRef& baseRef,
			const GenAssociations& genMatches) const;
  

      bool          addElecID_;
      typedef std::pair<std::string, edm::InputTag> NameTag;
      std::vector<NameTag> elecIDSrcs_;

      // tools
      GreaterByPt<Electron>       pTComparator_;

      pat::helper::MultiIsolator isolator_; 
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      std::vector<std::pair<pat::IsolationKeys,edm::InputTag> > isoDepositLabels_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;
      
      bool addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Electron>      userDataHelper_;
      
      
  };


}

#endif
