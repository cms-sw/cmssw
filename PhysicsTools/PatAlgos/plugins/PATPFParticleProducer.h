//
//

#ifndef PhysicsTools_PatAlgos_PATPFParticleProducer_h
#define PhysicsTools_PatAlgos_PATPFParticleProducer_h

/**
  \class    pat::PATPFParticleProducer PATPFParticleProducer.h "PhysicsTools/PatAlgos/interface/PATPFParticleProducer.h"
  \brief    Produces pat::PFParticle's

   The PATPFParticleProducer produces analysis-level pat::PFParticle's starting from
   a collection of objects of reco::PFCandidate.

  \author   Steven Lowette, Roger Wolf
  \version  $Id: PATPFParticleProducer.h,v 1.8 2012/05/26 10:42:53 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "DataFormats/PatCandidates/interface/PFParticle.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include <string>


namespace pat {

  class LeptonLRCalc;

  class PATPFParticleProducer : public edm::stream::EDProducer<> {

    public:

      explicit PATPFParticleProducer(const edm::ParameterSet & iConfig);
      ~PATPFParticleProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

    private:

      // configurables
      edm::EDGetTokenT<edm::View<reco::PFCandidate> > pfCandidateToken_;
      bool          embedPFCandidate_;
      bool          addGenMatch_;
      bool          embedGenMatch_;
      std::vector<edm::EDGetTokenT<edm::Association<reco::GenParticleCollection> > > genMatchTokens_;
      // tools
      GreaterByPt<PFParticle>      pTComparator_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;

      bool addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::PFParticle> userDataHelper_;


  };


}

#endif
