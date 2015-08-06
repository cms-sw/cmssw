//
//

#ifndef PhysicsTools_PatAlgos_PATMETProducer_h
#define PhysicsTools_PatAlgos_PATMETProducer_h

/**
  \class    pat::PATMETProducer PATMETProducer.h "PhysicsTools/PatAlgos/interface/PATMETProducer.h"
  \brief    Produces the pat::MET

   The PATMETProducer produces the analysis-level pat::MET starting from
   a collection of objects of METType.

  \author   Steven Lowette
  \version  $Id: PATMETProducer.h,v 1.10 2009/06/25 23:49:35 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/Utils/interface/EtComparator.h"

#include "DataFormats/PatCandidates/interface/MET.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"


#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoMET/METAlgorithms/interface/METSignificance.h"

namespace pat {

  class PATMETProducer : public edm::stream::EDProducer<> {

    public:

      explicit PATMETProducer(const edm::ParameterSet & iConfig);
      ~PATMETProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:

      // configurables
      edm::InputTag metSrc_;
      edm::EDGetTokenT<edm::View<reco::MET> > metToken_;
      bool          addGenMET_;
      edm::EDGetTokenT<edm::View<reco::GenMET> > genMETToken_;
      bool          addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;
      bool          addMuonCorr_;
      edm::InputTag muonSrc_;
      // tools
      GreaterByEt<MET> eTComparator_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::MET>      userDataHelper_;

    //MET Significance
    bool calculateMETSignificance_;
    metsig::METSignificance* metSigAlgo_;
    edm::EDGetTokenT<edm::View<reco::Jet> > jetToken_;
    edm::EDGetTokenT<edm::View<reco::Candidate> > pfCandToken_;
    std::vector< edm::EDGetTokenT<edm::View<reco::Candidate> > > lepTokens_;

    const reco::METCovMatrix getMETCovMatrix(const edm::Event& event) const;

  };


}

#endif
