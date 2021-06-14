/**
  \class    pat::PATCompositeCandidateProducer PATCompositeCandidateProducer.h "PhysicsTools/PatAlgos/interface/PATCompositeCandidateProducer.h"
  \brief    Produces the pat::CompositeCandidate

   The PATCompositeCandidateProducer produces the analysis-level pat::CompositeCandidate starting from
   any collection of Candidates

  \author   Salvatore Rappoccio
  \version  $Id: PATCompositeCandidateProducer.h,v 1.3 2009/06/25 23:49:35 gpetrucc Exp $
*/

#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "CommonTools/Utils/interface/EtComparator.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"
#include "PhysicsTools/PatAlgos/interface/VertexingHelper.h"

namespace pat {

  class PATCompositeCandidateProducer : public edm::stream::EDProducer<> {
  public:
    explicit PATCompositeCandidateProducer(const edm::ParameterSet& iConfig);
    ~PATCompositeCandidateProducer() override;

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  private:
    // configurables
    const edm::EDGetTokenT<edm::View<reco::CompositeCandidate> > srcToken_;  // list of reco::CompositeCandidates

    const bool useUserData_;
    pat::PATUserDataHelper<pat::CompositeCandidate> userDataHelper_;

    const bool addEfficiencies_;
    pat::helper::EfficiencyLoader efficiencyLoader_;

    const bool addResolutions_;
    pat::helper::KinResolutionsLoader resolutionLoader_;
  };

}  // namespace pat

using namespace pat;
using namespace std;
using namespace edm;

PATCompositeCandidateProducer::PATCompositeCandidateProducer(const ParameterSet& iConfig)
    : srcToken_(consumes<edm::View<reco::CompositeCandidate> >(iConfig.getParameter<InputTag>("src"))),
      useUserData_(iConfig.exists("userData")),
      userDataHelper_(iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector()),
      addEfficiencies_(iConfig.getParameter<bool>("addEfficiencies")),
      addResolutions_(iConfig.getParameter<bool>("addResolutions")) {
  // Efficiency configurables
  if (addEfficiencies_) {
    efficiencyLoader_ =
        pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"), consumesCollector());
  }

  // Resolution configurables
  if (addResolutions_) {
    resolutionLoader_ = pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"));
  }

  // produces vector of particles
  produces<vector<pat::CompositeCandidate> >();
}

PATCompositeCandidateProducer::~PATCompositeCandidateProducer() {}

void PATCompositeCandidateProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  // Get the vector of CompositeCandidate's from the event
  Handle<View<reco::CompositeCandidate> > cands;
  iEvent.getByToken(srcToken_, cands);

  if (efficiencyLoader_.enabled())
    efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled())
    resolutionLoader_.newEvent(iEvent, iSetup);

  auto myCompositeCandidates = std::make_unique<vector<pat::CompositeCandidate> >();

  if (cands.isValid()) {
    View<reco::CompositeCandidate>::const_iterator ibegin = cands->begin(), iend = cands->end(), i = ibegin;
    for (; i != iend; ++i) {
      pat::CompositeCandidate cand(*i);

      if (useUserData_) {
        userDataHelper_.add(cand, iEvent, iSetup);
      }

      if (efficiencyLoader_.enabled())
        efficiencyLoader_.setEfficiencies(cand, cands->refAt(i - cands->begin()));
      if (resolutionLoader_.enabled())
        resolutionLoader_.setResolutions(cand);

      myCompositeCandidates->push_back(std::move(cand));
    }

  }  // end if the two handles are valid

  iEvent.put(std::move(myCompositeCandidates));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATCompositeCandidateProducer);
