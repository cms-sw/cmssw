// makes CaloTowerCandidates from CaloTowers
// original author: L.Lista INFN, modifyed by: F.Ratnikov UMd
// Author for regionality A. Nikitenko
// Modified by S. Gennai

#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTauTag/HLTProducers/interface/CaloTowerCreatorForTauHLT.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include <cmath>

using namespace edm;
using namespace reco;
using namespace std;
using namespace l1extra;

CaloTowerCreatorForTauHLT::CaloTowerCreatorForTauHLT(const ParameterSet& p)
    : mVerbose(p.getUntrackedParameter<int>("verbose", 0)),
      mtowers_token(consumes<CaloTowerCollection>(p.getParameter<InputTag>("towers"))),
      mCone(p.getParameter<double>("UseTowersInCone")),
      mTauTrigger_token(consumes<L1JetParticleCollection>(p.getParameter<InputTag>("TauTrigger"))),
      mEtThreshold(p.getParameter<double>("minimumEt")),
      mEThreshold(p.getParameter<double>("minimumE")),
      mTauId(p.getParameter<int>("TauId")) {
  produces<CaloTowerCollection>();
}

CaloTowerCreatorForTauHLT::~CaloTowerCreatorForTauHLT() {}

void CaloTowerCreatorForTauHLT::produce(StreamID sid, Event& evt, const EventSetup& stp) const {
  edm::Handle<CaloTowerCollection> caloTowers;
  evt.getByToken(mtowers_token, caloTowers);

  // imitate L1 seeds
  edm::Handle<L1JetParticleCollection> jetsgen;
  evt.getByToken(mTauTrigger_token, jetsgen);

  std::unique_ptr<CaloTowerCollection> cands(new CaloTowerCollection);
  cands->reserve(caloTowers->size());

  int idTau = 0;
  L1JetParticleCollection::const_iterator myL1Jet = jetsgen->begin();
  for (; myL1Jet != jetsgen->end(); myL1Jet++) {
    if (idTau == mTauId) {
      unsigned idx = 0;
      for (; idx < caloTowers->size(); idx++) {
        const CaloTower* cal = &((*caloTowers)[idx]);
        bool isAccepted = false;
        if (mVerbose == 2) {
          edm::LogInfo("JetDebugInfo") << "CaloTowerCreatorForTauHLT::produce-> " << idx
                                       << " tower et/eta/phi/e: " << cal->et() << '/' << cal->eta() << '/' << cal->phi()
                                       << '/' << cal->energy() << " is...";
        }
        if (cal->et() >= mEtThreshold && cal->energy() >= mEThreshold) {
          math::PtEtaPhiELorentzVector p(cal->et(), cal->eta(), cal->phi(), cal->energy());
          double delta = ROOT::Math::VectorUtil::DeltaR((*myL1Jet).p4().Vect(), p);

          if (delta < mCone) {
            isAccepted = true;
            cands->push_back(*cal);
          }
        }
        if (mVerbose == 2) {
          if (isAccepted)
            edm::LogInfo("JetDebugInfo") << "accepted \n";
          else
            edm::LogInfo("JetDebugInfo") << "rejected \n";
        }
      }
    }
    idTau++;
  }

  evt.put(std::move(cands));
}

void CaloTowerCreatorForTauHLT::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription aDesc;
  aDesc.add<edm::InputTag>("TauTrigger", edm::InputTag("l1extraParticles", "Tau"))
      ->setComment("L1ExtraJet collection for seeding");
  aDesc.add<int>("TauId", 0)->setComment("Item from L1ExtraJet collection used for seeding");
  aDesc.add<edm::InputTag>("towers", edm::InputTag("towerMaker"))->setComment("Input tower collection");
  aDesc.add<double>("UseTowersInCone", 0.8)->setComment("Radius of cone around seed");
  aDesc.add<double>("minimumE", 0.8)->setComment("Minimum tower energy");
  aDesc.add<double>("minimumEt", 0.5)->setComment("Minimum tower ET");
  aDesc.addUntracked<int>("verbose", 0)->setComment("Verbosity level; 0=silent");
  desc.setComment("Produce tower collection around L1ExtraJetParticle seed.");
  desc.add("caloTowerMakerHLT", aDesc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CaloTowerCreatorForTauHLT);
