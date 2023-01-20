// makes CaloTowerCandidates from CaloTowers
// original author: L.Lista INFN, modifyed by: F.Ratnikov UMd
// Author for regionality A. Nikitenko
// Modified by S. Gennai

#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTauTag/HLTProducers/interface/CaloTowerFromL1TCreatorForTauHLT.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include <cmath>

using namespace edm;
using namespace reco;
using namespace std;

CaloTowerFromL1TCreatorForTauHLT::CaloTowerFromL1TCreatorForTauHLT(const ParameterSet& p)
    : mBX(p.getParameter<int>("BX")),
      mVerbose(p.getUntrackedParameter<int>("verbose", 0)),
      mtowers_token(consumes<CaloTowerCollection>(p.getParameter<InputTag>("towers"))),
      mCone(p.getParameter<double>("UseTowersInCone")),
      mTauTrigger_token(consumes<l1t::TauBxCollection>(p.getParameter<InputTag>("TauTrigger"))),
      mEtThreshold(p.getParameter<double>("minimumEt")),
      mEThreshold(p.getParameter<double>("minimumE")),
      mTauId(p.getParameter<int>("TauId")) {
  produces<CaloTowerCollection>();
}

CaloTowerFromL1TCreatorForTauHLT::~CaloTowerFromL1TCreatorForTauHLT() {}

void CaloTowerFromL1TCreatorForTauHLT::produce(StreamID sid, Event& evt, const EventSetup& stp) const {
  edm::Handle<CaloTowerCollection> caloTowers;
  evt.getByToken(mtowers_token, caloTowers);

  // imitate L1 seeds
  edm::Handle<l1t::TauBxCollection> jetsgen;
  evt.getByToken(mTauTrigger_token, jetsgen);

  std::unique_ptr<CaloTowerCollection> cands(new CaloTowerCollection);
  cands->reserve(caloTowers->size());

  int idTau = 0;
  if (jetsgen.isValid()) {
    for (auto myL1Jet = jetsgen->begin(mBX); myL1Jet != jetsgen->end(mBX); myL1Jet++) {
      if (idTau == mTauId) {
        unsigned idx = 0;
        for (; idx < caloTowers->size(); idx++) {
          const CaloTower* cal = &((*caloTowers)[idx]);
          bool isAccepted = false;
          if (mVerbose == 2) {
            edm::LogInfo("JetDebugInfo") << "CaloTowerFromL1TCreatorForTauHLT::produce-> " << idx
                                         << " tower et/eta/phi/e: " << cal->et() << '/' << cal->eta() << '/'
                                         << cal->phi() << '/' << cal->energy() << " is...";
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
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade jet bx collection not found." << std::endl;
  }

  evt.put(std::move(cands));
}

void CaloTowerFromL1TCreatorForTauHLT::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription aDesc;

  aDesc.add<edm::InputTag>("TauTrigger", edm::InputTag("caloStage2Digis"))->setComment("L1 Tau collection for seeding");
  aDesc.add<edm::InputTag>("towers", edm::InputTag("towerMaker"))->setComment("Input tower collection");
  aDesc.add<int>("TauId", 0)->setComment("Item from L1 Tau collection used for seeding. From 0 to 11");
  aDesc.add<double>("UseTowersInCone", 0.8)->setComment("Radius of cone around seed");
  aDesc.add<double>("minimumE", 0.8)->setComment("Minimum tower energy");
  aDesc.add<double>("minimumEt", 0.5)->setComment("Minimum tower ET");
  aDesc.add<int>("BX", 0)->setComment("Set bunch crossing; 0 = in time, -1 = previous, 1 = following");
  aDesc.addUntracked<int>("verbose", 0)->setComment("Verbosity level; 0=silent");

  desc.add("CaloTowerFromL1TCreatorForTauHLT", aDesc);
  desc.setComment("Produce tower collection around L1 particle seed.");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CaloTowerFromL1TCreatorForTauHLT);
