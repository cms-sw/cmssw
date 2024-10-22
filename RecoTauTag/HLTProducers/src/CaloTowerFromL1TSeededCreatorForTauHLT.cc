// makes CaloTowerCandidates from CaloTowers
// original author: L.Lista INFN, modifyed by: F.Ratnikov UMd
// Author for regionality A. Nikitenko
// Modified by S. Gennai + T. Strebler

#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTauTag/HLTProducers/interface/CaloTowerFromL1TSeededCreatorForTauHLT.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include <cmath>

using namespace edm;
using namespace reco;
using namespace std;

CaloTowerFromL1TSeededCreatorForTauHLT::CaloTowerFromL1TSeededCreatorForTauHLT(const ParameterSet& p)
    : m_verbose(p.getUntrackedParameter<int>("verbose", 0)),
      m_towers_token(consumes<CaloTowerCollection>(p.getParameter<InputTag>("towers"))),
      m_cone(p.getParameter<double>("UseTowersInCone")),
      m_tauTrigger_token(consumes<trigger::TriggerFilterObjectWithRefs>(p.getParameter<InputTag>("TauTrigger"))),
      m_EtThreshold(p.getParameter<double>("minimumEt")),
      m_EThreshold(p.getParameter<double>("minimumE")) {
  produces<CaloTowerCollection>();
}

CaloTowerFromL1TSeededCreatorForTauHLT::~CaloTowerFromL1TSeededCreatorForTauHLT() = default;

void CaloTowerFromL1TSeededCreatorForTauHLT::produce(StreamID sid, Event& evt, const EventSetup& stp) const {
  edm::Handle<CaloTowerCollection> caloTowers;
  evt.getByToken(m_towers_token, caloTowers);

  double m_cone2 = m_cone * m_cone;

  // L1 seeds
  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredTaus;
  evt.getByToken(m_tauTrigger_token, l1TriggeredTaus);

  auto cands = std::make_unique<CaloTowerCollection>();
  cands->reserve(caloTowers->size());

  l1t::TauVectorRef tauCandRefVec;
  l1TriggeredTaus->getObjects(trigger::TriggerL1Tau, tauCandRefVec);

  for (auto const& tauCandRef : tauCandRefVec) {
    for (auto const& cal : *caloTowers) {
      bool isAccepted = false;
      if (m_verbose == 2) {
        edm::LogInfo("JetDebugInfo") << "CaloTowerFromL1TSeededCreatorForTauHLT::produce->  tower et/eta/phi/e: "
                                     << cal.et() << '/' << cal.eta() << '/' << cal.phi() << '/' << cal.energy()
                                     << " is...";
      }
      if (cal.et() >= m_EtThreshold && cal.energy() >= m_EThreshold) {
        math::PtEtaPhiELorentzVector p(cal.et(), cal.eta(), cal.phi(), cal.energy());
        double delta2 = deltaR2((tauCandRef->p4()).Vect(), p);
        if (delta2 < m_cone2) {
          isAccepted = true;
          cands->push_back(cal);
        }
      }

      if (m_verbose == 2) {
        if (isAccepted)
          edm::LogInfo("JetDebugInfo") << "accepted \n";
        else
          edm::LogInfo("JetDebugInfo") << "rejected \n";
      }
    }
  }

  evt.put(std::move(cands));
}

void CaloTowerFromL1TSeededCreatorForTauHLT::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription aDesc;

  aDesc.add<edm::InputTag>("TauTrigger", edm::InputTag("hltL1sDoubleIsoTau40er"))
      ->setComment("Name of trigger filter for L1 seeds");
  aDesc.add<edm::InputTag>("towers", edm::InputTag("towerMaker"))->setComment("Input tower collection");
  aDesc.add<double>("UseTowersInCone", 0.8)->setComment("Radius of cone around seed");
  aDesc.add<double>("minimumE", 0.8)->setComment("Minimum tower energy");
  aDesc.add<double>("minimumEt", 0.5)->setComment("Minimum tower ET");
  aDesc.addUntracked<int>("verbose", 0)->setComment("Verbosity level; 0=silent");

  desc.add("CaloTowerFromL1TSeededCreatorForTauHLT", aDesc);
  desc.setComment("Produce tower collection around L1 particle seed.");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CaloTowerFromL1TSeededCreatorForTauHLT);
