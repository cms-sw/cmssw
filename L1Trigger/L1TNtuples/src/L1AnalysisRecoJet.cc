#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoJet.h"

//#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

L1Analysis::L1AnalysisRecoJet::L1AnalysisRecoJet()
{
}

L1Analysis::L1AnalysisRecoJet::~L1AnalysisRecoJet()
{
}


void L1Analysis::L1AnalysisRecoJet::SetCaloJet(const edm::Event& event,
                 const edm::EventSetup& setup,
                 edm::Handle<reco::CaloJetCollection> caloJets,
                 unsigned maxJet)
{

  recoJet_.nJets=0;

  for(reco::CaloJetCollection::const_iterator it=caloJets->begin();
      it!=caloJets->end() && recoJet_.nJets < maxJet;
      ++it) {

    recoJet_.et.push_back(it->et());
    // recoJet_.etCorr.push_back(it->et());// * scale);
    // recoJet_.corrFactor.push_back(1.);//scale);
    recoJet_.eta.push_back(it->eta());
    recoJet_.phi.push_back(it->phi());
    recoJet_.e.push_back(it->energy());
    recoJet_.eEMF.push_back(it->emEnergyFraction());
    recoJet_.eEmEB.push_back(it->emEnergyInEB());
    recoJet_.eEmEE.push_back(it->emEnergyInEE());
    recoJet_.eEmHF.push_back(it->emEnergyInHF());
    recoJet_.eHadHB.push_back(it->hadEnergyInHB());
    recoJet_.eHadHE.push_back(it->hadEnergyInHE());
    recoJet_.eHadHO.push_back(it->hadEnergyInHO());
    recoJet_.eHadHF.push_back(it->hadEnergyInHF());
    recoJet_.eMaxEcalTow.push_back(it->maxEInEmTowers());
    recoJet_.eMaxHcalTow.push_back(it->maxEInHadTowers());
    recoJet_.towerArea.push_back(it->towersArea());
    //    recoJet_.towerSize.push_back(static_cast<int>(it->getCaloConstituents().size()));
    // recoJet_.n60.push_back(it->n60());
    // recoJet_.n90.push_back(it->n90());

    // recoJet_.n90hits.push_back(1.); //int((*jetsID)[jetRef].n90Hits));
    // recoJet_.fHPD.push_back(1.); //(*jetsID)[jetRef].fHPD);
    // recoJet_.fRBX.push_back(1.); //(*jetsID)[jetRef].fRBX);

    recoJet_.nJets++;

  }
}


// void L1Analysis::L1AnalysisRecoJet::SetPFJet(const edm::Event& event,
//                  const edm::EventSetup& setup,
//                  edm::Handle<reco::PFJetCollection> caloJets,
//                  unsigned maxJet)
// {

//   recoPFJet_.nJets=0;

//   for(reco::CaloJetCollection::const_iterator it=caloJets->begin();
//       it!=caloJets->end() && recoJet_.nJets < maxJet;
//       ++it) {

//     recoJet_.et.push_back(it->et());
//     // recoJet_.etCorr.push_back(it->et());// * scale);
//     // recoJet_.corrFactor.push_back(1.);//scale);
//     recoJet_.eta.push_back(it->eta());
//     recoJet_.phi.push_back(it->phi());
//     recoJet_.e.push_back(it->energy());
//     // recoJet_.eEMF.push_back(it->emEnergyFraction());
//     // recoJet_.eEmEB.push_back(it->emEnergyInEB());
//     // recoJet_.eEmEE.push_back(it->emEnergyInEE());
//     // recoJet_.eEmHF.push_back(it->emEnergyInHF());
//     // recoJet_.eHadHB.push_back(it->hadEnergyInHB());
//     // recoJet_.eHadHE.push_back(it->hadEnergyInHE());
//     // recoJet_.eHadHO.push_back(it->hadEnergyInHO());
//     // recoJet_.eHadHF.push_back(it->hadEnergyInHF());
//     // recoJet_.eMaxEcalTow.push_back(it->maxEInEmTowers());
//     // recoJet_.eMaxHcalTow.push_back(it->maxEInHadTowers());
//     // recoJet_.towerArea.push_back(it->towersArea());
//     //    recoJet_.towerSize.push_back(static_cast<int>(it->getCaloConstituents().size()));
//     // recoJet_.n60.push_back(it->n60());
//     // recoJet_.n90.push_back(it->n90());

//     // recoJet_.n90hits.push_back(1.); //int((*jetsID)[jetRef].n90Hits));
//     // recoJet_.fHPD.push_back(1.); //(*jetsID)[jetRef].fHPD);
//     // recoJet_.fRBX.push_back(1.); //(*jetsID)[jetRef].fRBX);

//     recoJet_.nJets++;

//   }
// }
