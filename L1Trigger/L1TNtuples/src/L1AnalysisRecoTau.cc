#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoTau.h"

//#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

using namespace std;

L1Analysis::L1AnalysisRecoTau::L1AnalysisRecoTau()
{
}


L1Analysis::L1AnalysisRecoTau::~L1AnalysisRecoTau()
{
}


void L1Analysis::L1AnalysisRecoTau::SetTau(const edm::Event& event,
					   const edm::EventSetup& setup,
					   edm::Handle<reco::PFTauCollection> taus, edm::Handle<reco::PFTauDiscriminator> DMFindingOldTaus, edm::Handle<reco::PFTauDiscriminator> DMFindingTaus, edm::Handle<reco::PFTauDiscriminator> TightIsoTaus, edm::Handle<reco::PFTauDiscriminator> LooseIsoTaus, edm::Handle<reco::PFTauDiscriminator> LooseAntiMuon, edm::Handle<reco::PFTauDiscriminator> TightAntiMuon, edm::Handle<reco::PFTauDiscriminator> VLooseAntiElectron, edm::Handle<reco::PFTauDiscriminator> LooseAntiElectron, edm::Handle<reco::PFTauDiscriminator> TightAntiElectron, unsigned maxTau)
{
  recoTau_.nTaus=0;

  for(reco::PFTauCollection::const_iterator it=taus->begin();
      it!=taus->end() && recoTau_.nTaus < maxTau;
      ++it) {

    recoTau_.e.push_back(it->energy());    
    recoTau_.pt.push_back(it->pt());    
    recoTau_.et.push_back(it->et());    
    recoTau_.eta.push_back(it->eta());
    recoTau_.phi.push_back(it->phi());

    edm::Ref<reco::PFTauCollection> tauEdmRef(taus,recoTau_.nTaus);
    recoTau_.TightIsoFlag.push_back((*TightIsoTaus)[tauEdmRef]);
    recoTau_.LooseIsoFlag.push_back((*LooseIsoTaus)[tauEdmRef]);
    recoTau_.LooseAntiMuonFlag.push_back((*LooseAntiMuon)[tauEdmRef]);
    recoTau_.TightAntiMuonFlag.push_back((*TightAntiMuon)[tauEdmRef]);
    recoTau_.VLooseAntiElectronFlag.push_back((*VLooseAntiElectron)[tauEdmRef]);
    recoTau_.LooseAntiElectronFlag.push_back((*LooseAntiElectron)[tauEdmRef]);
    recoTau_.TightAntiElectronFlag.push_back((*TightAntiElectron)[tauEdmRef]);
    recoTau_.DMFindingOldDMs.push_back((*DMFindingOldTaus)[tauEdmRef]);
    recoTau_.DMFindingNewDMs.push_back((*DMFindingTaus)[tauEdmRef]);

    recoTau_.nTaus++;

  }
}


// void L1Analysis::L1AnalysisRecoTau::SetPFJet(const edm::Event& event,
//                  const edm::EventSetup& setup,
//                  edm::Handle<reco::PFJetCollection> taus,
//                  unsigned maxTau)
// {

//   recoPFJet_.nTaus=0;

//   for(reco::CaloJetCollection::const_iterator it=taus->begin();
//       it!=taus->end() && recoTau_.nTaus < maxTau;
//       ++it) {

//     recoTau_.et.push_back(it->et());
//     // recoTau_.etCorr.push_back(it->et());// * scale);
//     // recoTau_.corrFactor.push_back(1.);//scale);
//     recoTau_.eta.push_back(it->eta());
//     recoTau_.phi.push_back(it->phi());
//     recoTau_.e.push_back(it->energy());
//     // recoTau_.eEMF.push_back(it->emEnergyFraction());
//     // recoTau_.eEmEB.push_back(it->emEnergyInEB());
//     // recoTau_.eEmEE.push_back(it->emEnergyInEE());
//     // recoTau_.eEmHF.push_back(it->emEnergyInHF());
//     // recoTau_.eHadHB.push_back(it->hadEnergyInHB());
//     // recoTau_.eHadHE.push_back(it->hadEnergyInHE());
//     // recoTau_.eHadHO.push_back(it->hadEnergyInHO());
//     // recoTau_.eHadHF.push_back(it->hadEnergyInHF());
//     // recoTau_.eMaxEcalTow.push_back(it->maxEInEmTowers());
//     // recoTau_.eMaxHcalTow.push_back(it->maxEInHadTowers());
//     // recoTau_.towerArea.push_back(it->towersArea());
//     //    recoTau_.towerSize.push_back(static_cast<int>(it->getCaloConstituents().size()));
//     // recoTau_.n60.push_back(it->n60());
//     // recoTau_.n90.push_back(it->n90());

//     // recoTau_.n90hits.push_back(1.); //int((*jetsID)[jetRef].n90Hits));
//     // recoTau_.fHPD.push_back(1.); //(*jetsID)[jetRef].fHPD);
//     // recoTau_.fRBX.push_back(1.); //(*jetsID)[jetRef].fRBX);

//     recoTau_.nTaus++;

//   }
// }
