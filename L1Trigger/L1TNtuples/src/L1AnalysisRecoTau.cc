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
  unsigned int nTausTightIso=0;

  for(reco::PFTauDiscriminator::const_iterator it=TightIsoTaus->begin();
      it!=TightIsoTaus->end() && nTausTightIso < maxTau;
      ++it) {

    recoTau_.TightIsoFlag.push_back((*it).second);
    nTausTightIso++;
  }

  unsigned int nTausLooseIso=0;

  for(reco::PFTauDiscriminator::const_iterator it=LooseIsoTaus->begin();
      it!=LooseIsoTaus->end() && nTausLooseIso < maxTau;
      ++it) {

    recoTau_.LooseIsoFlag.push_back((*it).second);
    nTausLooseIso++;
  }

  unsigned int nTausLooseAntiMuon=0;

  for(reco::PFTauDiscriminator::const_iterator it=LooseAntiMuon->begin();
      it!=LooseAntiMuon->end() && nTausLooseAntiMuon < maxTau;
      ++it) {

    recoTau_.LooseAntiMuonFlag.push_back((*it).second);
    nTausLooseAntiMuon++;
  }

  unsigned int nTausTightAntiMuon=0;

  for(reco::PFTauDiscriminator::const_iterator it=TightAntiMuon->begin();
      it!=TightAntiMuon->end() && nTausTightAntiMuon < maxTau;
      ++it) {

    recoTau_.TightAntiMuonFlag.push_back((*it).second);
    nTausTightAntiMuon++;
  }

  unsigned int nTausVLooseAntiElectron=0;

  for(reco::PFTauDiscriminator::const_iterator it=VLooseAntiElectron->begin();
      it!=VLooseAntiElectron->end() && nTausVLooseAntiElectron < maxTau;
      ++it) {

    recoTau_.VLooseAntiElectronFlag.push_back((*it).second);
    nTausVLooseAntiElectron++;
  }

  unsigned int nTausLooseAntiElectron=0;

  for(reco::PFTauDiscriminator::const_iterator it=LooseAntiElectron->begin();
      it!=LooseAntiElectron->end() && nTausLooseAntiElectron < maxTau;
      ++it) {

    recoTau_.LooseAntiElectronFlag.push_back((*it).second);
    nTausLooseAntiElectron++;
  }

  unsigned int nTausTightAntiElectron=0;

  for(reco::PFTauDiscriminator::const_iterator it=TightAntiElectron->begin();
      it!=TightAntiElectron->end() && nTausTightAntiElectron < maxTau;
      ++it) {

    recoTau_.TightAntiElectronFlag.push_back((*it).second);
    nTausTightAntiElectron++;
  }

  unsigned int nTausDMFindingOld=0;

  for(reco::PFTauDiscriminator::const_iterator it=DMFindingOldTaus->begin();
      it!=DMFindingOldTaus->end() &&  nTausDMFindingOld < maxTau;
      ++it) {

    recoTau_.DMFindingOldDMs.push_back((*it).second);
    nTausDMFindingOld++;
  }

  unsigned int nTausDMFinding=0;

  for(reco::PFTauDiscriminator::const_iterator it=DMFindingTaus->begin();
      it!=DMFindingTaus->end() &&  nTausDMFinding < maxTau;
      ++it) {

    recoTau_.DMFindingNewDMs.push_back((*it).second);
    nTausDMFinding++;
  }

  recoTau_.nTaus=0;

  for(reco::PFTauCollection::const_iterator it=taus->begin();
      it!=taus->end() && recoTau_.nTaus < maxTau;
      ++it) {

    recoTau_.e.push_back(it->energy());    
    recoTau_.pt.push_back(it->pt());    
    recoTau_.et.push_back(it->et());    
    recoTau_.eta.push_back(it->eta());
    recoTau_.phi.push_back(it->phi());

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
