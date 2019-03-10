#include "L1Trigger/L1TNtuples/interface/L1AnalysisPhaseII.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"

L1Analysis::L1AnalysisPhaseII::L1AnalysisPhaseII()
{
}

L1Analysis::L1AnalysisPhaseII::~L1AnalysisPhaseII()
{

}

void L1Analysis::L1AnalysisPhaseII::SetVertices(float z0Puppi, float z0VertexTDR, const edm::Handle< std::vector<l1t::Vertex> > l1vertices, const edm::Handle< std::vector<l1t::L1TkPrimaryVertex> > l1TkPrimaryVertex){
      edm::Handle<std::vector<l1t::Vertex> > l1vertextdr;
      l1extra_.z0Puppi=z0Puppi;
      l1extra_.z0VertexTDR=z0VertexTDR;

      for (unsigned int i=0; i<l1vertices->size() ; i++){
                  l1extra_.z0Vertices.push_back(l1vertices->at(i).z0());
                  l1extra_.nVertices++;
      }
      for (unsigned int i=0; i<l1TkPrimaryVertex->size() ; i++){
                  l1extra_.z0L1TkPV.push_back(l1TkPrimaryVertex->at(i).getZvertex());
                  l1extra_.sumL1TkPV.push_back(l1TkPrimaryVertex->at(i).getSum());
                  l1extra_.nL1TkPVs++;
      }


}


void L1Analysis::L1AnalysisPhaseII::SetTau(const edm::Handle<l1t::TauBxCollection> tau, unsigned maxL1Extra)
{
  for (int ibx = tau->getFirstBX(); ibx <= tau->getLastBX(); ++ibx) {
    for (l1t::TauBxCollection::const_iterator it=tau->begin(ibx); it!=tau->end(ibx) && l1extra_.nTaus<maxL1Extra; it++){
      if (it->pt() > 0){
	l1extra_.tauEt .push_back(it->et());
	l1extra_.tauEta.push_back(it->eta());
	l1extra_.tauPhi.push_back(it->phi());
	l1extra_.tauIEt .push_back(it->hwPt());
	l1extra_.tauIEta.push_back(it->hwEta());
	l1extra_.tauIPhi.push_back(it->hwPhi());
	l1extra_.tauIso.push_back(it->hwIso());
	l1extra_.tauBx .push_back(ibx);
	l1extra_.tauTowerIPhi.push_back(it->towerIPhi());
	l1extra_.tauTowerIEta.push_back(it->towerIEta());
	l1extra_.tauRawEt.push_back(it->rawEt());
	l1extra_.tauIsoEt.push_back(it->isoEt());
	l1extra_.tauNTT.push_back(it->nTT());
	l1extra_.tauHasEM.push_back(it->hasEM());
	l1extra_.tauIsMerged.push_back(it->isMerged());
	l1extra_.tauHwQual.push_back(it->hwQual());
	l1extra_.nTaus++;
      }
    }
  }
}


void L1Analysis::L1AnalysisPhaseII::SetJet(const edm::Handle<l1t::JetBxCollection> jet, unsigned maxL1Extra)
{
  for (int ibx = jet->getFirstBX(); ibx <= jet->getLastBX(); ++ibx) {
    for (l1t::JetBxCollection::const_iterator it=jet->begin(ibx); it!=jet->end(ibx) && l1extra_.nJets<maxL1Extra; it++){
      if (it->pt() > 0){
	l1extra_.jetEt .push_back(it->et());
	l1extra_.jetEta.push_back(it->eta());
	l1extra_.jetPhi.push_back(it->phi());
	l1extra_.jetIEt .push_back(it->hwPt());
	l1extra_.jetIEta.push_back(it->hwEta());
	l1extra_.jetIPhi.push_back(it->hwPhi());
	l1extra_.jetBx .push_back(ibx);
	l1extra_.jetRawEt.push_back(it->rawEt());
	l1extra_.jetSeedEt.push_back(it->seedEt());
	l1extra_.jetTowerIEta.push_back(it->towerIEta());
	l1extra_.jetTowerIPhi.push_back(it->towerIPhi());
	l1extra_.jetPUEt.push_back(it->puEt());
	l1extra_.jetPUDonutEt0.push_back(it->puDonutEt(0));
	l1extra_.jetPUDonutEt1.push_back(it->puDonutEt(1));
	l1extra_.jetPUDonutEt2.push_back(it->puDonutEt(2));
	l1extra_.jetPUDonutEt3.push_back(it->puDonutEt(3));
	l1extra_.nJets++;
      }
    }
  }
}


void L1Analysis::L1AnalysisPhaseII::SetCaloJet(const edm::Handle<l1t::JetBxCollection> jet, unsigned maxL1Extra)
{
  for (int ibx = jet->getFirstBX(); ibx <= jet->getLastBX(); ++ibx) {
    for (l1t::JetBxCollection::const_iterator it=jet->begin(ibx); it!=jet->end(ibx) && l1extra_.nCaloJets<maxL1Extra; it++){
      if (it->pt() > 0){
      l1extra_.caloJetEt .push_back(it->et());
      l1extra_.caloJetEta.push_back(it->eta());
      l1extra_.caloJetPhi.push_back(it->phi());
      l1extra_.caloJetBx .push_back(ibx);
      l1extra_.nCaloJets++;
      }
    }
  }
}


void L1Analysis::L1AnalysisPhaseII::SetMuon(const edm::Handle<l1t::MuonBxCollection> muon, unsigned maxL1Extra)
{
  for (int ibx = muon->getFirstBX(); ibx <= muon->getLastBX(); ++ibx) {
    for (l1t::MuonBxCollection::const_iterator it=muon->begin(ibx); it!=muon->end(ibx) && l1extra_.nGlobalMuons<maxL1Extra; it++){
      if (it->pt() > 0){
	l1extra_.globalMuonPt .push_back(it->et());
	l1extra_.globalMuonEta.push_back(it->eta());
	l1extra_.globalMuonPhi.push_back(it->phi());
	l1extra_.globalMuonEtaAtVtx.push_back(it->etaAtVtx());
	l1extra_.globalMuonPhiAtVtx.push_back(it->phiAtVtx());
	l1extra_.globalMuonIEt .push_back(it->hwPt());
	l1extra_.globalMuonIEta.push_back(it->hwEta());
	l1extra_.globalMuonIPhi.push_back(it->hwPhi());
	l1extra_.globalMuonIEtaAtVtx.push_back(it->hwEtaAtVtx());
	l1extra_.globalMuonIPhiAtVtx.push_back(it->hwPhiAtVtx());
	l1extra_.globalMuonIDEta.push_back(it->hwDEtaExtra());
	l1extra_.globalMuonIDPhi.push_back(it->hwDPhiExtra());
	l1extra_.globalMuonChg.push_back(it->charge());
	l1extra_.globalMuonIso.push_back(it->hwIso());
	l1extra_.globalMuonQual.push_back(it->hwQual());
	l1extra_.globalMuonTfMuonIdx.push_back(it->tfMuonIndex());
	l1extra_.globalMuonBx .push_back(ibx);
	l1extra_.nGlobalMuons++;
      }
    }
  }
}

void L1Analysis::L1AnalysisPhaseII::SetMuonKF(const edm::Handle<l1t::RegionalMuonCandBxCollection> standaloneMuon, unsigned maxL1Extra, unsigned int muonDetector)
{
  for (int ibx = standaloneMuon->getFirstBX(); ibx <= standaloneMuon->getLastBX(); ++ibx) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator it=standaloneMuon->begin(ibx); it!=standaloneMuon->end(ibx) && l1extra_.nStandaloneMuons<maxL1Extra; it++){
      if (it->hwPt() > 0){
//      std::cout<<"hwPt vs hwPt2?"<<it->hwPt()*0.5<<" "<<it->hwPt2()<<"   "<<it->hwSign()<<"   "<<muonDetector<<std::endl;
      l1extra_.standaloneMuonPt .push_back(it->hwPt()*0.5);
      l1extra_.standaloneMuonPt2 .push_back(it->hwPt2());
      l1extra_.standaloneMuonDXY .push_back(it->hwDXY());
      l1extra_.standaloneMuonEta.push_back(it->hwEta()*0.010875);
      l1extra_.standaloneMuonPhi.push_back(l1t::MicroGMTConfiguration::calcGlobalPhi( it->hwPhi(), it->trackFinderType(), it->processor() )*2*M_PI/576);
      l1extra_.standaloneMuonChg.push_back( pow(-1,it->hwSign() ) );
      l1extra_.standaloneMuonQual.push_back(it->hwQual());
      l1extra_.standaloneMuonRegion.push_back(muonDetector);
      l1extra_.standaloneMuonBx .push_back(ibx);
      l1extra_.nStandaloneMuons++;
      }
    }
  }
}
  // RegionalMuons are a bit ugly... why not global muons?? 
  /// Get compressed pT (returned int * 0.5 = pT (GeV))
  //    const int hwPt() const { return m_hwPt; };
  //        /// Get compressed local phi (returned int * 2*pi/576 = local phi in rad)
  //            const int hwPhi() const { return m_hwPhi; };
  //                /// Get compressed eta (returned int * 0.010875 = eta)
  //                    const int hwEta() const { return m_hwEta; };
  //                        /// Get charge sign bit (charge = (-1)^(sign))
  //                        const int hwSign() const { return m_hwSign; };


void L1Analysis::L1AnalysisPhaseII::SetSum(const edm::Handle<l1t::EtSumBxCollection> sums, unsigned maxL1Extra)
{
  for (int ibx = sums->getFirstBX(); ibx <= sums->getLastBX(); ++ibx) {
    for (l1t::EtSumBxCollection::const_iterator it=sums->begin(ibx); it!=sums->end(ibx) && l1extra_.nSums<maxL1Extra; it++) {
      int type = static_cast<int>( it->getType() ); 
      l1extra_.sumType. push_back( type ); 
      l1extra_.sumEt. push_back( it->et() ); 
      l1extra_.sumPhi.push_back( it->phi() );
      l1extra_.sumIEt. push_back( it->hwPt() ); 
      l1extra_.sumIPhi.push_back( it->hwPhi() );
      l1extra_.sumBx. push_back( ibx );
      l1extra_.nSums++;
    }
  }
}

//EG (seeded by Phase 2 Objects )
void L1Analysis::L1AnalysisPhaseII::SetEG(const edm::Handle<l1t::EGammaBxCollection> EG, const edm::Handle<l1t::EGammaBxCollection> EGHGC,unsigned maxL1Extra)
{
  for(l1t::EGammaBxCollection::const_iterator it=EG->begin(); it!=EG->end() && l1extra_.nEG<maxL1Extra; it++){
    if (it->et() > 5){
    l1extra_.EGEt .push_back(it->et());
    l1extra_.EGEta.push_back(it->eta());
    l1extra_.EGPhi.push_back(it->phi());
    l1extra_.EGIso.push_back(it->isoEt());
    l1extra_.EGHwQual.push_back(it->hwQual());
    l1extra_.EGBx.push_back(0);//it->bx());
    l1extra_.EGHGC.push_back(0);
    bool quality= ( ( it->hwQual() >> 1 ) & 1   ) > 0 ;
    l1extra_.EGPassesLooseTrackID.push_back(quality); 
    quality= ( ( it->hwQual() >> 2 ) & 1 ) > 0 ;
    l1extra_.EGPassesPhotonID.push_back(quality);
    l1extra_.nEG++;
  }
  }

  for(l1t::EGammaBxCollection::const_iterator it=EGHGC->begin(); it!=EGHGC->end() && l1extra_.nEG<maxL1Extra; it++){
    if (it->et() > 10){
    l1extra_.EGEt .push_back(it->et());
    l1extra_.EGEta.push_back(it->eta());
    l1extra_.EGPhi.push_back(it->phi());
    l1extra_.EGIso.push_back(it->isoEt());
    l1extra_.EGHwQual.push_back(it->hwQual());
    l1extra_.EGBx.push_back(0);//it->bx());
    l1extra_.EGHGC.push_back(1);
    bool quality= (it->hwQual() ==2 ) ;
    l1extra_.EGPassesLooseTrackID.push_back(quality);  
    l1extra_.EGPassesPhotonID.push_back(quality);
    l1extra_.nEG++;
  }
  }
}

// TrkEG (seeded by Phase 2 Objects)
void L1Analysis::L1AnalysisPhaseII::SetTkEG(const edm::Handle<l1t::L1TkElectronParticleCollection> tkElectron, const edm::Handle<l1t::L1TkElectronParticleCollection> tkElectronHGC,  unsigned maxL1Extra)
{
  for(l1t::L1TkElectronParticleCollection::const_iterator it=tkElectron->begin(); it!=tkElectron->end() && l1extra_.nTkElectrons<maxL1Extra; it++){
    if (it->et() > 5){
    l1extra_.tkElectronEt .push_back(it->et());
    l1extra_.tkElectronEta.push_back(it->eta());
    l1extra_.tkElectronPhi.push_back(it->phi());
        int chargeFromCurvature = (it->trackCurvature() > 0)? 1 : -1 ; // ThisIsACheck
    l1extra_.tkElectronChg.push_back(chargeFromCurvature);
    l1extra_.tkElectronzVtx.push_back(it->getTrkzVtx());
    l1extra_.tkElectronTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkElectronHwQual.push_back(it->getEGRef()->hwQual());
    l1extra_.tkElectronEGRefPt.push_back(it->getEGRef()->et());
    l1extra_.tkElectronEGRefEta.push_back(it->getEGRef()->eta());
    l1extra_.tkElectronEGRefPhi.push_back(it->getEGRef()->phi());
    l1extra_.tkElectronBx.push_back(0);//it->bx());
    l1extra_.tkElectronHGC.push_back(0);
    bool quality=  ( ( it->getEGRef()->hwQual() >> 1 ) & 1  )> 0; // LooseTrackID should be the second bit 
    l1extra_.tkElectronPassesLooseTrackID.push_back(quality);
    quality=  ( ( it->getEGRef()->hwQual() >> 2 ) & 1  )> 0; // LooseTrackID should be the second bit 
    l1extra_.tkElectronPassesPhotonID.push_back(quality);
    l1extra_.nTkElectrons++;
  }}

  for(l1t::L1TkElectronParticleCollection::const_iterator it=tkElectronHGC->begin(); it!=tkElectronHGC->end() && l1extra_.nTkElectrons<maxL1Extra; it++){
    if (it->et() > 5){
    l1extra_.tkElectronEt .push_back(it->et());
    l1extra_.tkElectronEta.push_back(it->eta());
    l1extra_.tkElectronPhi.push_back(it->phi());
        int chargeFromCurvature = (it->trackCurvature() > 0)? 1 : -1 ; // ThisIsACheck
    l1extra_.tkElectronChg.push_back(chargeFromCurvature);
    l1extra_.tkElectronzVtx.push_back(it->getTrkzVtx());
    l1extra_.tkElectronTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkElectronHwQual.push_back(it->getEGRef()->hwQual());
    l1extra_.tkElectronEGRefPt.push_back(it->getEGRef()->et());
    l1extra_.tkElectronEGRefEta.push_back(it->getEGRef()->eta());
    l1extra_.tkElectronEGRefPhi.push_back(it->getEGRef()->phi());
    l1extra_.tkElectronBx.push_back(0);//it->bx());
    l1extra_.tkElectronHGC.push_back(1);
    bool quality= (it->getEGRef()->hwQual() ==2 ) ;
    l1extra_.tkElectronPassesLooseTrackID.push_back(quality);
    l1extra_.tkElectronPassesPhotonID.push_back(quality);
    l1extra_.nTkElectrons++;
  }}

}

void L1Analysis::L1AnalysisPhaseII::SetTkEGLoose(const edm::Handle<l1t::L1TkElectronParticleCollection> tkElectronLoose, const edm::Handle<l1t::L1TkElectronParticleCollection> tkElectronLooseHGC,unsigned maxL1Extra)
{
  for(l1t::L1TkElectronParticleCollection::const_iterator it=tkElectronLoose->begin(); it!=tkElectronLoose->end() && l1extra_.nTkElectronsLoose<maxL1Extra; it++){
    if (it->et() > 5){
    l1extra_.tkElectronLooseEt .push_back(it->et());
    l1extra_.tkElectronLooseEta.push_back(it->eta());
    l1extra_.tkElectronLoosePhi.push_back(it->phi());
        int chargeFromCurvature = (it->trackCurvature() > 0)? 1 : -1 ; // ThisIsACheck
    l1extra_.tkElectronLooseChg.push_back(chargeFromCurvature);
    l1extra_.tkElectronLoosezVtx.push_back(it->getTrkzVtx());
    l1extra_.tkElectronLooseTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkElectronLooseHwQual.push_back(it->getEGRef()->hwQual());
    l1extra_.tkElectronLooseEGRefPt.push_back(it->getEGRef()->et());
    l1extra_.tkElectronLooseEGRefEta.push_back(it->getEGRef()->eta());
    l1extra_.tkElectronLooseEGRefPhi.push_back(it->getEGRef()->phi());
    l1extra_.tkElectronLooseBx.push_back(0);//it->bx());
    l1extra_.tkElectronLooseHGC.push_back(0);
    bool quality=( ( it->getEGRef()->hwQual() >> 1 ) & 1   )> 0;  
    l1extra_.tkElectronLoosePassesLooseTrackID.push_back(quality);
    quality=( ( it->getEGRef()->hwQual() >> 2 ) & 1   )> 0;  
    l1extra_.tkElectronLoosePassesPhotonID.push_back(quality);
    l1extra_.nTkElectronsLoose++;
  }}

  for(l1t::L1TkElectronParticleCollection::const_iterator it=tkElectronLooseHGC->begin(); it!=tkElectronLooseHGC->end() && l1extra_.nTkElectronsLoose<maxL1Extra; it++){
    if (it->et() > 5){
    l1extra_.tkElectronLooseEt .push_back(it->et());
    l1extra_.tkElectronLooseEta.push_back(it->eta());
    l1extra_.tkElectronLoosePhi.push_back(it->phi());
        int chargeFromCurvature = (it->trackCurvature() > 0)? 1 : -1 ; // ThisIsACheck
    l1extra_.tkElectronLooseChg.push_back(chargeFromCurvature);
    l1extra_.tkElectronLoosezVtx.push_back(it->getTrkzVtx());
    l1extra_.tkElectronLooseTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkElectronLooseHwQual.push_back(it->getEGRef()->hwQual());
    l1extra_.tkElectronLooseEGRefPt.push_back(it->getEGRef()->et());
    l1extra_.tkElectronLooseEGRefEta.push_back(it->getEGRef()->eta());
    l1extra_.tkElectronLooseEGRefPhi.push_back(it->getEGRef()->phi());
    l1extra_.tkElectronLooseBx.push_back(0);//it->bx());
    l1extra_.tkElectronLooseHGC.push_back(1);
    bool quality= (it->getEGRef()->hwQual() ==2 ) ;
    l1extra_.tkElectronLoosePassesLooseTrackID.push_back(quality);
    l1extra_.tkElectronLoosePassesPhotonID.push_back(quality);
    l1extra_.nTkElectronsLoose++;
  }}


}

void L1Analysis::L1AnalysisPhaseII::SetTkEM(const edm::Handle<l1t::L1TkEmParticleCollection> tkPhoton, const edm::Handle<l1t::L1TkEmParticleCollection> tkPhotonHGC, unsigned maxL1Extra)
{
  for(l1t::L1TkEmParticleCollection::const_iterator it=tkPhoton->begin(); it!=tkPhoton->end() && l1extra_.nTkPhotons<maxL1Extra; it++){
    if (it->et() > 10){
    l1extra_.tkPhotonEt .push_back(it->et());
    l1extra_.tkPhotonEta.push_back(it->eta());
    l1extra_.tkPhotonPhi.push_back(it->phi());
    l1extra_.tkPhotonTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkPhotonTrkIsoPV.push_back(it->getTrkIsolPV());
    l1extra_.tkPhotonBx.push_back(0);//it->bx());
    l1extra_.tkPhotonHwQual.push_back(it->getEGRef()->hwQual());
    l1extra_.tkPhotonEGRefPt.push_back(it->getEGRef()->et());
    l1extra_.tkPhotonEGRefEta.push_back(it->getEGRef()->eta());
    l1extra_.tkPhotonEGRefPhi.push_back(it->getEGRef()->phi());
    l1extra_.tkPhotonHGC.push_back( 0 );
    bool quality= ( ( it->getEGRef()->hwQual() >> 2 ) & 1   ) > 0 ; 
    l1extra_.tkPhotonPassesLooseTrackID.push_back(quality);
    quality= ( ( it->getEGRef()->hwQual() >> 1 ) & 1   ) > 0 ; // Photon Id should be the third bit 
    l1extra_.tkPhotonPassesPhotonID.push_back(quality);
    l1extra_.nTkPhotons++;
  }}
  for(l1t::L1TkEmParticleCollection::const_iterator it=tkPhotonHGC->begin(); it!=tkPhotonHGC->end() && l1extra_.nTkPhotons<maxL1Extra; it++){
    if (it->et() > 10){
    l1extra_.tkPhotonEt .push_back(it->et());
    l1extra_.tkPhotonEta.push_back(it->eta());
    l1extra_.tkPhotonPhi.push_back(it->phi());
    l1extra_.tkPhotonTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkPhotonTrkIsoPV.push_back(it->getTrkIsolPV());
    l1extra_.tkPhotonBx.push_back(0);//it->bx());
    l1extra_.tkPhotonHwQual.push_back(it->getEGRef()->hwQual());
    l1extra_.tkPhotonEGRefPt.push_back(it->getEGRef()->et());
    l1extra_.tkPhotonEGRefEta.push_back(it->getEGRef()->eta());
    l1extra_.tkPhotonEGRefPhi.push_back(it->getEGRef()->phi());
    l1extra_.tkPhotonHGC.push_back( 1 );
    bool quality= (it->getEGRef()->hwQual() ==2 ) ;
    l1extra_.tkPhotonPassesLooseTrackID.push_back(quality);
    l1extra_.tkPhotonPassesPhotonID.push_back(quality);
    l1extra_.nTkPhotons++;
  for(l1t::L1TkEmParticleCollection::const_iterator it=tkEM->begin(); it!=tkEM->end() && l1extra_.nTkEM<maxL1Extra; it++){
    if (it->et() > 5){
    l1extra_.tkEMEt .push_back(it->et());
    l1extra_.tkEMEta.push_back(it->eta());
    l1extra_.tkEMPhi.push_back(it->phi());
    l1extra_.tkEMTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkEMBx.push_back(0);//it->bx());
    l1extra_.tkEMHwQual.push_back(it->getEGRef()->hwQual());
    l1extra_.tkEMEGRefPt.push_back(it->getEGRef()->et());
    l1extra_.tkEMEGRefEta.push_back(it->getEGRef()->eta());
    l1extra_.tkEMEGRefPhi.push_back(it->getEGRef()->phi());
   l1extra_.nTkEM++;
  }}
}

void L1Analysis::L1AnalysisPhaseII::SetTrkTau(const edm::Handle<l1t::L1TrkTauParticleCollection> tkTau, unsigned maxL1Extra)
{

  for(l1t::L1TrkTauParticleCollection::const_iterator it=tkTau->begin(); it!=tkTau->end() && l1extra_.nTkTau<maxL1Extra; it++){

    l1extra_.tkTauEt.push_back(it->et());
    l1extra_.tkTauEta.push_back(it->eta());
    l1extra_.tkTauPhi.push_back(it->phi());
    l1extra_.tkTauTrkIso.push_back(it->getVtxIso());
    l1extra_.tkTauBx.push_back(0);//it->bx());
    l1extra_.nTkTau++;
  }
}

void L1Analysis::L1AnalysisPhaseII::SetCaloTkTau(const edm::Handle<l1t::L1CaloTkTauParticleCollection> caloTkTau, unsigned maxL1Extra)
{

  for(l1t::L1CaloTkTauParticleCollection::const_iterator it=caloTkTau->begin(); it!=caloTkTau->end() && l1extra_.nCaloTkTau<maxL1Extra; it++){

    l1extra_.caloTkTauEt.push_back(it->et());
    l1extra_.caloTkTauEta.push_back(it->eta());
    l1extra_.caloTkTauPhi.push_back(it->phi());
    l1extra_.caloTkTauTrkIso.push_back(it->getVtxIso());
    l1extra_.caloTkTauBx.push_back(0);//it->bx());
    l1extra_.nCaloTkTau++;
  }
}

void L1Analysis::L1AnalysisPhaseII::SetTkEGTau(const edm::Handle<l1t::L1TkEGTauParticleCollection> tkEGTau, unsigned maxL1Extra)
{

  for(l1t::L1TkEGTauParticleCollection::const_iterator it=tkEGTau->begin(); it!=tkEGTau->end() && l1extra_.nTkEGTau<maxL1Extra; it++){

    l1extra_.tkEGTauEt.push_back(it->et());
    l1extra_.tkEGTauEta.push_back(it->eta());
    l1extra_.tkEGTauPhi.push_back(it->phi());
    l1extra_.tkEGTauTrkIso.push_back(it->getVtxIso());
    l1extra_.tkEGTauBx.push_back(0);//it->bx());
    l1extra_.nTkEGTau++;
  }
}


// TkJet
void L1Analysis::L1AnalysisPhaseII::SetTkJet(const edm::Handle<l1t::L1TkJetParticleCollection> trackerJet, unsigned maxL1Extra)
{

  for(l1t::L1TkJetParticleCollection::const_iterator it=trackerJet->begin(); it!=trackerJet->end() && l1extra_.nTrackerJets<maxL1Extra; it++){
    l1extra_.trackerJetEt .push_back(it->et());
    l1extra_.trackerJetEta.push_back(it->eta());
    l1extra_.trackerJetPhi.push_back(it->phi());
    l1extra_.trackerJetzVtx.push_back(it->getJetVtx());
    l1extra_.trackerJetBx .push_back(0);//it->bx());
    l1extra_.nTrackerJets++;
  }
}

void L1Analysis::L1AnalysisPhaseII::SetTkCaloJet(const edm::Handle<l1t::L1TkJetParticleCollection> tkCaloJet, unsigned maxL1Extra)
{

  for(l1t::L1TkJetParticleCollection::const_iterator it=tkCaloJet->begin(); it!=tkCaloJet->end() && l1extra_.nTkCaloJets<maxL1Extra; it++){
    l1extra_.tkCaloJetEt .push_back(it->et());
    l1extra_.tkCaloJetEta.push_back(it->eta());
    l1extra_.tkCaloJetPhi.push_back(it->phi());
    l1extra_.tkCaloJetzVtx.push_back(it->getJetVtx());
    l1extra_.tkCaloJetBx .push_back(0);//it->bx());
    l1extra_.nTkCaloJets++;
  }
}

void L1Analysis::L1AnalysisPhaseII::SetTkMuon(const edm::Handle<l1t::L1TkMuonParticleCollection> muon, unsigned maxL1Extra)
{

  for(l1t::L1TkMuonParticleCollection::const_iterator it=muon->begin(); it!=muon->end() && l1extra_.nTkMuons<maxL1Extra; it++){

    l1extra_.tkMuonPt .push_back(it->pt());
    l1extra_.tkMuonEta.push_back(it->eta());
    l1extra_.tkMuonPhi.push_back(it->phi());
    int chargeFromCurvature = (it->trackCurvature() > 0)? 1 : -1 ; // ThisIsACheck
    l1extra_.tkMuonChg.push_back( chargeFromCurvature);
    l1extra_.tkMuonTrkIso.push_back(it->getTrkIsol());
    if(it->muonDetector()!=3){
    l1extra_.tkMuonMuRefPt.push_back(it->getMuRef()->hwPt()*0.5);
    l1extra_.tkMuonMuRefEta.push_back(it->getMuRef()->hwEta()*0.010875);
    l1extra_.tkMuonMuRefPhi.push_back(l1t::MicroGMTConfiguration::calcGlobalPhi( it->getMuRef()->hwPhi(), it->getMuRef()->trackFinderType(), it->getMuRef()->processor() )*2*M_PI/576);
    l1extra_.tkMuonDRMuTrack.push_back(it->dR());
    l1extra_.tkMuonNMatchedTracks.push_back(it->nTracksMatched());
    l1extra_.tkMuonQual .push_back(it->getMuRef()->hwQual());
    l1extra_.tkMuonMuRefChg.push_back(pow(-1,it->getMuRef()->hwSign() ) );
    }else {
    l1extra_.tkMuonMuRefPt.push_back(-777);
    l1extra_.tkMuonMuRefEta.push_back(-777);
    l1extra_.tkMuonMuRefPhi.push_back(-777);
    l1extra_.tkMuonDRMuTrack.push_back(-777);
    l1extra_.tkMuonNMatchedTracks.push_back(0);
    l1extra_.tkMuonQuality .push_back(999);
    l1extra_.tkMuonMuRefChg.push_back(0);
    }
    l1extra_.tkMuonRegion.push_back(it->muonDetector());
    l1extra_.tkMuonzVtx.push_back(it->getTrkzVtx());
    l1extra_.tkMuonBx .push_back(0); //it->bx());
    l1extra_.nTkMuons++;
  }
}


void L1Analysis::L1AnalysisPhaseII::SetTkMuonStubs(const edm::Handle<l1t::L1TkMuonParticleCollection> muon, unsigned maxL1Extra, unsigned int muonDetector)
{

  for(l1t::L1TkMuonParticleCollection::const_iterator it=muon->begin(); it!=muon->end() && l1extra_.nTkMuonStubs<maxL1Extra; it++){

    l1extra_.tkMuonStubsPt .push_back( it->pt());
    l1extra_.tkMuonStubsEta.push_back(it->eta());
    l1extra_.tkMuonStubsPhi.push_back(it->phi());
    l1extra_.tkMuonStubsChg.push_back(it->charge());
    l1extra_.tkMuonStubsTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkMuonStubszVtx.push_back(it->getTrkzVtx());
    l1extra_.tkMuonStubsBx .push_back(0); //it->bx());
    l1extra_.tkMuonStubsQuality .push_back(1);
    l1extra_.tkMuonStubsBarrelStubs.push_back(it->getBarrelStubs().size());
    l1extra_.tkMuonStubsRegion.push_back(muonDetector);
    l1extra_.nTkMuonStubs++;
  }
}




void L1Analysis::L1AnalysisPhaseII::SetTkGlbMuon(const edm::Handle<l1t::L1TkGlbMuonParticleCollection> muon, unsigned maxL1Extra)
{
  for(l1t::L1TkGlbMuonParticleCollection::const_iterator it=muon->begin(); it!=muon->end() && l1extra_.nTkGlbMuons<maxL1Extra; it++){

    l1extra_.tkGlbMuonPt .push_back( it->pt());
    l1extra_.tkGlbMuonEta.push_back(it->eta());
    l1extra_.tkGlbMuonPhi.push_back(it->phi());
    l1extra_.tkGlbMuonChg.push_back(it->charge());
    l1extra_.tkGlbMuonTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkGlbMuonMuRefPt.push_back(it->getMuRef()->pt());
    l1extra_.tkGlbMuonMuRefEta.push_back(it->getMuRef()->eta());
    l1extra_.tkGlbMuonMuRefPhi.push_back(it->getMuRef()->phi());
    l1extra_.tkGlbMuonDRMuTrack.push_back(it->dR());
    l1extra_.tkGlbMuonNMatchedTracks.push_back(it->nTracksMatched());
    l1extra_.tkGlbMuonzVtx.push_back(it->getTrkzVtx());
    l1extra_.tkGlbMuonBx .push_back(0); //it->bx());
    l1extra_.tkGlbMuonQual .push_back(it->getMuRef()->hwQual());
    l1extra_.nTkGlbMuons++;
  }
}

// trackerMet
void L1Analysis::L1AnalysisPhaseII::SetTkMET(const edm::Handle<l1t::L1TkEtMissParticleCollection> trackerMets)
{
  for(l1t::L1TkEtMissParticleCollection::const_iterator it=trackerMets->begin(); it!=trackerMets->end(); it++) {
    l1extra_.trackerMetSumEt.    push_back( it->etTotal() );
    l1extra_.trackerMetEt.   push_back( it->etMiss() );
    l1extra_.trackerMetPhi.push_back( it->phi() );
    l1extra_.trackerMetBx. push_back( it->bx() );
    l1extra_.nTrackerMet++;
  }
}


void L1Analysis::L1AnalysisPhaseII::SetTkMHT(const edm::Handle<l1t::L1TkHTMissParticleCollection > trackerMHTs){
   // Hardcoding it like this, but this needs to be changed to a vector

  for(l1t::L1TkHTMissParticleCollection::const_iterator it=trackerMHTs->begin(); it!=trackerMHTs->end(); it++) {
    l1extra_.trackerHT.    push_back( it->EtTotal() );
    l1extra_.trackerMHT.   push_back( it->EtMiss() );
    l1extra_.trackerMHTPhi.push_back( it->phi() );
    l1extra_.nTrackerMHT++;
  }

}


/*
void L1Analysis::L1AnalysisPhaseII::SetPFJetForMET(const edm::Handle<l1t::PFJetCollection> PFJet, unsigned maxL1Extra)
{

  for(l1t::PFJetCollection::const_iterator it=PFJet->begin(); it!=PFJet->end() && l1extra_.nPuppiJetForMETs<maxL1Extra; it++){
    l1extra_.puppiJetForMETEt .push_back(it->pt());
    l1extra_.puppiJetForMETEtUnCorr .push_back(it->rawPt());
    l1extra_.puppiJetForMETEta.push_back(it->eta());
    l1extra_.puppiJetForMETPhi.push_back(it->phi());
//    l1extra_.puppiJetForMETzVtx.push_back(it->getJetVtx());
    l1extra_.puppiJetForMETBx .push_back(0);//it->bx());
    l1extra_.nPuppiJetForMETs++;
  }
}
*/

void L1Analysis::L1AnalysisPhaseII::SetPFJet(const edm::Handle<l1t::PFJetCollection> PFJet, unsigned maxL1Extra)
{
  double mHT15_px=0, mHT15_py=0, HT15=0;
  double mHT20_px=0, mHT20_py=0, HT20=0;
  double mHT30_px=0, mHT30_py=0, HT30=0;
  double mHT30_15_px=0, mHT30_15_py=0, HT30_15=0;

  for(l1t::PFJetCollection::const_iterator it=PFJet->begin(); it!=PFJet->end() && l1extra_.nPuppiJets<maxL1Extra; it++){
    l1extra_.puppiJetEt .push_back(it->pt());
    l1extra_.puppiJetEtUnCorr .push_back(it->rawPt());
    l1extra_.puppiJetEta.push_back(it->eta());
    l1extra_.puppiJetPhi.push_back(it->phi());
//    l1extra_.puppiJetzVtx.push_back(it->getJetVtx());
    l1extra_.puppiJetBx .push_back(0);//it->bx());
    l1extra_.nPuppiJets++;
    if(it->pt()>15 && fabs(it->eta())<2.4) { // this needs to be done in a nicer way
                   HT15+=it->pt();
                  mHT15_px+=it->px();
                  mHT15_py+=it->py();
    }
    if(it->pt()>20 && fabs(it->eta())<2.4) {
                  HT20+=it->pt();
                  mHT20_px+=it->px();
                  mHT20_py+=it->py();

    }
    if(it->pt()>30 && fabs(it->eta())<2.4) { 
                  HT30+=it->pt();
                  mHT30_px+=it->px();
                  mHT30_py+=it->py();
      }
    if(it->pt()>30 && fabs(it->eta())<1.5) {
                  HT30_15+=it->pt();
                  mHT30_15_px+=it->px();
                  mHT30_15_py+=it->py();
      }
  }
  l1extra_.puppiMHTEt.push_back( sqrt(mHT15_px*mHT15_px+mHT15_py*mHT15_py) );
  l1extra_.puppiMHTPhi.push_back( atan(mHT15_py/mHT15_px) );
  l1extra_.puppiHT.push_back( HT15 );

  l1extra_.puppiMHTEt.push_back( sqrt(mHT20_px*mHT20_px+mHT20_py*mHT20_py) );
  l1extra_.puppiMHTPhi.push_back( atan(mHT20_py/mHT20_px) );
  l1extra_.puppiHT.push_back( HT20 );

  l1extra_.puppiMHTEt.push_back( sqrt(mHT30_px*mHT30_px+mHT30_py*mHT30_py) );
  l1extra_.puppiMHTPhi.push_back( atan(mHT30_py/mHT30_px) );
  l1extra_.puppiHT.push_back( HT30 );

  l1extra_.puppiMHTEt.push_back( sqrt(mHT30_15_px*mHT30_15_px+mHT30_15_py*mHT30_15_py) );
  l1extra_.puppiMHTPhi.push_back( atan(mHT30_15_py/mHT30_15_px) );
  l1extra_.puppiHT.push_back( HT30_15 );

  l1extra_.nPuppiMHT=4;

}

void L1Analysis::L1AnalysisPhaseII::SetL1METPF(const edm::Handle< std::vector<reco::PFMET> > l1MetPF)
{
  reco::PFMET met=l1MetPF->at(0);
  l1extra_.puppiMETEt = met.et();
  l1extra_.puppiMETPhi = met.phi();
}

void L1Analysis::L1AnalysisPhaseII::SetPFObjects(const edm::Handle< vector<l1t::PFCandidate> >  l1pfCandidates,  unsigned maxL1Extra)
{
      for (unsigned int i=0; i<l1pfCandidates->size() && l1extra_.nPFMuons<maxL1Extra; i++){
           //enum Kind { ChargedHadron=0, Electron=1, NeutralHadron=2, Photon=3, Muon=4 };
            if(abs(l1pfCandidates->at(i).id())==4){
                  l1extra_.pfMuonPt.push_back(l1pfCandidates->at(i).pt()); 
                  l1extra_.pfMuonChg.push_back(l1pfCandidates->at(i).charge());
                  l1extra_.pfMuonEta.push_back(l1pfCandidates->at(i).eta());
                  l1extra_.pfMuonPhi.push_back(l1pfCandidates->at(i).phi());
                  l1extra_.pfMuonzVtx.push_back(l1pfCandidates->at(i).pfTrack()->track()->getPOCA(4).z()); // check with Giovanni, there has to be a cleaner way to do this. nParam_=4 should not be hardcoded
                  l1extra_.nPFMuons++;
            }
      }

      for (unsigned int i=0; i<l1pfCandidates->size(); i++){
           //enum Kind { ChargedHadron=0, Electron=1, NeutralHadron=2, Photon=3, Muon=4 };
            if(abs(l1pfCandidates->at(i).id())!=4){
              //  std::cout<<"pf cand id: "<<l1pfCandidates->at(i).id()<<std::endl;
                  l1extra_.pfCandId.push_back(l1pfCandidates->at(i).id()); 
                  l1extra_.pfCandEt.push_back(l1pfCandidates->at(i).pt()); 
                  l1extra_.pfCandChg.push_back(l1pfCandidates->at(i).charge());
                  l1extra_.pfCandEta.push_back(l1pfCandidates->at(i).eta());
                  l1extra_.pfCandPhi.push_back(l1pfCandidates->at(i).phi());
                  if (l1pfCandidates->at(i).id()==0) {
                      l1extra_.pfCandzVtx.push_back(l1pfCandidates->at(i).pfTrack()->track()->getPOCA(4).z()); 
                  } else {
                      l1extra_.pfCandzVtx.push_back(9999.0);
                  };

                  l1extra_.nPFCands++;
            }
      }


}

void L1Analysis::L1AnalysisPhaseII::SetPFTaus(const edm::Handle< vector<l1t::L1PFTau> >  l1pfTaus,  unsigned maxL1Extra)
{

      for (unsigned int i=0; i<l1pfTaus->size() && l1extra_.nPFTaus<maxL1Extra; i++){
                   if(l1pfTaus->at(i).pt()<10) continue;
                   l1extra_.pfTauEt.push_back(l1pfTaus->at(i).pt());
                   l1extra_.pfTauEta.push_back(l1pfTaus->at(i).eta());
                   l1extra_.pfTauPhi.push_back(l1pfTaus->at(i).phi());
                   l1extra_.pfTauChg.push_back(l1pfTaus->at(i).charge());
                   l1extra_.pfTauType.push_back(l1pfTaus->at(i).tauType());
                   l1extra_.pfTauChargedIso.push_back(l1pfTaus->at(i).chargedIso());
                   unsigned int isoflag=l1pfTaus->at(i).tauIsoQuality();
                   l1extra_.pfTauIsoFlag.push_back(isoflag);
                   //std::cout<<l1pfTaus->at(i).pt()<<"   "<<l1pfTaus->at(i).chargedIso()<<"  "<<l1pfTaus->at(i).passTightIso()<<"  "<<l1extra_.pfTauIsoFlag[l1extra_.nPFTaus]<<"  "<<isoflag<<std::endl;
                   // VeryLoose: <50; Loose < 20; Medium<10; Tight<5 
                   isoflag=l1pfTaus->at(i).tauRelIsoQuality();
                   l1extra_.pfTauRelIsoFlag.push_back(isoflag);
                   l1extra_.pfTauPassesMediumIso.push_back(l1pfTaus->at(i).passMediumIso());
                   l1extra_.nPFTaus++;
      }

}
