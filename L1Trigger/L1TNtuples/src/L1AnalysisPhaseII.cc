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


void L1Analysis::L1AnalysisPhaseII::SetMuon(const edm::Handle<l1t::MuonBxCollection> muon, unsigned maxL1Extra)
{
  for (int ibx = muon->getFirstBX(); ibx <= muon->getLastBX(); ++ibx) {
    for (l1t::MuonBxCollection::const_iterator it=muon->begin(ibx); it!=muon->end(ibx) && l1extra_.nMuons<maxL1Extra; it++){
      if (it->pt() > 0){
	l1extra_.muonEt .push_back(it->et());
	l1extra_.muonEta.push_back(it->eta());
	l1extra_.muonPhi.push_back(it->phi());
	l1extra_.muonEtaAtVtx.push_back(it->etaAtVtx());
	l1extra_.muonPhiAtVtx.push_back(it->phiAtVtx());
	l1extra_.muonIEt .push_back(it->hwPt());
	l1extra_.muonIEta.push_back(it->hwEta());
	l1extra_.muonIPhi.push_back(it->hwPhi());
	l1extra_.muonIEtaAtVtx.push_back(it->hwEtaAtVtx());
	l1extra_.muonIPhiAtVtx.push_back(it->hwPhiAtVtx());
	l1extra_.muonIDEta.push_back(it->hwDEtaExtra());
	l1extra_.muonIDPhi.push_back(it->hwDPhiExtra());
	l1extra_.muonChg.push_back(it->charge());
	l1extra_.muonIso.push_back(it->hwIso());
	l1extra_.muonQual.push_back(it->hwQual());
	l1extra_.muonTfMuonIdx.push_back(it->tfMuonIndex());
	l1extra_.muonBx .push_back(ibx);
	l1extra_.nMuons++;
      }
    }
  }
}

void L1Analysis::L1AnalysisPhaseII::SetMuonKF(const edm::Handle<l1t::RegionalMuonCandBxCollection> muonKF, unsigned maxL1Extra)
{
  for (int ibx = muonKF->getFirstBX(); ibx <= muonKF->getLastBX(); ++ibx) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator it=muonKF->begin(ibx); it!=muonKF->end(ibx) && l1extra_.nMuonsKF<maxL1Extra; it++){
      if (it->hwPt() > 0){
      l1extra_.muonKFEt .push_back(it->hwPt()*0.5);
      l1extra_.muonKFEta.push_back(it->hwEta()*0.010875);
      l1extra_.muonKFPhi.push_back(l1t::MicroGMTConfiguration::calcGlobalPhi( it->hwPhi(), it->trackFinderType(), it->processor() )*2*M_PI/576);
      l1extra_.muonKFChg.push_back(pow(-1,it->hwSign()));
      l1extra_.muonKFQual.push_back(it->hwQual());
      l1extra_.muonKFBx .push_back(ibx);
      l1extra_.nMuonsKF++;
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
void L1Analysis::L1AnalysisPhaseII::SetEG(const edm::Handle<l1t::EGammaBxCollection> EG, unsigned maxL1Extra)
{
  for(l1t::EGammaBxCollection::const_iterator it=EG->begin(); it!=EG->end() && l1extra_.nEG<maxL1Extra; it++){
    if (it->et() > 10){
    l1extra_.EGEt .push_back(it->et());
    l1extra_.EGEta.push_back(it->eta());
    l1extra_.EGPhi.push_back(it->phi());
    l1extra_.EGIso.push_back(it->isoEt());
    l1extra_.EGHwQual.push_back(it->hwQual());
    l1extra_.EGBx.push_back(0);//it->bx());
    l1extra_.nEG++;
  }
}
}

// TrkEG (seeded by Phase 2 Objects)
void L1Analysis::L1AnalysisPhaseII::SetTkEG(const edm::Handle<l1t::L1TkElectronParticleCollection> tkEG, unsigned maxL1Extra)
{
  for(l1t::L1TkElectronParticleCollection::const_iterator it=tkEG->begin(); it!=tkEG->end() && l1extra_.nTkEG<maxL1Extra; it++){
    if (it->et() > 5){
    l1extra_.tkEGEt .push_back(it->et());
    l1extra_.tkEGEta.push_back(it->eta());
    l1extra_.tkEGPhi.push_back(it->phi());
    l1extra_.tkEGzVtx.push_back(it->getTrkzVtx());
    l1extra_.tkEGTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkEGHwQual.push_back(it->getEGRef()->hwQual());
    l1extra_.tkEGEGRefPt.push_back(it->getEGRef()->et());
    l1extra_.tkEGEGRefEta.push_back(it->getEGRef()->eta());
    l1extra_.tkEGEGRefPhi.push_back(it->getEGRef()->phi());
    l1extra_.tkEGBx.push_back(0);//it->bx());
    l1extra_.nTkEG++;
  }}
}

void L1Analysis::L1AnalysisPhaseII::SetTkEGLoose(const edm::Handle<l1t::L1TkElectronParticleCollection> tkEGLoose, unsigned maxL1Extra)
{
  for(l1t::L1TkElectronParticleCollection::const_iterator it=tkEGLoose->begin(); it!=tkEGLoose->end() && l1extra_.ntkEGLoose<maxL1Extra; it++){
    if (it->et() > 5){
    l1extra_.tkEGLooseEt .push_back(it->et());
    l1extra_.tkEGLooseEta.push_back(it->eta());
    l1extra_.tkEGLoosePhi.push_back(it->phi());
    l1extra_.tkEGLoosezVtx.push_back(it->getTrkzVtx());
    l1extra_.tkEGLooseTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkEGLooseHwQual.push_back(it->getEGRef()->hwQual());
    l1extra_.tkEGLooseEGRefPt.push_back(it->getEGRef()->et());
    l1extra_.tkEGLooseEGRefEta.push_back(it->getEGRef()->eta());
    l1extra_.tkEGLooseEGRefPhi.push_back(it->getEGRef()->phi());
    l1extra_.tkEGLooseBx.push_back(0);//it->bx());
    l1extra_.ntkEGLoose++;
  }}
}

void L1Analysis::L1AnalysisPhaseII::SetTkEM(const edm::Handle<l1t::L1TkEmParticleCollection> tkEM, unsigned maxL1Extra)
{
  for(l1t::L1TkEmParticleCollection::const_iterator it=tkEM->begin(); it!=tkEM->end() && l1extra_.nTkEM<maxL1Extra; it++){
    if (it->et() > 10){
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

//TkTau
void L1Analysis::L1AnalysisPhaseII::SetTkTau(const edm::Handle<l1t::L1TkTauParticleCollection> tkTau, unsigned maxL1Extra)
{

  for(l1t::L1TkTauParticleCollection::const_iterator it=tkTau->begin(); it!=tkTau->end() && l1extra_.nTkTau<maxL1Extra; it++){

    l1extra_.tkTauEt.push_back(it->et());
    l1extra_.tkTauEta.push_back(it->eta());
    l1extra_.tkTauPhi.push_back(it->phi());
    l1extra_.tkTauzVtx.push_back(it->getTrkzVtx());
    l1extra_.tkTauTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkTauBx.push_back(0);//it->bx());
    l1extra_.nTkTau++;
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

    l1extra_.tkMuonEt .push_back( it->pt());
    l1extra_.tkMuonEta.push_back(it->eta());
    l1extra_.tkMuonPhi.push_back(it->phi());
    l1extra_.tkMuonChg.push_back(it->charge());
    l1extra_.tkMuonTrkIso.push_back(it->getTrkIsol());
    l1extra_.tkMuonMuRefPt.push_back(it->getMuRef()->hwPt()*0.5);
    l1extra_.tkMuonMuRefEta.push_back(it->getMuRef()->hwEta()*0.010875);
    l1extra_.tkMuonMuRefPhi.push_back(l1t::MicroGMTConfiguration::calcGlobalPhi( it->getMuRef()->hwPhi(), it->getMuRef()->trackFinderType(), it->getMuRef()->processor() )*2*M_PI/576);
    l1extra_.tkMuonDRMuTrack.push_back(it->dR());
    l1extra_.tkMuonNMatchedTracks.push_back(it->nTracksMatched());
    l1extra_.tkMuonzVtx.push_back(it->getTrkzVtx());
    l1extra_.tkMuonBx .push_back(0); //it->bx());
    l1extra_.tkMuonQuality .push_back(it->getMuRef()->hwQual());

    l1extra_.nTkMuons++;
  }
}

void L1Analysis::L1AnalysisPhaseII::SetTkGlbMuon(const edm::Handle<l1t::L1TkGlbMuonParticleCollection> muon, unsigned maxL1Extra)
{
  for(l1t::L1TkGlbMuonParticleCollection::const_iterator it=muon->begin(); it!=muon->end() && l1extra_.nTkGlbMuons<maxL1Extra; it++){

    l1extra_.tkGlbMuonEt .push_back( it->pt());
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
    l1extra_.tkGlbMuonQuality .push_back(it->getMuRef()->hwQual());
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

void L1Analysis::L1AnalysisPhaseII::SetPFJet(const edm::Handle<reco::PFJetCollection> PFJet, unsigned maxL1Extra)
{
  double mHT15_px=0, mHT15_py=0, HT15=0;
  double mHT20_px=0, mHT20_py=0, HT20=0;
  double mHT30_px=0, mHT30_py=0, HT30=0;

  // This should not be hardcoded here!!! 
   double offset[10]={-12.058, -12.399, -11.728, -10.557, -5.391,  4.586,  3.542,  1.825, -6.946, -17.857};
   double scale[10]={ 1.127,  1.155,  1.124,  1.192,  1.289,  0.912,  1.008,  1.298,  1.650,  1.402};
   double etaBin[10]={0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  3.500,  4.000,  4.500,  5.000};

  for(reco::PFJetCollection::const_iterator it=PFJet->begin(); it!=PFJet->end() && l1extra_.nPuppiJets<maxL1Extra; it++){
    double corrBin=1, offsetBin=0;
    for(int j=0; j<10; j++) {if(corrBin==1 && fabs(it->eta())<etaBin[j]) {corrBin=scale[j]; offsetBin=offset[j];}}
    double ptcorr=(it->pt()-offsetBin)/corrBin;
    reco::Particle::PolarLorentzVector corrP4= reco::Particle::PolarLorentzVector(ptcorr,it->eta(),it->phi(),it->mass());
    l1extra_.puppiJetEt .push_back(corrP4.pt());
    l1extra_.puppiJetEtUnCorr .push_back(it->et());
    l1extra_.puppiJetEta.push_back(it->eta());
    l1extra_.puppiJetPhi.push_back(it->phi());
//    l1extra_.puppiJetzVtx.push_back(it->getJetVtx());
    l1extra_.puppiJetBx .push_back(0);//it->bx());
    l1extra_.nPuppiJets++;
    if(corrP4.pt()>15 && fabs(it->eta())<2.4) { // this needs to be done in a nicer way
                   HT15+=corrP4.pt();
                  mHT15_px+=corrP4.px();
                  mHT15_py+=corrP4.py();
    }
    if(corrP4.pt()>20 && fabs(it->eta())<2.4) {
                  HT20+=corrP4.pt();
                  mHT20_px+=corrP4.px();
                  mHT20_py+=corrP4.py();

    }
    if(corrP4.pt()>30 && fabs(it->eta())<2.4) { 
                  HT30+=corrP4.pt();
                  mHT30_px+=corrP4.px();
                  mHT30_py+=corrP4.py();
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

  l1extra_.nPuppiMHT=3;

}

void L1Analysis::L1AnalysisPhaseII::SetL1METPF(const edm::Handle< std::vector<reco::PFMET> > l1MetPF)
{
  reco::PFMET met=l1MetPF->at(0);
  l1extra_.puppiMETEt = met.et();
  l1extra_.puppiMETPhi = met.phi();
}


