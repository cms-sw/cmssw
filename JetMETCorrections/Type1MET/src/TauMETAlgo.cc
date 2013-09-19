// File: TauMETAlgo.cc
// Description:  see TauMETAlgo.h
// Authors: Alfredo Gurrola, C.N.Nguyen

#include "JetMETCorrections/Type1MET/interface/TauMETAlgo.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace std;
using namespace reco;

typedef math::XYZTLorentzVector LorentzVector;
typedef math::XYZPoint Point;

  TauMETAlgo::TauMETAlgo() {}
  TauMETAlgo::~TauMETAlgo() {}

  void TauMETAlgo::run(edm::Event& iEvent,const edm::EventSetup& iSetup,
                       edm::Handle<PFTauCollection> tauHandle,edm::Handle<CaloJetCollection> calojetHandle,
                       double jetPTthreshold,double jetEMfracLimit,
                       const JetCorrector& correctedjets,const std::vector<CaloMET>& uncorMET,double jetMatchDeltaR,
                       double tauMinEt,double tauEtaMax,bool useSeedTrack,double seedTrackPt,bool useTrackIsolation,double trackIsolationMinPt,
                       bool useECALIsolation,double gammaIsolationMinPt,bool useProngStructure,std::vector<CaloMET>* corrMET) {

    //    std::cerr << "TauMETAlgo::run -> Test.. " << std::endl;

    double DeltaPx = 0.0;
    double DeltaPy = 0.0;
    double DeltaSumET = 0.0;
    for(PFTauCollection::size_type iPFTau=0;iPFTau<tauHandle->size();iPFTau++) {
      PFTauRef thePFTau(tauHandle,iPFTau);
      bool matchFlag = false;
      bool goodTau = false;
      if((fabs(thePFTau->eta()) <= tauEtaMax) && (thePFTau->et() >= tauMinEt)) {
        goodTau = true;
        if(useSeedTrack) {
          if(!(thePFTau->leadPFChargedHadrCand())) {goodTau = false;}
          else {
            if(thePFTau->leadPFChargedHadrCand()->et() < seedTrackPt) {goodTau = false;}
          }
        }
        if(useTrackIsolation) {
          vector<reco::PFCandidatePtr> PFTauProdIsolCHCands = (*thePFTau).isolationPFChargedHadrCands();
          for(vector<reco::PFCandidatePtr>::const_iterator iChargedHadrCand=PFTauProdIsolCHCands.begin();
              iChargedHadrCand!=PFTauProdIsolCHCands.end();++iChargedHadrCand) {
            if((**iChargedHadrCand).pt() > trackIsolationMinPt) {goodTau = false;}
          }
        }
        if(useECALIsolation) {
          vector<reco::PFCandidatePtr> PFTauProdIsolGammaCands = (*thePFTau).isolationPFGammaCands();
          for(vector<reco::PFCandidatePtr>::const_iterator iGammaCand=PFTauProdIsolGammaCands.begin();
              iGammaCand!=PFTauProdIsolGammaCands.end();++iGammaCand) {
            if((**iGammaCand).et() > gammaIsolationMinPt) {goodTau = false;}
          }
        }
        if(useProngStructure) {
          if((thePFTau->signalPFChargedHadrCands().size() != 1) && (thePFTau->signalPFChargedHadrCands().size() != 3)) {goodTau = false;}
        }
      }
      if(goodTau) {
        for(CaloJetCollection::const_iterator calojetIter = calojetHandle->begin();calojetIter != calojetHandle->end();++calojetIter) {
          if(deltaR(calojetIter->p4(),thePFTau->p4())<jetMatchDeltaR) {
            if( calojetIter->pt() > jetPTthreshold && calojetIter->emEnergyFraction() < jetEMfracLimit ) {
              double correct = correctedjets.correction (*calojetIter);
              DeltaPx += ((correct * calojetIter->px()) - thePFTau->px());
              DeltaPy += ((correct * calojetIter->py()) - thePFTau->py());
              DeltaSumET += ((correct * calojetIter->et()) - thePFTau->et());
            } else {
              DeltaPx += (calojetIter->px() - thePFTau->px());
              DeltaPy += (calojetIter->py() - thePFTau->py());
              DeltaSumET += (calojetIter->et() - thePFTau->et());
            }
            if(matchFlag) {std::cerr << "### TauMETAlgo - ERROR:  Multiple jet matches!!!! " << std::endl;}
            matchFlag = true;
          }
        }
      }
    }
    CorrMETData delta;
    delta.mex = DeltaPx;
    delta.mey = DeltaPy;
    delta.sumet = - DeltaSumET;
    const CaloMET* u = &(uncorMET.front());
    double corrMetPx = u->px() + delta.mex;
    double corrMetPy = u->py() + delta.mey;
    MET::LorentzVector correctedMET4vector( corrMetPx, corrMetPy, 0., sqrt(corrMetPx*corrMetPx + corrMetPy*corrMetPy));
    std::vector<CorrMETData> corrections = u->mEtCorr();
    corrections.push_back(delta);
    CaloMET result;
    result = CaloMET(u->getSpecific(), (u->sumEt() + delta.sumet), corrections, correctedMET4vector, u->vertex());
    corrMET->push_back(result);
  }

  void TauMETAlgo::run(edm::Event& iEvent,const edm::EventSetup& iSetup,
                       edm::Handle<PFTauCollection> tauHandle,edm::Handle<CaloJetCollection> calojetHandle,
                       double jetPTthreshold,double jetEMfracLimit,
                       const JetCorrector& correctedjets,const std::vector<MET>& uncorMET,double jetMatchDeltaR,
                       double tauMinEt,double tauEtaMax,bool useSeedTrack,double seedTrackPt,bool useTrackIsolation,double trackIsolationMinPt,
                       bool useECALIsolation,double gammaIsolationMinPt,bool useProngStructure,std::vector<MET>* corrMET) {

    std::cerr << "TauMETAlgo::run -> Test.. " << std::endl;

    double DeltaPx = 0.0;
    double DeltaPy = 0.0;
    double DeltaSumET = 0.0;
    for(PFTauCollection::size_type iPFTau=0;iPFTau<tauHandle->size();iPFTau++) {
      PFTauRef thePFTau(tauHandle,iPFTau);
      bool matchFlag = false;
      bool goodTau = false;
      if((fabs(thePFTau->eta()) <= tauEtaMax) && (thePFTau->et() >= tauMinEt)) {
        goodTau = true;
        if(useSeedTrack) {
          if(!(thePFTau->leadPFChargedHadrCand())) {goodTau = false;}
          else {
            if(thePFTau->leadPFChargedHadrCand()->et() < seedTrackPt) {goodTau = false;}
          }
        }
        if(useTrackIsolation) {
          vector<reco::PFCandidatePtr> PFTauProdIsolCHCands = (*thePFTau).isolationPFChargedHadrCands();
          for(vector<reco::PFCandidatePtr>::const_iterator iChargedHadrCand=PFTauProdIsolCHCands.begin();
              iChargedHadrCand!=PFTauProdIsolCHCands.end();++iChargedHadrCand) {
            if((**iChargedHadrCand).pt() > trackIsolationMinPt) {goodTau = false;}
          }
        }
        if(useECALIsolation) {
          vector<reco::PFCandidatePtr> PFTauProdIsolGammaCands = (*thePFTau).isolationPFGammaCands();
          for(vector<reco::PFCandidatePtr>::const_iterator iGammaCand=PFTauProdIsolGammaCands.begin();
              iGammaCand!=PFTauProdIsolGammaCands.end();++iGammaCand) {
            if((**iGammaCand).et() > gammaIsolationMinPt) {goodTau = false;}
          }
        }
        if(useProngStructure) {
          if((thePFTau->signalPFChargedHadrCands().size() != 1) && (thePFTau->signalPFChargedHadrCands().size() != 3)) {goodTau = false;}
        }
      }
      if(goodTau) {
        for(CaloJetCollection::const_iterator calojetIter = calojetHandle->begin();calojetIter != calojetHandle->end();++calojetIter) {
          if(deltaR(calojetIter->p4(),thePFTau->p4())<jetMatchDeltaR) {
            if( calojetIter->pt() > jetPTthreshold && calojetIter->emEnergyFraction() < jetEMfracLimit ) {
              double correct = correctedjets.correction (*calojetIter);
              DeltaPx += ((correct * calojetIter->px()) - thePFTau->px());
              DeltaPy += ((correct * calojetIter->py()) - thePFTau->py());
              DeltaSumET += ((correct * calojetIter->et()) - thePFTau->et());
            } else {
              DeltaPx += (calojetIter->px() - thePFTau->px());
              DeltaPy += (calojetIter->py() - thePFTau->py());
              DeltaSumET += (calojetIter->et() - thePFTau->et());
            }
            if(matchFlag) {std::cerr << "### TauMETAlgo - ERROR:  Multiple jet matches!!!! " << std::endl;}
            matchFlag = true;
          }
        }
      }
    }
    CorrMETData delta;
    delta.mex = DeltaPx;
    delta.mey = DeltaPy;
    delta.sumet = - DeltaSumET;
    const MET* u = &(uncorMET.front());
    double corrMetPx = u->px() + delta.mex;
    double corrMetPy = u->py() + delta.mey;
    MET::LorentzVector correctedMET4vector( corrMetPx, corrMetPy, 0., sqrt(corrMetPx*corrMetPx + corrMetPy*corrMetPy));
    std::vector<CorrMETData> corrections = u->mEtCorr();
    corrections.push_back(delta);
    MET result;
    result = MET((u->sumEt() + delta.sumet), corrections, correctedMET4vector, u->vertex());
    corrMET->push_back(result);
  }

