#ifndef DATACERTIFICATIONJETMET_H
#define DATACERTIFICATIONJETMET_H

// author: Kenichi Hatakeyama (Rockefeller U.)

// system include files
#include <memory>
#include <cstdio>
#include <cmath>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
//
// class decleration
//

class DataCertificationJetMET : public DQMEDHarvester {
   public:
      explicit DataCertificationJetMET(const edm::ParameterSet&);
      ~DataCertificationJetMET() override;

   private:
      void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override ;

      MonitorElement*  reportSummary;
      MonitorElement*  CertificationSummary;
      MonitorElement*  reportSummaryMap;
      MonitorElement*  CertificationSummaryMap;

   // ----------member data ---------------------------

   edm::ParameterSet conf_;
   edm::Service<TFileService> fs_;
   int verbose_;
   bool InMemory_;
   bool isData;
   std::string metFolder;
   std::string jetAlgo;

   edm::InputTag inputMETLabelRECO_;
   edm::InputTag inputMETLabelRECOUncleaned_;
   edm::InputTag inputMETLabelMiniAOD_;
   edm::InputTag inputJetLabelRECO_;
   edm::InputTag inputJetLabelMiniAOD_;

   int    nbinsPV_;
   double nPVMin_; 
   double nPVMax_;

   int etaBin_;
   double etaMin_;
   double etaMax_;

   int ptBin_;
   double ptMin_;
   double ptMax_;

   std::string folderName;

   bool caloJetMeanTest;
   bool caloJetKSTest;
   bool pfJetMeanTest;
   bool pfJetKSTest;
   bool jptJetMeanTest;
   bool jptJetKSTest;
   bool caloMETMeanTest;
   bool caloMETKSTest;
   bool pfMETMeanTest;
   bool pfMETKSTest;
   bool tcMETMeanTest;
   bool tcMETKSTest;

   bool jetTests[5][2];  //one for each type of jet certification/test type
   bool metTests[5][2];  //one for each type of met certification/test type

   bool isHI;

  //MET: filter efficiencies, started from uncleaned directories
   MonitorElement* mMET_EffHBHENoiseFilter;
   MonitorElement* mMET_EffCSCTightHaloFilter;
   MonitorElement* mMET_EffeeBadScFilter;
   MonitorElement* mMET_EffEcalDeadCellTriggerFilter;
   MonitorElement* mMET_EffEcalDeadCellBoundaryFilter;
   MonitorElement* mMET_EffHBHEIsoNoiseFilter;
   MonitorElement* mMET_EffCSCTightHalo2015Filter;
   MonitorElement* mMET_EffHcalStripHaloFilter;

   //MET: RECO vs MiniAOD histos
   MonitorElement* mMET_MiniAOD_over_Reco;
   MonitorElement* mMEy_MiniAOD_over_Reco;
   MonitorElement* mSumET_MiniAOD_over_Reco;
   MonitorElement* mMETPhi_MiniAOD_over_Reco;
   MonitorElement* mMET_logx_MiniAOD_over_Reco;
   MonitorElement* mSumET_logx_MiniAOD_over_Reco;
   MonitorElement* mChargedHadronEtFraction_MiniAOD_over_Reco;
   MonitorElement* mNeutralHadronEtFraction_MiniAOD_over_Reco;
   MonitorElement* mPhotonEtFraction_MiniAOD_over_Reco;
   MonitorElement* mHFHadronEtFraction_MiniAOD_over_Reco;
   MonitorElement* mHFEMEtFraction_MiniAOD_over_Reco;
   MonitorElement* mMET_nVtx_profile_MiniAOD_over_Reco;
   MonitorElement* mSumET_nVtx_profile_MiniAOD_over_Reco;
   MonitorElement* mChargedHadronEtFraction_nVtx_profile_MiniAOD_over_Reco;
   MonitorElement* mNeutralHadronEtFraction_nVtx_profile_MiniAOD_over_Reco;
   MonitorElement* mPhotonEtFraction_nVtx_profile_MiniAOD_over_Reco;

   //Jets: RECO vs MiniAOD histos
   MonitorElement* mPt_MiniAOD_over_Reco;
   MonitorElement* mEta_MiniAOD_over_Reco;
   MonitorElement* mPhi_MiniAOD_over_Reco;
   MonitorElement* mNjets_MiniAOD_over_Reco;
   MonitorElement* mPt_uncor_MiniAOD_over_Reco;
   MonitorElement* mEta_uncor_MiniAOD_over_Reco;
   MonitorElement* mPhi_uncor_MiniAOD_over_Reco;
   MonitorElement* mJetEnergyCorr_MiniAOD_over_Reco;
   MonitorElement* mJetEnergyCorrVSeta_MiniAOD_over_Reco;
   MonitorElement* mDPhi_MiniAOD_over_Reco;
   MonitorElement* mLooseJIDPassFractionVSeta_MiniAOD_over_Reco;
   MonitorElement* mPt_Barrel_MiniAOD_over_Reco;
   MonitorElement* mPt_EndCap_MiniAOD_over_Reco;
   MonitorElement* mPt_Forward_MiniAOD_over_Reco;
   MonitorElement* mMVAPUJIDDiscriminant_lowPt_Barrel_MiniAOD_over_Reco;
   MonitorElement* mMVAPUJIDDiscriminant_lowPt_EndCap_MiniAOD_over_Reco;
   MonitorElement* mMVAPUJIDDiscriminant_lowPt_Forward_MiniAOD_over_Reco;
   MonitorElement* mMVAPUJIDDiscriminant_mediumPt_EndCap_MiniAOD_over_Reco;
   MonitorElement* mMVAPUJIDDiscriminant_highPt_Barrel_MiniAOD_over_Reco;
   MonitorElement* mCHFracVSpT_Barrel_MiniAOD_over_Reco;
   MonitorElement* mNHFracVSpT_EndCap_MiniAOD_over_Reco;
   MonitorElement* mPhFracVSpT_Barrel_MiniAOD_over_Reco;
   MonitorElement* mHFHFracVSpT_Forward_MiniAOD_over_Reco;
   MonitorElement* mHFEFracVSpT_Forward_MiniAOD_over_Reco;
   MonitorElement* mCHFrac_MiniAOD_over_Reco;
   MonitorElement* mNHFrac_MiniAOD_over_Reco;
   MonitorElement* mPhFrac_MiniAOD_over_Reco;
   MonitorElement* mChargedMultiplicity_MiniAOD_over_Reco;
   MonitorElement* mNeutralMultiplicity_MiniAOD_over_Reco;
   MonitorElement* mMuonMultiplicity_MiniAOD_over_Reco;
   MonitorElement* mNeutralFraction_MiniAOD_over_Reco;

};



#endif
