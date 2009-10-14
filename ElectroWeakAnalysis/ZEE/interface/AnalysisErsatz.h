// -*- C++ -*-
//
// Package:    AnalysisErsatz
// Class:      AnalysisErsatz
// 
/**\class AnalysisErsatz AnalysisErsatz.cc ElectroWeakAnalysis/AnalysisErsatz/src/AnalysisErsatz.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  David Wardrope
//         Created:  Fri Nov 14 15:00:43 GMT 2008
// $Id: AnalysisErsatz.h,v 1.1 2009/04/23 14:19:41 wardrope Exp $
//
//


// system include files
#include <memory>
//Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
//Random Number Generator
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandFlat.h"
//Egamma 
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
//ECAL 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEgamma/EgammaTools/interface/ECALPositionCalculator.h"
//OtherObjects
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/GenMETFwd.h"
//PhysicsTools
//#include "PhysicsTools/Utilities/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "PhysicsTools/CandUtils/interface/CenterOfMassBooster.h"
#include "Math/GenVector/Boost.h"

//Helper Functions
#include "ElectroWeakAnalysis/ZEE/interface/UniqueElectrons.h"
#include "ElectroWeakAnalysis/ZEE/interface/ElectronSelector.h"
#include "ElectroWeakAnalysis/ZEE/interface/CaloVectors.h"
//ROOT
#include "TTree.h"

#define nEntries_arr_ 4
//
// class declaration
//

//namespace CLHEP{
//	class RandFlat;
//}

class AnalysisErsatz : public edm::EDAnalyzer {
   public:
      explicit AnalysisErsatz(const edm::ParameterSet&);
      ~AnalysisErsatz();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
	edm::InputTag MCTruthCollection_;
	edm::InputTag ElectronCollection_, EBRecHitCollection_, EERecHitCollection_; 
	edm::InputTag eIsoTrack_, eIsoEcal_, eIsoHcal_;
	edm::InputTag TrackCollection_, CaloMEtCollection_, GenMEtCollection_;
	bool ErsatzEvent_, C_Fiducial_;
        enum cut_index_t { EtCut_, EB_sIhIh_, EB_dEtaIn_, EB_dPhiIn_, EB_TrckIso_, EB_EcalIso_, EB_HcalIso_,
                                EE_sIhIh_, EE_dEtaIn_, EE_dPhiIn_, EE_TrckIso_, EE_EcalIso_, EE_HcalIso_};
        std::vector<double> CutVector_;
	double mW_, mZ_;
        edm::InputTag TriggerEvent_, TriggerResults_, TriggerPath_;
	std::string TriggerName_;

	TTree* t_;
	double Boson_pt_, Boson_y_, Boson_m_, Boson_mt_;
	double Boson_phi_;
	double McElec3_pt_[nEntries_arr_], McElec3_eta_[nEntries_arr_];
	double McElec1_pt_[nEntries_arr_], McElec1_eta_[nEntries_arr_];
	int RndmInt_;
	double RndmMcElec_pt_, RndmMcElec_eta_, RndmMcElec_phi_;
	double RndmMcElec_Rescaled_pt_, RndmMcElec_Rescaled_eta_, RndmMcElec_Rescaled_phi_;
	double RndmMcElecTRIG_pt_, RndmMcElecTRIG_eta_, RndmMcElecRECO_pt_, RndmMcElecRECO_eta_;
	double OthrMcElec_pt_, OthrMcElec_eta_, OthrMcElec_phi_;
	double OthrMcElecTRIG_pt_, OthrMcElecTRIG_eta_, OthrMcElecRECO_pt_, OthrMcElecRECO_eta_;
	double OthrMcElec_Rescaled_pt_, OthrMcElec_Rescaled_eta_, OthrMcElec_Rescaled_phi_;
	int  RndmTrig_, RndmReco_, OthrTrig_, OthrReco_;
	double McNu_pt_, McNu_eta_, McNu_phi_, McNu_ECALeta_;
	double McNu_vx_, McNu_vy_, McNu_vz_;
	double McLeptons_dPhi_, McLeptons_dEta_, McLeptons_dR_;

	double elec_pt_[nEntries_arr_], elec_eta_[nEntries_arr_], elec_phi_[nEntries_arr_];
	double elec_pt25_, elec_eta25_, elec_phi25_;
        double elec_sIhIh_[nEntries_arr_], elec_dPhiIn_[nEntries_arr_], elec_dEtaIn_[nEntries_arr_];
        double elec_isoTrack_[nEntries_arr_], elec_isoEcal_[nEntries_arr_], elec_isoHcal_[nEntries_arr_];

	double Selected_nuPt_[nEntries_arr_],Selected_nuEta_[nEntries_arr_];
	double trackMEt_x_, trackMEt_y_;
	double caloMEt_, caloSumEt_, caloUESumEt_;
	double caloMEt25_, caloMEt30_;
	double caloMEtECAL25_, caloMEtECAL30_;//using ECAL eta to restrict neutrino
	double caloMEtPhi_, caloMEtPhi25_, caloMEtPhi30_;
	double caloMEtPhiECAL25_, caloMEtPhiECAL30_;//using ECAL eta to restrict neutrino
	double caloMt_[nEntries_arr_], caloMt25_[nEntries_arr_], caloMt30_[nEntries_arr_];
	double genMEt_, genMt_[nEntries_arr_], genUESumEt_, genMEt25_;
	int nHltObj_, nSelElecs_;
	double HltObj_pt_[nEntries_arr_], HltObj_eta_[nEntries_arr_];
};

