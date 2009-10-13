#ifndef ElectroWeakAnalysis_ErsatzMEt_h
#define ElectroWeakAnalysis_ErsatzMEt_h
// -*- C++ -*-
//
// Package:    ErsatzMEt
// Class:      ErsatzMEt
// 
/**\class ErsatzMEt ErsatzMEt.cc ElectroWeakAnalysis/ErsatzMEt/src/ErsatzMEt.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  David Wardrope
//         Created:  Tue Nov 11 16:47:29 GMT 2008
// $Id: ErsatzMEt.h,v 1.4 2009/10/12 23:57:09 rnandi Exp $
//
//


// system include files
#include <memory>

//Framework 
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
//Random Number Generator
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandFlat.h"
//CMSSW Containers
#include "DataFormats/Common/interface/ValueMap.h"
//Egamma Objects
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
//ECAL
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h" 
#include "RecoEcal/EgammaClusterAlgos/interface/EgammaSCEnergyCorrectionAlgo.h"
//Geometry
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
//DetIds
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
//Other Objects
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
//Physics Tools
#include "DataFormats/Math/interface/deltaR.h"

//Maths
#include "Math/GenVector/VectorUtil.h"
//ROOT
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
//Helper Functions
#include "ElectroWeakAnalysis/ZEE/interface/UniqueElectrons.h"
#include "ElectroWeakAnalysis/ZEE/interface/ElectronSelector.h"
#include "ElectroWeakAnalysis/ZEE/interface/CaloVectors.h"
#include "ElectroWeakAnalysis/ZEE/interface/SCEnergyCorrections.h"

#define nEntries_arr_ 4
//#include "DataFormats/EgammaReco/interface/SuperCluster.h"
//
// class declaration
//

class ErsatzMEt : public edm::EDAnalyzer {
   public:
      explicit ErsatzMEt(const edm::ParameterSet&);
      ~ErsatzMEt();


   private:
      	virtual void beginJob(const edm::EventSetup&) ;
      	virtual void analyze(const edm::Event&, const edm::EventSetup&);
//	std::map<reco::GsfElectronRef, reco::SuperClusterRef> probeFinder(const std::vector<reco::GsfElectronRef>&,
//							const edm::Handle<reco::SuperClusterCollection>&,
//							const edm::Handle<reco::SuperClusterCollection>&);
	std::map<reco::GsfElectronRef, reco::GsfElectronRef> probeFinder(const std::vector<reco::GsfElectronRef>&,
							const edm::Handle<reco::GsfElectronCollection>);
	reco::MET ersatzFabrik(const reco::GsfElectronRef&, const reco::SuperClusterRef&, 
					const edm::Handle<EcalRecHitCollection>&,
					const edm::Handle<reco::CaloMETCollection>&, const double&, const int);
      	virtual void endJob();

      	// ----------member data ---------------------------
	edm::InputTag MCTruthCollection_;
	edm::InputTag ElectronCollection_, HybridScCollection_, M5x5ScCollection_, EBRecHitCollection_, EERecHitCollection_;
	edm::InputTag eIdRobust_, eIdRobustTight_;
	edm::InputTag eIsoTrack_, eIsoEcal_, eIsoHcal_;
	edm::InputTag TrackCollection_, CaloMEtCollection_;
	edm::InputTag CaloTowerCollection_;
	edm::InputTag TriggerEvent_, TriggerResults_, TriggerPath_;
	std::string TriggerName_, ProcessName_;
	edm::TriggerNames TriggerNames_;
	edm::ParameterSet hyb_fCorrPSet_, m5x5_fCorrPSet_;
	double sigmaElectronicNoise_EB_;
	double mW_, mZ_, mTPmin_, mTPmax_;
	//should use enumerate to create indices for CutVector
	//at moment, 0 = Elec Pt, 1 = EB Track Isolation, 2 = EB Ecal Isolation, 3 = Hcal Isolation, 4 = EE Track Isolation
	//5 = EE Ecal Isolation, 6 = EE Hcal Isolation
	enum cut_index_t { EtCut_, EB_sIhIh_, EB_dEtaIn_, EB_dPhiIn_, EB_TrckIso_, EB_EcalIso_, EB_HcalIso_,
				EE_sIhIh_, EE_dEtaIn_, EE_dPhiIn_, EE_TrckIso_, EE_EcalIso_, EE_HcalIso_};

	std::vector<double> CutVector_;
		
	int etaWidth_, phiWidth_;
	bool Zevent_, HLTPathCheck_;
//	std::vector<double> EtaWeights_;
	HLTConfigProvider hltConfig_;
	
	edm::ESHandle<CaloGeometry> geoHandle_;
	edm::ESHandle<CaloTopology> pTopology_;
	//Output variables
	TTree* t_;
	double TotEClus_;
	int TotNProbes_;
	int nTags_, nProbes_;
	double recoCaloMEt_;
	int McElec_nZmum_, McElec_nFinal_;
	double McZ_m_, McZ_rescM_, McZ_Pt_, McZ_Phi_, McZ_Eta_;
	double McZ_rescPt_, McZ_rescEta_, McZ_rescPhi_;

	int nRecHitsInStrip_[nEntries_arr_], nRecHitsInCone_[nEntries_arr_];
	int probe_nClus_[nEntries_arr_], probe_elecMatch_[nEntries_arr_];
	int tag_q_[nEntries_arr_];
	double tag_pt_[nEntries_arr_], tag_eta_[nEntries_arr_], tag_phi_[nEntries_arr_];
	double tag_rescPt_[nEntries_arr_], tag_rescEta_[nEntries_arr_], tag_rescPhi_[nEntries_arr_];
	double tag_sIhIh_[nEntries_arr_], tag_dPhiIn_[nEntries_arr_], tag_dEtaIn_[nEntries_arr_];
	double tag_isoTrack_[nEntries_arr_], tag_isoEcal_[nEntries_arr_], tag_isoHcal_[nEntries_arr_];
	int probe_q_[nEntries_arr_];	
	double probe_pt_[nEntries_arr_], probe_eta_[nEntries_arr_], probe_phi_[nEntries_arr_];
	double probe_rescPt_[nEntries_arr_], probe_rescEta_[nEntries_arr_], probe_rescPhi_[nEntries_arr_];
	double probe_sIhIh_[nEntries_arr_], probe_isoTrack_[nEntries_arr_];
	double probe_e2x5Max_[nEntries_arr_], probe_e1x5Max_[nEntries_arr_], probe_e5x5_[nEntries_arr_];
	double rechit_E_[nEntries_arr_];
	double Z_pt_[nEntries_arr_];
	double Z_probe_dPhi_[nEntries_arr_];
	double ErsatzV1CaloMEt_[nEntries_arr_], ErsatzV1CaloMt_[nEntries_arr_];
	double ErsatzV1CaloMEtPhi_[nEntries_arr_];
	double ErsatzV1aCaloMEt_[nEntries_arr_], ErsatzV1aCaloMEtPhi_[nEntries_arr_], ErsatzV1aCaloMt_[nEntries_arr_];
	double ErsatzV1bCaloMEt_[nEntries_arr_], ErsatzV1bCaloMEtPhi_[nEntries_arr_], ErsatzV1bCaloMt_[nEntries_arr_];
	double ErsatzV1cCaloMEt_[nEntries_arr_], ErsatzV1cCaloMEtPhi_[nEntries_arr_], ErsatzV1cCaloMt_[nEntries_arr_];
	double ErsatzV2CaloMEt_[nEntries_arr_], ErsatzV2CaloMt_[nEntries_arr_];
	double ErsatzV2CaloMEtPhi_[nEntries_arr_];
	double Ersatz_Mesc_[nEntries_arr_], ErsatzV1_Mesc_[nEntries_arr_];
	double Ersatz_rescMesc_[nEntries_arr_], ErsatzV1_rescMesc_[nEntries_arr_];

	double McElec_pt_[nEntries_arr_], McElec_eta_[nEntries_arr_], McElec_phi_[nEntries_arr_],
		McElec_rescPt_[nEntries_arr_], McElec_rescEta_[nEntries_arr_], McElec_rescPhi_[nEntries_arr_]; 
	double McProbe_pt_[nEntries_arr_], McProbe_eta_[nEntries_arr_], McProbe_rescPt_[nEntries_arr_], McProbe_rescEta_[nEntries_arr_]; 
	double McProbe_phi_[nEntries_arr_], McProbe_rescPhi_[nEntries_arr_];
	double McElecProbe_dPhi_[nEntries_arr_], McElecProbe_dR_[nEntries_arr_];

	double probe_d_MCE_SCE_[nEntries_arr_];
	double probe_UnclusEcalE_[nEntries_arr_], probe_HcalEt01_[nEntries_arr_], probe_HcalEt015_[nEntries_arr_];
	double probe_HcalEt02_[nEntries_arr_],probe_HcalEt025_[nEntries_arr_];
	double probe_HcalE015_[nEntries_arr_];
	double probe_E_[nEntries_arr_], probe_rawE_[nEntries_arr_], probe_fEtaCorrE_[nEntries_arr_], probe_fBremCorrE_[nEntries_arr_];
	double probe_EAdd_[nEntries_arr_];

	int iComb_;
};
#endif

