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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
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
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
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
      	virtual void beginJob() ;
      	virtual void analyze(const edm::Event&, const edm::EventSetup&);
//	std::map<reco::GsfElectronRef, reco::SuperClusterRef> probeFinder(const std::vector<reco::GsfElectronRef>&,
//							const edm::Handle<reco::SuperClusterCollection>&,
//							const edm::Handle<reco::SuperClusterCollection>&);
	std::map<reco::GsfElectronRef, reco::GsfElectronRef> probeFinder(const std::vector<reco::GsfElectronRef>&,
							const edm::Handle<reco::GsfElectronCollection>);
	reco::MET ersatzFabrik(const reco::GsfElectronRef&, const reco::SuperCluster&,
					const reco::MET&, const int);
	reco::MET ersatzFabrik(const reco::GsfElectronRef&, const reco::GsfElectronRef&,
					const reco::MET&);
	bool isInBarrel(double);
	bool isInEndCap(double);
	bool isInFiducial(double);

      	virtual void endJob();

      	// ----------member data ---------------------------
	edm::EDGetTokenT<reco::GenParticleCollection> MCTruthCollection_;
	edm::EDGetTokenT<reco::GsfElectronCollection> ElectronCollection_;
	edm::EDGetTokenT<reco::SuperClusterCollection> HybridScCollection_;
	edm::EDGetTokenT<reco::SuperClusterCollection> M5x5ScCollection_;
	edm::EDGetTokenT<reco::GenMETCollection> GenMEtCollection_;
	edm::EDGetTokenT<reco::CaloMETCollection> CaloMEtCollection_;
	edm::EDGetTokenT<reco::METCollection> T1MEtCollection_;
	edm::EDGetTokenT<reco::PFMETCollection> PfMEtCollection_;
	edm::EDGetTokenT<reco::METCollection> TcMEtCollection_;
	edm::EDGetTokenT<trigger::TriggerEvent> TriggerEvent_;
	edm::EDGetTokenT<edm::TriggerResults> TriggerResults_;
	edm::InputTag TriggerPath_;
	std::string TriggerName_, ProcessName_;
	edm::ParameterSet hyb_fCorrPSet_, m5x5_fCorrPSet_;
	double mW_, mZ_, mTPmin_, mTPmax_;
	double BarrelEtaMax_, EndCapEtaMin_, EndCapEtaMax_;

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
	int nTags_, nProbes_;
	double CaloMEt_, T1MEt_, PfMEt_, TcMEt_;
	double CaloMEtphi_, T1MEtphi_, PfMEtphi_, TcMEtphi_;
	int McElec_nZmum_, McElec_nFinal_;
	double McZ_m_, McZ_pt_, McZ_phi_, McZ_eta_, McZ_y_;
	double McZ_rescM_, McZ_rescPt_, McZ_rescEta_, McZ_rescPhi_, McZ_rescY_;

	int probe_nClus_[nEntries_arr_];
	int tag_q_[nEntries_arr_];
	double tag_pt_[nEntries_arr_], tag_eta_[nEntries_arr_], tag_phi_[nEntries_arr_];
	//double tag_caloV1_rescPt_[nEntries_arr_], tag_caloV1_rescEta_[nEntries_arr_], tag_caloV1_rescPhi_[nEntries_arr_];
	//double tag_caloV2_rescPt_[nEntries_arr_], tag_caloV2_rescEta_[nEntries_arr_], tag_caloV2_rescPhi_[nEntries_arr_];
	//double tag_caloV3_rescPt_[nEntries_arr_], tag_caloV3_rescEta_[nEntries_arr_], tag_caloV3_rescPhi_[nEntries_arr_];
	//double tag_caloV4_rescPt_[nEntries_arr_], tag_caloV4_rescEta_[nEntries_arr_], tag_caloV4_rescPhi_[nEntries_arr_];
	double tag_rescPt_[nEntries_arr_], tag_rescEta_[nEntries_arr_], tag_rescPhi_[nEntries_arr_];
	double tag_sIhIh_[nEntries_arr_], tag_dPhiIn_[nEntries_arr_], tag_dEtaIn_[nEntries_arr_];
	double tag_trckIso_[nEntries_arr_], tag_ecalIso_[nEntries_arr_], tag_hcalIso_[nEntries_arr_];
	double tag_e2x5Max_[nEntries_arr_], tag_e1x5Max_[nEntries_arr_], tag_e5x5_[nEntries_arr_];
	double tag_hoe_[nEntries_arr_], tag_eop_[nEntries_arr_], tag_pin_[nEntries_arr_], tag_pout_[nEntries_arr_];
	int probe_q_[nEntries_arr_];
	double probe_pt_[nEntries_arr_], probe_eta_[nEntries_arr_], probe_phi_[nEntries_arr_];
	double probe_rescPt_[nEntries_arr_], probe_rescEta_[nEntries_arr_], probe_rescPhi_[nEntries_arr_];
	double probe_sIhIh_[nEntries_arr_], probe_dPhiIn_[nEntries_arr_], probe_dEtaIn_[nEntries_arr_];
	double probe_trckIso_[nEntries_arr_], probe_ecalIso_[nEntries_arr_], probe_hcalIso_[nEntries_arr_];
	double probe_e2x5Max_[nEntries_arr_], probe_e1x5Max_[nEntries_arr_], probe_e5x5_[nEntries_arr_];
	double probe_hoe_[nEntries_arr_], probe_eop_[nEntries_arr_], probe_pin_[nEntries_arr_], probe_pout_[nEntries_arr_];
	double Z_pt_[nEntries_arr_], Z_eta_[nEntries_arr_], Z_phi_[nEntries_arr_], Z_m_[nEntries_arr_], Z_y_[nEntries_arr_];
	double Z_rescPt_[nEntries_arr_], Z_rescEta_[nEntries_arr_], Z_rescPhi_[nEntries_arr_], Z_rescM_[nEntries_arr_], Z_rescY_[nEntries_arr_];
	double Z_probe_dPhi_[nEntries_arr_];
	double ErsatzV1CaloMEt_[nEntries_arr_], ErsatzV1CaloMt_[nEntries_arr_], ErsatzV1CaloMEtPhi_[nEntries_arr_];
	double ErsatzV2CaloMEt_[nEntries_arr_], ErsatzV2CaloMEtPhi_[nEntries_arr_], ErsatzV2CaloMt_[nEntries_arr_];
	double ErsatzV3CaloMEt_[nEntries_arr_], ErsatzV3CaloMEtPhi_[nEntries_arr_], ErsatzV3CaloMt_[nEntries_arr_];
	double ErsatzV4CaloMEt_[nEntries_arr_], ErsatzV4CaloMEtPhi_[nEntries_arr_], ErsatzV4CaloMt_[nEntries_arr_];
	double ErsatzV1T1MEt_[nEntries_arr_], ErsatzV1T1Mt_[nEntries_arr_], ErsatzV1T1MEtPhi_[nEntries_arr_];
	double ErsatzV1PfMEt_[nEntries_arr_], ErsatzV1PfMt_[nEntries_arr_], ErsatzV1PfMEtPhi_[nEntries_arr_];
	double ErsatzV1TcMEt_[nEntries_arr_], ErsatzV1TcMt_[nEntries_arr_], ErsatzV1TcMEtPhi_[nEntries_arr_];
	double ErsatzV1_Mesc_[nEntries_arr_], ErsatzV1_rescMesc_[nEntries_arr_];
	double ErsatzV2_Mesc_[nEntries_arr_], ErsatzV2_rescMesc_[nEntries_arr_];
	double ErsatzV3_Mesc_[nEntries_arr_], ErsatzV3_rescMesc_[nEntries_arr_];
	double ErsatzV4_Mesc_[nEntries_arr_], ErsatzV4_rescMesc_[nEntries_arr_];

	double McElec_pt_[nEntries_arr_], McElec_eta_[nEntries_arr_], McElec_phi_[nEntries_arr_];
	double McElec_rescPt_[nEntries_arr_], McElec_rescEta_[nEntries_arr_], McElec_rescPhi_[nEntries_arr_];
	double McProbe_pt_[nEntries_arr_], McProbe_eta_[nEntries_arr_], McProbe_phi_[nEntries_arr_];
	double McProbe_rescPt_[nEntries_arr_], McProbe_rescEta_[nEntries_arr_], McProbe_rescPhi_[nEntries_arr_];
	double McElecProbe_dPhi_[nEntries_arr_], McElecProbe_dEta_[nEntries_arr_], McElecProbe_dR_[nEntries_arr_];

	double probe_d_MCE_SCE_[nEntries_arr_];
	double probe_sc_pt_[nEntries_arr_], probe_sc_eta_[nEntries_arr_], probe_sc_phi_[nEntries_arr_];
	double probe_sc_E_[nEntries_arr_], probe_sc_rawE_[nEntries_arr_], probe_sc_nClus_[nEntries_arr_];
	double probe_scV2_E_[nEntries_arr_];
	double probe_scV3_E_[nEntries_arr_];
	double probe_scV4_E_[nEntries_arr_];

	int iComb_;
};
#endif

