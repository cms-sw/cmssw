#include "ElectroWeakAnalysis/ZEE/interface/ErsatzMEt.h"
#include "FWCore/Common/interface/TriggerNames.h"

ErsatzMEt::ErsatzMEt(const edm::ParameterSet& ps)
{
	MCTruthCollection_  = consumes<reco::GenParticleCollection>(ps.getParameter<edm::InputTag>("MCTruthCollection"));
	ElectronCollection_ = consumes<reco::GsfElectronCollection>(ps.getParameter<edm::InputTag>("ElectronCollection"));
	HybridScCollection_ = consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("HybridScCollection"));
	M5x5ScCollection_ = consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("M5x5ScCollection"));
	GenMEtCollection_  = consumes<reco::GenMETCollection>(ps.getParameter<edm::InputTag>("GenMEtCollection"));
	CaloMEtCollection_ = consumes<reco::CaloMETCollection>(ps.getParameter<edm::InputTag>("CaloMEtCollection"));
	//T1MEtCollection_ = consumes<reco::METCollection>(ps.getParameter<edm::InputTag>("T1MEtCollection"));
	PfMEtCollection_ = consumes<reco::PFMETCollection>(ps.getParameter<edm::InputTag>("PfMEtCollection"));
	TcMEtCollection_ = consumes<reco::METCollection>(ps.getParameter<edm::InputTag>("TcMEtCollection"));
	TriggerEvent_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("TriggerEvent"));
	TriggerPath_ = ps.getParameter<edm::InputTag>("TriggerPath");
	TriggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
	TriggerName_ = ps.getParameter<std::string>("TriggerName");
	HLTPathCheck_ = ps.getParameter<bool>("HLTPathCheck");

	Zevent_ = ps.getParameter<bool>("Zevent");
	mW_ = ps.getParameter<double>("mW");
        mZ_ = ps.getParameter<double>("mZ");
	mTPmin_ = ps.getParameter<double>("mTPmin");
	mTPmax_ = ps.getParameter<double>("mTPmax");
	BarrelEtaMax_ = ps.getParameter<double>("BarrelEtaMax");
	EndCapEtaMin_ = ps.getParameter<double>("EndCapEtaMin");
	EndCapEtaMax_ = ps.getParameter<double>("EndCapEtaMax");

	hyb_fCorrPSet_ = ps.getParameter<edm::ParameterSet>("hyb_fCorrPSet");
	m5x5_fCorrPSet_ = ps.getParameter<edm::ParameterSet>("m5x5_fCorrPSet");

 	double CElecPtMin = ps.getParameter<double>("CElecPtMin");
	double CEB_siEiE = ps.getParameter<double>("CEB_sigmaIEtaIEta");
	double CEB_dPhiIn = ps.getParameter<double>("CEB_deltaPhiIn");
	double CEB_dEtaIn = ps.getParameter<double>("CEB_deltaEtaIn");
	double CEB_EcalIso = ps.getParameter<double>("CEB_EcalIso");
	double CEB_HcalIso = ps.getParameter<double>("CEB_HcalIso");
	double CEB_TrckIso = ps.getParameter<double>("CEB_TrckIso");
	double CEE_siEiE = ps.getParameter<double>("CEE_sigmaIEtaIEta");
	double CEE_dPhiIn = ps.getParameter<double>("CEE_deltaPhiIn");
	double CEE_dEtaIn = ps.getParameter<double>("CEE_deltaEtaIn");
	double CEE_EcalIso = ps.getParameter<double>("CEE_EcalIso");
	double CEE_HcalIso = ps.getParameter<double>("CEE_HcalIso");
	double CEE_TrckIso = ps.getParameter<double>("CEE_TrckIso");

	CutVector_.resize(13);
	CutVector_[EtCut_] = CElecPtMin;
	CutVector_[EB_sIhIh_] = CEB_siEiE;
	CutVector_[EB_dPhiIn_] = CEB_dPhiIn;
	CutVector_[EB_dEtaIn_] = CEB_dEtaIn;
	CutVector_[EB_TrckIso_] = CEB_TrckIso;
	CutVector_[EB_EcalIso_] = CEB_EcalIso;
	CutVector_[EB_HcalIso_] = CEB_HcalIso;
	CutVector_[EE_sIhIh_] = CEE_siEiE;
	CutVector_[EE_dPhiIn_] = CEE_dPhiIn;
	CutVector_[EE_dEtaIn_] = CEE_dEtaIn;
	CutVector_[EE_TrckIso_] = CEE_TrckIso;
	CutVector_[EE_EcalIso_] = CEE_EcalIso;
	CutVector_[EE_HcalIso_] = CEE_HcalIso;

	for(std::vector<double>::const_iterator it = CutVector_.begin(); it != CutVector_.end(); ++it)
	{
		edm::LogDebug_("","",101)<<"CutVector_ = "<< *it;
	}
}

ErsatzMEt::~ErsatzMEt()
{
}

// ------------ method called once each job just before starting event loop  ------------
void ErsatzMEt::beginJob()
{
	edm::Service<TFileService> fs;

	t_ = fs->make<TTree>("ErsatzMEt", "Data on ErsatzMEt");

	edm::LogDebug_("","", 75)<<"Creating Ersatz MEt branches.";

	t_->Branch("nTags", &nTags_, "nTags/I");
	t_->Branch("nProbes", &nProbes_, "nProbes/I");

	t_->Branch("ErsatzV1CaloMEt", ErsatzV1CaloMEt_, "ErsatzV1CaloMEt[4]/D");
	t_->Branch("ErsatzV1CaloMt", ErsatzV1CaloMt_, "ErsatzV1CaloMt[4]/D");
	t_->Branch("ErsatzV1CaloMEtPhi", ErsatzV1CaloMEtPhi_, "ErsatzV1CaloMEtPhi[4]/D");
	t_->Branch("ErsatzV2CaloMEt", ErsatzV2CaloMEt_, "ErsatzV2CaloMEt[4]/D");
	t_->Branch("ErsatzV2CaloMEtPhi", ErsatzV2CaloMEtPhi_, "ErsatzV2CaloMEtPhi[4]/D");
	t_->Branch("ErsatzV2CaloMt", ErsatzV2CaloMt_, "ErsatzV2CaloMt[4]/D");
	t_->Branch("ErsatzV3CaloMEt", ErsatzV3CaloMEt_, "ErsatzV3CaloMEt[4]/D");
	t_->Branch("ErsatzV3CaloMEtPhi", ErsatzV3CaloMEtPhi_, "ErsatzV3CaloMEtPhi[4]/D");
	t_->Branch("ErsatzV3CaloMt", ErsatzV3CaloMt_, "ErsatzV3CaloMt[4]/D");
	t_->Branch("ErsatzV4CaloMEt", ErsatzV4CaloMEt_, "ErsatzV4CaloMEt[4]/D");
	t_->Branch("ErsatzV4CaloMEtPhi", ErsatzV4CaloMEtPhi_, "ErsatzV4CaloMEtPhi[4]/D");
	t_->Branch("ErsatzV4CaloMt", ErsatzV4CaloMt_, "ErsatzV4CaloMt[4]/D");
	t_->Branch("ErsatzV1T1MEt", ErsatzV1T1MEt_, "ErsatzV1T1MEt[4]/D");
	t_->Branch("ErsatzV1T1Mt", ErsatzV1T1Mt_, "ErsatzV1T1Mt[4]/D");
	t_->Branch("ErsatzV1T1MEtPhi", ErsatzV1T1MEtPhi_, "ErsatzV1T1MEtPhi[4]/D");
	t_->Branch("ErsatzV1PfMEt", ErsatzV1PfMEt_, "ErsatzV1PfMEt[4]/D");
	t_->Branch("ErsatzV1PfMt", ErsatzV1PfMt_, "ErsatzV1PfMt[4]/D");
	t_->Branch("ErsatzV1PfMEtPhi", ErsatzV1TcMEtPhi_, "ErsatzV1PfMEtPhi[4]/D");
	t_->Branch("ErsatzV1TcMEt", ErsatzV1TcMEt_, "ErsatzV1TcMEt[4]/D");
	t_->Branch("ErsatzV1TcMt", ErsatzV1TcMt_, "ErsatzV1TcMt[4]/D");
	t_->Branch("ErsatzV1TcMEtPhi", ErsatzV1TcMEtPhi_, "ErsatzV1TcMEtPhi[4]/D");

	t_->Branch("CaloMEt", &CaloMEt_, "CaloMEt/D");
	t_->Branch("CaloMEtphi", &CaloMEtphi_, "CaloMEtphi/D");
	t_->Branch("T1MEt", &T1MEt_, "T1MEt/D");
	t_->Branch("T1MEtphi", &T1MEtphi_, "T1MEtphi/D");
	t_->Branch("PfMEt", &PfMEt_, "PfMEt/D");
	t_->Branch("PfMEtphi", &PfMEtphi_, "PfMEtphi/D");
	t_->Branch("TcMEt", &TcMEt_, "TcMEt/D");
	t_->Branch("TcMEtphi", &TcMEtphi_, "TcMEtphi/D");

	edm::LogDebug_("","", 91)<<"Creating electron branches.";
	t_->Branch("tag_q", tag_q_,"tag_q[4]/I");
	t_->Branch("tag_pt", tag_pt_,"tag_pt[4]/D");
	t_->Branch("tag_eta", tag_eta_,"tag_eta[4]/D");
	t_->Branch("tag_phi", tag_phi_,"tag_phi[4]/D");
	t_->Branch("tag_sIhIh", tag_sIhIh_, "tag_sIhIh[4]/D");
	t_->Branch("tag_dPhiIn", tag_dPhiIn_, "tag_dPhiIn[4]/D");
	t_->Branch("tag_dEtaIn", tag_dEtaIn_, "tag_dEtaIn[4]/D");
	t_->Branch("tag_trckIso", tag_trckIso_,"tag_trckIso[4]/D");
	t_->Branch("tag_ecalIso", tag_ecalIso_,"tag_ecalIso[4]/D");
	t_->Branch("tag_hcalIso", tag_hcalIso_,"tag_hcalIso[4]/D");
	t_->Branch("tag_e2x5Max", tag_e2x5Max_,"tag_e2x5Max[4]/D");
	t_->Branch("tag_e1x5Max", tag_e1x5Max_,"tag_e1x5Max[4]/D");
	t_->Branch("tag_e5x5", tag_e5x5_,"tag_e5x5[4]/D");
	t_->Branch("tag_hoe", tag_hoe_,"tag_hoe[4]/D");
	t_->Branch("tag_eop", tag_eop_,"tag_eop[4]/D");
	t_->Branch("tag_pin", tag_pin_,"tag_pin[4]/D");
	t_->Branch("tag_pout", tag_pout_,"tag_pout[4]/D");
	t_->Branch("tag_rescPt", tag_rescPt_, "tag_rescPt[4]/D");
	t_->Branch("tag_rescEta", tag_rescEta_, "tag_rescEta[4]/D");
	t_->Branch("tag_rescPhi", tag_rescPhi_, "tag_rescPhi[4]/D");

	edm::LogDebug_("","", 103)<<"Creating ersatz neutrino branches.";
	t_->Branch("probe_q", probe_q_,"probe_q[4]/I");
	t_->Branch("probe_pt", probe_pt_,"probe_pt[4]/D");
	t_->Branch("probe_eta", probe_eta_,"probe_eta[4]/D");
	t_->Branch("probe_phi", probe_phi_,"probe_phi[4]/D");
	t_->Branch("probe_sIhIh", probe_sIhIh_, "probe_sIhIh[4]/D");
	t_->Branch("probe_dPhiIn", probe_dPhiIn_, "probe_dPhiIn[4]/D");
	t_->Branch("probe_dEtaIn", probe_dEtaIn_, "probe_dEtaIn[4]/D");
	t_->Branch("probe_trckIso", probe_trckIso_,"probe_trckIso[4]/D");
	t_->Branch("probe_ecalIso", probe_ecalIso_,"probe_ecalIso[4]/D");
	t_->Branch("probe_hcalIso", probe_hcalIso_,"probe_hcalIso[4]/D");
	t_->Branch("probe_e2x5Max", probe_e2x5Max_,"probe_e2x5Max[4]/D");
	t_->Branch("probe_e1x5Max", probe_e1x5Max_,"probe_e1x5Max[4]/D");
	t_->Branch("probe_e5x5", probe_e5x5_,"probe_e5x5[4]/D");
	t_->Branch("probe_hoe", probe_hoe_,"probe_hoe[4]/D");
	t_->Branch("probe_eop", probe_eop_,"probe_eop[4]/D");
	t_->Branch("probe_pin", probe_pin_,"probe_pin[4]/D");
	t_->Branch("probe_pout", probe_pout_,"probe_pout[4]/D");
	t_->Branch("probe_rescPt", probe_rescPt_, "probe_rescPt[4]/D");
	t_->Branch("probe_rescEta", probe_rescEta_, "probe_rescEta[4]/D");
	t_->Branch("probe_rescPhi", probe_rescPhi_, "probe_rescPhi[4]/D");

	t_->Branch("Z_m", Z_m_, "Z_m[4]/D");
	t_->Branch("Z_pt", Z_pt_, "Z_pt[4]/D");
	t_->Branch("Z_eta", Z_eta_, "Z_eta[4]/D");
	t_->Branch("Z_y", Z_y_, "Z_y[4]/D");
	t_->Branch("Z_phi", Z_phi_, "Z_phi[4]/D");
	t_->Branch("Z_rescM", Z_rescM_, "Z_rescM[4]/D");
	t_->Branch("Z_rescPt", Z_rescPt_, "Z_rescPt[4]/D");
	t_->Branch("Z_rescEta", Z_rescEta_, "Z_rescEta[4]/D");
	t_->Branch("Z_rescY", Z_rescY_, "Z_rescY[4]/D");
	t_->Branch("Z_rescPhi", Z_rescPhi_, "Z_rescPhi[4]/D");
	t_->Branch("Z_probe_dPhi",Z_probe_dPhi_,"Z_probe_dPhi[4]/D");

	t_->Branch("probe_sc_pt",probe_sc_pt_, "probe_sc_pt[4]/D");
	t_->Branch("probe_sc_eta",probe_sc_eta_, "probe_sc_eta[4]/D");
	t_->Branch("probe_sc_phi", probe_sc_phi_, "probe_sc_phi[4]/D");
	t_->Branch("probe_sc_E",probe_sc_E_, "probe_sc_E[4]/D");
	t_->Branch("probe_sc_rawE",probe_sc_rawE_, "probe_sc_rawE[4]/D");
	t_->Branch("probe_sc_nClus", probe_sc_nClus_, "probe_sc_nClus[4]/D");
	t_->Branch("probe_scV2_E",probe_scV2_E_, "probe_scV2_E[4]/D");
	t_->Branch("probe_scV3_E",probe_scV3_E_, "probe_scV3_E[4]/D");
	t_->Branch("probe_scV4_E",probe_scV4_E_, "probe_scV4_E[4]/D");
	t_->Branch("probe_d_MCE_SCE", probe_d_MCE_SCE_, "probe_d_MCE_SCE[4]/D");

	t_->Branch("ErsatzV1_Mesc", ErsatzV1_Mesc_, "ErsatzV1_Mesc[4]/D");
	t_->Branch("ErsatzV1_rescMesc", ErsatzV1_rescMesc_, "ErsatzV1_rescMesc[4]/D");

	t_->Branch("McElec_nFinal", &McElec_nFinal_, "McElec_nFinal/I");

	if(Zevent_){
		t_->Branch("McZ_m", &McZ_m_, "McZ_m/D");
		t_->Branch("McZ_rescM", &McZ_rescM_, "McZ_rescM/D");
		t_->Branch("McZ_Pt", &McZ_pt_, "McZ_Pt/D");
		t_->Branch("McZ_y", &McZ_y_, "McZ_y/D");
		t_->Branch("McZ_Eta", &McZ_eta_, "McZ_Eta/D");
		t_->Branch("McZ_Phi", &McZ_phi_, "McZ_Phi/D");
		t_->Branch("McZ_rescPt", &McZ_rescPt_, "McZ_Pt/D");
		t_->Branch("McZ_rescY", &McZ_rescY_, "McZ_rescY/D");
		t_->Branch("McZ_rescEta", &McZ_rescEta_, "McZ_Eta/D");
		t_->Branch("McZ_rescPhi", &McZ_rescPhi_, "McZ_Phi/D");
		t_->Branch("McElec_nZmum", &McElec_nZmum_, "McElec_nZmum/I");
		t_->Branch("McElec_eta", &McElec_eta_, "McElec_eta[4]/D");
		t_->Branch("McElec_pt", &McElec_pt_, "McElec_pt[4]/D");
		t_->Branch("McElec_phi", &McElec_phi_, "McElec_phi[4]/D");
		t_->Branch("McElec_rescEta", &McElec_rescEta_, "McElec_rescEta[4]/D");
		t_->Branch("McElec_rescPhi", &McElec_rescPhi_, "McElec_rescPhi[4]/D");
		t_->Branch("McElec_rescPt", &McElec_rescPt_, "McElec_rescPt[4]/D");
		t_->Branch("McProbe_eta", &McProbe_eta_, "McProbe_eta[4]/D");
		t_->Branch("McProbe_pt", &McProbe_pt_, "McProbe_pt[4]/D");
		t_->Branch("McProbe_phi", &McProbe_phi_, "McProbe_phi[4]/D");
		t_->Branch("McProbe_rescEta", &McProbe_rescEta_, "McProbe_rescEta[4]/D");
		t_->Branch("McProbe_rescPt", &McProbe_rescPt_, "McProbe_rescPt[4]/D");
		t_->Branch("McProbe_rescPhi", &McProbe_rescPhi_, "McProbe_rescPhi[4]/D");
		t_->Branch("McElecProbe_dPhi", &McElecProbe_dPhi_, "McElecProbe_dPhi/D");
		t_->Branch("McElecProbe_dEta", &McElecProbe_dEta_, "McElecProbe_dEta/D");
		t_->Branch("McElecProbe_dR", &McElecProbe_dR_, "McElecProbe_dR/D");
	}

}

void ErsatzMEt::analyze(const edm::Event& evt, const edm::EventSetup& es)
{

       es.get<CaloGeometryRecord>().get(geoHandle_);
       es.get<CaloTopologyRecord>().get(pTopology_);

	edm::LogDebug_("","", 151)<<"Initialising variables.";
	nTags_ = -99; nProbes_ = -99;
	CaloMEt_ = -99.; CaloMEtphi_ = -99.;
	T1MEt_ = -99.; T1MEtphi_ = -99.;
	PfMEt_ = -99.; PfMEtphi_ = -99.;
	TcMEt_ = -99.; TcMEtphi_ = -99.;
	if(Zevent_)
	{
		McZ_m_ = -99.; McZ_pt_ = -99.; McZ_y_ = -99.; McZ_eta_ = -99.; McZ_phi_ = -99.;
		McZ_rescM_ = -99.; McZ_rescPt_ = -99.; McZ_rescY_ = -99.; McZ_rescEta_ = -99.; McZ_rescPhi_ = -99.;
		McElec_nZmum_ = -99; McElec_nFinal_ = -99;
	}

	for(int i = 0; i < nEntries_arr_; ++i)
	{
		tag_q_[i] = -99;
		tag_pt_[i] = -99.; tag_eta_[i] = -99.; tag_phi_[i] = -99.;
		tag_rescPt_[i] = -99.; tag_rescEta_[i] = -99.; tag_rescPhi_[i] = -99.;
		tag_trckIso_[i] = -99.; tag_ecalIso_[i] = -99.; tag_hcalIso_[i] = -99.;
		tag_sIhIh_[i] = -99.; tag_dPhiIn_[i] = -99.; tag_dEtaIn_[i] = -99.;
		tag_e5x5_[i] = -99.; tag_e2x5Max_[i] = -99.; tag_e1x5Max_[i] = -99.;
		tag_hoe_[i] = -99.; tag_eop_[i] = -99.; tag_pin_[i] = -99.; tag_pout_[i] = -99.;

		probe_q_[i] = -99;
		probe_pt_[i] = -99.; probe_eta_[i] = -99.; probe_phi_[i] = -99.;
		probe_rescPt_[i] = -99.; probe_rescEta_[i] = -99.; probe_rescPhi_[i] = -99.;
		probe_trckIso_[i] = -99.; probe_ecalIso_[i] = -99.; probe_hcalIso_[i] = -99.;
		probe_sIhIh_[i] = -99.; probe_dPhiIn_[i] = -99.; probe_dEtaIn_[i] = -99.;
		probe_e5x5_[i] = -99.; probe_e2x5Max_[i] = -99.; probe_e1x5Max_[i] = -99.;
		probe_hoe_[i] = -99.; probe_eop_[i] = -99.; probe_pin_[i] = -99.; probe_pout_[i] = -99.;

		Z_pt_[i] = -99.; Z_y_[i] = -99.; Z_eta_[i] = -99.; Z_phi_[i] = -99.; Z_m_[i] = -99.;
		Z_rescPt_[i] = -99.; Z_rescY_[i] = -99.; Z_rescEta_[i] = -99.; Z_rescPhi_[i] = -99.; Z_rescM_[i] = -99.; Z_probe_dPhi_[i] = -99.;

		ErsatzV1_Mesc_[i] = -99.; ErsatzV1_rescMesc_[i] = -99.;
		ErsatzV2_Mesc_[i] = -99.; ErsatzV2_rescMesc_[i] = -99.;
		ErsatzV3_Mesc_[i] = -99.; ErsatzV3_rescMesc_[i] = -99.;
		ErsatzV4_Mesc_[i] = -99.; ErsatzV4_rescMesc_[i] = -99.;
		ErsatzV1CaloMEt_[i] = -99.; ErsatzV1CaloMt_[i] = -99.; ErsatzV1CaloMEtPhi_[i] = -99.;
		ErsatzV2CaloMEt_[i] = -99.; ErsatzV2CaloMt_[i] = -99.; ErsatzV2CaloMEtPhi_[i] = -99.;
		ErsatzV3CaloMEt_[i] = -99.; ErsatzV3CaloMt_[i] = -99.; ErsatzV3CaloMEtPhi_[i] = -99.;
		ErsatzV4CaloMEt_[i] = -99.; ErsatzV4CaloMt_[i] = -99.; ErsatzV4CaloMEtPhi_[i] = -99.;
		ErsatzV1T1MEt_[i] = -99.; ErsatzV1T1Mt_[i] = -99.; ErsatzV1T1MEtPhi_[i] = -99.;
		ErsatzV1PfMEt_[i] = -99.; ErsatzV1PfMt_[i] = -99.; ErsatzV1PfMEtPhi_[i] = -99.;
		ErsatzV1TcMEt_[i] = -99.; ErsatzV1TcMt_[i] = -99.; ErsatzV1TcMEtPhi_[i] = -99.;

		probe_sc_pt_[i] = -99.; probe_sc_eta_[i] = -99.; probe_sc_phi_[i] = -99.;
		probe_sc_E_[i] = -99.; probe_sc_rawE_[i] = -99.;
		probe_scV2_E_[i] = -99.;
		probe_scV3_E_[i] = -99.;
		probe_scV4_E_[i] = -99.;

		if(Zevent_)
		{
			McElec_pt_[i] = -99.; McElec_eta_[i] = -99.; McElec_phi_[i] = -99.;
			McElec_rescPt_[i] = -99.; McElec_rescEta_[i] = -99.; McElec_rescPhi_[i] = -99.;
			McProbe_pt_[i] = -99.; McProbe_eta_[i] = -99.; McProbe_phi_[i] = -99.;
			McProbe_rescPt_[i] = -99.; McProbe_rescEta_[i] = -99.; McProbe_rescPhi_[i] = -99.;
			McElecProbe_dPhi_[i] = -99.; McElecProbe_dEta_[i] = -99.; McElecProbe_dR_[i] = -99.;
		}

		edm::LogDebug_("","",180)<<"Initialisation of array index "<< i <<" completed.";
	}
	//Get Collections
	edm::Handle<reco::GenParticleCollection> pGenPart;
		evt.getByToken(MCTruthCollection_, pGenPart);
	edm::Handle<reco::GsfElectronCollection> pElectrons;
		evt.getByToken(ElectronCollection_, pElectrons);
	edm::Handle<reco::SuperClusterCollection> pHybrid;
		evt.getByToken(HybridScCollection_, pHybrid);
	edm::Handle<reco::SuperClusterCollection> pM5x5;
		evt.getByToken(M5x5ScCollection_, pM5x5);
	edm::Handle<reco::CaloMETCollection> pCaloMEt;
		evt.getByToken(CaloMEtCollection_, pCaloMEt);
	edm::Handle<reco::METCollection> pT1MEt;
//		evt.getByToken(T1MEtCollection_, pT1MEt);
	edm::Handle<reco::PFMETCollection> pPfMEt;
		evt.getByToken(PfMEtCollection_, pPfMEt);
	edm::Handle<reco::METCollection> pTcMEt;
		evt.getByToken(TcMEtCollection_, pTcMEt);
	edm::Handle<reco::GenMETCollection> pGenMEt;
		evt.getByToken(GenMEtCollection_, pGenMEt);
	edm::Handle<edm::TriggerResults> pTriggerResults;
		evt.getByToken(TriggerResults_, pTriggerResults);
	edm::Handle<trigger::TriggerEvent> pHLT;
		evt.getByToken(TriggerEvent_, pHLT);

	std::vector<math::XYZTLorentzVector>McElecs,McElecsFinalState;
	std::vector<math::XYZTLorentzVector> McElecsResc;
	if(Zevent_)
	{
		edm::LogDebug_("","",289)<<"Analysing MC properties.";
		const reco::GenParticleCollection *McCand = pGenPart.product();
		math::XYZTLorentzVector Zboson, RescZboson, McElec1, McElec2;
		for(reco::GenParticleCollection::const_iterator McP = McCand->begin(); McP != McCand->end(); ++McP)
		{
			const reco::Candidate* mum = McP->mother();
			if(abs(McP->pdgId())==11 && abs(mum->pdgId()) == 23)
			{
				McElecs.push_back(McP->p4());
				if(abs(mum->pdgId() == 23)) Zboson = mum->p4();

				std::cout <<"Found electron, ID = "<< McP->pdgId() <<"\t status = "<< McP->status()<<std::endl;
				if(McP->status() != 1)
				{
					const reco::Candidate* McPD = McP->daughter(0);
					McPD = McPD->mother();
					while(McPD->status() != 1)
					{
						int n = McPD->numberOfDaughters();
						std::cout<< McPD->pdgId() << " : status = "<<McPD->status()
							<<"\tNumber of Daughters = "<< n <<std::endl;
						for(int j = 0; j < n; ++ j)
						{
							const reco::Candidate *d = McPD->daughter( j );
							std::cout <<"Daughter "<< j <<"\t id = "<< d->pdgId() << std::endl;
							if(abs(d->pdgId()) == 11)
							{
								McPD = d;
								break;
							}
						}
					}
					std::cout<< McPD->pdgId() << " : status = "<<McPD->status()<<"\tAdding to vector!"<<std::endl;
					McElecsFinalState.push_back(McPD->p4());
				}else McElecsFinalState.push_back(McP->p4());
			}
		}
		McZ_m_ = Zboson.M(); McZ_pt_ = Zboson.Pt(); McZ_phi_ = Zboson.Phi(); McZ_eta_ = Zboson.Eta(); McZ_y_ = Zboson.Rapidity();
		McElec_nZmum_ =McElecs.size();
		McElec_nFinal_ =McElecsFinalState.size();
		edm::LogDebug_("","",309)<<"MC electrons with Z mother = "<< McElec_nZmum_
						<<"\tFinal state MC electrons = "<< McElec_nFinal_;

		McElecsResc.resize(2);
//		RescZboson.SetCoordinates(Zboson.Px(), Zboson.Py(), Zboson.Pz(), sqrt(Zboson.P2()+(mW_*mW_*Zboson.M2())/(mZ_*mZ_)));
		RescZboson.SetCoordinates(Zboson.Px()*mW_/mZ_, Zboson.Py()*mW_/mZ_, Zboson.Pz()*mW_/mZ_, Zboson.E()*mW_/mZ_);
		McZ_rescM_ = RescZboson.M(); McZ_rescPt_ = RescZboson.Pt(); McZ_rescEta_ = RescZboson.Eta(); McZ_rescPhi_ = RescZboson.Phi();
		McZ_rescY_ = RescZboson.Rapidity();
		ROOT::Math::Boost CoMBoost(Zboson.BoostToCM());

		math::XYZTLorentzVector RescMcElec0 = CoMBoost(McElecsFinalState[0]);
		math::XYZTLorentzVector RescMcElec1 = CoMBoost(McElecsFinalState[1]);
		RescMcElec0 *= mW_/mZ_;
		RescMcElec1 *= mW_/mZ_;

		double E_W = RescZboson.E();
		ROOT::Math::Boost BackToLab(RescZboson.Px()/E_W, RescZboson.Py()/E_W, RescZboson.Pz()/E_W);

		RescMcElec0 = BackToLab(RescMcElec0);
//		RndmMcElec_Rescaled_pt_ = RescMcElec0.Pt();
//		RndmMcElec_Rescaled_eta_ = RescMcElec0.Eta();
//		RndmMcElec_Rescaled_phi_ = RescMcElec0.Phi();

		RescMcElec1 = BackToLab(RescMcElec1);
//		OthrMcElec_Rescaled_pt_ = RescMcElec1.Pt();
//		OthrMcElec_Rescaled_eta_ = RescMcElec1.Eta();
//		OthrMcElec_Rescaled_phi_ = RescMcElec1.Phi();
		McElecsResc[0] = RescMcElec0;
		McElecsResc[1] = RescMcElec1;
		math::XYZTLorentzVector sum = RescMcElec1+RescMcElec0;
		edm::LogDebug_("","", 307)<<"McElecsResc[0] + McElecsResc[1] = ("<<sum.Px()<<", "<<sum.Py()<<", "
						<<sum.Pz()<<", "<<sum.E()<<")";
	}

	const edm::TriggerResults* HltRes = pTriggerResults.product();
	const edm::TriggerNames & triggerNames = evt.triggerNames(*HltRes);
	if(HLTPathCheck_)
	{
		for(unsigned int itrig = 0; itrig < HltRes->size(); ++itrig)
		{
			std::string nom = triggerNames.triggerName(itrig);
			edm::LogInfo("")<< itrig <<" : Name = "<< nom <<"\t Accepted = "<< HltRes->accept(itrig);
		}
	}
	if(HltRes->accept(34) ==0) edm::LogError("")<<"Event did not pass "<< triggerNames.triggerName(34)<<"!";
	if(HltRes->accept(34) !=0)
	{
	std::vector<reco::GsfElectronRef> UniqueElectrons;
	UniqueElectrons = uniqueElectronFinder(pElectrons);
	edm::LogDebug_("","ErsatzMEt",192)<<"Unique electron size = "<<UniqueElectrons.size();
	std::vector<reco::GsfElectronRef> SelectedElectrons;
	const unsigned int fId = pHLT->filterIndex(TriggerPath_);
	std::cout << "Filter Id = " << fId << std::endl;
	SelectedElectrons = electronSelector(UniqueElectrons, pHLT, fId, CutVector_);
	nTags_ = SelectedElectrons.size();
	edm::LogDebug_("","ErsatzMEt",197)<<"Selected electron size = "<<nTags_;

	iComb_ = 0;
	if(Zevent_)
	{
	//Match MC electrons to the selected electrons and store some of their properties in the tree.
	//The properties of the other MC electron (i.e. that not selected) are also stored.
		for(std::vector<reco::GsfElectronRef>::const_iterator elec = SelectedElectrons.begin();
								elec != SelectedElectrons.end(); ++elec)
		{
			for(int m = 0; m < 2; ++m)
			{
				double dRLimit = 99.;
				double dR = reco::deltaR(McElecs[m], *(*elec));
				if(dR < dRLimit)
				{
					dRLimit = dR;
					McElec_pt_[iComb_] = McElecs[m].pt();
					McElec_eta_[iComb_] = McElecs[m].eta();
					McElec_rescPt_[iComb_] = McElecsResc[m].pt();
					McElec_rescEta_[iComb_] = McElecsResc[m].eta();
				}
			}
		}
	}

	std::map<reco::GsfElectronRef, reco::GsfElectronRef> TagProbePairs;
	TagProbePairs = probeFinder(SelectedElectrons, pElectrons);
	nProbes_ = TagProbePairs.size();
	edm::LogDebug_("", "ErsatzMEt", 209)<<"Number of tag-probe pairs = "<< TagProbePairs.size();

	if(!TagProbePairs.empty())
	{
		const reco::CaloMETCollection* caloMEtCollection = pCaloMEt.product();
        	const reco::MET calomet = *(caloMEtCollection->begin());
		CaloMEt_ = calomet.pt();
		CaloMEtphi_ = calomet.phi();

		//const reco::METCollection* t1MEtCollection = pT1MEt.product();
        	//const reco::MET t1met = *(t1MEtCollection->begin());
		//T1MEt_ = t1met.pt();
		//T1MEtphi_ = t1met.phi();

		const reco::PFMETCollection* pfMEtCollection = pPfMEt.product();
        	const reco::MET pfmet = *(pfMEtCollection->begin());
		PfMEt_ = pfmet.pt();
		PfMEtphi_ = pfmet.phi();

		const reco::METCollection* tcMEtCollection = pTcMEt.product();
        	const reco::MET tcmet = *(tcMEtCollection->begin());
		TcMEt_ = tcmet.pt();
		TcMEtphi_ = tcmet.phi();

		reco::MET ersatzMEt;

		for(std::map<reco::GsfElectronRef, reco::GsfElectronRef>::const_iterator it = TagProbePairs.begin();
			it != TagProbePairs.end(); ++it)
		{
			edm::LogDebug_("","DelendumLoop", 293)<<"iComb_ = "<< iComb_;
			tag_q_[iComb_] = it->first->charge();
			edm::LogDebug_("","",360)<<"tag charge = "<< tag_q_[iComb_];
			tag_pt_[iComb_] = it->first->pt();
			tag_eta_[iComb_] = it->first->eta();
			tag_phi_[iComb_] = it->first->phi();
			edm::LogDebug_("","ErsatzMEt", 364)<<"tag pt = "<< tag_pt_[iComb_]
					<<"\teta = "<< tag_eta_[iComb_]<<"\tphi = "<< tag_phi_[iComb_];
                        tag_trckIso_[iComb_] = it->first->isolationVariables03().tkSumPt;
                        tag_ecalIso_[iComb_] = it->first->isolationVariables04().ecalRecHitSumEt;
                        tag_hcalIso_[iComb_] = it->first->isolationVariables04().hcalDepth1TowerSumEt
						+ it->first->isolationVariables04().hcalDepth2TowerSumEt;
			edm::LogDebug_("","ErsatzMEt", 370)<<"tag trackiso = "<< tag_trckIso_[iComb_]
					<<"\tecaliso = "<< tag_ecalIso_[iComb_]<<"\thcaliso = "<< tag_hcalIso_[iComb_];
			tag_sIhIh_[iComb_] = it->first->scSigmaIEtaIEta();
			tag_dPhiIn_[iComb_] = it->first->deltaPhiSuperClusterTrackAtVtx();
			tag_dEtaIn_[iComb_] = it->first->deltaEtaSuperClusterTrackAtVtx();
			edm::LogDebug_("","ErsatzMEt", 245)<<"tag sIhIh = "<< tag_sIhIh_[iComb_]
					<<"\tdPhiIn = "<< tag_dPhiIn_[iComb_]<<"\tdEtaIn = "<< tag_dEtaIn_[iComb_];
			tag_e5x5_[iComb_] = it->first->scE5x5();
			tag_e2x5Max_[iComb_] = it->first->scE2x5Max();
			tag_e2x5Max_[iComb_] = it->first->scE1x5();
			edm::LogDebug_("","ErsatzMEt", 245)<<"tag e5x5 = "<< tag_e5x5_[iComb_]
					<<"\te2x5Max = "<< tag_e2x5Max_[iComb_]<<"\te1x5Max = "<< tag_e1x5Max_[iComb_];
			tag_hoe_[iComb_] = it->first->hadronicOverEm();
			tag_eop_[iComb_] = it->first->eSuperClusterOverP();
			tag_pin_[iComb_] = it->first->trackMomentumAtVtx().R();
			tag_pout_[iComb_] = it->first->trackMomentumOut().R();
			edm::LogDebug_("","ErsatzMEt", 245)<<"tag hoe = "<<tag_hoe_[iComb_]<<"\tpoe = "<<tag_eop_[iComb_]
					<<"\tpin = "<< tag_pin_[iComb_]<<"\tpout = "<< tag_pout_[iComb_];
			probe_q_[iComb_] = it->first->charge();
			edm::LogDebug_("","",360)<<"probe charge = "<< probe_q_[iComb_];
			probe_pt_[iComb_] = it->second->pt();
			probe_eta_[iComb_] = it->second->eta();
			probe_phi_[iComb_] = it->second->phi();
			edm::LogDebug_("","ErsatzMEt", 245)<<"probe pt = "<< probe_pt_[iComb_]
					<<"\teta = "<< probe_eta_[iComb_]<<"\tphi = "<< probe_phi_[iComb_];
                        probe_trckIso_[iComb_] = it->second->isolationVariables03().tkSumPt;
                        probe_ecalIso_[iComb_] = it->second->isolationVariables04().ecalRecHitSumEt;
                        probe_hcalIso_[iComb_] = it->second->isolationVariables04().hcalDepth1TowerSumEt
						+ it->second->isolationVariables04().hcalDepth2TowerSumEt;
			edm::LogDebug_("","ErsatzMEt", 245)<<"probe trackiso = "<< probe_trckIso_[iComb_]
					<<"\tecaliso = "<< probe_ecalIso_[iComb_]<<"\thcaliso = "<< probe_phi_[iComb_];
			probe_sIhIh_[iComb_] = it->second->scSigmaIEtaIEta();
			probe_dPhiIn_[iComb_] = it->second->deltaPhiSuperClusterTrackAtVtx();
			probe_dEtaIn_[iComb_] = it->second->deltaEtaSuperClusterTrackAtVtx();
			edm::LogDebug_("","ErsatzMEt", 245)<<"probe sIhIh = "<< probe_sIhIh_[iComb_]
					<<"\tdPhiIn = "<< probe_dPhiIn_[iComb_]<<"\tdEtaIn = "<< probe_dEtaIn_[iComb_];
			probe_e5x5_[iComb_] = it->second->scE5x5();
			probe_e2x5Max_[iComb_] = it->second->scE2x5Max();
			probe_e2x5Max_[iComb_] = it->second->scE1x5();
			edm::LogDebug_("","ErsatzMEt", 245)<<"probe e5x5 = "<< probe_e5x5_[iComb_]
					<<"\te2x5Max = "<< probe_e2x5Max_[iComb_]<<"\te1x5Max = "<< probe_e1x5Max_[iComb_];
			probe_hoe_[iComb_] = it->second->hadronicOverEm();
			probe_eop_[iComb_] = it->second->eSuperClusterOverP();
			probe_pin_[iComb_] = it->second->trackMomentumAtVtx().R();
			probe_pout_[iComb_] = it->second->trackMomentumOut().R();
			edm::LogDebug_("","ErsatzMEt", 245)<<"probe hoe = "<<probe_hoe_[iComb_]<<"\tpoe = "<<probe_eop_[iComb_]
					<<"\tpin = "<< probe_pin_[iComb_]<<"\tpout = "<< probe_pout_[iComb_];

			double dRLimit = 0.2;
			for(unsigned int mcEId = 0; mcEId < McElecs.size(); ++mcEId)
			{
//				double dR = reco::deltaR((*(*mcEl)), probeVec);
				double dR = reco::deltaR(McElecs[mcEId], it->second->p4());
				if(dR < dRLimit)
				{
					dRLimit = dR;
					McProbe_pt_[iComb_] = McElecs[mcEId].pt();
					McProbe_eta_[iComb_] = McElecs[mcEId].eta();
					McProbe_phi_[iComb_] = McElecs[mcEId].phi();
					McProbe_rescPt_[iComb_] = McElecsResc[mcEId].pt();
					McProbe_rescEta_[iComb_] = McElecsResc[mcEId].eta();
					McProbe_rescPhi_[iComb_] = McElecsResc[mcEId].phi();
					probe_d_MCE_SCE_[iComb_] = McElecs[mcEId].energy() - it->second->superCluster()->rawEnergy();
					McElecProbe_dPhi_[iComb_] = reco::deltaPhi(McElecs[mcEId].phi(), McElecs[(mcEId+1)%2].phi());
					McElecProbe_dEta_[iComb_] = fabs(McElecs[mcEId].eta() - McElecs[(mcEId+1)%2].eta());
					McElecProbe_dR_[iComb_] = reco::deltaR(McElecs[mcEId], McElecs[(mcEId+1)%2]);
				}
			}

			// Uncorrected supercluster V1
			reco::SuperCluster scV1 = *(it->second->superCluster());
			math::XYZTLorentzVector	probe_scV1_detVec = DetectorVector(scV1);
			probe_sc_pt_[iComb_] = probe_scV1_detVec.pt();
			probe_sc_eta_[iComb_] = scV1.eta();
			probe_sc_phi_[iComb_] = scV1.phi();
			probe_sc_nClus_[iComb_] = scV1.clustersSize();
			probe_sc_E_[iComb_] = scV1.energy();
			probe_sc_rawE_[iComb_] = scV1.rawEnergy();

			ersatzMEt = ersatzFabrik(it->first, scV1, calomet, 1);
			ErsatzV1CaloMEt_[iComb_] = ersatzMEt.pt();
			ErsatzV1CaloMEtPhi_[iComb_] = ersatzMEt.phi();
			//ersatzMEt = ersatzFabrik(it->first, it->second, t1met);
			//ErsatzV1T1MEt_[iComb_] = ersatzMEt.pt();
			//ErsatzV1T1MEtPhi_[iComb_] = ersatzMEt.phi();
			ersatzMEt = ersatzFabrik(it->first, it->second, pfmet);
			ErsatzV1PfMEt_[iComb_] = ersatzMEt.pt();
			ErsatzV1PfMEtPhi_[iComb_] = ersatzMEt.phi();
			ersatzMEt = ersatzFabrik(it->first, it->second, tcmet);
			ErsatzV1TcMEt_[iComb_] = ersatzMEt.pt();
			ErsatzV1TcMEtPhi_[iComb_] = ersatzMEt.phi();

			// fEta corrected supercluster V2
			reco::SuperCluster scV2;
			if(fabs(probe_sc_eta_[iComb_]) < 1.479)
			{
				scV2 = fEtaScCorr(scV1);
			}else{
				scV2 = scV1;
			}
			probe_scV2_E_[iComb_] = scV2.energy();
			ersatzMEt = ersatzFabrik(it->first, scV2, calomet, 2);
			ErsatzV2CaloMEt_[iComb_] = ersatzMEt.pt();
			ErsatzV2CaloMEtPhi_[iComb_] = ersatzMEt.phi();

			// fBrem corrected supercluster V3
			reco::SuperCluster scV3;
			if(fabs(probe_sc_eta_[iComb_]) < 1.479)
			{
				scV3 = fBremScCorr(scV1, hyb_fCorrPSet_);
			}else{
				scV3 = fBremScCorr(scV1, m5x5_fCorrPSet_);
			}
			probe_scV3_E_[iComb_] = scV3.energy();
			ersatzMEt = ersatzFabrik(it->first, scV3, calomet, 3);
			ErsatzV3CaloMEt_[iComb_] = ersatzMEt.pt();
			ErsatzV3CaloMEtPhi_[iComb_] = ersatzMEt.phi();

			// Fully corrected supercluster V4
			reco::SuperCluster scV4;
			if(fabs(probe_sc_eta_[iComb_]) < 1.479)
			{
				scV4 = fBremScCorr(scV1, hyb_fCorrPSet_);
			}else{
				scV4 = fBremScCorr(scV1, m5x5_fCorrPSet_);
			}
			probe_scV4_E_[iComb_] = scV4.energy();
			ersatzMEt = ersatzFabrik(it->first, scV4, calomet, 4);
			ErsatzV4CaloMEt_[iComb_] = ersatzMEt.pt();
			ErsatzV4CaloMEtPhi_[iComb_] = ersatzMEt.phi();

			++iComb_;
		}
		t_->Fill();
	}
	}
}

std::map<reco::GsfElectronRef, reco::GsfElectronRef> ErsatzMEt::probeFinder(const std::vector<reco::GsfElectronRef>& tags,
							const edm::Handle<reco::GsfElectronCollection> pElectrons)
{
	const reco::GsfElectronCollection *probeCands = pElectrons.product();
	std::map<reco::GsfElectronRef, reco::GsfElectronRef> TagProbes;
	for(std::vector<reco::GsfElectronRef>::const_iterator tagelec = tags.begin(); tagelec != tags.end(); ++tagelec)
	{
		reco::GsfElectronRef tag = *tagelec;
		std::pair<reco::GsfElectronRef, reco::GsfElectronRef> TagProbePair;
		int nProbesPerTag = 0;
		int index = 0;
		for(reco::GsfElectronCollection::const_iterator probeelec = probeCands->begin(); probeelec != probeCands->end(); ++probeelec)
		{
			reco::GsfElectronRef probe(pElectrons, index);
			double probeScEta = probe->superCluster()->eta();
			if(probe->superCluster() != tag->superCluster() && fabs(probeScEta) < 2.5)
			{
				if(fabs(probeScEta) < 1.4442 || fabs(probeScEta) > 1.560)
				{
					double invmass = ROOT::Math::VectorUtil::InvariantMass(tag->p4(), probe->p4());
					if(mTPmin_ <= invmass && invmass <= mTPmax_)
					{
						TagProbePair = std::make_pair(tag, probe);
						++nProbesPerTag;
					}
				}
			}
			++index;
		}
		//nGsfElectrons_ = index;
		if(nProbesPerTag == 1) TagProbes.insert(TagProbePair);
	}
	return TagProbes;
}

reco::MET ErsatzMEt::ersatzFabrik(const reco::GsfElectronRef& elec,
					const reco::SuperCluster& sc,
					const reco::MET& met,
					const int corr)
{
	const math::XYZPoint ZVertex(elec->TrackPositionAtVtx().X(), elec->TrackPositionAtVtx().Y(),elec->TrackPositionAtVtx().Z());

	math::XYZTLorentzVector nu, boost_nu, ele, boost_ele;
	reco::SuperCluster elecSc = *(elec->superCluster());
	nu = PhysicsVectorRaw(met.vertex(), sc);
	boost_nu = PhysicsVectorRaw(ZVertex, sc);
	ele = PhysicsVectorRaw(met.vertex(), elecSc);
	boost_ele = ele;

	//Should use reco vertex for best Z->ee measurement.
        edm::LogDebug_("ersatzFabrikV1", "", 569)<<"elec  = ("<< elec->p4().Px() << ", "<< elec->p4().Py()<< ", "<< elec->p4().Pz() << ", "<< elec->p4().E()<<")";
	math::XYZTLorentzVector Zboson = boost_nu + elec->p4();
        edm::LogDebug_("ersatzFabrikV1", "", 569)<<"Z pt = "<< Zboson.Pt() << "Z boson mass = " << Zboson.M();
        edm::LogDebug_("ersatzFabrikV1","", 570)<<"Z boson in lab frame = ("<<Zboson.Px()<<", "<<Zboson.Py()<<", "
                                        <<Zboson.Pz()<<", "<<Zboson.E()<<")";
        math::XYZTLorentzVector RescZboson(Zboson.Px(), Zboson.Py(), Zboson.Pz(), sqrt(Zboson.P2()+(mW_*mW_*Zboson.M2())/(mZ_*mZ_)));
        edm::LogDebug_("ersatzFabrikV1","", 573)<<"W boson in lab frame = ("<<RescZboson.Px()<<", "<<RescZboson.Py()<<", "
                                        <<RescZboson.Pz()<<", "<<RescZboson.E()<<")";
        ROOT::Math::Boost BoostToZRestFrame(Zboson.BoostToCM());
        edm::LogDebug_("ersatzFabrikV1","", 576)<<"Electron in lab frame = ("<< boost_ele.Px()<<", "<< boost_ele.Py()<<", "
                                        << boost_ele.Pz()<<", "<< boost_ele.E()<<")";
        edm::LogDebug_("ersatzFabrikV1","", 578)<<"Ersatz Neutrino in lab frame = ("<< boost_nu.Px()<<", "<< boost_nu.Py()<<", "
                                        << boost_nu.Pz()<<", "<< boost_nu.E()<<")";
        boost_ele = BoostToZRestFrame(boost_ele);
        boost_nu = BoostToZRestFrame(boost_nu);
        edm::LogDebug_("ersatzFabrikV1","", 582)<<"Electron in Z rest frame = ("<<boost_ele.Px()<<", "<<boost_ele.Py()<<", "
                                        <<boost_ele.Pz()<<", "<<boost_ele.E()<<")";
        edm::LogDebug_("ersatzFabrikV1","", 584)<<"Ersatz Neutrino in Z rest frame = ("<<boost_nu.Px()<<", "<<boost_nu.Py()<<", "
                                        <<boost_nu.Pz()<<", "<<boost_nu.E()<<")";
        boost_ele *= mW_/mZ_;
        boost_nu *= mW_/mZ_;

        double E_W = RescZboson.E();
        ROOT::Math::Boost BackToLab(RescZboson.Px()/E_W, RescZboson.Py()/E_W, RescZboson.Pz()/E_W);
	math::XYZTLorentzVector metVec(-99999., -99999., -99., -99999.);
        boost_ele = BackToLab(boost_ele);

       	boost_nu = BackToLab(boost_nu);
        math::XYZTLorentzVector sum = boost_nu+boost_ele;
       	edm::LogDebug_("ersatzFabrikV1","", 597)<<"Electron back in lab frame = ("<<boost_ele.Px()<<", "<<boost_ele.Py()<<", "
                                        <<boost_ele.Pz()<<", "<<boost_ele.E()<<")";
        edm::LogDebug_("ersatzFabrikV1","", 599)<<"Ersatz Neutrino back in lab frame = ("<<boost_nu.Px()<<", "<<boost_nu.Py()<<", "
                                       <<boost_nu.Pz()<<", "<<boost_nu.E()<<")";
       	edm::LogDebug_("ersatzFabrikV1","", 601)<<"boost_ele + boost_nu = ("<<sum.Px()<<", "<<sum.Py()<<", "
                                        <<sum.Pz()<<", "<<sum.E()<<")";

	nu.SetXYZT(nu.X(), nu.Y(), 0., nu.T());
	ele.SetXYZT(ele.X(), ele.Y(), 0., ele.T());
	boost_ele.SetXYZT(boost_ele.X(), boost_ele.Y(), 0., boost_ele.T());
	metVec = met.p4() + nu + ele - boost_ele;

	reco::MET ersatzMEt(metVec, met.vertex());
	if (corr == 1)
	{
		//Z_caloV1_m_[iComb_] = Zboson.M();
		//Z_caloV1_pt_[iComb_] = Zboson.Pt();
		//Z_caloV1_y_[iComb_] = Zboson.Y();
		//Z_caloV1_eta_[iComb_] = Zboson.Eta();
		//Z_caloV1_phi_[iComb_] = Zboson.Phi();
		//Z_caloV1_rescM_[iComb_] = RescZboson.M();
		//Z_caloV1_rescPt_[iComb_] = RescZboson.Pt();
		//Z_caloV1_rescY_[iComb_] = RescZboson.Y();
		//Z_caloV1_rescEta_[iComb_] = RescZboson.Eta();
		//Z_caloV1_rescPhi_[iComb_] = RescZboson.Phi();
		//Z_caloV1_probe_dPhi_[iComb_] = reco::deltaPhi(Zboson.Phi(), elec->phi());
		//tag_caloV1_rescPt_[iComb_]  = boost_ele.Pt();
		//tag_caloV1_rescEta_[iComb_]  = boost_ele.Eta();
		//tag_caloV1_rescPhi_[iComb_]  = boost_ele.Phi();
		//probe_caloV1_rescPt_[iComb_]  = boost_nu.Pt();
		//probe_caloV1_rescEta_[iComb_]  = boost_nu.Eta();
		//probe_caloV1_rescPhi_[iComb_]  = boost_nu.Phi();
		ErsatzV1_Mesc_[iComb_]  = ROOT::Math::VectorUtil::InvariantMass(elec->p4(), boost_nu);
		ErsatzV1_rescMesc_[iComb_] = ROOT::Math::VectorUtil::InvariantMass(ele, nu);
		ErsatzV1CaloMt_[iComb_] = sqrt(2.*boost_ele.Pt()*ersatzMEt.pt()*
						(1-cos(reco::deltaPhi(boost_ele.Phi(), ersatzMEt.phi()))));
	}
	if (corr == 2)
	{
		//Z_caloV2_m_[iComb_] = Zboson.M();
		//Z_caloV2_pt_[iComb_] = Zboson.Pt();
		//Z_caloV2_y_[iComb_] = Zboson.Y();
		//Z_caloV2_eta_[iComb_] = Zboson.Eta();
		//Z_caloV2_phi_[iComb_] = Zboson.Phi();
		//Z_caloV2_rescM_[iComb_] = RescZboson.M();
		//Z_caloV2_rescPt_[iComb_] = RescZboson.Pt();
		//Z_caloV2_rescY_[iComb_] = RescZboson.Y();
		//Z_caloV2_rescEta_[iComb_] = RescZboson.Eta();
		//Z_caloV2_rescPhi_[iComb_] = RescZboson.Phi();
		//Z_caloV2_probe_dPhi_[iComb_] = reco::deltaPhi(Zboson.Phi(), boost_elec->phi());
		//tag_caloV2_rescPt_[iComb_]  = boost_ele.Pt();
		//tag_caloV2_rescEta_[iComb_]  = boost_ele.Eta();
		//tag_caloV2_rescPhi_[iComb_]  = boost_ele.Phi();
		//probe_caloV2_rescPt_[iComb_]  = boost_nu.Pt();
		//probe_caloV2_rescEta_[iComb_]  = boost_nu.Eta();
		//probe_caloV2_rescPhi_[iComb_]  = boost_nu.Phi();
		ErsatzV2_Mesc_[iComb_]  = ROOT::Math::VectorUtil::InvariantMass(elec->p4(), boost_nu);
		ErsatzV2_rescMesc_[iComb_] = ROOT::Math::VectorUtil::InvariantMass(ele, nu);
		ErsatzV2CaloMt_[iComb_] = sqrt(2.*boost_ele.Pt()*ersatzMEt.pt()*
						(1-cos(reco::deltaPhi(boost_ele.Phi(), ersatzMEt.phi()))));
	}
	if (corr == 3)
	{
		//Z_caloV3_m_[iComb_] = Zboson.M();
		//Z_caloV3_pt_[iComb_] = Zboson.Pt();
		//Z_caloV3_y_[iComb_] = Zboson.Y();
		//Z_caloV3_eta_[iComb_] = Zboson.Eta();
		//Z_caloV3_phi_[iComb_] = Zboson.Phi();
		//Z_caloV3_rescM_[iComb_] = RescZboson.M();
		//Z_caloV3_rescPt_[iComb_] = RescZboson.Pt();
		//Z_caloV3_rescY_[iComb_] = RescZboson.Y();
		//Z_caloV3_rescEta_[iComb_] = RescZboson.Eta();
		//Z_caloV3_rescPhi_[iComb_] = RescZboson.Phi();
		//Z_caloV3_probe_dPhi_[iComb_] = reco::deltaPhi(Zboson.Phi(), boost_elec->phi());
		//tag_caloV3_rescPt_[iComb_]  = boost_ele.Pt();
		//tag_caloV3_rescEta_[iComb_]  = boost_ele.Eta();
		//tag_caloV3_rescPhi_[iComb_]  = boost_ele.Phi();
		//probe_caloV3_rescPt_[iComb_]  = boost_nu.Pt();
		//probe_caloV3_rescEta_[iComb_]  = boost_nu.Eta();
		//probe_caloV3_rescPhi_[iComb_]  = boost_nu.Phi();
		ErsatzV3_Mesc_[iComb_]  = ROOT::Math::VectorUtil::InvariantMass(elec->p4(), boost_nu);
		ErsatzV3_rescMesc_[iComb_] = ROOT::Math::VectorUtil::InvariantMass(ele, nu);
		ErsatzV3CaloMt_[iComb_] = sqrt(2.*boost_ele.Pt()*ersatzMEt.pt()*
						(1-cos(reco::deltaPhi(boost_ele.Phi(), ersatzMEt.phi()))));
	}
	if (corr == 4)
	{
		//Z_caloV4_m_[iComb_] = Zboson.M();
		//Z_caloV4_pt_[iComb_] = Zboson.Pt();
		//Z_caloV4_y_[iComb_] = Zboson.Y();
		//Z_caloV4_eta_[iComb_] = Zboson.Eta();
		//Z_caloV4_phi_[iComb_] = Zboson.Phi();
		//Z_caloV4_rescM_[iComb_] = RescZboson.M();
		//Z_caloV4_rescPt_[iComb_] = RescZboson.Pt();
		//Z_caloV4_rescY_[iComb_] = RescZboson.Y();
		//Z_caloV4_rescEta_[iComb_] = RescZboson.Eta();
		//Z_caloV4_rescPhi_[iComb_] = RescZboson.Phi();
		//Z_caloV4_probe_dPhi_[iComb_] = reco::deltaPhi(Zboson.Phi(), boost_elec->phi());
		//tag_caloV4_rescPt_[iComb_]  = boost_ele.Pt();
		//tag_caloV4_rescEta_[iComb_]  = boost_ele.Eta();
		//tag_caloV4_rescPhi_[iComb_]  = boost_ele.Phi();
		//probe_caloV4_rescPt_[iComb_]  = boost_nu.Pt();
		//probe_caloV4_rescEta_[iComb_]  = boost_nu.Eta();
		//probe_caloV4_rescPhi_[iComb_]  = boost_nu.Phi();
		ErsatzV4_Mesc_[iComb_]  = ROOT::Math::VectorUtil::InvariantMass(elec->p4(), boost_nu);
		ErsatzV4_rescMesc_[iComb_] = ROOT::Math::VectorUtil::InvariantMass(ele, nu);
		ErsatzV4CaloMt_[iComb_] = sqrt(2.*boost_ele.Pt()*ersatzMEt.pt()*
						(1-cos(reco::deltaPhi(boost_ele.Phi(), ersatzMEt.phi()))));
	}
	return ersatzMEt;
}

reco::MET ErsatzMEt::ersatzFabrik(const reco::GsfElectronRef& tag,
					const reco::GsfElectronRef& probe,
					const reco::MET& met)
{
	math::XYZTLorentzVector elec, nu, boost_elec, boost_nu;
	boost_elec = tag->p4();
        edm::LogDebug_("ersatzFabrikV1", "", 858)<<"boost_elec  = ("<< boost_elec.Px() << ", "<< boost_elec.Py()<< ", "<< boost_elec.Pz() << ", "<< boost_elec.E()<<")";
	boost_nu = probe->p4();
        edm::LogDebug_("ersatzFabrikV1", "", 860)<<"boost_nu  = ("<< boost_nu.Px() << ", "<< boost_nu.Py()<< ", "<< boost_nu.Pz() << ", "<< boost_nu.E()<<")";
	math::XYZTLorentzVector Zboson = boost_elec + boost_nu;
        edm::LogDebug_("ersatzFabrikV1", "", 862)<<"Zboson  = ("<< Zboson.Px() << ", "<< Zboson.Py()<< ", "<< Zboson.Pz() << ", "<< Zboson.E()<<")";
        math::XYZTLorentzVector RescZboson(Zboson.Px(), Zboson.Py(), Zboson.Pz(), sqrt(Zboson.P2()+(mW_*mW_*Zboson.M2())/(mZ_*mZ_)));
        edm::LogDebug_("ersatzFabrikV1", "", 864)<<"RescZboson  = ("<< RescZboson.Px() << ", "<< RescZboson.Py()<< ", "<< RescZboson.Pz() << ", "<< RescZboson.E()<<")";
        ROOT::Math::Boost BoostToZRestFrame(Zboson.BoostToCM());
        elec = BoostToZRestFrame(boost_elec);
        edm::LogDebug_("ersatzFabrikV1", "", 867)<<"boost_elec (in Z rest frame) = ("<< elec.Px() << ", "<< elec.Py()<< ", "<< elec.Pz() << ", "<< elec.E()<<")";
        nu = BoostToZRestFrame(boost_nu);
        edm::LogDebug_("ersatzFabrikV1", "", 869)<<"boost_nu (in Z rest frame) = ("<< nu.Px() << ", "<< nu.Py()<< ", "<< nu.Pz() << ", "<< nu.E()<<")";
        elec *= mW_/mZ_;
        edm::LogDebug_("ersatzFabrikV1", "", 871)<<"elec (in Z rest frame) = ("<< elec.Px() << ", "<< elec.Py()<< ", "<< elec.Pz() << ", "<< elec.E()<<")";
        nu *= mW_/mZ_;
        edm::LogDebug_("ersatzFabrikV1", "", 873)<<"nu (in Z rest frame) = ("<< nu.Px() << ", "<< nu.Py()<< ", "<< nu.Pz() << ", "<< nu.E()<<")";
        ROOT::Math::Boost BoostBackToLab(RescZboson.Px()/RescZboson.E(), RescZboson.Py()/RescZboson.E(), RescZboson.Pz()/RescZboson.E());
	math::XYZTLorentzVector metVec(-99999., -99999., -99., -99999.);
        elec = BoostBackToLab(elec);
        edm::LogDebug_("ersatzFabrikV1", "", 877)<<"elec = ("<< elec.Px() << ", "<< elec.Py()<< ", "<< elec.Pz() << ", "<< elec.E()<<")";
       	nu = BoostBackToLab(nu);
        edm::LogDebug_("ersatzFabrikV1", "", 879)<<"nu = ("<< nu.Px() << ", "<< nu.Py()<< ", "<< nu.Pz() << ", "<< nu.E()<<")";
	Z_m_[iComb_] = Zboson.M();
	Z_pt_[iComb_] = Zboson.Pt();
	Z_y_[iComb_] = Zboson.Y();
	Z_eta_[iComb_] = Zboson.Eta();
	Z_phi_[iComb_] = Zboson.Phi();
	Z_rescM_[iComb_] = RescZboson.M();
	Z_rescPt_[iComb_] = RescZboson.Pt();
	Z_rescY_[iComb_] = RescZboson.Y();
	Z_rescEta_[iComb_] = RescZboson.Eta();
	Z_rescPhi_[iComb_] = RescZboson.Phi();
	Z_probe_dPhi_[iComb_] = reco::deltaPhi(Zboson.Phi(), boost_elec.phi());
	tag_rescPt_[iComb_]  = elec.Pt();
	tag_rescEta_[iComb_]  = elec.Eta();
	tag_rescPhi_[iComb_]  = elec.Phi();
	probe_rescPt_[iComb_]  = nu.Pt();
	probe_rescEta_[iComb_]  = nu.Eta();
	probe_rescPhi_[iComb_]  = nu.Phi();
	elec.SetXYZT(elec.X(), elec.Y(), 0., elec.T());
	nu.SetXYZT(nu.X(), nu.Y(), 0., nu.T());
	boost_elec.SetXYZT(boost_elec.X(), boost_elec.Y(), 0., boost_elec.T());
	metVec = met.p4() + nu + elec - boost_elec;
	reco::MET ersatzMEt(metVec, met.vertex());
	return ersatzMEt;
}

bool ErsatzMEt::isInBarrel(double eta)
{
	return (fabs(eta) < BarrelEtaMax_);
}

bool ErsatzMEt::isInEndCap(double eta)
{
	return (fabs(eta) < EndCapEtaMax_ && fabs(eta) > EndCapEtaMin_);
}

bool ErsatzMEt::isInFiducial(double eta)
{
	return isInBarrel(eta) || isInEndCap(eta);
}

// ------------ method called once each job just after ending the event loop  ------------
void ErsatzMEt::endJob() {
}
//define this as a plug-in
DEFINE_FWK_MODULE(ErsatzMEt);

