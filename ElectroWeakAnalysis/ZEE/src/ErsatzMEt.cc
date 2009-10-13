#include "ElectroWeakAnalysis/ZEE/interface/ErsatzMEt.h"

ErsatzMEt::ErsatzMEt(const edm::ParameterSet& ps)
{
	MCTruthCollection_  = ps.getParameter<edm::InputTag>("MCTruthCollection");
	ElectronCollection_ = ps.getParameter<edm::InputTag>("ElectronCollection");
	HybridScCollection_ = ps.getParameter<edm::InputTag>("HybridScCollection");
	M5x5ScCollection_ = ps.getParameter<edm::InputTag>("M5x5ScCollection");
	EBRecHitCollection_ = ps.getParameter<edm::InputTag>("EBRecHitCollection");
	EERecHitCollection_ = ps.getParameter<edm::InputTag>("EERecHitCollection");
	eIsoTrack_ = ps.getParameter<edm::InputTag>("eIsoTrack");
	eIsoEcal_ = ps.getParameter<edm::InputTag>("eIsoEcal");
	eIsoHcal_ = ps.getParameter<edm::InputTag>("eIsoHcal");
        TrackCollection_ = ps.getParameter<edm::InputTag>("TrackCollection");
	CaloMEtCollection_ = ps.getParameter<edm::InputTag>("CaloMEtCollection");
	CaloTowerCollection_ = ps.getParameter<edm::InputTag>("CaloTowerCollection");
	TriggerEvent_ = ps.getParameter<edm::InputTag>("TriggerEvent");
	TriggerPath_ = ps.getParameter<edm::InputTag>("TriggerPath");
	TriggerResults_ = ps.getParameter<edm::InputTag>("TriggerResults");
	TriggerName_ = ps.getParameter<std::string>("TriggerName");
	Zevent_ = ps.getParameter<bool>("Zevent");
	mW_ = ps.getParameter<double>("mW");
        mZ_ = ps.getParameter<double>("mZ");
	mTPmin_ = ps.getParameter<double>("mTPmin");
	mTPmax_ = ps.getParameter<double>("mTPmax");
	//SC Correction Parameters
	sigmaElectronicNoise_EB_ = ps.getParameter<double>("sigmaElectronicNoise_EB");
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
	etaWidth_ = ps.getParameter<int>("etaWidth");
	phiWidth_ = ps.getParameter<int>("phiWidth");
	HLTPathCheck_ = ps.getParameter<bool>("HLTPathCheck");

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
/*
	CutVector_[0] = CElecPtMin;
	CutVector_[1] = CEB_siEiE;
	CutVector_[2] = CEB_dPhiIn;
	CutVector_[3] = CEB_dEtaIn;
	CutVector_[4] = CEB_TrckIso;
	CutVector_[5] = CEB_EcalIso;
	CutVector_[6] = CEB_HcalIso;
	CutVector_[7] = CEE_siEiE;
	CutVector_[8] = CEE_dPhiIn;
	CutVector_[9] = CEE_dEtaIn;
	CutVector_[10] = CEE_TrckIso;
	CutVector_[11] = CEE_EcalIso;
	CutVector_[12] = CEE_HcalIso;
*/
	for(std::vector<double>::const_iterator it = CutVector_.begin(); it != CutVector_.end(); ++it)
	{
		edm::LogDebug_("","",101)<<"CutVector_ = "<< *it;
	}
}

ErsatzMEt::~ErsatzMEt()
{
}

// ------------ method called once each job just before starting event loop  ------------
void ErsatzMEt::beginJob(const edm::EventSetup& setup)
{
	edm::Service<TFileService> fs;

	t_ = fs->make<TTree>("ErsatzMEt", "Data on ErsatzMEt");
	
	setup.get<CaloGeometryRecord>().get(geoHandle_);
	setup.get<CaloTopologyRecord>().get(pTopology_);
	
	edm::LogDebug_("","", 75)<<"Creating Ersatz MEt branches.";
	t_->Branch("nTags", &nTags_, "nTags/I");
	t_->Branch("nProbes", &nProbes_, "nProbes/I");
	t_->Branch("nRecHitsInStrip", nRecHitsInStrip_, "nRecHitsInStrip[4]/I");
	t_->Branch("nRecHitsInCone", nRecHitsInCone_, "nRecHitsInCone[4]/I");
	t_->Branch("ErsatzV1CaloMEt", ErsatzV1CaloMEt_, "ErsatzV1CaloMEt[4]/D");
	t_->Branch("ErsatzV1CaloMt", ErsatzV1CaloMt_, "ErsatzV1CaloMt[4]/D");
	t_->Branch("ErsatzV1CaloMEtPhi", ErsatzV1CaloMEtPhi_, "ErsatzV1CaloMEtPhi[4]/D");
	t_->Branch("ErsatzV1aCaloMEt", ErsatzV1aCaloMEt_, "ErsatzV1aCaloMEt[4]/D");
	t_->Branch("ErsatzV1aCaloMEtPhi", ErsatzV1aCaloMEtPhi_, "ErsatzV1aCaloMEtPhi[4]/D");
	t_->Branch("ErsatzV1aCaloMt", ErsatzV1aCaloMt_, "ErsatzV1aCaloMt[4]/D");
	t_->Branch("ErsatzV1bCaloMEt", ErsatzV1bCaloMEt_, "ErsatzV1bCaloMEt[4]/D");
	t_->Branch("ErsatzV1bCaloMEtPhi", ErsatzV1bCaloMEtPhi_, "ErsatzV1bCaloMEtPhi[4]/D");
	t_->Branch("ErsatzV1bCaloMt", ErsatzV1bCaloMt_, "ErsatzV1bCaloMt[4]/D");
	t_->Branch("ErsatzV1cCaloMEt", ErsatzV1cCaloMEt_, "ErsatzV1cCaloMEt[4]/D");
	t_->Branch("ErsatzV1cCaloMEtPhi", ErsatzV1cCaloMEtPhi_, "ErsatzV1cCaloMEtPhi[4]/D");
	t_->Branch("ErsatzV2CaloMEt", ErsatzV2CaloMEt_, "ErsatzV2CaloMEt[4]/D");
	t_->Branch("ErsatzV2CaloMt", ErsatzV2CaloMt_, "ErsatzV2CaloMt[4]/D");
	t_->Branch("ErsatzV2CaloMEtPhi", ErsatzV2CaloMEtPhi_, "ErsatzV2CaloMEtPhi[4]/D");

	t_->Branch("recoCaloMEt", &recoCaloMEt_, "recoCaloMEt/D");

	edm::LogDebug_("","", 91)<<"Creating electron branches.";
	t_->Branch("tag_q", tag_q_,"tag_q[4]/I");
	t_->Branch("tag_pt", tag_pt_,"tag_pt[4]/D");
	t_->Branch("tag_eta", tag_eta_,"tag_eta[4]/D");
	t_->Branch("tag_phi", tag_phi_,"tag_phi[4]/D");
	t_->Branch("tag_sIhIh", tag_sIhIh_, "tag_sIhIh[4]/D");
	t_->Branch("tag_dPhiIn", tag_dPhiIn_, "tag_dPhiIn[4]/D");
	t_->Branch("tag_dEtaIn", tag_dEtaIn_, "tag_dEtaIn[4]/D");
	t_->Branch("tag_isoTrack", tag_isoTrack_,"tag_isoTrack[4]/D");
	t_->Branch("tag_isoEcal", tag_isoEcal_,"tag_isoEcal[4]/D");
	t_->Branch("tag_isoHcal", tag_isoHcal_,"tag_isoHcal[4]/D");
	t_->Branch("tag_rescPt", tag_rescPt_, "tag_rescPt[4]/D");
	t_->Branch("tag_rescEta", tag_rescEta_, "tag_rescEta[4]/D");
	edm::LogDebug_("","", 103)<<"Creating ersatz neutrino branches.";
	t_->Branch("probe_q", probe_q_,"probe_q[4]/I");
	t_->Branch("probe_pt", probe_pt_,"probe_pt[4]/D");
	t_->Branch("probe_eta", probe_eta_,"probe_eta[4]/D");
	t_->Branch("probe_phi", probe_phi_,"probe_phi[4]/D");

	//t_->Branch("probe_elecMatch", probe_elecMatch_, "probe_elecMatch[4]/I");
	t_->Branch("probe_isoTrack", probe_isoTrack_,"probe_isoTrack[4]/D");
	t_->Branch("probe_sIhIh", probe_sIhIh_,"probe_sIhIh[4]/D");
	t_->Branch("probe_e2x5Max", probe_e2x5Max_,"probe_e2x5Max[4]/D");
	t_->Branch("probe_e1x5Max", probe_e1x5Max_,"probe_e1x5Max[4]/D");
	t_->Branch("probe_e5x5", probe_e5x5_,"probe_e5x5[4]/D");
	t_->Branch("probe_rescPt", probe_rescPt_, "probe_rescPt[4]/D");
	t_->Branch("probe_rescEta", probe_rescEta_, "probe_rescEta[4]/D");
	t_->Branch("probe_rescPhi", probe_rescPhi_, "probe_rescPhi[4]/D");
	t_->Branch("probe_nClus", probe_nClus_, "probe_nClus[4]/D");

	t_->Branch("Z_pt", Z_pt_, "Z_pt[4]/D");
	t_->Branch("Z_probe_dPhi",Z_probe_dPhi_,"Z_probe_dPhi[4]/D"); 
	//t_->Branch("rechit_E", rechit_E_, "rechit_E[4]/D");
	t_->Branch("probe_E",probe_E_, "probe_E[4]/D");
	t_->Branch("probe_EAdd",probe_EAdd_, "probe_EAdd[4]/D");
	t_->Branch("probe_rawE",probe_rawE_, "probe_rawE[4]/D");
	t_->Branch("probe_fEtaCorrE",probe_fEtaCorrE_, "probe_fEtaCorrE[4]/D");
	t_->Branch("probe_fBremCorrE",probe_fBremCorrE_, "probe_fBremCorrE[4]/D");
	t_->Branch("probe_d_MCE_SCE", probe_d_MCE_SCE_, "probe_d_MCE_SCE[4]/D");
	t_->Branch("probe_UnclusEcalE", probe_UnclusEcalE_, "probe_UnclusEcalE[4]/D");
	t_->Branch("probe_HcalEt015", probe_HcalEt015_, "probe_HcalEt015[4]/D");
	t_->Branch("probe_HcalE015", probe_HcalE015_, "probe_HcalE015[4]/D");

	edm::LogDebug_("","", 103)<<"Creating electron - ersatz neutrino combination branches.";
	t_->Branch("ErsatzV1_Mesc", ErsatzV1_Mesc_, "ErsatzV1_Mesc[4]/D");
	t_->Branch("ErsatzV1_rescMesc", ErsatzV1_rescMesc_, "ErsatzV1_rescMesc[4]/D");

	edm::LogDebug_("","", 103)<<"Creating MC branches.";
	t_->Branch("McElec_nFinal", &McElec_nFinal_, "McElec_nFinal/D");


	if(Zevent_){
		t_->Branch("McZ_m", &McZ_m_, "McZ_m/D");
		t_->Branch("McZ_rescM", &McZ_rescM_, "McZ_rescM/D");
		t_->Branch("McZ_Pt", &McZ_Pt_, "McZ_Pt/D");
		t_->Branch("McZ_Eta", &McZ_Eta_, "McZ_Eta/D");
		t_->Branch("McZ_Phi", &McZ_Phi_, "McZ_Phi/D");
		t_->Branch("McZ_rescPt", &McZ_rescPt_, "McZ_Pt/D");
		t_->Branch("McZ_rescEta", &McZ_rescEta_, "McZ_Eta/D");
		t_->Branch("McZ_rescPhi", &McZ_rescPhi_, "McZ_Phi/D");
		t_->Branch("McElec_nZmum", &McElec_nZmum_, "McElec_nZmum/D");
		t_->Branch("McElec_eta", McElec_eta_, "McElec_eta[4]/D");
		t_->Branch("McElec_pt", McElec_pt_, "McElec_pt[4]/D");
		t_->Branch("McElec_phi", McElec_phi_, "McElec_phi[4]/D");
		t_->Branch("McElec_rescEta", McElec_rescEta_, "McElec_rescEta[4]/D");
		t_->Branch("McElec_rescPhi", McElec_rescPhi_, "McElec_rescPhi[4]/D");
		t_->Branch("McElec_rescPt", McElec_rescPt_, "McElec_rescPt[4]/D");
		t_->Branch("McProbe_eta", McProbe_eta_, "McProbe_eta[4]/D");
		t_->Branch("McProbe_pt", McProbe_pt_, "McProbe_pt[4]/D");
		t_->Branch("McProbe_phi", McProbe_phi_, "McProbe_phi[4]/D");
		t_->Branch("McProbe_rescEta", McProbe_rescEta_, "McProbe_rescEta[4]/D");
		t_->Branch("McProbe_rescPt", McProbe_rescPt_, "McProbe_rescPt[4]/D");
		t_->Branch("McProbe_rescPhi", McProbe_rescPhi_, "McProbe_rescPhi[4]/D");
		t_->Branch("McElecProbe_dPhi", McElecProbe_dPhi_, "McElecProbe_dPhi[4]/D");
		t_->Branch("McElecProbe_dR", McElecProbe_dR_, "McElecProbe_dR[4]/D");
	}

	TotNProbes_ = 0;
	TotEClus_ = 0.;
}
void ErsatzMEt::analyze(const edm::Event& evt, const edm::EventSetup& es)
{
	edm::LogDebug_("","", 151)<<"Initialising variables.";
	nTags_ = -99; nProbes_ = -99;
	recoCaloMEt_ = -99.;
	McZ_m_ = -99.; McZ_rescM_ = -99.; McZ_Pt_ = -99.; McZ_Phi_ = -99.;McZ_Eta_ = -99.;
	McZ_rescPt_ = -99.;McZ_rescEta_ = -99.;McZ_rescPhi_ = -99.; 
	McElec_nZmum_ = -99; McElec_nFinal_ = -99;
	for(int i = 0; i < nEntries_arr_; ++i)
	{
		nRecHitsInStrip_[i] = -99; nRecHitsInCone_[i] = -99;
		tag_q_[i] = -99;
		tag_pt_[i] = -99.;tag_eta_[i] = -99.;tag_phi_[i] = -99.;
		tag_rescPt_[i] = -99.;tag_rescEta_[i] = -99.;tag_rescPhi_[i] = -99.;
		tag_isoTrack_[i] = -99.; tag_isoEcal_[i] = -99.; tag_isoHcal_[i] = -99.;
		tag_sIhIh_[i] = -99.; tag_dPhiIn_[i] = -99.; tag_dEtaIn_[i] = -99.;
		//probe_elecMatch_[i] = -99;
		probe_isoTrack_[i] = -99.;
		probe_q_[i] = -99;
		probe_pt_[i] = -99.;probe_eta_[i] = -99.;probe_phi_[i] = -99.;
		probe_sIhIh_[i] = -99.;probe_e2x5Max_[i] = -99.;probe_e1x5Max_[i] = -99.;
		probe_rescPt_[i] = -99.;probe_rescEta_[i] = -99.;probe_nClus_[i] = -99;
		Z_pt_[i] = -99.; Z_probe_dPhi_[i] = -99.;
		
		//rechit_E_[i] = -99.; 
		ErsatzV1_Mesc_[i] = -99.; ErsatzV1_rescMesc_[i] = -99.;
		ErsatzV1CaloMEt_[i] = -99.; ErsatzV1CaloMt_[i] = -99.;ErsatzV1CaloMEtPhi_[i] = -99.; 
		ErsatzV1aCaloMEt_[i] = -99.; ErsatzV1aCaloMt_[i] = -99.;ErsatzV1aCaloMEtPhi_[i] = -99.; 
		ErsatzV1bCaloMEt_[i] = -99.; ErsatzV1bCaloMt_[i] = -99.;ErsatzV1bCaloMEtPhi_[i] = -99.; 
		ErsatzV1cCaloMEt_[i] = -99.; ErsatzV1cCaloMt_[i] = -99.;ErsatzV1cCaloMEtPhi_[i] = -99.; 
		ErsatzV2CaloMEt_[i] = -99.; ErsatzV2CaloMt_[i] = -99.;ErsatzV2CaloMEtPhi_[i] = -99.; 
		McElec_pt_[i] = -99.; McElec_eta_[i] = -99.;
		McElec_rescPt_[i] = -99.; McElec_rescEta_[i] = -99.; 
		McProbe_pt_[i] = -99.; McProbe_eta_[i] = -99.; McProbe_rescPt_[i] = -99.; McProbe_rescEta_[i] = -99.; 
		McProbe_phi_[i] = -99.; McProbe_rescPhi_[i] = -99.;
		McElecProbe_dPhi_[i] = -99.; McElecProbe_dR_[i] = -99.;
		probe_E_[i] = -99.; probe_EAdd_[i] = -99.; probe_fBremCorrE_[i] = -99.;
		probe_rawE_[i] = -99.; probe_fEtaCorrE_[i] = -99.;
		probe_d_MCE_SCE_[i] = -99.; probe_UnclusEcalE_[i] = -99.;
		probe_HcalEt015_[i] = -99.; probe_HcalE015_[i] = -99.;

		edm::LogDebug_("","",180)<<"Initialisation of array index "<< i <<" completed.";
	}	
	//Get Collections
	edm::Handle<reco::GenParticleCollection> pGenPart;
	try
	{
		evt.getByLabel(MCTruthCollection_, pGenPart);
	}catch(cms::Exception& ex)
	{
		edm::LogError("")<<"Error! Can't get collection with label "<< MCTruthCollection_.label();
	}
	edm::Handle<reco::GsfElectronCollection> pElectrons;
	try
	{
		evt.getByLabel(ElectronCollection_, pElectrons);
	}catch(cms::Exception &ex)
	{
		edm::LogError("analyze") <<"Can't get collection with label "<< ElectronCollection_.label();
	}
	edm::Handle<reco::SuperClusterCollection> pHybrid;
	try
	{
		evt.getByLabel(HybridScCollection_, pHybrid);
	}catch(cms::Exception &ex)
	{
		edm::LogError("analyze") <<"Can't get collection with label "<< HybridScCollection_.label();
	}
	edm::Handle<reco::SuperClusterCollection> pM5x5;
	try
	{
		evt.getByLabel(M5x5ScCollection_, pM5x5);
	}catch(cms::Exception &ex)
	{
		edm::LogError("analyze") <<"Can't get collection with label "<< M5x5ScCollection_.label();
	}
	edm::Handle<EcalRecHitCollection> pEBRecHits;
	try
	{
		evt.getByLabel(EBRecHitCollection_, pEBRecHits);
	}catch(cms::Exception &ex)
	{
		edm::LogError("analyze")<<"Can't get collection with label "<< EBRecHitCollection_.label();
	}
	edm::Handle<EcalRecHitCollection> pEERecHits;
	try
	{
		evt.getByLabel(EERecHitCollection_, pEERecHits);
	}catch(cms::Exception &ex)
	{
		edm::LogError("analyze")<<"Can't get collection with label "<< EERecHitCollection_.label();
	}
	std::vector<edm::Handle<edm::ValueMap<double> > > eIsoValueMap(3);
	try
	{
		evt.getByLabel(eIsoTrack_, eIsoValueMap[0]);
	}catch(cms::Exception &ex)
	{
		edm::LogError("analyze")<< "Can't get collection with label " << eIsoTrack_.label();
	}
	try
	{
		evt.getByLabel(eIsoEcal_, eIsoValueMap[1]);
	}catch(cms::Exception &ex)
	{
		edm::LogError("analyze")<< "Can't get collection with label " << eIsoEcal_.label();
	}
	try
	{
		evt.getByLabel(eIsoHcal_, eIsoValueMap[2]);
	}catch(cms::Exception &ex)
	{
		edm::LogError("analyze")<< "Can't get collection with label " << eIsoHcal_.label();
	}
        edm::Handle<reco::TrackCollection> pTracks;
        try
        {
                evt.getByLabel(TrackCollection_, pTracks);
        }catch(cms::Exception &ex)
        {
                edm::LogError("analyze") <<"Can't get collection with label "<< TrackCollection_.label();
        }
	edm::Handle<reco::CaloMETCollection> pCaloMEt;
	try
	{
		evt.getByLabel(CaloMEtCollection_, pCaloMEt);
	}catch(cms::Exception &ex)
	{
		edm::LogError("analyze")<<"Can't get collection with label "<< CaloMEtCollection_.label();
	}
    	edm::Handle<CaloTowerCollection> pTowers;
    	try
    	{
		evt.getByLabel(CaloTowerCollection_, pTowers);
    	}
    	catch(cms::Exception& ex)
    	{
		edm::LogError("TPAnalyzer") << "Error! Can't get collection with label " << CaloTowerCollection_.label();
	}

	edm::Handle<edm::TriggerResults> pTriggerResults;
	try
	{
		evt.getByLabel(TriggerResults_, pTriggerResults);
	}catch(cms::Exception &ex)
	{
		edm::LogError("analyze")<<"Cant get collection with label "<< TriggerResults_.label();
	}
	edm::Handle<trigger::TriggerEvent> pHLT;
	try
	{
		evt.getByLabel(TriggerEvent_, pHLT);
	}catch(cms::Exception &ex)
	{
		edm::LogError("analyze")<<"Can't get collection with label "<< TriggerEvent_.label();
	}

	//Find leptons to match to ersatz neutrinos. Can then investigate if this is where differences are introduced between Z->ee and W->enu
//	std::vector<reco::GenParticleCollection::const_iterator>McElecs;
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
		McZ_m_ = Zboson.M(); McZ_Pt_ = Zboson.Pt(); McZ_Phi_ = Zboson.Phi(); McZ_Eta_ = Zboson.Eta();
		McElec_nZmum_ =McElecs.size();
		McElec_nFinal_ =McElecsFinalState.size();
		edm::LogDebug_("","",309)<<"MC electrons with Z mother = "<< McElec_nZmum_ 
						<<"\tFinal state MC electrons = "<< McElec_nFinal_;
		
		McElecsResc.resize(2);
//		RescZboson.SetCoordinates(Zboson.Px(), Zboson.Py(), Zboson.Pz(), sqrt(Zboson.P2()+(mW_*mW_*Zboson.M2())/(mZ_*mZ_)));
		RescZboson.SetCoordinates(Zboson.Px(), Zboson.Py(), Zboson.Pz(), Zboson.E());
		McZ_rescM_ = RescZboson.M(); McZ_rescPt_ = RescZboson.Pt(); McZ_rescEta_ = RescZboson.Eta(); McZ_rescPhi_ = RescZboson.Phi();
		ROOT::Math::Boost CoMBoost(Zboson.BoostToCM());
		
		math::XYZTLorentzVector RescMcElec0 = CoMBoost(McElecsFinalState[0]);
		math::XYZTLorentzVector RescMcElec1 = CoMBoost(McElecsFinalState[1]);
		RescMcElec0 *= mW_/mZ_;
		RescMcElec1 *= mW_/mZ_;

		double E_W = RescZboson.E();
		ROOT::Math::Boost BackToLab(RescZboson.Px()/E_W, RescZboson.Py()/E_W, RescZboson.Pz()/E_W);

		RescMcElec0 = BackToLab(RescMcElec0);
	/*	RndmMcElec_Rescaled_pt_ = RescMcElec0.Pt();
		RndmMcElec_Rescaled_eta_ = RescMcElec0.Eta();
		RndmMcElec_Rescaled_phi_ = RescMcElec0.Phi();
	*/
		RescMcElec1 = BackToLab(RescMcElec1);
	/*	OthrMcElec_Rescaled_pt_ = RescMcElec1.Pt();
		OthrMcElec_Rescaled_eta_ = RescMcElec1.Eta();
		OthrMcElec_Rescaled_phi_ = RescMcElec1.Phi();
	*/	McElecsResc[0] = RescMcElec0;
		McElecsResc[1] = RescMcElec1;
		math::XYZTLorentzVector sum = RescMcElec1+RescMcElec0;
		edm::LogDebug_("","", 307)<<"McElecsResc[0] + McElecsResc[1] = ("<<sum.Px()<<", "<<sum.Py()<<", "
						<<sum.Pz()<<", "<<sum.E()<<")";
	}	

	const edm::TriggerResults* HltRes = pTriggerResults.product();
	TriggerNames_.init(*HltRes);
	if(HLTPathCheck_)
	{
		for(uint itrig = 0; itrig < HltRes->size(); ++itrig)
		{
			std::string nom = TriggerNames_.triggerName(itrig);
			edm::LogInfo("")<< itrig <<" : Name = "<< nom <<"\t Accepted = "<< HltRes->accept(itrig);
		}
	}
	if(HltRes->accept(34) ==0) edm::LogError("")<<"Event did not pass "<< TriggerNames_.triggerName(34)<<"!";
	if(HltRes->accept(34) !=0)
	{
	std::vector<reco::GsfElectronRef> UniqueElectrons;
	UniqueElectrons = uniqueElectronFinder(pElectrons);
	edm::LogDebug_("","ErsatzMEt",192)<<"Unique electron size = "<<UniqueElectrons.size();
	std::vector<reco::GsfElectronRef> SelectedElectrons;
	const unsigned int fId = pHLT->filterIndex(TriggerPath_);
	std::cout << "Filter Id = " << fId << std::endl;
	EcalClusterLazyTools lazyTools(evt,es,EBRecHitCollection_, EERecHitCollection_);
	std::cout << "ECAL cluster lazy tools done" << std::endl; 
	SelectedElectrons = electronSelector(UniqueElectrons, lazyTools, eIsoValueMap, pHLT, fId, CutVector_);
//	SelectedElectrons = UniqueElectrons;
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

	std::map<reco::GsfElectronRef, reco::GsfElectronRef> delendumMap;
	delendumMap = probeFinder(SelectedElectrons, pElectrons);
	nProbes_ = delendumMap.size();	
	edm::LogDebug_("", "ErsatzMEt", 209)<<"Number of tag-probe pairs = "<< delendumMap.size();

	if(!delendumMap.empty())
	{
		const reco::CaloMETCollection* MEtCollection = pCaloMEt.product();
        	reco::CaloMETCollection::const_iterator met = MEtCollection->begin();
		recoCaloMEt_ = met->pt();

		reco::MET ersatzV1MEt;

	        const reco::TrackCollection* Tracks = pTracks.product();

		for(std::map<reco::GsfElectronRef, reco::GsfElectronRef>::const_iterator it = delendumMap.begin();
			it != delendumMap.end(); ++it)
		{
			reco::SuperClusterRef sc = it->second->superCluster();
			edm::LogDebug_("","DelendumLoop", 293)<<"iComb_ = "<< iComb_;
			tag_q_[iComb_] = it->first->charge();
			probe_q_[iComb_] = it->second->charge();
			tag_pt_[iComb_] = it->first->pt();
			tag_eta_[iComb_] = it->first->eta();
			tag_phi_[iComb_] = it->first->phi();
			edm::LogDebug_("","ErsatzMEt", 245)<<"tag pt["<<iComb_<<"] = "<< tag_pt_[iComb_] 
					<<"\ttag_eta_ = "<< tag_eta_[iComb_]<<"\ttag_phi_[iComb_] = "<< tag_phi_[iComb_];
                        const edm::ValueMap<double>& eIsoMapTrk = *eIsoValueMap[0];
                        const edm::ValueMap<double>& eIsoMapEcal = *eIsoValueMap[1];
                        const edm::ValueMap<double>& eIsoMapHcal = *eIsoValueMap[2];
                        tag_isoTrack_[iComb_] = eIsoMapTrk[it->first];
                        tag_isoEcal_[iComb_] = eIsoMapEcal[it->first];
                        tag_isoHcal_[iComb_] = eIsoMapHcal[it->first];
			std::vector<float> vCov = lazyTools.localCovariances(*(it->first->superCluster()->seed())) ;
			tag_sIhIh_[iComb_] = vCov[0];
			tag_dPhiIn_[iComb_] = it->first->deltaPhiSuperClusterTrackAtVtx();
			tag_dEtaIn_[iComb_] = it->first->deltaEtaSuperClusterTrackAtVtx();
			math::XYZTLorentzVector	probe_detVec = DetectorVector(sc);
			probe_pt_[iComb_] = probe_detVec.pt(); 
			probe_eta_[iComb_] = sc->eta();
			probe_phi_[iComb_] = sc->phi();
			probe_nClus_[iComb_] = sc->clustersSize();
			probe_E_[iComb_] = sc->energy();
			probe_rawE_[iComb_] = sc->rawEnergy();

			if(fabs(probe_eta_[iComb_]) < 1.479){
				reco::SuperCluster corrSC = fEtaScCorr(sc);
			//	std::cout <<"New fEtaScCorr SC energy = "<< corrSC.energy() << std::endl;
				probe_fEtaCorrE_[iComb_] = corrSC.energy();
				corrSC = fBremScCorr(sc, hyb_fCorrPSet_);
				probe_fBremCorrE_[iComb_] = corrSC.energy();
			}else{
				reco::SuperCluster corrSC = fBremScCorr(sc, m5x5_fCorrPSet_);
				probe_fBremCorrE_[iComb_] = corrSC.energy();
			}

			reco::CaloCluster probeSeed = *(sc->seed());
			probe_sIhIh_[iComb_] = sqrt(lazyTools.localCovariances(probeSeed)[0]);
			probe_e2x5Max_[iComb_] = lazyTools.e2x5Max(probeSeed);
			probe_e1x5Max_[iComb_] = lazyTools.e1x5(probeSeed);
			probe_e5x5_[iComb_] = lazyTools.e5x5(probeSeed);
		
			edm::LogDebug_("","ErsatzMEt", 444)<<"Initialising CaloNavigator";
			std::vector<std::pair<DetId, float> > SChits = sc->hitsAndFractions();
			std::vector<DetId> SChitsId;
			for (std::vector<std::pair<DetId, float> >::const_iterator haf = SChits.begin(); haf != SChits.end(); ++haf)
			{
				SChitsId.push_back(haf->first);
			}
			DetId seedXtalId = probeSeed.hitsAndFractions()[0].first;
		        const CaloTopology *topology = pTopology_.product();	
			CaloNavigator<DetId> cursor = CaloNavigator<DetId>(seedXtalId, topology->getSubdetectorTopology(seedXtalId));
			// x == eta, y == phi in EB
			int ArrayBound = 9;
			if(fabs(probe_eta_[iComb_]) < (1.479-0.0174*ArrayBound))
			{
				const CaloSubdetectorGeometry* ebtmp = geoHandle_->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
				const EcalBarrelGeometry* ecalBarrelGeometry = dynamic_cast< const EcalBarrelGeometry* >(ebtmp); 
				assert(ecalBarrelGeometry);
				const CaloCellGeometry* center_cell = ecalBarrelGeometry->getGeometry(seedXtalId);
		  		GlobalPoint p = center_cell->getPosition();
//				std::cout << "SC phi = "<< sc->phi() <<"\tseed = "<< p.phi();
				TotEClus_ += sc->rawEnergy();
				++TotNProbes_;
				probe_UnclusEcalE_[iComb_] = 0.;
//				std::cout << "Crystal iphi, ieta values : "<<std::endl;
				for(int i = -ArrayBound; i <= ArrayBound; ++i )
				{
					for(int j = -ArrayBound; j <= ArrayBound; ++j ) 
					{
				//		if(j%5 == 0) std::cout << std::endl;
						cursor.home();
						cursor.offsetBy( i, j );
	//					double energy = recHitEnergy( *cursor, recHits );
						const EcalRecHitCollection* Hits = pEBRecHits.product();
						EcalRecHitCollection::const_iterator hit = Hits->find(*cursor);
						double energy = 0.;
						if(hit != Hits->end()) energy = hit->energy(); 
//						std::cout<<"("<< j << ", "<< i <<") = "<<energy <<"\t";
						std::vector<DetId>::const_iterator SChit;
						SChit = find(SChitsId.begin(), SChitsId.end(), *cursor);
						if(SChit == SChitsId.end()) 
						{
							probe_UnclusEcalE_[iComb_]  += energy; 
						}
					}
//					std::cout << std::endl;
				}
			}
			double dRLimit = 0.2;
			const math::XYZPoint Vertex(it->first->TrackPositionAtVtx().X(),
							it->first->TrackPositionAtVtx().Y(),
							it->first->TrackPositionAtVtx().Z());
			math::XYZTLorentzVector probeVec = PhysicsVector(Vertex, sc);
			probe_isoTrack_[iComb_] = 0.;
			for(reco::TrackCollection::const_iterator tr = Tracks->begin(); tr != Tracks->end(); ++tr)
			{
				double dRt = reco::deltaR(*tr, probeVec); 
				if(dRt < 0.4 && dRt > 0.02) probe_isoTrack_[iComb_] += tr->pt();
			}
//			for(std::vector<math::XYZTLorentzVector>::const_iterator
//						mcEl =McElecs.begin(); mcEl !=McElecs.end(); ++mcEl)
			for(uint mcEId = 0; mcEId < McElecs.size(); ++mcEId)
			{
//				double dR = reco::deltaR((*(*mcEl)), probeVec); 
				double dR = reco::deltaR(McElecs[mcEId], probeVec); 
				if(dR < dRLimit)
				{
					dRLimit = dR;
					McProbe_pt_[iComb_] = McElecs[mcEId].pt();
					McProbe_eta_[iComb_] = McElecs[mcEId].eta();
					McProbe_phi_[iComb_] = McElecs[mcEId].phi();
					McProbe_rescPt_[iComb_] = McElecsResc[mcEId].pt();
					McProbe_rescEta_[iComb_] = McElecsResc[mcEId].eta();
					McProbe_rescPhi_[iComb_] = McElecsResc[mcEId].phi();
					probe_d_MCE_SCE_[iComb_] = McElecs[mcEId].energy() - sc->rawEnergy();
					McElecProbe_dPhi_[iComb_] = reco::deltaPhi(McElecs[mcEId].phi(), McElecs[(mcEId+1)%2].phi());
					McElecProbe_dR_[iComb_] = reco::deltaR(McElecs[mcEId], McElecs[(mcEId+1)%2]);
				} 
			}

			const CaloTowerCollection* Towers = pTowers.product();
			probe_HcalEt015_[iComb_] = 0.;
			probe_HcalE015_[iComb_] = 0.;
			for(CaloTowerCollection::const_iterator tower = Towers->begin(); tower != Towers->end(); ++tower)
			{
				if(reco::deltaR((*tower), probe_detVec) < 0.15){
					probe_HcalEt015_[iComb_] += tower->hadEt();
					probe_HcalE015_[iComb_] += tower->hadEnergy();
				}
			}
			//The integers arguments to ersatzFabrik define the corrections applied to the superclusters
			//0 = raw; 1 = f(eta) corrected; 2 = f(brem) corrected; 3 = fully corrected.
			ersatzV1MEt = ersatzFabrik(it->first, sc, pEBRecHits, pCaloMEt, CutVector_[0], 3);
			ErsatzV1CaloMEt_[iComb_] = ersatzV1MEt.pt();	
			ErsatzV1CaloMEtPhi_[iComb_] = ersatzV1MEt.phi();	
			if(ersatzV1MEt.pz() < -98.)
			{
				ErsatzV1CaloMEt_[iComb_] = -99.;
				ErsatzV1CaloMEtPhi_[iComb_] = -99.;
				ErsatzV1CaloMt_[iComb_] = -99.;
			}

			ersatzV1MEt = ersatzFabrik(it->first, sc, pEBRecHits, pCaloMEt, CutVector_[0], 0);
			ErsatzV1aCaloMEt_[iComb_] = ersatzV1MEt.pt();	
			ErsatzV1aCaloMEtPhi_[iComb_] = ersatzV1MEt.phi();	
			if(ersatzV1MEt.pz() < -98.)
			{
				ErsatzV1aCaloMEt_[iComb_] = -99.;
				ErsatzV1aCaloMt_[iComb_] = -99.;
				ErsatzV1aCaloMEtPhi_[iComb_] = -99.;
			}

			ersatzV1MEt = ersatzFabrik(it->first, sc, pEBRecHits, pCaloMEt, CutVector_[0], 1);
			ErsatzV1bCaloMEt_[iComb_] = ersatzV1MEt.pt();	
			ErsatzV1bCaloMEtPhi_[iComb_] = ersatzV1MEt.phi();	
			if(ersatzV1MEt.pz() < -98.)
			{
				ErsatzV1bCaloMEt_[iComb_] = -99.;
				ErsatzV1bCaloMt_[iComb_] = -99.;
				ErsatzV1bCaloMEtPhi_[iComb_] = -99.;
			}
			reco::MET ersatzV2MEt = ersatzFabrik(it->first, sc, pEBRecHits, pCaloMEt, CutVector_[0], 4);
			ErsatzV2CaloMEt_[iComb_] = ersatzV2MEt.pt();	
			ErsatzV2CaloMEtPhi_[iComb_] = ersatzV2MEt.phi();	
			if(ersatzV2MEt.pz() < -98.)
			{
				ErsatzV2CaloMEt_[iComb_] = -99.;
				ErsatzV2CaloMt_[iComb_] = -99.;
				ErsatzV2CaloMEtPhi_[iComb_] = -99.;
			}
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

reco::MET ErsatzMEt::ersatzFabrik(const reco::GsfElectronRef& elec, const reco::SuperClusterRef& sc,
							const edm::Handle<EcalRecHitCollection>& RecHits,
							const edm::Handle<reco::CaloMETCollection>& pCaloMEt,
							const double& EtCut, const int corr)
{
	const reco::CaloMETCollection* MEtCollection = pCaloMEt.product();
        reco::CaloMETCollection::const_iterator met = MEtCollection->begin();
	const math::XYZPoint ZVertex(elec->TrackPositionAtVtx().X(), elec->TrackPositionAtVtx().Y(),elec->TrackPositionAtVtx().Z());

	math::XYZTLorentzVector nu, boost_nu, ele, boost_ele;
	reco::SuperClusterRef elecSC = elec->superCluster();
	double scEta = sc->eta(); double elecScEta = elecSC->eta(); 
//	reco::SuperClusterCollection corrClus;
//	int scId = 0;
	if(corr == 0){
		nu = PhysicsVectorRaw(met->vertex(), sc);
		boost_nu = PhysicsVectorRaw(ZVertex, sc);
		ele = PhysicsVectorRaw(met->vertex(), elecSC);
		boost_ele = ele;
	}
	if(corr == 1){
		if(fabs(scEta) < 1.479){
			reco::SuperCluster corrSC = fEtaScCorr(sc);
//        		reco::SuperClusterRef scRef(corrClus, scId);
//			++scId;
			nu = PhysicsVector(met->vertex(), corrSC);
			boost_nu = PhysicsVector(ZVertex, corrSC);
		}else{
			nu = PhysicsVectorRaw(met->vertex(), sc);
			boost_nu = PhysicsVectorRaw(ZVertex, sc);
		}
		if(fabs(elecScEta) < 1.479){
			reco::SuperCluster corrSC = fEtaScCorr(elecSC);
			ele = PhysicsVector(met->vertex(), corrSC);
//			corrClus.push_back(corrSC);
  //      		const reco::SuperClusterRef scRef(corrClus, scId);
//			++scId;
//yy			ele = PhysicsVector(met->vertex(), scRef);
			boost_ele = ele;
		}else{
			ele = PhysicsVectorRaw(met->vertex(), elecSC);
			boost_ele = ele;
		}
	}
	if(corr == 2){
		if(fabs(scEta) < 1.479){
			reco::SuperCluster corrSC = fBremScCorr(sc, hyb_fCorrPSet_);
			nu = PhysicsVector(met->vertex(), corrSC);
			boost_nu = PhysicsVector(ZVertex, corrSC);
		}else{
			reco::SuperCluster corrSC = fBremScCorr(sc, m5x5_fCorrPSet_);
			nu = PhysicsVector(met->vertex(), corrSC);
			boost_nu = PhysicsVector(ZVertex, corrSC);
		}
		if(fabs(elecScEta) < 1.479){
			reco::SuperCluster corrSC = fBremScCorr(elecSC, hyb_fCorrPSet_);
			ele = PhysicsVector(met->vertex(), corrSC);
			boost_ele = ele;
		}else{
			reco::SuperCluster corrSC = fBremScCorr(elecSC, m5x5_fCorrPSet_);
			ele = PhysicsVector(met->vertex(), corrSC);
			boost_ele = ele;
		}
	}
	if(corr == 3){
		nu = PhysicsVector(met->vertex(), sc);
		boost_nu = PhysicsVector(ZVertex, sc);
		ele = PhysicsVector(met->vertex(), elec->superCluster());
		boost_ele = ele;
	}
	if(corr == 4){
		if(fabs(scEta) < 1.479)
		{
			double energyCorr = 0.;
			const EcalRecHitCollection* Hits = RecHits.product();
		        const CaloTopology *topology = pTopology_.product();	
		
			const CaloSubdetectorGeometry* ebtmp = geoHandle_->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
			const EcalBarrelGeometry* ecalBarrelGeometry = dynamic_cast< const EcalBarrelGeometry* >(ebtmp); 
			assert(ecalBarrelGeometry);
			
			GlobalPoint scLoc(sc->position().x(), sc->position().y(), sc->position().z());
			DetId scCentreId = ecalBarrelGeometry->getClosestCell(scLoc);
			CaloNavigator<DetId> cursor = CaloNavigator<DetId>(scCentreId, topology->getSubdetectorTopology(scCentreId));
			for(int i = -3; i <= 3; i+=6 )
			{
				for(int j = -2; j <= 2; ++j ) 
				{
//					if(j%5 == 0) std::cout << std::endl;
					cursor.home();
					cursor.offsetBy( i, j );
					EcalRecHitCollection::const_iterator hit = Hits->find(*cursor);
					if(hit != Hits->end()){
						double energy = hit->energy(); 
//						std::cout<<"("<< j << ", "<< i <<") = "<<energy <<"\t";
						energyCorr += energy;
					}
				}
//					std::cout << std::endl;
			}
			reco::SuperCluster corrSC = fEAddScCorr(sc, energyCorr);
			probe_EAdd_[iComb_] = corrSC.energy();	
			nu = PhysicsVector(met->vertex(), corrSC);
			boost_nu = PhysicsVector(ZVertex, corrSC);
		}else{
			nu = PhysicsVectorRaw(met->vertex(), sc);
			boost_nu = PhysicsVectorRaw(ZVertex, sc);
		}


	//reco::SuperClusterRef elecSC = elec->superCluster();
	//double scEta = sc->eta(); double elecScEta = elecSC->eta(); 
		if(fabs(elecScEta) < 1.479)
		{
			double energyCorr = 0.;
			const EcalRecHitCollection* Hits = RecHits.product();
		        const CaloTopology *topology = pTopology_.product();	
		
			const CaloSubdetectorGeometry* ebtmp = geoHandle_->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
			const EcalBarrelGeometry* ecalBarrelGeometry = dynamic_cast< const EcalBarrelGeometry* >(ebtmp); 
			assert(ecalBarrelGeometry);
			
			GlobalPoint scLoc(elecSC->position().x(), elecSC->position().y(), elecSC->position().z());
			DetId scCentreId = ecalBarrelGeometry->getClosestCell(scLoc);
			CaloNavigator<DetId> cursor = CaloNavigator<DetId>(scCentreId, topology->getSubdetectorTopology(scCentreId));
			for(int i = -3; i <= 3; i+=6 )
			{
				for(int j = -2; j <= 2; ++j ) 
				{
//					if(j%5 == 0) std::cout << std::endl;
					cursor.home();
					cursor.offsetBy( i, j );
					EcalRecHitCollection::const_iterator hit = Hits->find(*cursor);
					if(hit != Hits->end()){
						double energy = hit->energy(); 
//						std::cout<<"("<< j << ", "<< i <<") = "<<energy <<"\t";
						energyCorr += energy;
					}
				}
//					std::cout << std::endl;
			}
			reco::SuperCluster corrSC = fEAddScCorr(elecSC, energyCorr);
			ele = PhysicsVector(met->vertex(), corrSC);
			boost_ele = ele;
		}else{
			ele = PhysicsVectorRaw(met->vertex(), elecSC);
			boost_ele = ele;
		}
	}

	ErsatzV1_Mesc_[iComb_]  = ROOT::Math::VectorUtil::InvariantMass(elec->p4(), boost_nu);
	//Should use reco vertex for best Z->ee measurement. 
        edm::LogDebug_("ersatzFabrikV1", "", 569)<<"elec  = ("<< elec->p4().Px() << ", "<< elec->p4().Py()<< ", "<< elec->p4().Pz() << ", "<< elec->p4().E()<<")";
	math::XYZTLorentzVector Zboson = boost_nu + elec->p4();
	Z_pt_[iComb_] = Zboson.Pt();
	Z_probe_dPhi_[iComb_] = reco::deltaPhi(Zboson.Phi(), elec->phi());
        edm::LogDebug_("ersatzFabrikV1", "", 569)<<"Z pt = "<< Zboson.Pt() << "Z boson mass = " << Zboson.M();
        edm::LogDebug_("ersatzFabrikV1","", 570)<<"Z boson in lab frame = ("<<Zboson.Px()<<", "<<Zboson.Py()<<", "
                                        <<Zboson.Pz()<<", "<<Zboson.E()<<")";
        math::XYZTLorentzVector RescZboson(Zboson.Px(), Zboson.Py(), Zboson.Pz(), sqrt(Zboson.P2()+(mW_*mW_*Zboson.M2())/(mZ_*mZ_)));
        edm::LogDebug_("ersatzFabrikV1","", 573)<<"W boson in lab frame = ("<<RescZboson.Px()<<", "<<RescZboson.Py()<<", "
                                        <<RescZboson.Pz()<<", "<<RescZboson.E()<<")";
        ROOT::Math::Boost CoMBoost(Zboson.BoostToCM());
        edm::LogDebug_("ersatzFabrikV1","", 576)<<"Electron in lab frame = ("<< boost_ele.Px()<<", "<< boost_ele.Py()<<", "
                                        << boost_ele.Pz()<<", "<< boost_ele.E()<<")";
        edm::LogDebug_("ersatzFabrikV1","", 578)<<"Ersatz Neutrino in lab frame = ("<< boost_nu.Px()<<", "<< boost_nu.Py()<<", "
                                        << boost_nu.Pz()<<", "<< boost_nu.E()<<")";
        boost_ele = CoMBoost(boost_ele);
        boost_nu = CoMBoost(boost_nu);
        edm::LogDebug_("ersatzFabrikV1","", 582)<<"Electron in Z rest frame = ("<<boost_ele.Px()<<", "<<boost_ele.Py()<<", "
                                        <<boost_ele.Pz()<<", "<<boost_ele.E()<<")";
        edm::LogDebug_("ersatzFabrikV1","", 584)<<"Ersatz Neutrino in Z rest frame = ("<<boost_nu.Px()<<", "<<boost_nu.Py()<<", "
                                        <<boost_nu.Pz()<<", "<<boost_nu.E()<<")";
        boost_ele *= mW_/mZ_;
        boost_nu *= mW_/mZ_;

	ErsatzV1_Mesc_[iComb_] = RescZboson.M();
        double E_W = RescZboson.E();
        ROOT::Math::Boost BackToLab(RescZboson.Px()/E_W, RescZboson.Py()/E_W, RescZboson.Pz()/E_W);
	math::XYZTLorentzVector metVec(-99999., -99999., -99., -99999.);
        boost_ele = BackToLab(boost_ele);
	if(boost_ele.Pt() > EtCut)
	{
		tag_rescPt_[iComb_]  = boost_ele.Pt();
		tag_rescEta_[iComb_]  = boost_ele.Eta();
		tag_rescPhi_[iComb_]  = boost_ele.Phi();
		probe_rescPt_[iComb_]  = boost_nu.Pt();
		probe_rescEta_[iComb_]  = boost_nu.Eta();
		probe_rescPhi_[iComb_]  = boost_nu.Phi();

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
		metVec = met->p4() + nu + ele - boost_ele;
	}
	
	reco::MET ersatzMEt(metVec, met->vertex()); 	
	if(corr == 3)	ErsatzV1CaloMt_[iComb_] = sqrt(2.*boost_ele.Pt()*ersatzMEt.pt()*
						(1-cos(reco::deltaPhi(boost_ele.Phi(), ersatzMEt.phi()))));
	if(corr == 0)	ErsatzV1aCaloMt_[iComb_] = sqrt(2.*boost_ele.Pt()*ersatzMEt.pt()*
						(1-cos(reco::deltaPhi(boost_ele.Phi(), ersatzMEt.phi()))));
	if(corr == 1)	ErsatzV1bCaloMt_[iComb_] = sqrt(2.*boost_ele.Pt()*ersatzMEt.pt()*
						(1-cos(reco::deltaPhi(boost_ele.Phi(), ersatzMEt.phi()))));
	if(corr == 2)	ErsatzV1cCaloMt_[iComb_] = sqrt(2.*boost_ele.Pt()*ersatzMEt.pt()*
						(1-cos(reco::deltaPhi(boost_ele.Phi(), ersatzMEt.phi()))));
	if(corr == 4)	ErsatzV2CaloMt_[iComb_] = sqrt(2.*boost_ele.Pt()*ersatzMEt.pt()*
						(1-cos(reco::deltaPhi(boost_ele.Phi(), ersatzMEt.phi()))));
	return ersatzMEt;
}

// ------------ method called once each job just after ending the event loop  ------------
void ErsatzMEt::endJob() {
}
//define this as a plug-in
DEFINE_FWK_MODULE(ErsatzMEt);
