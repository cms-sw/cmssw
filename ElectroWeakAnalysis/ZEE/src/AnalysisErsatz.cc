/*#include "ElectroWeakAnalysis/ZEE/interface/AnalysisErsatz.h"
AnalysisErsatz::AnalysisErsatz(const edm::ParameterSet& ps)
{
	MCTruthCollection_  = consumes<reco::GenParticleCollection>(ps.getParameter<edm::InputTag>("MCTruthCollection"));
	ElectronCollection_ = consumes<reco::GsfElectronCollection>(ps.getParameter<edm::InputTag>("ElectronCollection"));
	GenMEtCollection_  = consumes<reco::GenMETCollection>(ps.getParameter<edm::InputTag>("GenMEtCollection"));
	//T1MEtCollection_  = consumes<reco::METCollection>(ps.getParameter<edm::InputTag>("T1MEtCollection"));
	PfMEtCollection_  = consumes<reco::PFMETCollection>(ps.getParameter<edm::InputTag>("PfMEtCollection"));
	TcMEtCollection_  = consumes<reco::METCollection>(ps.getParameter<edm::InputTag>("TcMEtCollection"));
	CaloMEtCollection_  = consumes<reco::CaloMETCollection>(ps.getParameter<edm::InputTag>("CaloMEtCollection"));
        TriggerEvent_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("TriggerEvent"));
        TriggerPath_ = ps.getParameter<edm::InputTag>("TriggerPath");
        TriggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
        TriggerName_ = ps.getParameter<std::string>("TriggerName");
	ErsatzEvent_ = ps.getParameter<bool>("ErsatzEvent");
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
	mW_ = ps.getParameter<double>("mW");
	mZ_ = ps.getParameter<double>("mZ");

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

}


AnalysisErsatz::~AnalysisErsatz()
{
}


// ------------ method called once each job just before starting event loop  ------------
void AnalysisErsatz::beginJob()
{
	edm::Service<TFileService> fs;
	t_ = fs->make<TTree>("AnalysisData", "Analysis of Ersatz MEt Properties");

	t_->Branch("Boson_pt", &Boson_pt_,"Boson_pt/D");
	t_->Branch("Boson_y", &Boson_y_,"Boson_y/D");
	t_->Branch("Boson_phi", &Boson_phi_,"Boson_phi/D");
	t_->Branch("Boson_m", &Boson_m_,"Boson_m/D");
	t_->Branch("McElec1_pt", McElec1_pt_,"McElec1_pt[4]/D");
	t_->Branch("McElec1_eta", McElec1_eta_,"McElec1_eta[4]/D");
	t_->Branch("McElec3_pt", McElec3_pt_,"McElec3_pt[4]/D");
	t_->Branch("McElec3_eta", McElec3_eta_,"McElec3_eta[4]/D");
	t_->Branch("RndmInt", &RndmInt_, "RndmInt/I");
	t_->Branch("RndmTrig", &RndmTrig_, "RndmTrig/I");
	t_->Branch("RndmReco", &RndmReco_, "RndmReco/I");
	t_->Branch("OthrTrig", &OthrTrig_, "OthrTrig/I");
	t_->Branch("OthrReco", &OthrReco_, "OthrReco/I");
	t_->Branch("RndmMcElec_pt", &RndmMcElec_pt_,"RndmMcElec_pt/D");
	t_->Branch("RndmMcElec_eta", &RndmMcElec_eta_,"RndmMcElec_eta/D");
	t_->Branch("RndmMcElec_phi", &RndmMcElec_phi_,"RndmMcElec_phi/D");
	t_->Branch("RndmMcElec_Rescaled_pt", &RndmMcElec_Rescaled_pt_,"RndmMcElec_Rescaled_pt/D");
	t_->Branch("RndmMcElec_Rescaled_eta", &RndmMcElec_Rescaled_eta_,"RndmMcElec_Rescaled_eta/D");
	t_->Branch("RndmMcElec_Rescaled_phi", &RndmMcElec_Rescaled_phi_,"RndmMcElec_Rescaled_phi/D");
	t_->Branch("RndmMcElecTRIG_pt", &RndmMcElecTRIG_pt_,"RndmMcElecTRIG_pt/D");
	t_->Branch("RndmMcElecTRIG_eta", &RndmMcElecTRIG_eta_,"RndmMcElecTRIG_eta/D");
	t_->Branch("RndmMcElecRECO_pt", &RndmMcElecRECO_pt_,"RndmMcElecRECO_pt/D");
	t_->Branch("RndmMcElecRECO_eta", &RndmMcElecRECO_eta_,"RndmMcElecRECO_eta/D");
	t_->Branch("OthrMcElec_pt", &OthrMcElec_pt_,"OthrMcElec_pt/D");
	t_->Branch("OthrMcElec_eta", &OthrMcElec_eta_,"OthrMcElec_eta/D");
	t_->Branch("OthrMcElec_phi", &OthrMcElec_phi_,"OthrMcElec_phi/D");
	t_->Branch("OthrMcElec_Rescaled_pt", &OthrMcElec_Rescaled_pt_,"OthrMcElec_Rescaled_pt/D");
	t_->Branch("OthrMcElec_Rescaled_eta", &OthrMcElec_Rescaled_eta_,"OthrMcElec_Rescaled_eta/D");
	t_->Branch("OthrMcElec_Rescaled_phi", &OthrMcElec_Rescaled_phi_,"OthrMcElec_Rescaled_phi/D");
	t_->Branch("OthrMcElecTRIG_pt", &OthrMcElecTRIG_pt_,"OthrMcElecTRIG_pt/D");
	t_->Branch("OthrMcElecTRIG_eta", &OthrMcElecTRIG_eta_,"OthrMcElecTRIG_eta/D");
	t_->Branch("OthrMcElecRECO_pt", &OthrMcElecRECO_pt_,"OthrMcElecRECO_pt/D");
	t_->Branch("OthrMcElecRECO_eta", &OthrMcElecRECO_eta_,"OthrMcElecRECO_eta/D");
	t_->Branch("McNu_pt", &McNu_pt_,"McNu_pt/D");
	t_->Branch("McNu_eta", &McNu_eta_,"McNu_eta/D");
	t_->Branch("McNu_phi", &McNu_phi_,"McNu_phi/D");
	t_->Branch("McNu_vx", &McNu_vx_,"McNu_vx/D");
	t_->Branch("McNu_vy", &McNu_vy_,"McNu_vy/D");
	t_->Branch("McNu_vz", &McNu_vz_,"McNu_vz/D");
	t_->Branch("McLeptons_dPhi", &McLeptons_dPhi_,"McLeptons_dPhi/D");
	t_->Branch("McLeptons_dEta", &McLeptons_dEta_,"McLeptons_dEta/D");
	t_->Branch("McLeptons_dR", &McLeptons_dR_,"McLeptons_dR/D");
	t_->Branch("nSelElecs", &nSelElecs_,"nSelElecs/I");
	t_->Branch("elec_q", elec_q_,"elec_q[4]/D");
	t_->Branch("elec_pt", elec_pt_,"elec_pt[4]/D");
	t_->Branch("elec_eta", elec_eta_,"elec_eta[4]/D");
	t_->Branch("elec_phi", elec_phi_,"elec_phi[4]/D");
	t_->Branch("elec_pt25", &elec_pt25_,"elec_pt25/D");
	t_->Branch("elec_eta25", &elec_eta25_,"elec_eta25/D");
	t_->Branch("elec_phi25", &elec_phi25_,"elec_phi25/D");
        t_->Branch("elec_sIhIh", elec_sIhIh_, "elec_sIhIh[4]/D");
        t_->Branch("elec_dPhiIn", elec_dPhiIn_, "elec_dPhiIn[4]/D");
        t_->Branch("elec_dEtaIn", elec_dEtaIn_, "elec_dEtaIn[4]/D");
        t_->Branch("elec_trckIso", elec_trckIso_,"elec_trckIso[4]/D");
        t_->Branch("elec_ecalIso", elec_ecalIso_,"elec_ecalIso[4]/D");
        t_->Branch("elec_hcalIso", elec_hcalIso_,"elec_hcalIso[4]/D");
	t_->Branch("elec_e2x5Max", elec_e2x5Max_,"elec_e2x5Max[4]/D");
	t_->Branch("elec_e1x5Max", elec_e1x5Max_,"elec_e1x5Max[4]/D");
	t_->Branch("elec_e5x5", elec_e5x5_,"elec_e5x5[4]/D");
	t_->Branch("elec_hoe", elec_hoe_,"elec_hoe[4]/D");
	t_->Branch("elec_eop", elec_eop_,"elec_eop[4]/D");
	t_->Branch("elec_pin", elec_pin_,"elec_pin[4]/D");
	t_->Branch("elec_pout", elec_pout_,"elec_pout[4]/D");

	t_->Branch("Selected_nuPt", Selected_nuPt_, "Selected_nuPt[4]/D");
	t_->Branch("Selected_nuEta", Selected_nuEta_, "Selected_nuEta[4]/D");
	t_->Branch("Selected_nuPhi", Selected_nuPhi_, "Selected_nuPhi[4]/D");
	t_->Branch("caloMEt", &caloMEt_,"caloMEt/D");
	t_->Branch("t1MEt", &t1MEt_,"t1MEt/D");
	t_->Branch("t1MEtPhi", &t1MEtPhi_,"t1MEtPhi/D");
	t_->Branch("t1SumEt", &t1SumEt_,"t1SumEt/D");
	t_->Branch("pfMEt", &pfMEt_,"pfMEt/D");
	t_->Branch("pfMEtPhi", &pfMEtPhi_,"pfMEtPhi/D");
	t_->Branch("pfSumEt", &pfSumEt_,"pfSumEt/D");
	t_->Branch("tcMEt", &tcMEt_,"tcMEt/D");
	t_->Branch("tcMEtPhi", &tcMEtPhi_,"tcMEtPhi/D");
	t_->Branch("tcSumEt", &tcSumEt_,"tcSumEt/D");
	t_->Branch("caloSumEt", &caloSumEt_,"caloSumEt/D");
	t_->Branch("caloMEt25", &caloMEt25_,"caloMEt25/D");
	t_->Branch("caloMEt30", &caloMEt30_,"caloMEt30/D");
	t_->Branch("caloMEtECAL25", &caloMEtECAL25_,"caloMEtECAL25/D");
	t_->Branch("caloMEtECAL30", &caloMEtECAL30_,"caloMEtECAL30/D");
	t_->Branch("caloMEtPhi", &caloMEtPhi_,"caloMEtPhi/D");
	t_->Branch("caloMEtPhi25", &caloMEtPhi25_,"caloMEtPhi25/D");
	t_->Branch("caloMEtPhi30", &caloMEtPhi30_,"caloMEtPhi30/D");
	t_->Branch("caloMEtPhiECAL25", &caloMEtPhiECAL25_,"caloMEtPhiECAL25/D");
	t_->Branch("caloMEtPhiECAL30", &caloMEtPhiECAL30_,"caloMEtPhiECAL30/D");
	t_->Branch("caloMt", caloMt_,"caloMt[4]/D");
	t_->Branch("caloMt25", caloMt25_,"caloMt25[4]/D");
	t_->Branch("caloMt30", caloMt30_,"caloMt30[4]/D");
	t_->Branch("caloUESumEt", &caloUESumEt_, "caloUESumEt_/D");
	t_->Branch("nHltObj", &nHltObj_, "nHltObj/I");
	t_->Branch("HltObj_pt_", HltObj_pt_,"HltObj_pt_[4]/D");
	t_->Branch("HltObj_eta_", HltObj_eta_,"HltObj_eta_[4]/D");
	t_->Branch("genMEt", &genMEt_,"genMEt/D");
	t_->Branch("genUESumEt", &genUESumEt_, "genUESumEt_/D");

}
// ------------ method called to for each event  ------------
void AnalysisErsatz::analyze(const edm::Event& evt, const edm::EventSetup& es)
{
	caloMEt_ = -99.; caloSumEt_ = -99.; caloUESumEt_ = -99.;
	caloMEt25_ = -99.; caloMEt30_ = -99.;
	caloMEtECAL25_ = -99.; caloMEtECAL30_ =- 99.;
	caloMEtPhi_ = -99.; caloMEtPhi25_ = -99.; caloMEtPhi30_ = -99.;
	caloMEtPhiECAL25_ = -99.; caloMEtPhiECAL30_ =- 99.;
	genMEt_ = -99.; genUESumEt_ = -99.; genMEt25_ = -99.;
	t1MEt_ = -99.; t1MEtPhi_ = -99.; t1SumEt_ = -99.;
	pfMEt_ = -99.; pfMEtPhi_ = -99.; pfSumEt_ = -99.;
	tcMEt_ = -99.; tcMEtPhi_ = -99.; tcSumEt_ = -99.;
	nHltObj_ = -99; nSelElecs_ = -99;
	Boson_pt_ = -99.; Boson_y_ = -99.; Boson_m_ = -99.; Boson_mt_ = -99.; Boson_phi_ = -99.;
	McNu_pt_ = -99.; McNu_eta_ = -99.; McNu_phi_ = -99.;
	McNu_vx_= -99.; McNu_vy_= -99.; McNu_vz_ = -99.;
	McLeptons_dPhi_ = -99.; McLeptons_dEta_ = -99.; McLeptons_dR_ = -99.;
	RndmMcElec_pt_ = -99.; RndmMcElec_eta_ = -99.; RndmMcElec_phi_ = -99.;
	RndmMcElec_Rescaled_pt_ = -99.; RndmMcElec_Rescaled_eta_ = -99.; RndmMcElec_Rescaled_phi_ = -99.;
	RndmMcElecTRIG_pt_ = -99.; RndmMcElecTRIG_eta_ = -99.; RndmMcElecRECO_pt_ = -99.; RndmMcElecRECO_eta_ = -99.;
	OthrMcElec_pt_ = -99.; OthrMcElec_eta_ = -99.; OthrMcElec_phi_ = -99.;
	OthrMcElec_Rescaled_pt_ = -99.; OthrMcElec_Rescaled_eta_ = -99.; OthrMcElec_Rescaled_phi_ = -99.;
	OthrMcElecTRIG_pt_ = -99.; OthrMcElecTRIG_eta_ = -99.; OthrMcElecRECO_pt_ = -99.; OthrMcElecRECO_eta_ = -99.;
	RndmTrig_ = 0; RndmReco_ = 0; OthrTrig_ = 0; OthrReco_ = 0;
	elec_pt25_ = -99.; elec_eta25_= -99.; elec_eta25_= -99.;
	for(int i = 0; i < nEntries_arr_; ++i)
	{
		McElec1_pt_[i] = -99.; McElec1_eta_[i] = -99.;
		McElec3_pt_[i] = -99.; McElec3_eta_[i] = -99.;
		elec_q_[i] = -99.; elec_pt_[i] = -99.; elec_eta_[i]= -99.; elec_phi_[i]= -99.;
                elec_trckIso_[i] = -99.; elec_ecalIso_[i] = -99.; elec_hcalIso_[i] = -99.;
                elec_sIhIh_[i] = -99.; elec_dPhiIn_[i] = -99.; elec_dEtaIn_[i] = -99.;
		elec_e5x5_[i] = -99.; elec_e2x5Max_[i] = -99.; elec_e1x5Max_[i] = -99.;
		elec_hoe_[i] = -99.; elec_eop_[i] = -99.; elec_pin_[i] = -99.; elec_pout_[i] = -99.;
		Selected_nuPt_[i] = -99.; Selected_nuEta_[i] = -99.; Selected_nuPhi_[i] = -99.;
		caloMt_[i] = -99.; caloMt25_[i] = -99.; caloMt30_[i] = -99.;
		genMt_[i] = -99.;
	}
	edm::Handle<reco::GenParticleCollection> pGenPart;
		evt.getByToken(MCTruthCollection_, pGenPart);
        edm::Handle<reco::GsfElectronCollection> pElectrons;
                evt.getByToken(ElectronCollection_, pElectrons);
        edm::Handle<reco::CaloMETCollection> pCaloMEt;
                evt.getByToken(CaloMEtCollection_, pCaloMEt);
	//edm::Handle<reco::METCollection> pT1MEt;
	//	evt.getByToken(T1MEtCollection_, pT1MEt);
	edm::Handle<reco::PFMETCollection> pPfMEt;
		evt.getByToken(PfMEtCollection_, pPfMEt);
	edm::Handle<reco::METCollection> pTcMEt;
		evt.getByToken(TcMEtCollection_, pTcMEt);
	edm::Handle<reco::GenMETCollection> pGenMEt;
		evt.getByToken(GenMEtCollection_, pGenMEt);
        edm::Handle<edm::TriggerResults> pTriggerResults;
                evt.getByToken(TriggerResults_, pTriggerResults);
        const edm::TriggerResults* HltRes = pTriggerResults.product();
       	edm::Handle<trigger::TriggerEvent> pHLT;
        if(HltRes->accept(34) !=0)
	{
                	evt.getByToken(TriggerEvent_, pHLT);
	}
	edm::LogDebug_("analyse","", 143)<<"Have obtained collections."<<std::endl;
	int m = 0; int n = 0; int k = 0;
	bool BosonAnalysed = false; bool nuInEta25 = false; bool nuInEta30 = false;
	double elecEt = 0;
	std::vector<reco::GenParticleCollection::const_iterator> Leptons;
	const reco::GenParticleCollection *McCand = pGenPart.product();
	math::XYZTLorentzVector RndmMcElec, OthrMcElec;
	edm::Service<edm::RandomNumberGenerator> rng;
	math::XYZTLorentzVector RndmMcElec_alt, OthrMcElec_alt, Zboson;
	CLHEP::RandFlat flatDistribution(rng->getEngine(), 0, 2);
	double RandomNum = flatDistribution.fire();
	RndmInt_ = int(RandomNum);
//	std::cout<<"Random Number = "<< RandomNum <<"\t int = "<< int(RandomNum)<<std::endl;
	for(reco::GenParticleCollection::const_iterator McP = McCand->begin(); McP != McCand->end(); ++McP)
	{
		const reco::Candidate* mum = McP->mother();
		if(McP->pdgId()==11)
		{
			edm::LogDebug_("", "", 216)<<"Found electron, mother = "<< mum->pdgId() <<"\t status = "<< McP->status()
					<<"\tpt = "<< McP->pt() <<"\teta = "<< McP->eta();
		}
		if(McP->pdgId()==-11)
		{
			edm::LogDebug_("", "", 221)<<"Found positron, mother = "<< mum->pdgId() <<"\t status = "<< McP->status()
					<<"\tpt = "<< McP->pt() <<"\teta = "<< McP->eta();
	  	}
		if(abs(McP->pdgId())==12)
		{
			edm::LogDebug_("", "", 216)<<"Found neutrino, mother = "<< mum->pdgId() <<"\t status = "<< McP->status()
					<<"\tpt = "<< McP->pt() <<"\teta = "<< McP->eta();
		}
		if(abs(McP->pdgId())==11 && (abs(mum->pdgId()) == 24|| abs(mum->pdgId()) == 23))
		{
//			Leptons.push_back(McP);
			if(!BosonAnalysed)
			{
				Boson_pt_ = mum->pt();
				Boson_y_ = mum->y();
				Boson_phi_ = mum->phi();
				Boson_m_ = mum->mass();
				Boson_mt_ = mum->mt();
				if(abs(mum->pdgId() == 23)) Zboson = mum->p4();
				BosonAnalysed = true;
			}
			McElec3_pt_[k] = McP->pt();
			McElec3_eta_[k] = McP->eta();
			if(k == int(RandomNum))
			{
				RndmMcElec_alt = McP->p4();
			}else OthrMcElec_alt = McP->p4();
			++k;
		}
	  	if(abs(McP->pdgId())==12 && abs(mum->pdgId()) == 24)
		{
			Leptons.push_back(McP);
			edm::LogDebug_("","",328)<<"Pushed neutrino back into Leptons. Leptons.size() = "<< Leptons.size();
			McNu_pt_ = McP->pt();
			McNu_eta_ = McP->eta();
			edm::LogDebug_("","",332)<<"ECAL eta = "<< McNu_ECALeta_;
			McNu_phi_ = McP->phi();
			McNu_vx_ = McP->vx();
			McNu_vy_ = McP->vy();
			McNu_vz_ = McP->vz();
			if(fabs(McNu_eta_) < 2.5) nuInEta25 = true;
			if(fabs(McNu_eta_) < 3.0) nuInEta30 = true;
			++n;
		}
		if(abs(McP->pdgId())==11 && McP->status() == 1 && (abs(mum->pdgId()) == 11))
		{
			Leptons.push_back(McP);
			edm::LogDebug_("","",344)<<"Pushed electron back into Leptons. Leptons.size() = "<< Leptons.size();
			McElec1_pt_[m] = McP->pt();
			McElec1_eta_[m] = McP->eta();
			if(m == int(RandomNum))
			{
				RndmMcElec = McP->p4();
				RndmMcElec_pt_ = McElec1_pt_[m];
				RndmMcElec_eta_ = McElec1_eta_[m];
				RndmMcElec_phi_ = McP->phi();
			}
			else{
				OthrMcElec = McP->p4();
				OthrMcElec_pt_ = McElec1_pt_[m];
				OthrMcElec_eta_ = McElec1_eta_[m];
				OthrMcElec_phi_ = McP->phi();
			}

			elecEt += McP->pt();
			++m;
		}
	}
	edm::LogDebug_("", "", 362)<<"Size of Leptons = "<< Leptons.size();
	McLeptons_dPhi_ = reco::deltaPhi(Leptons[0]->phi(), Leptons[1]->phi());
	McLeptons_dEta_ = Leptons[0]->eta() - Leptons[1]->eta();
	McLeptons_dR_ = reco::deltaR(*Leptons[0], *Leptons[1]);
	edm::LogDebug_("","",369)<<"McLeptons_dR_ = "<< McLeptons_dR_;

	math::XYZTLorentzVector Wboson;
	if(McNu_pt_ < -98.)
	{
        	edm::LogDebug_("", "", 303)<<"Z pt = "<< Zboson.Pt() << "Z boson mass = " << Zboson.M();
	        edm::LogDebug_("","", 307)<<"Z boson in lab frame = ("<<Zboson.Px()<<", "<<Zboson.Py()<<", "
					<<Zboson.Pz()<<", "<<Zboson.E()<<")";

		Wboson.SetCoordinates(Zboson.Px(), Zboson.Py(), Zboson.Pz(), sqrt(Zboson.P2()+(mW_*mW_*Zboson.M2())/(mZ_*mZ_)));
        	edm::LogDebug_("","", 307)<<"W boson in lab frame = ("<<Wboson.Px()<<", "<<Wboson.Py()<<", "
					<<Wboson.Pz()<<", "<<Wboson.E()<<")";
        	ROOT::Math::Boost CoMBoost(Zboson.BoostToCM());
	        edm::LogDebug_("","", 307)<<"RndmElec in lab frame = ("<<RndmMcElec_alt.Px()<<", "<<RndmMcElec_alt.Py()<<", "
					<<RndmMcElec_alt.Pz()<<", "<<RndmMcElec_alt.E()<<")";
	        edm::LogDebug_("","", 307)<<"OthrElec in lab frame = ("<<OthrMcElec_alt.Px()<<", "<<OthrMcElec_alt.Py()<<", "
					<<OthrMcElec_alt.Pz()<<", "<<OthrMcElec_alt.E()<<")";
        	RndmMcElec_alt = CoMBoost(RndmMcElec_alt);
 		OthrMcElec_alt = CoMBoost(OthrMcElec_alt);
        	edm::LogDebug_("","", 307)<<"RndmElec in Z rest frame = ("<<RndmMcElec_alt.Px()<<", "<<RndmMcElec_alt.Py()<<", "
					<<RndmMcElec_alt.Pz()<<", "<<RndmMcElec_alt.E()<<")";
	        edm::LogDebug_("","", 307)<<"OthrElec in Z rest frame = ("<<OthrMcElec_alt.Px()<<", "<<OthrMcElec_alt.Py()<<", "
					<<OthrMcElec_alt.Pz()<<", "<<OthrMcElec_alt.E()<<")";
		RndmMcElec_alt *= mW_/mZ_;
        	OthrMcElec_alt *= mW_/mZ_;

		double E_W = Wboson.E();
		ROOT::Math::Boost BackToLab(Wboson.Px()/E_W, Wboson.Py()/E_W, Wboson.Pz()/E_W);

		RndmMcElec_alt = BackToLab(RndmMcElec_alt);
		RndmMcElec_Rescaled_pt_ = RndmMcElec_alt.Pt();
		RndmMcElec_Rescaled_eta_ = RndmMcElec_alt.Eta();
		RndmMcElec_Rescaled_phi_ = RndmMcElec_alt.Phi();

		OthrMcElec_alt = BackToLab(OthrMcElec_alt);
		OthrMcElec_Rescaled_pt_ = OthrMcElec_alt.Pt();
		OthrMcElec_Rescaled_eta_ = OthrMcElec_alt.Eta();
		OthrMcElec_Rescaled_phi_ = OthrMcElec_alt.Phi();

		math::XYZTLorentzVector sum = OthrMcElec_alt+RndmMcElec_alt;
        	edm::LogDebug_("","", 307)<<"RndmElec back in lab frame = ("<<RndmMcElec_alt.Px()<<", "<<RndmMcElec_alt.Py()<<", "
						<<RndmMcElec_alt.Pz()<<", "<<RndmMcElec_alt.E()<<")";
	        edm::LogDebug_("","", 307)<<"OthrElec back in lab frame = ("<<OthrMcElec_alt.Px()<<", "<<OthrMcElec_alt.Py()<<", "
						<<OthrMcElec_alt.Pz()<<", "<<OthrMcElec_alt.E()<<")";
	        edm::LogDebug_("","", 307)<<"OthrElec +RndmElec = ("<<sum.Px()<<", "<<sum.Py()<<", "
						<<sum.Pz()<<", "<<sum.E()<<")";
        }else{
		edm::LogDebug_("","", 416)<<"McNu_pt_ = "<<McNu_pt_;
		RndmMcElec_Rescaled_pt_ = RndmMcElec_pt_;
		edm::LogDebug_("","",416)<<" RndmMcElec_Rescaled_pt_ = "<< RndmMcElec_Rescaled_pt_;
		OthrMcElec_Rescaled_pt_ = OthrMcElec_pt_;
	}
	//TriggerNames_.init(*HltRes);

	edm::LogDebug_("","", 420)<<"HltRes->accept() = "<< HltRes->accept(34);
        if(HltRes->accept(34) ==0) edm::LogError("")<<"Event did not pass HLT path 34, assumed to be "<< TriggerName_ <<"!";
	unsigned int fId = 999;
	if(HltRes->accept(34) !=0)
	{
	        fId = pHLT->filterIndex(TriggerPath_); // something wrong with this step
		edm::LogDebug_("","",426)<<"fId = pHLT->filterIndex("<< TriggerPath_<<") = "<< fId;
        	const trigger::Keys& ring = pHLT->filterKeys(fId);
	        const trigger::TriggerObjectCollection& HltObjColl = pHLT->getObjects();
		nHltObj_ = ring.size();
	        for(int k = 0; k < nHltObj_; ++k)
        	{
        		const trigger::TriggerObject& HltObj = HltObjColl[ring[k]];
                        if(reco::deltaR(RndmMcElec, HltObj) < 0.1) RndmTrig_ = 1;
                        if(reco::deltaR(OthrMcElec, HltObj) < 0.1) OthrTrig_ = 1;
			if(k < 4)
			{
				HltObj_pt_[k] = HltObj.pt();
				HltObj_eta_[k] = HltObj.eta();
			}
		}
	}
	if(RndmTrig_ != 0)
	{
		RndmMcElecTRIG_pt_ = RndmMcElec_pt_;
		RndmMcElecTRIG_eta_ = RndmMcElec_eta_;
	}
	if(OthrTrig_ != 0)
	{
		OthrMcElecTRIG_pt_ = OthrMcElec_pt_;
		OthrMcElecTRIG_eta_ = OthrMcElec_eta_;
	}
	const reco::GenMETCollection* genMEtCollection = pGenMEt.product();
	reco::GenMETCollection::const_iterator genMEt = genMEtCollection->begin();
	genMEt_ = genMEt->pt();
	genUESumEt_ = genMEt->sumEt() - elecEt;
//	std::cout<<"genMEt->sumEt() - elecEt = "<< genMEt->sumEt()<<" - "<< elecEt <<" = "<< genUESumEt_;

        const reco::CaloMETCollection* caloMEtCollection = pCaloMEt.product();
        reco::CaloMETCollection::const_iterator met = caloMEtCollection->begin();
        caloMEt_ = met->pt();
	edm::LogDebug_("","",462)<<"caloMEt_ = "<< caloMEt_;
	caloMEtPhi_ = met->phi();
	caloSumEt_ = met->sumEt();

        //const reco::METCollection* t1MEtCollection = pT1MEt.product();
        //reco::METCollection::const_iterator t1met = t1MEtCollection->begin();
        //t1MEt_ = t1met->pt();
	//edm::LogDebug_("","",462)<<"t1MEt_ = "<< t1MEt_;
	//t1MEtPhi_ = t1met->phi();
	//t1SumEt_ = t1met->sumEt();

        const reco::PFMETCollection* pfMEtCollection = pPfMEt.product();
        reco::PFMETCollection::const_iterator pfmet = pfMEtCollection->begin();
        pfMEt_ = pfmet->pt();
	edm::LogDebug_("","",462)<<"pfMEt_ = "<< pfMEt_;
	pfMEtPhi_ = pfmet->phi();
	pfSumEt_ = pfmet->sumEt();

        const reco::METCollection* tcMEtCollection = pTcMEt.product();
        reco::METCollection::const_iterator tcmet = tcMEtCollection->begin();
        tcMEt_ = tcmet->pt();
	edm::LogDebug_("","",462)<<"tcMEt_ = "<< tcMEt_;
	tcMEtPhi_ = tcmet->phi();
	tcSumEt_ = tcmet->sumEt();

	if(fabs(McNu_ECALeta_) < 2.5)
	{
		caloMEtECAL25_ = met->pt();
		caloMEtPhiECAL25_ = met->phi();
	}
	if(fabs(McNu_ECALeta_) < 3.0)
	{
		caloMEtECAL30_ = met->pt();
		caloMEtPhiECAL30_ = met->phi();
	}
	if(nuInEta25)
	{
		genMEt25_ = genMEt->pt();
		caloMEt25_ = met->pt();
		caloMEtPhi25_ = met->phi();
	}
	if(nuInEta30){
		caloMEt30_ = met->pt();
		caloMEtPhi30_ = met->phi();
	}

	std::vector<reco::GsfElectronRef> UniqueElectrons = uniqueElectronFinder(pElectrons);
	caloUESumEt_ = met->sumEt();
	for(std::vector<reco::GsfElectronRef>::const_iterator Relec = UniqueElectrons.begin(); Relec != UniqueElectrons.end(); ++Relec)
	{
		reco::GsfElectronRef elec = *Relec;
		math::XYZTLorentzVector sc = PhysicsVector(met->vertex(), *(elec->superCluster()));
		if(reco::deltaR(RndmMcElec, *elec) < 0.1) RndmReco_ = 1;
		if(reco::deltaR(OthrMcElec, *elec) < 0.1) OthrReco_ = 1;
		caloUESumEt_ -= sc.Pt();
	}
	if(RndmReco_ != 0)
	{
		RndmMcElecRECO_pt_ = RndmMcElec_pt_;
		RndmMcElecRECO_eta_ = RndmMcElec_eta_;
	}
	if(OthrReco_ != 0)
	{
		OthrMcElecRECO_pt_ = OthrMcElec_pt_;
		OthrMcElecRECO_eta_ = OthrMcElec_eta_;
	}
	edm::LogDebug_("analyse","", 230)<<"Analysed UE information"<<std::endl;
        if(HltRes->accept(34) != 0)
	{
		std::vector<reco::GsfElectronRef> SelectedElectrons = electronSelector(UniqueElectrons, pHLT, fId,  CutVector_);
		nSelElecs_ = SelectedElectrons.size();
		m = 0;
		for(std::vector<reco::GsfElectronRef>::const_iterator Relec = SelectedElectrons.begin(); Relec != SelectedElectrons.end(); ++Relec)
		{
			reco::GsfElectronRef elec = *Relec;
			if(elec->pt() > CutVector_[0])
			{
				elec_q_[m] = elec->charge();
				elec_pt_[m] = elec->pt();
				elec_eta_[m] = elec->eta();
				elec_phi_[m] = elec->phi();
                        	elec_trckIso_[m] = elec->isolationVariables03().tkSumPt;
                        	elec_ecalIso_[m] = elec->isolationVariables04().ecalRecHitSumEt;
                        	elec_hcalIso_[m] = elec->isolationVariables04().hcalDepth1TowerSumEt
						+ elec->isolationVariables04().hcalDepth2TowerSumEt;
				elec_sIhIh_[m] = elec->scSigmaIEtaIEta();
				elec_dPhiIn_[m] = elec->deltaPhiSuperClusterTrackAtVtx();
				elec_dEtaIn_[m] = elec->deltaEtaSuperClusterTrackAtVtx();
				elec_e5x5_[m] = elec->scE5x5();
				elec_e2x5Max_[m] = elec->scE2x5Max();
				elec_e2x5Max_[m] = elec->scE1x5();
				elec_hoe_[m] = elec->hadronicOverEm();
				elec_eop_[m] = elec->eSuperClusterOverP();
				elec_pin_[m] = elec->trackMomentumAtVtx().R();
				elec_pout_[m] = elec->trackMomentumOut().R();

				Selected_nuPt_[m] = McNu_pt_;
				Selected_nuEta_[m] = McNu_eta_;
				Selected_nuPhi_[m] = McNu_phi_;
				caloMt_[m] = sqrt(2.*elec->pt()*met->pt()*(1-cos(reco::deltaPhi(elec->phi(), met->phi()))));
				if(nuInEta25)
				{
					caloMt25_[m] = caloMt_[m];
					elec_pt25_ = elec->pt();
					elec_eta25_ = elec->eta();
					elec_phi25_ = elec->phi();
				}
				if(nuInEta30) caloMt30_[m] = caloMt_[m];
				++m;
			}
		}
	}

	edm::LogDebug_("analyse","", 248)<<"Analysed final selection information"<<std::endl;
	t_->Fill();
}



// ------------ method called once each job just after ending the event loop  ------------
void AnalysisErsatz::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(AnalysisErsatz);
*/
