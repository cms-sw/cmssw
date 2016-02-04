#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TF1.h" 
#include "TRandom3.h"	
#include "TString.h"
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;

void PerformAnalysis(TString process, double eventweight, TString datapath)//, ofstream& results) //int main()
{
	cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%\%" << endl;
	cout << "\% " << "Perform Analysis" << endl; 
	cout << "\% " << process << endl;
	cout << "\% " << "Event Weight = " << eventweight << endl;
	cout << "\% " << "Data Path = " << datapath << endl;
	cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%\%" << endl;

	// Declare electron cut value variables	
	double cMEt = 30.;
	double cPt = 30.;
	double cECALiso_EB = 4.2, cECALiso_EE = 3.4;
	double cHCALiso_EB = 2.0, cHCALiso_EE = 1.3;
	double cTrackiso_EB = 2.2, cTrackiso_EE = 1.1;
	double cDeltaEta_EB = 0.0040, cDeltaEta_EE = 0.0066;
	double cDeltaPhi_EB = 0.025, cDeltaPhi_EE = 0.020;
	double csIhIh_EB = 0.0099, csIhIh_EE = 0.0280;
	// Declare neutrino cut value variables
	double cHCAL = 6.2;
	double cHCALEt = 12;
	double cf1x5 = 0.83, cf2x5 = 0.93;
	int celecmatch = 0;
	double cnusIhIh = 0.027;

	cout << "Cut Values:" << endl;	

	cout << "MEt cut " << cMEt << endl;
	
	cout << "Electron selection cuts:" << endl;
	cout << "Pt cut " << cPt << endl;
	cout << "ECAL Isolation cut (EB) " << cECALiso_EB << endl;
	cout << "ECAL Isolation cut (EE) " << cECALiso_EE << endl;
	cout << "HCAL Isolation cut (EB) " << cHCALiso_EB << endl;
	cout << "HCAL Isolation cut (EE) " << cHCALiso_EE << endl;
	cout << "Track Isolation cut (EB) " << cTrackiso_EB << endl;
	cout << "Track Isolation cut (EE) " << cTrackiso_EE << endl;
	cout << "Delta Eta cut (EB) " << cDeltaEta_EB << endl;
	cout << "Delta Eta cut (EE) " << cDeltaEta_EE << endl;
	cout << "Delta Phi cut (EB) " << cDeltaPhi_EB << endl;
	cout << "Delta Phi cut (EE) " << cDeltaPhi_EE << endl;
	cout << "Sigma iEta iEta cut (EB) " << csIhIh_EB << endl;
	cout << "Sigma iEta iEta cut (EE) " << csIhIh_EE  << endl;
	
	cout << "Probe selection cuts:" << endl;
	cout << "HCAL Energy cut " << cHCAL << endl;
	cout << "HCAL Transverse Energy cut " << cHCALEt << endl;
	cout << "Fraction of energy in 1x5 cut " << cf1x5 << endl;
	cout << "Fraction of energy in 2x5 cut " << cf2x5 << endl;
	cout << "Require electron match " << celecmatch << endl;
	cout << "Sigma iEta iEta cut " << cnusIhIh << endl;
	
	cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%\%" << endl;
	// Import probe selection efficiency weights
	double nueff[345];
	ifstream weightsin;
	weightsin.open("EtaWeights.txt", ifstream::in);

	for(int eta=0; eta < 345; ++eta)
	{
		double weight;
		weightsin >> weight;
		//cout << eta << "\t" << weight << endl; 
		nueff[eta] = weight;
	}
	weightsin.close();
	cout << "Imported probe selection efficiencies" << endl;	

	TString OutFileName = process+".root";
	TFile* outfile = TFile::Open(OutFileName, "recreate");

	cout << "Created output file \"" << OutFileName << "\"" << endl;

	TH1F* h_dataWMEt_pass_EB = new TH1F("dataWMEt_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_dataWMEt_pass_EE = new TH1F("dataWMEt_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_dataWMEt_fail_EB = new TH1F("dataWMEt_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_dataWMEt_fail_EE = new TH1F("dataWMEt_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_mcWMEtin_pass_EB = new TH1F("mcWMEtin_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWMEtin_pass_EE = new TH1F("mcWMEtin_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWMEtin_fail_EB = new TH1F("mcWMEtin_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWMEtin_fail_EE = new TH1F("mcWMEtin_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWMEtout_pass_EB = new TH1F("mcWMEtout_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWMEtout_pass_EE = new TH1F("mcWMEtout_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWMEtout_fail_EB = new TH1F("mcWMEtout_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWMEtout_fail_EE = new TH1F("mcWMEtout_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	
	TH1F* h_ErsatzMEt_pass_EB = new TH1F("ErsatzMEt_pass_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_pass_EE = new TH1F("ErsatzMEt_pass_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fail_EB = new TH1F("ErsatzMEt_fail_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fail_EE = new TH1F("ErsatzMEt_fail_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_ErsatzMEt_probept = new TH1F("ErsatzMEt_probept","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_uncorr = new TH1F("ErsatzMEt_uncorr","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fetacorr = new TH1F("ErsatzMEt_fetacorr","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fbremcorr = new TH1F("ErsatzMEt_fbremcorr","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_ErsatzMEt_pass_EB_peakfit = new TH1F("ErsatzMEt_pass_EB_peakfit","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_pass_EE_peakfit = new TH1F("ErsatzMEt_pass_EE_peakfit","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WMEt_pass_EB_peakfit = new TH1F("WMEt_pass_EB_peakfit","W MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WMEt_pass_EE_peakfit = new TH1F("WMEt_pass_EE_peakfit","W MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_ErsatzMEt_pass_EB_shifted = new TH1F("ErsatzMEt_pass_EB_shifted","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_pass_EE_shifted = new TH1F("ErsatzMEt_pass_EE_shifted","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fail_EB_shifted = new TH1F("ErsatzMEt_fail_EB_shifted","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fail_EE_shifted = new TH1F("ErsatzMEt_fail_EE_shifted","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	
	TH1F* h_acceptance_correction_pass_EB = new TH1F("acceptacne_correction_pass_EB", "Acceptance Correction pass EB;#slash{E_{T}};Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_pass_EE = new TH1F("acceptacne_correction_pass_EE", "Acceptance Correction pass EE;#slash{E_{T}};Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_fail_EB = new TH1F("acceptacne_correction_fail_EB", "Acceptance Correction fail EB;#slash{E_{T}};Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_fail_EE = new TH1F("acceptacne_correction_fail_EE", "Acceptance Correction fail EE;#slash{E_{T}};Acceptance Correction", 100, 0., 100.);

	TH1F* h_ErsatzMEt_pass_EB_corr = new TH1F("ErsatzMEt_pass_EB_corr","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_pass_EE_corr = new TH1F("ErsatzMEt_pass_EE_corr","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fail_EB_corr = new TH1F("ErsatzMEt_fail_EB_corr","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fail_EE_corr = new TH1F("ErsatzMEt_fail_EE_corr","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	
	cout << "Declared Histograms" << endl;	

	vector<double> ErsatzMEt_pass_EB;
	vector<double> ErsatzMEt_pass_EE;
	vector<double> ErsatzMEt_fail_EB;
	vector<double> ErsatzMEt_fail_EE;
	
	vector<double> Weight_pass_EB;
	vector<double> Weight_pass_EE;
	vector<double> Weight_fail_EB;
	vector<double> Weight_fail_EE;

	TRandom3 r;
	
	TString WFileName = datapath+"WenuTrue.root";
	TFile *fileW = TFile::Open(WFileName);
	cout << "Opened W Monte Carlo file" << endl;
	TTree *t = (TTree*) fileW->Get("analyse/AnalysisData");
	cout << "Got W TTree" << endl;

	long nEntries = t->GetEntriesFast();
	cout << "Total number of W events = " << nEntries << endl;

	double elec_pt_W[4], elec_eta_W[4];
        double elec_trckIso_W[4]/*, elec_EcalIso_W[4], elec_HcalIso_W[4]*/;
        double elec_sigIhIh_W[4]/*, elec_dPhi_W[4], elec_dEta_W[4]*/;

	double nu_pt_W, nu_eta_W, nu_ECALeta_W, nu_phi_W;

	//double McW_pt, McW_phi; 
	double CaloMEt_W;//, CaloMEt25, CaloMEt30;
	//double CaloMt[4];// CaloMt25[4], CaloMt30[4];

//	TBranch* bMcW_pt = t->GetBranch("Boson_pt");
//      bMcW_pt->SetAddress(&McW_pt);   
//	TBranch* bMcW_phi = t->GetBranch("Boson_phi");
//      bMcW_phi->SetAddress(&McW_phi);   
//	TBranch* bNSelElecs = t->GetBranch("nSelElecs");
//	bNSelElecs->SetAddress(&nSelElecs);
	TBranch* bElec_eta = t->GetBranch("elec_eta");
	bElec_eta->SetAddress(&elec_eta_W);
	TBranch* bElec_pt = t->GetBranch("elec_pt");
	bElec_pt->SetAddress(&elec_pt_W);

        TBranch* bTag_sIhIh_W = t->GetBranch("elec_sIhIh");
        bTag_sIhIh_W->SetAddress(&elec_sigIhIh_W);
	//TBranch* bTag_dPhi = t->GetBranch("elec_dPhiIn");
        //bTag_dPhi->SetAddress(&elec_dPhi_W);
        //TBranch* bTag_dEta = t->GetBranch("elec_dEtaIn");
        //bTag_dEta->SetAddress(&elec_dEta_W);
        TBranch* bTag_tIso_W = t->GetBranch("elec_isoTrack");
        bTag_tIso_W->SetAddress(&elec_trckIso_W);
        //TBranch* bTag_eIso = t->GetBranch("elec_isoEcal");
        //bTag_eIso->SetAddress(&elec_EcalIso_W);
        //TBranch* bTag_hIso = t->GetBranch("elec_isoHcal");
        //bTag_hIso->SetAddress(&elec_HcalIso_W);

	//      TBranch* = t->GetBranch("");
//      ->SetAddress(&);   
	TBranch* bMcNu_pt = t->GetBranch("McNu_pt");
	bMcNu_pt->SetAddress(&nu_pt_W);
	TBranch* bMcNu_phi = t->GetBranch("McNu_phi");
	bMcNu_phi->SetAddress(&nu_phi_W);
	TBranch* bMcNu_eta = t->GetBranch("McNu_eta");
	bMcNu_eta->SetAddress(&nu_eta_W);
	TBranch* bMcNu_ECALeta = t->GetBranch("McNu_ECALeta");
	bMcNu_ECALeta->SetAddress(&nu_ECALeta_W);
	TBranch* bCalo_MEt = t->GetBranch("caloMEt");
	bCalo_MEt->SetAddress(&CaloMEt_W);
//	TBranch* bCalo_MEt25 = t->GetBranch("caloMEt25");
//	bCalo_MEt25->SetAddress(&CaloMEt25);
//	TBranch* bCalo_MEt30 = t->GetBranch("caloMEt30");
//	bCalo_MEt30->SetAddress(&CaloMEt30);
//	TBranch* bCalo_Mt = t->GetBranch("caloMt");
//	bCalo_Mt->SetAddress(&CaloMt);
	cout << "Set up branches" << endl;

	//long nentries = t->GetEntries();
	//int index = 0;
	int aaa = 0, bbb = 0, ccc = 0, ddd = 0;
	for(long i = 0; i < nEntries; ++i)
	{
		if(i%100000 == 0) cout <<"Analysing event "<< i << endl;
		//if (i == ChosenEvents[index])
		//{
		//index++;
		//bool iIsChosen = (i == ChosenEvents[index]);
		t->GetEntry(i);
		if(elec_pt_W[0] > cPt)
		{
			bool pass_e_cuts = false;
			bool pass_trkiso_cut = false;
			bool inBarrel = false;
			bool inEndcap = false;
			if(fabs(elec_eta_W[0]) < 1.4442)
			{
				pass_e_cuts = (elec_sigIhIh_W[0] < csIhIh_EB); 
				pass_trkiso_cut = (elec_trckIso_W[0] < cTrackiso_EB);
				inBarrel = true;
			}else if(fabs(elec_eta_W[0]) < 2.5)
			{	
				pass_e_cuts = (elec_sigIhIh_W[0] < csIhIh_EE); 
				pass_trkiso_cut = (elec_trckIso_W[0] < cTrackiso_EE);
				inEndcap = true;
			}
			if(pass_e_cuts)
			{
				if(pass_trkiso_cut)
				{ 
					if(inBarrel) 
					{
						h_dataWMEt_pass_EB->Fill(CaloMEt_W); 
						h_WMEt_pass_EB_peakfit->Fill(CaloMEt_W);
						aaa++;
						if(fabs(nu_eta_W) < 2.5) 
						{
							h_mcWMEtin_pass_EB->Fill(CaloMEt_W);
						}else{ 
							h_mcWMEtout_pass_EB->Fill(CaloMEt_W); 
						} 
					}
					if(inEndcap) 
					{ 
						h_dataWMEt_pass_EE->Fill(CaloMEt_W);
						h_WMEt_pass_EE_peakfit->Fill(CaloMEt_W);
						bbb++;
						if(fabs(nu_eta_W) < 2.5) 
						{
							h_mcWMEtin_pass_EE->Fill(CaloMEt_W);
						}else{ 
							h_mcWMEtout_pass_EE->Fill(CaloMEt_W); 
						} 
					}
				}else
				{
					if(inBarrel) 
					{ 
						h_dataWMEt_fail_EB->Fill(CaloMEt_W);
						ccc++;
						if(fabs(nu_eta_W) < 2.5) 
						{
							h_mcWMEtin_fail_EB->Fill(CaloMEt_W);
						}else{ 
							h_mcWMEtout_fail_EB->Fill(CaloMEt_W); 
						} 
					}
					if(inEndcap) 
					{ 
						h_dataWMEt_fail_EE->Fill(CaloMEt_W);
						ddd++;
						if(fabs(nu_eta_W) < 2.5) 
						{
							h_mcWMEtin_fail_EE->Fill(CaloMEt_W);
						}else{ 
							h_mcWMEtout_fail_EE->Fill(CaloMEt_W); 
						} 
					}
				}
			}
		}
	}
	fileW->Close();
	cout << "Closed W Monte Carlo file" << endl;

	cout << "Number of W events passing selection cuts = " << aaa+bbb+ccc+ddd << endl;	
	cout << "Number Pass EB = " << aaa << endl;	
	cout << "Number Pass EE = " << bbb << endl;	
	cout << "Number Fail EB = " << ccc << endl;	
	cout << "Number Fail EE = " << ddd << endl;	
        
	TString ErsatzFileName = datapath+process+".root";
	TFile *fileZ = TFile::Open(ErsatzFileName);
        cout << "Opened Ersatz data file" << endl;
	t = (TTree*) fileZ->Get("ErsatzMEt/ErsatzMEt");
	cout << "Got ersatz TTree" << endl;	
	nEntries = t->GetEntries();
	cout << "Total number of ersatz events = " << nEntries << endl;	

	int nErNu;
	double tag_pt[4], tag_eta[4], tag_phi[4], probe_pt[4], probe_eta[4], probe_phi[4];
	double ErsatzV1MEt[4], ErsatzV1aMEt[4], ErsatzV1bMEt[4];
	double elec_trkIso[4], elec_ECALIso[4], elec_HCALIso[4];
	double elec_sigIhIh[4], elec_dPhi[4], elec_dEta[4];
	double tag_rescPt[4], mesc[4];
	double nu_e1x5[4], nu_e2x5[4], nu_e5x5[4], nu_sigIhIh[4];
	double nu_HCALEt[4], nu_HCAL[4], nu_trckIso[4];
	double caloMEt;
	//int nu_elec[4];
	
	TBranch* bNum_ErNu = t->GetBranch("nProbes");
	bNum_ErNu->SetAddress(&nErNu);
	//W selected electron properties
	TBranch* bTag_eta = t->GetBranch("tag_eta");
	bTag_eta->SetAddress(&tag_eta);
	TBranch* bTag_pt = t->GetBranch("tag_pt");
	bTag_pt->SetAddress(&tag_pt);
	TBranch* bTag_phi = t->GetBranch("tag_phi");
	bTag_phi->SetAddress(&tag_phi);
	TBranch* bTag_rescPt = t->GetBranch("tag_rescPt");
	bTag_rescPt->SetAddress(&tag_rescPt);
//	TBranch* = t->GetBranch("");
//	->SetAddress(&);
	TBranch* bTag_sIhIh = t->GetBranch("tag_sIhIh");
	bTag_sIhIh->SetAddress(&elec_sigIhIh);
	TBranch* bTag_dPhi = t->GetBranch("tag_dPhiIn");
	bTag_dPhi->SetAddress(&elec_dPhi);
	TBranch* bTag_dEta = t->GetBranch("tag_dEtaIn");
	bTag_dEta->SetAddress(&elec_dEta);
	TBranch* bTag_tIso = t->GetBranch("tag_isoTrack");
	bTag_tIso->SetAddress(&elec_trkIso);
	TBranch* bTag_eIso = t->GetBranch("tag_isoEcal");
	bTag_eIso->SetAddress(&elec_ECALIso);
	TBranch* bTag_hIso = t->GetBranch("tag_isoHcal");
	bTag_hIso->SetAddress(&elec_HCALIso);
	//ersatz neutrino properties
	TBranch* bProbe_pt = t->GetBranch("probe_pt");
	bProbe_pt->SetAddress(&probe_pt);
	TBranch* bProbe_eta = t->GetBranch("probe_eta");
	bProbe_eta->SetAddress(&probe_eta);
	TBranch* bProbe_phi = t->GetBranch("probe_phi");
	bProbe_phi->SetAddress(&probe_phi);
	//TBranch* bProbe_elec = t->GetBranch("probe_elecMatch");
	//bProbe_elec->SetAddress(&nu_elec);
	TBranch* bProbe_trckIso = t->GetBranch("probe_isoTrack");
	bProbe_trckIso->SetAddress(&nu_trckIso);
	//TBranch* bProbe_ECALIso = t->GetBranch("probe_isoECAL");
	//bProbe_ECALIso->SetAddress(&nu_ECALIso);
	//TBranch* bProbe_HCALIso = t->GetBranch("probe_isoHCAL");
	//bProbe_HCALIso->SetAddress(&nu_HCALIso);
	TBranch* bProbe_sIhIh = t->GetBranch("probe_sIhIh");
	bProbe_sIhIh->SetAddress(&nu_sigIhIh);
	//TBranch* bProbe_DeltaEta = t->GetBranch("probe_DeltaEta");
	//bProbe_DeltaEta->SetAddress(&nu_DeltaEta);
	//TBranch* bProbe_DeltaPhiIso = t->GetBranch("probe_DeltaPhi");
	//bProbe_DeltaPhi->SetAddress(&nu_DeltaPhi);
	TBranch* bProbe_e1x5 = t->GetBranch("probe_e1x5Max");
	bProbe_e1x5->SetAddress(&nu_e1x5);
	TBranch* bProbe_e2x5 = t->GetBranch("probe_e2x5Max");
	bProbe_e2x5->SetAddress(&nu_e2x5);
	TBranch* bProbe_e5x5 = t->GetBranch("probe_e5x5");
	bProbe_e5x5->SetAddress(&nu_e5x5);
	TBranch* bProbe_HCAL = t->GetBranch("probe_HcalE015");
	bProbe_HCAL->SetAddress(&nu_HCAL);
	TBranch* bProbe_HCALEt = t->GetBranch("probe_HcalEt015");
	bProbe_HCALEt->SetAddress(&nu_HCALEt);
	//Ersatz MEt results
	TBranch* bErsatzV1_MEt = t->GetBranch("ErsatzV1CaloMEt");
	bErsatzV1_MEt->SetAddress(&ErsatzV1MEt);
	TBranch* bErsatzV1a_MEt = t->GetBranch("ErsatzV1aCaloMEt");
	bErsatzV1a_MEt->SetAddress(&ErsatzV1aMEt);
	TBranch* bErsatzV1b_MEt = t->GetBranch("ErsatzV1bCaloMEt");
	bErsatzV1b_MEt->SetAddress(&ErsatzV1bMEt);
	TBranch* bMesc = t->GetBranch("ErsatzV1_Mesc");
	bMesc->SetAddress(&mesc); 
	TBranch* bCaloMEt = t->GetBranch("recoCaloMEt");
	bCaloMEt->SetAddress(&caloMEt);
	cout << "Set up Branches" << endl;

	aaa=0, bbb=0, ccc=0, ddd=0;
	for(int i=0; i < nEntries; ++i)
	{
		if(i%100000 == 0) cout <<"Processing event "<< i << endl;
		t->GetEntry(i);
		for(int j = 0; j < nErNu; ++j)
		{ 
			bool passEtCut = false;
			if(tag_rescPt[j] > cPt) passEtCut = true;
			/*
			if(process == "Zee" || process == "BCtoE_30to80" || process == "BCtoE_80to170"){
				if(tag_rescPt[j] > cPt) passEtCut = true;
			}else{
				if(tag_pt[j] > (91.188/80.398)*cPt) passEtCut = true;
			}
			*/
			if(passEtCut)
			{
                                if(fabs(mesc[j]-91.1876) < 21.1876)
				{
					bool pass_e_cuts = false;
					bool pass_trkiso_cut = false;
					bool inBarrel = false;
					bool inEndcap = false;
					if(fabs(tag_eta[j])<1.4442)
					{
						pass_e_cuts = (elec_ECALIso[j] < cECALiso_EB && elec_HCALIso[j] < cHCALiso_EB
								&& elec_sigIhIh[j] < csIhIh_EB && elec_dPhi[j] < cDeltaPhi_EB
								&& elec_dEta[j] < cDeltaEta_EB);
						pass_trkiso_cut = (elec_trkIso[j] < cTrackiso_EB);
						inBarrel = true;
					}else if(fabs(tag_eta[j] < 2.5))
					{
						pass_e_cuts = (elec_ECALIso[j] < cECALiso_EE && elec_HCALIso[j] < cHCALiso_EE
								&& elec_sigIhIh[j] < csIhIh_EE && elec_dPhi[j] < cDeltaPhi_EE
								&& elec_dEta[j] < cDeltaEta_EE);
						pass_trkiso_cut = (elec_trkIso[j] < cTrackiso_EE);
						inEndcap = true;
					}
					if(pass_e_cuts)
					{
						bool pass_nu_cuts = false;
						double f1x5 = nu_e1x5[j]/nu_e5x5[j];
						double f2x5 = nu_e2x5[j]/nu_e5x5[j];
						if(fabs(probe_eta[j]) < 1.4442)
						{
							pass_nu_cuts = (nu_HCAL[j] < cHCAL && (f1x5 > cf1x5 || f2x5 > cf2x5)
									/*&& nu_elec[j] == celecmatch*/);
						}else if(fabs(probe_eta[j] < 2.5)){
							pass_nu_cuts = (nu_HCALEt[j] < cHCALEt && nu_sigIhIh[j] < cnusIhIh
									/*&& nu_elec[j] == celecmatch*/);
						}
						if(pass_nu_cuts)
						{
							int EtaInt = int((probe_eta[j] + 3.)/0.01739);
							double weight = eventweight/nueff[EtaInt];
							if(pass_trkiso_cut)
							{
								if(inBarrel) 
								{
									aaa++;	
									ErsatzMEt_pass_EB.push_back(ErsatzV1bMEt[j]);
									Weight_pass_EB.push_back(weight);
									h_ErsatzMEt_pass_EB->Fill(ErsatzV1bMEt[j], weight);
									h_ErsatzMEt_pass_EB_peakfit->Fill(ErsatzV1bMEt[j], weight);
									h_ErsatzMEt_probept->Fill(probe_pt[j], weight);
									h_ErsatzMEt_uncorr->Fill(ErsatzV1MEt[j], weight);
									h_ErsatzMEt_fetacorr->Fill(ErsatzV1aMEt[j], weight);
									h_ErsatzMEt_fbremcorr->Fill(ErsatzV1bMEt[j], weight);
								}
								if(inEndcap)
								{
									bbb++;
									ErsatzMEt_pass_EE.push_back(ErsatzV1bMEt[j]);
									Weight_pass_EE.push_back(weight);
									h_ErsatzMEt_pass_EE->Fill(ErsatzV1bMEt[j], weight);
									h_ErsatzMEt_pass_EE_peakfit->Fill(ErsatzV1bMEt[j], weight);
									h_ErsatzMEt_probept->Fill(probe_pt[j], weight);
									h_ErsatzMEt_uncorr->Fill(ErsatzV1MEt[j], weight);
									h_ErsatzMEt_fetacorr->Fill(ErsatzV1aMEt[j], weight);
									h_ErsatzMEt_fbremcorr->Fill(ErsatzV1bMEt[j], weight);
								}	
							}else{
								if(inBarrel)
								{
									ccc++;
									ErsatzMEt_fail_EB.push_back(ErsatzV1bMEt[j]);
									Weight_fail_EB.push_back(weight);
									h_ErsatzMEt_fail_EB->Fill(ErsatzV1bMEt[j], weight);
								}
								if(inEndcap)
								{
									ddd++;
									ErsatzMEt_fail_EE.push_back(ErsatzV1bMEt[j]);
									Weight_fail_EE.push_back(weight);
									h_ErsatzMEt_fail_EE->Fill(ErsatzV1bMEt[j], weight);
								}	
							}
						}		
					}
				}
			}
		}
	}
	fileZ->Close();
	cout << "Closed Ersatz data file" << endl;

	cout << "Number of events passing selection cuts = " << aaa+bbb+ccc+ddd << endl;
	cout << "Number Pass EB = " << aaa << endl;	
	cout << "Number Pass EE = " << bbb << endl;	
	cout << "Number Fail EB = " << ccc << endl;	
	cout << "Number Fail EE = " << ddd << endl;	
        
	cout << "Apply shift correction ..." << endl;

	int maxbin;

	h_WMEt_pass_EB_peakfit->Scale(1./h_WMEt_pass_EB_peakfit->Integral(0,100));
	maxbin = h_WMEt_pass_EB_peakfit->GetMaximumBin();
	TF1 peakW_EB = TF1("peakW_EB", "gaus", maxbin-4, maxbin+4); 
	h_WMEt_pass_EB_peakfit->Fit("peakW_EB", "MR");
	cout << "W MEt maximum bin = " << maxbin << "\tPeak of Gaussian = " << peakW_EB.GetParameter(1) << endl;
	h_WMEt_pass_EB_peakfit->Draw();
	
	h_ErsatzMEt_pass_EB_peakfit->Scale(1./h_ErsatzMEt_pass_EB_peakfit->Integral(0,100));
	maxbin = h_ErsatzMEt_pass_EB_peakfit->GetMaximumBin();
	TF1 peakZ_EB = TF1("peakZ_EB", "gaus", maxbin-4, maxbin+4); 
	h_ErsatzMEt_pass_EB_peakfit->Fit("peakZ_EB", "MR");
	cout << "Ersatz maximum bin = " << maxbin << "\tPeak of Gaussian = " << peakZ_EB.GetParameter(1) << endl; 
	h_ErsatzMEt_pass_EB_peakfit->Draw();

	double shift_EB = peakW_EB.GetParameter(1) - peakZ_EB.GetParameter(1);
	cout << "EB Shift = " << shift_EB << endl;

	h_WMEt_pass_EE_peakfit->Scale(1./h_WMEt_pass_EE_peakfit->Integral(0,100));
	maxbin = h_WMEt_pass_EE_peakfit->GetMaximumBin();
	TF1 peakW_EE = TF1("peakW_EE", "gaus", maxbin-4, maxbin+4); 
	h_WMEt_pass_EE_peakfit->Fit("peakW_EE", "MR");
	cout << "W MEt maximum bin = " << maxbin << "\tPeak of Gaussian = " << peakW_EE.GetParameter(1) << endl;
	h_WMEt_pass_EE_peakfit->Draw();
	
	h_ErsatzMEt_pass_EE_peakfit->Scale(1./h_ErsatzMEt_pass_EE_peakfit->Integral(0,100));
	maxbin = h_ErsatzMEt_pass_EE_peakfit->GetMaximumBin();
	TF1 peakZ_EE = TF1("peakZ_EE", "gaus", maxbin-4, maxbin+4); 
	h_ErsatzMEt_pass_EE_peakfit->Fit("peakZ_EE", "MR");
	cout << "Ersatz maximum bin = " << maxbin << "\tPeak of Gaussian = " << peakZ_EE.GetParameter(1) << endl; 
	h_ErsatzMEt_pass_EE_peakfit->Draw();

	double shift_EE = peakW_EE.GetParameter(1) - peakZ_EE.GetParameter(1);
	cout << "EE Shift = " << shift_EE << endl;

	for(unsigned int i=0; i < ErsatzMEt_pass_EB.size(); i++)
	{
		ErsatzMEt_pass_EB[i] += shift_EB;
		h_ErsatzMEt_pass_EB_shifted->Fill(ErsatzMEt_pass_EB[i], Weight_pass_EB[i]);
	}
	for(unsigned int i=0; i < ErsatzMEt_pass_EE.size(); i++)
	{
		ErsatzMEt_pass_EE[i] += shift_EE;
		h_ErsatzMEt_pass_EE_shifted->Fill(ErsatzMEt_pass_EE[i], Weight_pass_EE[i]);
	}
	for(unsigned int i=0; i < ErsatzMEt_fail_EB.size(); i++)
	{
		ErsatzMEt_fail_EB[i] += shift_EB;
		h_ErsatzMEt_fail_EB_shifted->Fill(ErsatzMEt_fail_EB[i], Weight_fail_EB[i]);
	}
	for(unsigned int i=0; i < ErsatzMEt_fail_EE.size(); i++)
	{
		ErsatzMEt_fail_EE[i] += shift_EE;
		h_ErsatzMEt_fail_EE_shifted->Fill(ErsatzMEt_fail_EE[i], Weight_fail_EE[i]);
	}

	cout << "Apply acceptance correction ..." << endl;	
	
	TH1F* h_ones = new TH1F("ones", "Histogram of Ones;;", 100, 0., 100.);
	for (int i=0; i<100; i++)
	{
		h_ones->Fill(i+0.5);
	}
	
	h_acceptance_correction_pass_EB->Divide(h_mcWMEtout_pass_EB, h_mcWMEtin_pass_EB);
	h_acceptance_correction_pass_EB->Add(h_ones);
	h_ErsatzMEt_pass_EB_corr->Multiply(h_ErsatzMEt_pass_EB_shifted, h_acceptance_correction_pass_EB);
	
	h_acceptance_correction_pass_EE->Divide(h_mcWMEtout_pass_EE, h_mcWMEtin_pass_EE);
	h_acceptance_correction_pass_EE->Add(h_ones);
	h_ErsatzMEt_pass_EE_corr->Multiply(h_ErsatzMEt_pass_EE_shifted, h_acceptance_correction_pass_EE);
	
	h_acceptance_correction_fail_EB->Divide(h_mcWMEtout_fail_EB, h_mcWMEtin_fail_EB);
	h_acceptance_correction_fail_EB->Add(h_ones);
	h_ErsatzMEt_fail_EB_corr->Multiply(h_ErsatzMEt_fail_EB_shifted, h_acceptance_correction_fail_EB);
	
	h_acceptance_correction_fail_EE->Divide(h_mcWMEtout_fail_EE, h_mcWMEtin_fail_EE);
	h_acceptance_correction_fail_EE->Add(h_ones);
	h_ErsatzMEt_fail_EE_corr->Multiply(h_ErsatzMEt_fail_EE_shifted, h_acceptance_correction_fail_EE);

	cout << "Calculating f ..." << endl;

	double N_pass_EB = h_ErsatzMEt_pass_EB_corr->Integral(0,100);
	double A_EB = h_ErsatzMEt_pass_EB_corr->Integral(int(cMEt)+1,100); 
	double B_EB = h_ErsatzMEt_pass_EB_corr->Integral(0,int(cMEt));
	double N_pass_EE = h_ErsatzMEt_pass_EE_corr->Integral(0,100);
	double A_EE = h_ErsatzMEt_pass_EE_corr->Integral(int(cMEt)+1,100); 
	double B_EE = h_ErsatzMEt_pass_EE_corr->Integral(0,int(cMEt));
	double N_fail_EB = h_ErsatzMEt_fail_EB_corr->Integral(0,100);
	double D_EB = h_ErsatzMEt_fail_EB_corr->Integral(int(cMEt)+1,100); 
	double C_EB = h_ErsatzMEt_fail_EB_corr->Integral(0,int(cMEt));
	double N_fail_EE = h_ErsatzMEt_fail_EE_corr->Integral(0,100);
	double D_EE = h_ErsatzMEt_fail_EE_corr->Integral(int(cMEt)+1,100); 
	double C_EE = h_ErsatzMEt_fail_EE_corr->Integral(0,int(cMEt));

	double A = A_EB + A_EE;
	double B = B_EB + B_EE;
	double C = C_EB + C_EE;
	double D = D_EB + D_EE;
	double N_pass = N_pass_EB + N_pass_EE;
	double N_fail = N_fail_EB + N_fail_EE;
	//int N = N_pass + N_fail;

	double eff = 1.0*A/(A+B);
	double efferror = sqrt(eff*(1.-eff)/N_pass);
	double f = 1.0*A/B;
	double ferror = efferror/((1.-eff)*(1.-eff)); 
	
	double effprime = 1.0*D/(D+C);
	double effprimeerror = sqrt(effprime*(1.-effprime)/N_fail);
	double fprime = 1.0*D/C;
	double fprimeerror = effprimeerror/((1.-effprime)*(1.-effprime));

	cout << "f\tferror\teff\tefferror\tA\tB\tN_pass" << endl;
	cout << f << "\t" << ferror << "\t" << eff << "\t" << efferror << "\t" << A << "\t" << B << "\t" << N_pass << "\n" << endl; 
	
	cout << "fprime\tfprimeerror\teffprime\teffprimeerror\tD\tC\tN_fail" << endl;
	cout << fprime << "\t" << fprimeerror << "\t" << effprime << "\t" << effprimeerror << "\t" << D << "\t" << C << "\t" << N_fail << "\n" << endl; 

//	results << process << "\t" << f << "\t" << ferror << "\t" << A << "\t" << B << "\t" << fprime << "\t" << fprimeerror << "\t" << D << "\t" << C << endl;

	outfile->Write();
	outfile->Close();
}
