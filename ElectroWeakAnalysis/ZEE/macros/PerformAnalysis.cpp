#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TF1.h" 
#include "TCanvas.h"
#include "TRandom3.h"	
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;

void PerformAnalysis() //int main()
{
	// Declare electron cut value variables	
	double cMEt;
	double cPt;
	double cECALiso_EB, cECALiso_EE;
	double cHCALiso_EB, cHCALiso_EE;
	double cTrackiso_EB, cTrackiso_EE;
	double cDeltaEta_EB, cDeltaEta_EE;
	double cDeltaPhi_EB, cDeltaPhi_EE;
	double csIhIh_EB, csIhIh_EE;
	// Declare neutrino cut value variables
	double cHCAL;
	double cHCALEt;
	double cf1x5, cf2x5;
	int celecmatch;
	double cnusIhIh;

	cout << "Importing cut values ...\n" << endl;	

	// Import cut values from file
	ifstream cuts;
	cuts.open("CutValues.txt", ifstream::in);
	cuts >> cMEt;
	cout << "MEt cut " << cMEt << "\n" << endl;

	cout << "Electron selection cuts:" << endl;
	cuts >> cPt;
	cout << "Pt cut " << cPt << endl;
	cuts >> cECALiso_EB;
	cout << "ECAL Isolation cut (EB) " << cECALiso_EB << endl;
	cuts >> cECALiso_EE;
	cout << "ECAL Isolation cut (EE) " << cECALiso_EE << endl;
	cuts >> cHCALiso_EB;
	cout << "HCAL Isolation cut (EB) " << cHCALiso_EB << endl;
	cuts >> cHCALiso_EE;
	cout << "HCAL Isolation cut (EE) " << cHCALiso_EE << endl;
	cuts >> cTrackiso_EB;
	cout << "Track Isolation cut (EB) " << cTrackiso_EB << endl;
	cuts >> cTrackiso_EE;
	cout << "Track Isolation cut (EE) " << cTrackiso_EE << endl;
	cuts >> cDeltaEta_EB;
	cout << "Delta Eta cut (EB) " << cDeltaEta_EB << endl;
	cuts >> cDeltaEta_EE;
	cout << "Delta Eta cut (EE) " << cDeltaEta_EE << endl;
	cuts >> cDeltaPhi_EB;
	cout << "Delta Phi cut (EB) " << cDeltaPhi_EB << endl;
	cuts >> cDeltaPhi_EE;
	cout << "Delta Phi cut (EE) " << cDeltaPhi_EE << endl;
	cuts >> csIhIh_EB;
	cout << "Sigma iEta iEta cut (EB) " << csIhIh_EB << endl;
	cuts >> csIhIh_EE;
	cout << "Sigma iEta iEta cut (EE) " << csIhIh_EE << "\n" << endl;
	
	cout << "Probe selection cuts:" << endl;
	cuts >> cHCAL;
	cout << "HCAL Energy cut " << cHCAL << endl;
	cuts >> cf1x5;
	cout << "Fraction of energy in 1x5 cut " << cf1x5 << endl;
	cuts >> cf2x5;
	cout << "Fraction of energy in 2x5 cut " << cf2x5 << endl;
	cuts >> celecmatch;
	cout << "Require electron match " << celecmatch << endl;
	cuts >> cnusIhIh;
	cout << "Sigma iEta iEta cut " << cnusIhIh << endl;
	cuts >> cHCALEt;
	cout << "HCAL Transverse Energy cut " << cHCALEt << "\n" << endl;
	cuts.close();
	
	cout << "Importing probe selection efficiencies ..." << endl;

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
	
	TFile* outfile = TFile::Open("results.root", "recreate");

	TH1F* h_dataWMEt_pass_EB = new TH1F("dataWMEt_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_dataWMEt_pass_EE = new TH1F("dataWMEt_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_dataWMEt_fail_EB = new TH1F("dataWMEt_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_dataWMEt_fail_EE = new TH1F("dataWMEt_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
//	TH1F* h_mcWMEtin_pass_EB = new TH1F("mcWMEtin_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
//	TH1F* h_mcWMEtin_pass_EE = new TH1F("mcWMEtin_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
//	TH1F* h_mcWMEtin_fail_EB = new TH1F("mcWMEtin_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
//	TH1F* h_mcWMEtin_fail_EE = new TH1F("mcWMEtin_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
//	TH1F* h_mcWMEtout_pass_EB = new TH1F("mcWMEtout_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
//	TH1F* h_mcWMEtout_pass_EE = new TH1F("mcWMEtout_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
//	TH1F* h_mcWMEtout_fail_EB = new TH1F("mcWMEtout_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
//	TH1F* h_mcWMEtout_fail_EE = new TH1F("mcWMEtout_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	
	TH1F* h_ErsatzMEt_pass_EB = new TH1F("ErsatzMEt_pass_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_pass_EE = new TH1F("ErsatzMEt_pass_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fail_EB = new TH1F("ErsatzMEt_fail_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fail_EE = new TH1F("ErsatzMEt_fail_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

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
	
	vector<double> ErsatzMEt_pass_EB;
	vector<double> ErsatzMEt_pass_EE;
	vector<double> ErsatzMEt_fail_EB;
	vector<double> ErsatzMEt_fail_EE;
	
	vector<double> Weight_pass_EB;
	vector<double> Weight_pass_EE;
	vector<double> Weight_fail_EB;
	vector<double> Weight_fail_EE;

	TRandom3 r;
	
	TFile *fileW = TFile::Open("/tmp/rnandi/WenuTrue.root");
	TTree *t = (TTree*) fileW->Get("analyse/AnalysisData");
	
	long nEntries = t->GetEntries();
	double requiredintLumi = 10.;
	double fullmcintLumi = 1000.;
	long nSample = long(nEntries*requiredintLumi/fullmcintLumi);
	
	cout << "Sampling " << requiredintLumi << "pb^{-1} of W data and performing selections..." << endl;

	cout << "Total number of events = " << nEntries << endl;
	cout << "Sample size = " << nSample << endl;
	vector<int> VectorofNumbers;
	vector<int> ChosenEvents;
	for (long i = 0; i < nEntries; i++)
	{
		//if (i%100000==0) cout << i << endl;
		VectorofNumbers.push_back(i);
	}
	cout << "Created Vector of Numbers" << endl;
	for (int i=0; i < nSample; i++)
	{
		int random_number = int((nEntries - i)*r.Rndm());
		ChosenEvents.push_back(VectorofNumbers[random_number]);	
		//if (i%1000==0) cout << i << "\t" << random_number << endl;
		VectorofNumbers.erase(VectorofNumbers.begin() + random_number);
	}
	//cout << nSample << endl;
	sort(ChosenEvents.begin(), ChosenEvents.end());
	//cout << "Chosen Events" << "\t" << ChosenEvents.size() << endl;
	//cout << ChosenEvents[0] << "\t" << ChosenEvents[1] << "\t" << ChosenEvents[2] << endl;
	//cout << "Done sorting" << endl;
	//cout << "Last chosen event = " << ChosenEvents[nSample-1] << endl;
	//cout << "Capacity = " << ChosenEvents.capacity() << endl;
	//cout << "Max Size = " << ChosenEvents.max_size() << endl;

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

	//long nentries = t->GetEntries();
	int index = 0;
	int aaa = 0, bbb = 0, ccc = 0, ddd = 0;
	for(long i = 0; i < nEntries; ++i)
	{
		if(i%100000 == 0) cout <<"Analysing event "<< i << endl;
		if (i == ChosenEvents[index])
		{
		index++;
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
						//if(iIsChosen) 
						//{
							h_dataWMEt_pass_EB->Fill(CaloMEt_W); 
							aaa++;
						//}
						//if(fabs(nu_eta_W) < 2.5) 
						//{
						//	h_mcWMEtin_pass_EB->Fill(CaloMEt_W);
						//}else{ 
						//	h_mcWMEtout_pass_EB->Fill(CaloMEt_W); 
						//} 
					}
					if(inEndcap) 
					{ 
						//if(iIsChosen) 
						//{
							h_dataWMEt_pass_EE->Fill(CaloMEt_W);
							bbb++;
						//}
						//if(fabs(nu_eta_W) < 2.5) 
						//{
						//	h_mcWMEtin_pass_EE->Fill(CaloMEt_W);
						//}else{ 
						//	h_mcWMEtout_pass_EE->Fill(CaloMEt_W); 
						//} 
					}
				}else
				{
					if(inBarrel) 
					{ 
						//if(iIsChosen) 
						//{
							h_dataWMEt_fail_EB->Fill(CaloMEt_W);
							ccc++;
						//}
						//if(fabs(nu_eta_W) < 2.5) 
						//{
						//	h_mcWMEtin_fail_EB->Fill(CaloMEt_W);
						//}else{ 
						//	h_mcWMEtout_fail_EB->Fill(CaloMEt_W); 
						//} 
					}
					if(inEndcap) 
					{ 
						//if(iIsChosen) 
						//{
							h_dataWMEt_fail_EE->Fill(CaloMEt_W);
							ddd++;
						//}
						//if(fabs(nu_eta_W) < 2.5) 
						//{
						//	h_mcWMEtin_fail_EE->Fill(CaloMEt_W);
						//}else{ 
						//	h_mcWMEtout_fail_EE->Fill(CaloMEt_W); 
						//} 
					}
				}
			}
		}
		}
	}
	fileW->Close();
	
	cout << "Number of chosen events processed" << index << endl;
	cout << "Total number in sample = " << ChosenEvents.size() << endl;
	cout << "Number Pass EB = " << aaa << endl;	
	cout << "Number Pass EE = " << bbb << endl;	
	cout << "Number Fail EB = " << ccc << endl;	
	cout << "Number Fail EE = " << ddd << endl;	
        
	TFile *fileZ = TFile::Open("/tmp/rnandi/Zee.root");
        t = (TTree*) fileZ->Get("ErsatzMEt/ErsatzMEt");
	
	nEntries = t->GetEntries();
	requiredintLumi = 10.;
	fullmcintLumi = 1000.;
	nSample = int(nEntries*requiredintLumi/fullmcintLumi);

	cout << "Sampling " << requiredintLumi << "pb^{-1} of Z data and performing selections..." << endl;

	VectorofNumbers.resize(0);
	ChosenEvents.resize(0);
	for (long i = 0; i < nEntries; i++)
	{
		VectorofNumbers.push_back(i);
	}
	for (int i=0; i < nSample; i++)
	{
		int random_number = int((nEntries - i)*r.Rndm());
		ChosenEvents.push_back(VectorofNumbers[random_number]);	
		VectorofNumbers.erase(VectorofNumbers.begin() + random_number);
	}
	sort(ChosenEvents.begin(), ChosenEvents.end());
	cout << ChosenEvents[0] << "\t" << ChosenEvents[1] << "\t" << ChosenEvents[2] << endl;
	cout << "Chosen Events" << "\t" << ChosenEvents.size() << endl;	
	int nErNu;
	double tag_pt[4], tag_eta[4], tag_phi[4], probe_pt[4], probe_eta[4], probe_phi[4];
	double ErsatzV1bMEt[4];
	double elec_trkIso[4], elec_ECALIso[4], elec_HCALIso[4];
	double elec_sigIhIh[4], elec_dPhi[4], elec_dEta[4];
	double tag_rescPt[4], mesc[4];
	double nu_e1x5[4], nu_e2x5[4], nu_e5x5[4], nu_sigIhIh[4];
	double nu_HCALEt[4], nu_HCAL[4], nu_trckIso[4];
	double caloMEt;
	int nu_elec[4];
	cout << "Declared Variables" << endl;
	
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
	TBranch* bProbe_elec = t->GetBranch("probe_elecMatch");
	bProbe_elec->SetAddress(&nu_elec);
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
	TBranch* bErsatzV1b_MEt = t->GetBranch("ErsatzV1bCaloMEt");
	bErsatzV1b_MEt->SetAddress(&ErsatzV1bMEt);
	TBranch* bMesc = t->GetBranch("ErsatzV1_Mesc");
	bMesc->SetAddress(&mesc); 
	TBranch* bCaloMEt = t->GetBranch("recoCaloMEt");
	bCaloMEt->SetAddress(&caloMEt);
	cout << "Set up Branches" << endl;

	aaa=0, bbb=0, ccc=0, ddd=0;
	for(int i=0; i < nSample; ++i)
	{
		if(i%100000 == 0) cout <<"Processing event "<< i <<"."<< endl;
		t->GetEntry(ChosenEvents[i]);
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
							if(pass_trkiso_cut)
							{
								if(inBarrel) 
								{
									aaa++;	
									ErsatzMEt_pass_EB.push_back(ErsatzV1bMEt[j]);
									Weight_pass_EB.push_back(1./nueff[EtaInt]);
									h_ErsatzMEt_pass_EB->Fill(ErsatzV1bMEt[j], 1./nueff[EtaInt]);
								}
								if(inEndcap)
								{
									bbb++;
									ErsatzMEt_pass_EE.push_back(ErsatzV1bMEt[j]);
									Weight_pass_EE.push_back(1./nueff[EtaInt]);
									h_ErsatzMEt_pass_EE->Fill(ErsatzV1bMEt[j], 1./nueff[EtaInt]);
								}	
							}else{
								if(inBarrel)
								{
									ccc++;
									ErsatzMEt_fail_EB.push_back(ErsatzV1bMEt[j]);
									Weight_fail_EB.push_back(1./nueff[EtaInt]);
									h_ErsatzMEt_fail_EB->Fill(ErsatzV1bMEt[j], 1./nueff[EtaInt]);
								}
								if(inEndcap)
								{
									ddd++;
									ErsatzMEt_fail_EE.push_back(ErsatzV1bMEt[j]);
									Weight_fail_EE.push_back(1./nueff[EtaInt]);
									h_ErsatzMEt_fail_EE->Fill(ErsatzV1bMEt[j], 1./nueff[EtaInt]);
								}	
							}
						}		
					}
				}
			}
		}
	}
	fileZ->Close();

	cout << "Total number in sample = " << ChosenEvents.size() << endl;
	cout << "Number Pass EB = " << aaa << endl;	
	cout << "Number Pass EE = " << bbb << endl;	
	cout << "Number Fail EB = " << ccc << endl;	
	cout << "Number Fail EE = " << ddd << endl;	
        
	cout << "Apply shift correction ..." << endl;

	TCanvas* c0_EB = new TCanvas("c0_EB", "EB Shift", 800, 600);
	//TCanvas* c0_Pass_EE = new TCanvas("c0_Pass_EE", "Before Shift", );
	//TCanvas* c0_Fail_EB = new TCanvas("c0_Fail_EB", "Before Shift", );
	//TCanvas* c0_Fail_EE = new TCanvas("c0_Fail_EE", "Before Shift", );
	
	//TCanvas* c1_Pass_EB = new TCanvas("c1_Pass_EB", "After Shift", );
	//TCanvas* c1_Pass_EE = new TCanvas("c1_Pass_EE", "After Shift", );
	//TCanvas* c1_Fail_EB = new TCanvas("c1_Fail_EB", "After Shift", );
	//TCanvas* c1_Fail_EE = new TCanvas("c1_Fail_EE", "After Shift", );

	int maxbin;

	c0_EB->cd();
	TH1F* h_W_EB = h_dataWMEt_pass_EB;
	h_W_EB->Scale(1./h_W_EB->Integral(0,100));
	h_W_EB->SetLineColor(2);
	h_W_EB->Draw();
	maxbin = h_W_EB->GetMaximumBin();
	TF1 peakW_EB = TF1("peakW_EB", "gaus", maxbin-4, maxbin+4); 
	h_W_EB->Fit("peakW_EB", "MR");
	cout << "W MEt maximum bin = " << maxbin << "\tPeak of Gaussian = " << peakW_EB.GetParameter(1) << endl;
	
	TH1F* h_Z_EB = h_ErsatzMEt_pass_EB;
	h_Z_EB->Scale(1./h_Z_EB->Integral(0,100));
	h_Z_EB->SetLineColor(4);
	h_Z_EB->Draw("same");
	maxbin = h_Z_EB->GetMaximumBin();
	TF1 peakZ_EB = TF1("peakZ_EB", "gaus", maxbin-4, maxbin+4); 
	h_Z_EB->Fit("peakZ_EB", "MR");
	cout << "Ersatz maximum bin = " << maxbin << "\tPeak of Gaussian = " << peakZ_EB.GetParameter(1) << endl; 

	double shift_EB = peakW_EB.GetParameter(1) - peakZ_EB.GetParameter(1);
	cout << "EB Shift = " << shift_EB << endl;

	c0_EB->SaveAs("Shift_EB.png");
//	delete c0_EB;
	
	TCanvas* c0_EE = new TCanvas("c0_EE", "EE Shift", 800, 600);
	
	TH1F* h_W_EE = h_dataWMEt_pass_EE;
	h_W_EE->Scale(1./h_W_EE->Integral(0,100));
	h_W_EE->SetLineColor(2);
	h_W_EE->Draw();
	maxbin = h_W_EE->GetMaximumBin();
	TF1 peakW_EE = TF1("peakW_EE", "gaus", maxbin-4, maxbin+4); 
	h_W_EE->Fit("peakW_EE", "MR");
	cout << "W MEt maximum bin = " << maxbin << "\tPeak of Gaussian = " << peakW_EE.GetParameter(1) << endl;
	
	TH1F* h_Z_EE = h_ErsatzMEt_pass_EE;
	h_Z_EE->Scale(1./h_Z_EE->Integral(0,100));
	h_Z_EE->SetLineColor(4);
	h_Z_EE->Draw("same");
	maxbin = h_Z_EE->GetMaximumBin();
	TF1 peakZ_EE = TF1("peakZ_EE", "gaus", maxbin-4, maxbin+4); 
	h_Z_EE->Fit("peakZ_EE", "MR");
	cout << "Ersatz maximum bin = " << maxbin << "\tPeak of Gaussian = " << peakZ_EE.GetParameter(1) << endl; 

	double shift_EE = peakW_EE.GetParameter(1) - peakZ_EE.GetParameter(1);
	cout << "EE Shift = " << shift_EE << endl;

	c0_EE->SaveAs("Shift_EE.png");
//	delete c0_EE;
/*	
	h_dataWMEt_fail_EB->Scale(1./h_dataWMEt_fail_EB->Integral(0,100));
	maxbin = h_dataWMEt_fail_EB->GetMaximumBin();
	TF1 peakW_fail_EB = TF1("peakW_fail_EB", "gaus", maxbin-4, maxbin+4); 
	h_dataWMEt_fail_EB->Fit("peakW_fail_EB", "MR");
	cout << "W MEt maximum bin = " << maxbin << "\tPeak of Gaussian = " << peakW_fail_EB.GetParameter(1) << endl;
	
	h_ErsatzMEt_fail_EB->Scale(1./h_ErsatzMEt_fail_EB->Integral(0,100));
	maxbin = h_ErsatzMEt_fail_EB->GetMaximumBin();
	TF1 peakZ_fail_EB = TF1("peakZ_fail_EB", "gaus", maxbin-4, maxbin+4); 
	h_ErsatzMEt_fail_EB->Fit("peakZ_fail_EB", "MR");
	cout << "Ersatz maximum bin = " << maxbin << "\tPeak of Gaussian = " << peakZ_fail_EB.GetParameter(1) << endl; 

	double shift_fail_EB = peakW_fail_EB.GetParameter(1) - peakZ_fail_EB.GetParameter(1);
	
	h_dataWMEt_fail_EE->Scale(1./h_dataWMEt_fail_EE->Integral(0,100));
	maxbin = h_dataWMEt_fail_EE->GetMaximumBin();
	TF1 peakW_fail_EE = TF1("peakW_fail_EE", "gaus", maxbin-4, maxbin+4); 
	h_dataWMEt_fail_EE->Fit("peakW_fail_EE", "MR");
	cout << "W MEt maximum bin = " << maxbin << "\tPeak of Gaussian = " << peakW_fail_EE.GetParameter(1) << endl;
	
	h_ErsatzMEt_fail_EE->Scale(1./h_ErsatzMEt_fail_EE->Integral(0,100));
	maxbin = h_ErsatzMEt_fail_EE->GetMaximumBin();
	TF1 peakZ_fail_EE = TF1("peakZ_fail_EE", "gaus", maxbin-4, maxbin+4); 
	h_ErsatzMEt_fail_EE->Fit("peakZ_fail_EE", "MR");
	cout << "Ersatz maximum bin = " << maxbin << "\tPeak of Gaussian = " << peakZ_fail_EE.GetParameter(1) << endl; 
	
	double shift_fail_EE = peakW_fail_EE.GetParameter(1) - peakZ_fail_EE.GetParameter(1);
*/
/*
	for(vector<double>::const_iterator it = ErsatzMEt_pass_EB.begin(); it != ErsatzMEt_pass_EB.end(); ++it)
	{
		*it += shift_pass_EB;
		//h_ErsatzMEt_pass_EB_shifted->Fill(*it);//fill with weight!!!!!!!
	}
	for(vector<double>::const_iterator it = ErsatzMEt_pass_EE.begin(); it != ErsatzMEt_pass_EE.end(); ++it)
	{
		*it += shift_pass_EE;
		//h_ErsatzMEt_pass_EE_shifted->Fill(*it);
	}
	for(vector<double>::const_iterator it = ErsatzMEt_fail_EB.begin(); it != ErsatzMEt_fail_EB.end(); ++it)
	{
		*it += shift_fail_EB;
		//h_ErsatzMEt_fail_EB_shifted->Fill(*it);
	}
	for(vector<double>::const_iterator it = ErsatzMEt_fail_EE.begin(); it != ErsatzMEt_fail_EE.end(); ++it)
	{
		*it += shift_fail_EE;
		//h_ErsatzMEt_fail_EE_shifted->Fill(*it);
	}
*/	
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
	
	TFile *fileMC = TFile::Open("mcWMEtHistograms.root");
	TH1F* h_mcWMEtin_pass_EB = (TH1F*) fileMC->Get("mcWMEtin_pass_EB"); 
	TH1F* h_mcWMEtin_pass_EE = (TH1F*) fileMC->Get("mcWMEtin_pass_EE");
	TH1F* h_mcWMEtin_fail_EB = (TH1F*) fileMC->Get("mcWMEtin_fail_EB");
	TH1F* h_mcWMEtin_fail_EE = (TH1F*) fileMC->Get("mcWMEtin_fail_EE");
	TH1F* h_mcWMEtout_pass_EB = (TH1F*) fileMC->Get("mcWMEtout_pass_EB");
	TH1F* h_mcWMEtout_pass_EE = (TH1F*) fileMC->Get("mcWMEtout_pass_EE");
	TH1F* h_mcWMEtout_fail_EB = (TH1F*) fileMC->Get("mcWMEtout_fail_EB");
	TH1F* h_mcWMEtout_fail_EE = (TH1F*) fileMC->Get("mcWMEtout_fail_EE");

	TH1F* h_ones = new TH1F("ones", "Histogram of Ones;;", 100, 0., 100.);

	for (int i=0; i<100; i++)
	{
		h_ones->Fill(i+0.5);
	}
	for (int i=0; i<100; i++)
	{
		double out = h_mcWMEtout_pass_EB->GetBinContent(i);
		double in = h_mcWMEtin_pass_EB->GetBinContent(i);
		cout << out << "\t" << in << "\t" << out/in << endl;
	} 
	h_acceptance_correction_pass_EB->Divide(h_mcWMEtout_pass_EB, h_mcWMEtin_pass_EB);
	h_acceptance_correction_pass_EB->Add(h_ones);
	h_ErsatzMEt_pass_EB_corr->Multiply(h_ErsatzMEt_pass_EB_shifted, h_acceptance_correction_pass_EB);
	//h_ErsatzMEt_pass_EB_corr->Add(h_ErsatzMEt_pass_EB_shifted, h_ErsatzMEt_pass_EB_corr);
	cout << "Done pass EB" << endl;	
	
	for (int i=0; i<100; i++)
	{
		double out = h_mcWMEtout_pass_EE->GetBinContent(i);
		double in = h_mcWMEtin_pass_EE->GetBinContent(i);
		cout << out << "\t" << in << "\t" << out/in << endl;
	} 
	h_acceptance_correction_pass_EE->Divide(h_mcWMEtout_pass_EE, h_mcWMEtin_pass_EE);
	h_acceptance_correction_pass_EE->Add(h_ones);
	h_ErsatzMEt_pass_EE_corr->Multiply(h_ErsatzMEt_pass_EE_shifted, h_acceptance_correction_pass_EE);
	//h_ErsatzMEt_pass_EE_corr->Add(h_ErsatzMEt_pass_EE_shifted, h_ErsatzMEt_pass_EE_corr);
	cout << "Done pass EE" << endl;	

	for (int i=0; i<100; i++)
	{
		double out = h_mcWMEtout_fail_EB->GetBinContent(i);
		double in = h_mcWMEtin_fail_EB->GetBinContent(i);
		cout << out << "\t" << in << "\t" << out/in << endl;
	} 
	h_acceptance_correction_fail_EB->Divide(h_mcWMEtout_fail_EB, h_mcWMEtin_fail_EB);
	h_acceptance_correction_fail_EB->Add(h_ones);
	h_ErsatzMEt_fail_EB_corr->Multiply(h_ErsatzMEt_fail_EB_shifted, h_acceptance_correction_fail_EB);
	//h_ErsatzMEt_fail_EB_corr->Add(h_ErsatzMEt_fail_EB_shifted, h_ErsatzMEt_fail_EB_corr);
	cout << "Done fail EB" << endl;	

	for (int i=0; i<100; i++)
	{
		double out = h_mcWMEtout_fail_EE->GetBinContent(i);
		double in = h_mcWMEtin_fail_EE->GetBinContent(i);
		cout << out << "\t" << in << "\t" << out/in << endl;
	} 
	h_acceptance_correction_fail_EE->Divide(h_mcWMEtout_fail_EE, h_mcWMEtin_fail_EE);
	h_acceptance_correction_fail_EE->Add(h_ones);
	h_ErsatzMEt_fail_EE_corr->Multiply(h_ErsatzMEt_fail_EE_shifted, h_acceptance_correction_fail_EE);
	//h_ErsatzMEt_fail_EB_corr->Add(h_ErsatzMEt_fail_EE_shifted, h_ErsatzMEt_fail_EE_corr);
	cout << "Done fail EE" << endl;	

	fileMC->Close();

	cout << "Calculating f ..." << endl;

	int N_pass_EB = int(h_ErsatzMEt_pass_EB_corr->Integral(0,100));
	int A_EB = int(h_ErsatzMEt_pass_EB_corr->Integral(int(cMEt),100)); 
	int B_EB = int(h_ErsatzMEt_pass_EB_corr->Integral(0,int(cMEt)));
	int N_pass_EE = int(h_ErsatzMEt_pass_EE_corr->Integral(0,100));
	int A_EE = int(h_ErsatzMEt_pass_EE_corr->Integral(int(cMEt),100)); 
	int B_EE = int(h_ErsatzMEt_pass_EE_corr->Integral(0,int(cMEt)));
	int N_fail_EB = int(h_ErsatzMEt_fail_EB_corr->Integral(0,100));
	int D_EB = int(h_ErsatzMEt_fail_EB_corr->Integral(int(cMEt),100)); 
	int C_EB = int(h_ErsatzMEt_fail_EB_corr->Integral(0,int(cMEt)));
	int N_fail_EE = int(h_ErsatzMEt_fail_EE_corr->Integral(0,100));
	int D_EE = int(h_ErsatzMEt_fail_EE_corr->Integral(int(cMEt),100)); 
	int C_EE = int(h_ErsatzMEt_fail_EE_corr->Integral(0,int(cMEt)));

	int A = A_EB + A_EE;
	int B = B_EB + B_EE;
	int C = C_EB + C_EE;
	int D = D_EB + D_EE;
	int N_pass = N_pass_EB + N_pass_EE;
	int N_fail = N_fail_EB + N_fail_EE;
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

	outfile->Write();
	outfile->Close();
}
