#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TLegend.h"
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

	TFile* outfile = TFile::Open("ZeePlots.root", "recreate");

	TH1F* h_McW_m = new TH1F("McW_m","MC Boson Mass;m;Arbitrary Units", 100, 40., 130.);
	TH1F* h_McW_pt = new TH1F("McW_pt","MC Boson p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_McW_y = new TH1F("McW_y","MC Boson Rapidity;y;Arbitrary Units", 100, -5., 5.);

	TH1F* h_McZ_m = new TH1F("McZ_m","MC Boson Mass;m (GeV);Arbitrary Units", 100, 40., 130.);
	TH1F* h_McZ_pt = new TH1F("McZ_pt","MC Boson p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_McZ_y = new TH1F("McZ_y","MC Boson Rapidity;y;Arbitrary Units", 100, -5., 5.);
	TH1F* h_McZ_rescM = new TH1F("McZ_rescM","Rescaled MC Boson Mass;m (GeV);Arbitrary Units", 100, 40., 130.);
	TH1F* h_McZ_rescPt = new TH1F("McZ_rescPt","Rescaled MC Boson p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_McZ_rescY = new TH1F("McZ_rescY","Rescaled MC Boson Rapidity;y;Arbitrary Units", 100, -5., 5.);
	
	TH1F* h_McElec_pt = new TH1F("McElec_pt","MC Electron p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_McElec_eta = new TH1F("McElec_eta","MC Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);

	TH1F* h_McTag_pt = new TH1F("McTag_pt","MC Electron p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_McTag_eta = new TH1F("McTag_eta","MC Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McTag_rescPt = new TH1F("McTag_rescPt","Rescaled MC Electron p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_McTag_rescEta = new TH1F("McTag_rescEta","Rescaled MC Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);

	TH1F* h_McNu_pt = new TH1F("McNu_pt","MC Neutrino p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_McNu_eta = new TH1F("McNu_eta","MC Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);

	TH1F* h_McProbe_pt = new TH1F("McProbe_pt","MC Neutrino p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_McProbe_eta = new TH1F("McProbe_eta","MC Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McProbe_rescPt = new TH1F("McProbe_rescPt","Rescaled MC Neutrino p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_McProbe_rescEta = new TH1F("McProbe_rescEta","Rescaled MC Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);

	TH1F* h_McTagProbe_dPhi = new TH1F("McElecNu_dPhi", ";#Delta#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_McTagProbe_dEta = new TH1F("McElecNu_dEta", ";#Delta#eta;Arbitrary Units", 100, 0., 5.);
	TH1F* h_McTagProbe_dR = new TH1F("McElecNu_dR", ";#DeltaR;Arbitrary Units", 100, 0., 3.);
	TH1F* h_McElecNu_dPhi = new TH1F("McElecNu_dPhi", ";#Delta#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_McElecNu_dEta = new TH1F("McElecNu_dEta", ";#Delta#eta;Arbitrary Units", 100, 0., 5.);
	TH1F* h_McElecNu_dR = new TH1F("McElecNu_dR", ";#DeltaR;Arbitrary Units", 100, 0., 3.);

	TH1F* h_W_pt = new TH1F("W_pt","Boson p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_W_y = new TH1F("W_y","Boson Rapidity;y;Arbitrary Units", 100, -5., 5.);
	TH1F* h_W_rescPt = new TH1F("W_rescPt","Rescaled Boson p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_W_rescY = new TH1F("W_rescY","Rescaled Boson Rapidity;y;Arbitrary Units", 100, -5., 5.);

	TH1F* h_Z_pt = new TH1F("Z_pt","Boson p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_Z_y = new TH1F("Z_y","Boson Rapidity;y;Arbitrary Units", 100, -5., 5.);
	TH1F* h_Z_rescPt = new TH1F("Z_rescPt","Rescaled Boson p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_Z_rescY = new TH1F("Z_rescY","Rescaled Boson Rapidity;y;Arbitrary Units", 100, -5., 5.);
	
	TH1F* h_tag_q = new TH1F("tag_q","Electron Charge;q;Arbitrary Units", 3, -1.5, 1.5);
	TH1F* h_tag_pt = new TH1F("tag_pt","Electron p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_tag_rescPt = new TH1F("tag_rescPt","Rescaled Electron p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_tag_eta = new TH1F("tag_eta","Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_tag_rescEta = new TH1F("tag_rescEta","Rescaled Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_tag_ecaliso = new TH1F("tag_ecaliso","Electron ECAL isolation;ECAL isolation;Arbitrary Units", 100, 0., 10.);
	TH1F* h_tag_hcaliso = new TH1F("tag_hcaliso","Electron HCAL isolation;HCAL isolation;Arbitrary Units", 100, 0., 5.);
	TH1F* h_tag_trackiso = new TH1F("tag_trackiso","Electron Track isolation;Track isolation;Arbitrary Units", 100, 0., 5.);
	TH1F* h_tag_dPhi = new TH1F("tag_dPhi","Electron #Delta #phi_{in};#Delta #phi_{in};Arbitrary Units", 100, 0., 0.05);
	TH1F* h_tag_dEta = new TH1F("tag_dEta","Electron #Delta #eta_{in};#Delta #eta_{in};Arbitrary Units", 100, 0., 0.01);
	TH1F* h_tag_sIhIh = new TH1F("tag_sIhIh","Electron #sigma_{i#eta i#eta};#sigma_{i#eta i#eta};Arbitrary Units", 100, 0., 0.02);

	TH1F* h_elec_q = new TH1F("elec_q","Electron Charge;q;Arbitrary Units", 3, -1.5, 1.5);
	TH1F* h_elec_pt = new TH1F("elec_pt","Electron p_{T};p_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_elec_eta = new TH1F("elec_eta","Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_elec_ecaliso = new TH1F("elec_ecaliso","Electron ECAL isolation;ECAL isolation;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_hcaliso = new TH1F("elec_hcaliso","Electron HCAL isolation;HCAL isolation;Arbitrary Units", 100, 0., 5.);
	TH1F* h_elec_trackiso = new TH1F("elec_trackiso","Electron Track isolation;Track isolation;Arbitrary Units", 100, 0., 5.);
	TH1F* h_elec_dPhi = new TH1F("elec_dPhi","Electron #Delta #phi_{in};#Delta #phi_{in};Arbitrary Units", 100, 0., 0.05);
	TH1F* h_elec_dEta = new TH1F("elec_dEta","Electron #Delta #eta_{in};#Delta #eta_{in};Arbitrary Units", 100, 0., 0.01);
	TH1F* h_elec_sIhIh = new TH1F("elec_sIhIh","Electron #sigma_{i#eta i#eta};#sigma_{i#eta i#eta};Arbitrary Units", 100, 0., 0.02);

	TH1F* h_mcWGenMEtin_pass_EB = new TH1F("mcWGenMEtin_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWGenMEtin_pass_EE = new TH1F("mcWGenMEtin_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWGenMEtin_fail_EB = new TH1F("mcWGenMEtin_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWGenMEtin_fail_EE = new TH1F("mcWGenMEtin_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWGenMEtout_pass_EB = new TH1F("mcWGenMEtout_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWGenMEtout_pass_EE = new TH1F("mcWGenMEtout_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWGenMEtout_fail_EB = new TH1F("mcWGenMEtout_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWGenMEtout_fail_EE = new TH1F("mcWGenMEtout_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	
	TH1F* h_mcWCaloMEtin_pass_EB = new TH1F("mcWCaloMEtin_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWCaloMEtin_pass_EE = new TH1F("mcWCaloMEtin_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWCaloMEtin_fail_EB = new TH1F("mcWCaloMEtin_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWCaloMEtin_fail_EE = new TH1F("mcWCaloMEtin_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWCaloMEtout_pass_EB = new TH1F("mcWCaloMEtout_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWCaloMEtout_pass_EE = new TH1F("mcWCaloMEtout_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWCaloMEtout_fail_EB = new TH1F("mcWCaloMEtout_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWCaloMEtout_fail_EE = new TH1F("mcWCaloMEtout_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	
	TH1F* h_mcWTcMEtin_pass_EB = new TH1F("mcWTcMEtin_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWTcMEtin_pass_EE = new TH1F("mcWTcMEtin_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWTcMEtin_fail_EB = new TH1F("mcWTcMEtin_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWTcMEtin_fail_EE = new TH1F("mcWTcMEtin_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWTcMEtout_pass_EB = new TH1F("mcWTcMEtout_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWTcMEtout_pass_EE = new TH1F("mcWTcMEtout_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWTcMEtout_fail_EB = new TH1F("mcWTcMEtout_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWTcMEtout_fail_EE = new TH1F("mcWTcMEtout_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	
	TH1F* h_mcWPfMEtin_pass_EB = new TH1F("mcWPfMEtin_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWPfMEtin_pass_EE = new TH1F("mcWPfMEtin_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWPfMEtin_fail_EB = new TH1F("mcWPfMEtin_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWPfMEtin_fail_EE = new TH1F("mcWPfMEtin_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWPfMEtout_pass_EB = new TH1F("mcWPfMEtout_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWPfMEtout_pass_EE = new TH1F("mcWPfMEtout_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWPfMEtout_fail_EB = new TH1F("mcWPfMEtout_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_mcWPfMEtout_fail_EE = new TH1F("mcWPfMEtout_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	
	TH1F* h_WGenMEt_pass_EB = new TH1F("WGenMEt_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WGenMEt_pass_EE = new TH1F("WGenMEt_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WGenMEt_fail_EB = new TH1F("WGenMEt_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WGenMEt_fail_EE = new TH1F("WGenMEt_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WGenMEt_pass = new TH1F("WGenMEt_pass","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WGenMEt_fail = new TH1F("WGenMEt_fail","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_WCaloMEt_pass_EB = new TH1F("WCaloMEt_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WCaloMEt_pass_EE = new TH1F("WCaloMEt_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WCaloMEt_fail_EB = new TH1F("WCaloMEt_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WCaloMEt_fail_EE = new TH1F("WCaloMEt_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WCaloMEt_pass = new TH1F("WCaloMEt_pass","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WCaloMEt_fail = new TH1F("WCaloMEt_fail","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_WTcMEt_pass_EB = new TH1F("WTcMEt_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WTcMEt_pass_EE = new TH1F("WTcMEt_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WTcMEt_fail_EB = new TH1F("WTcMEt_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WTcMEt_fail_EE = new TH1F("WTcMEt_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WTcMEt_pass = new TH1F("WTcMEt_pass","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WTcMEt_fail = new TH1F("WTcMEt_fail","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_WPfMEt_pass_EB = new TH1F("WPfMEt_pass_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WPfMEt_pass_EE = new TH1F("WPfMEt_pass_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WPfMEt_fail_EB = new TH1F("WPfMEt_fail_EB","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WPfMEt_fail_EE = new TH1F("WPfMEt_fail_EE","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WPfMEt_pass = new TH1F("WPfMEt_pass","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_WPfMEt_fail = new TH1F("WPfMEt_fail","W#rightarrow e#nu MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_ErsatzMEt_pass_EB = new TH1F("ErsatzMEt_pass_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_pass_EE = new TH1F("ErsatzMEt_pass_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fail_EB = new TH1F("ErsatzMEt_fail_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fail_EE = new TH1F("ErsatzMEt_fail_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_pass = new TH1F("ErsatzMEt_pass","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzMEt_fail = new TH1F("ErsatzMEt_fail","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_ErsatzGenMEt_pass_EB = new TH1F("ErsatzGenMEt_pass_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzGenMEt_pass_EE = new TH1F("ErsatzGenMEt_pass_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzGenMEt_fail_EB = new TH1F("ErsatzGenMEt_fail_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzGenMEt_fail_EE = new TH1F("ErsatzGenMEt_fail_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzGenMEt_pass = new TH1F("ErsatzGenMEt_pass","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzGenMEt_fail = new TH1F("ErsatzGenMEt_fail","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_ErsatzCaloMEt_pass_EB = new TH1F("ErsatzCaloMEt_pass_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzCaloMEt_pass_EE = new TH1F("ErsatzCaloMEt_pass_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzCaloMEt_fail_EB = new TH1F("ErsatzCaloMEt_fail_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzCaloMEt_fail_EE = new TH1F("ErsatzCaloMEt_fail_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzCaloMEt_pass = new TH1F("ErsatzCaloMEt_pass","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzCaloMEt_fail = new TH1F("ErsatzCaloMEt_fail","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_ErsatzTcMEt_pass_EB = new TH1F("ErsatzTcMEt_pass_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzTcMEt_pass_EE = new TH1F("ErsatzTcMEt_pass_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzTcMEt_fail_EB = new TH1F("ErsatzTcMEt_fail_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzTcMEt_fail_EE = new TH1F("ErsatzTcMEt_fail_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzTcMEt_pass = new TH1F("ErsatzTcMEt_pass","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzTcMEt_fail = new TH1F("ErsatzTcMEt_fail","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_ErsatzPfMEt_pass_EB = new TH1F("ErsatzPfMEt_pass_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzPfMEt_pass_EE = new TH1F("ErsatzPfMEt_pass_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzPfMEt_fail_EB = new TH1F("ErsatzPfMEt_fail_EB","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzPfMEt_fail_EE = new TH1F("ErsatzPfMEt_fail_EE","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzPfMEt_pass = new TH1F("ErsatzPfMEt_pass","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzPfMEt_fail = new TH1F("ErsatzPfMEt_fail","Ersatz MET;#slash{E}_{T} (GeV);Arbitrary Units", 100, 0., 100.);

	TH1F* h_acceptance_correction_genMEt_pass_EB = new TH1F("acceptance_correction_genMEt_pass_EB", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_genMEt_pass_EE = new TH1F("acceptance_correction_genMEt_pass_EE", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_genMEt_fail_EB = new TH1F("acceptance_correction_genMEt_fail_EB", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_genMEt_fail_EE = new TH1F("acceptance_correction_genMEt_fail_EE", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);

	TH1F* h_acceptance_correction_caloMEt_pass_EB = new TH1F("acceptance_correction_caloMEt_pass_EB", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_caloMEt_pass_EE = new TH1F("acceptance_correction_caloMEt_pass_EE", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_caloMEt_fail_EB = new TH1F("acceptance_correction_caloMEt_fail_EB", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_caloMEt_fail_EE = new TH1F("acceptance_correction_caloMEt_fail_EE", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);

	TH1F* h_acceptance_correction_tcMEt_pass_EB = new TH1F("acceptance_correction_tcMEt_pass_EB", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_tcMEt_pass_EE = new TH1F("acceptance_correction_tcMEt_pass_EE", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_tcMEt_fail_EB = new TH1F("acceptance_correction_tcMEt_fail_EB", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_tcMEt_fail_EE = new TH1F("acceptance_correction_tcMEt_fail_EE", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);

	TH1F* h_acceptance_correction_pfMEt_pass_EB = new TH1F("acceptance_correction_pfMEt_pass_EB", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_pfMEt_pass_EE = new TH1F("acceptance_correction_pfMEt_pass_EE", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_pfMEt_fail_EB = new TH1F("acceptance_correction_pfMEt_fail_EB", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	TH1F* h_acceptance_correction_pfMEt_fail_EE = new TH1F("acceptance_correction_pfMEt_fail_EE", "Acceptance Correction;#slash{E}_{T} (GeV);Acceptance Correction", 100, 0., 100.);
	
	TH1F* h_dpt_pf_gsf = new TH1F("h_dpt_pf_gsf", "pf p_{T} - gsf p_{T};#Delta p_{T} (GeV);Arbitrary Units", 100, -10., 10.);
	TH1F* h_dpt_pf_gsf_EB = new TH1F("h_dpt_pf_gsf_EB", "pf p_{T} - gsf p_{T};#Delta p_{T} (GeV);Arbitrary Units", 100, -10., 10.);
	TH1F* h_dpt_pf_gsf_EE = new TH1F("h_dpt_pf_gsf_EE", "pf p_{T} - gsf p_{T};#Delta p_{T} (GeV);Arbitrary Units", 100, -10., 10.);
	TH1F* h_nMatched = new TH1F("h_nMatched", "Particle Flow Matching;nMatched;Arbitrary Units", 4, 0., 4.);
	TH1F* h_nMatched_EB = new TH1F("h_nMatched_EB", "Particle Flow Matching;nMatched;Arbitrary Units", 4, 0., 4.);
	TH1F* h_nMatched_EE = new TH1F("h_nMatched_EE", "Particle Flow Matching;nMatched;Arbitrary Units", 4, 0., 4.);

	cout << "Declared Histograms" << endl;	

	TRandom3 r;
	
	TString WFileName = datapath+"WenuTrue.root";
	TFile *fileW = TFile::Open(WFileName);
	cout << "Opened W Monte Carlo file" << endl;
	TTree *t = (TTree*) fileW->Get("analyse/AnalysisData");
	cout << "Got W TTree" << endl;

	long nEntries = t->GetEntriesFast();
	cout << "Total number of W events = " << nEntries << endl;

	int nSelElecs;

	int elec_q[4];
	double elec_pt[4], elec_eta[4], elec_phi[4];
	double elec_trackiso[4], elec_ecaliso[4], elec_hcaliso[4];
	double elec_sIhIh[4], elec_dPhi[4], elec_dEta[4];
	double elec_eop[4], elec_hoe[4], elec_pin[4], elec_pout[4];
	double elec_e1x5[4], elec_e2x5[4], elec_e5x5[4];

	double McElec_pt, McElec_eta, McElec_phi;
	double McNu_pt, McNu_eta, McNu_phi;
	double McElecNu_dPhi, McElecNu_dEta, McElecNu_dR;
	double McW_m, McW_pt, McW_y, McW_phi; 
	double GenMEt_W, CaloMEt_W, TcMEt_W, PfMEt_W;//, CaloMEt25, CaloMEt30;
	//double CaloMt[4];// CaloMt25[4], CaloMt30[4];

	TBranch* bMcW_m = t->GetBranch("Boson_m");
        bMcW_m->SetAddress(&McW_m);   
	TBranch* bMcW_pt = t->GetBranch("Boson_pt");
        bMcW_pt->SetAddress(&McW_pt);   
	TBranch* bMcW_y = t->GetBranch("Boson_y");
        bMcW_y->SetAddress(&McW_y);   
	TBranch* bnSelElecs = t->GetBranch("nSelElecs");
	bnSelElecs->SetAddress(&nSelElecs);
	//TBranch* belec_q = t->GetBranch("elec_q");
	//belec_q->SetAddress(&elec_q);
	TBranch* belec_pt = t->GetBranch("elec_pt");
	belec_pt->SetAddress(&elec_pt);
	TBranch* belec_eta = t->GetBranch("elec_eta");
	belec_eta->SetAddress(&elec_eta);
	TBranch* belec_phi = t->GetBranch("elec_phi");
	belec_phi->SetAddress(&elec_phi);
        TBranch* belec_trackiso = t->GetBranch("elec_trckIso");
        belec_trackiso->SetAddress(&elec_trackiso);
        TBranch* belec_ecaliso = t->GetBranch("elec_ecalIso");
        belec_ecaliso->SetAddress(&elec_ecaliso);
        TBranch* belec_hcaliso = t->GetBranch("elec_hcalIso");
        belec_hcaliso->SetAddress(&elec_hcaliso);
        TBranch* bElec_sIhIh = t->GetBranch("elec_sIhIh");
        bElec_sIhIh->SetAddress(&elec_sIhIh);
	TBranch* bElec_dPhi = t->GetBranch("elec_dPhiIn");
        bElec_dPhi->SetAddress(&elec_dPhi);
        TBranch* bElec_dEta = t->GetBranch("elec_dEtaIn");
        bElec_dEta->SetAddress(&elec_dEta);
	cout << "moooooooooooooo" << endl;
	TBranch* bMcElec_pt = t->GetBranch("RndmMcElec_pt");
	bMcElec_pt->SetAddress(&McElec_pt);
	TBranch* bMcElec_eta = t->GetBranch("RndmMcElec_eta");
	bMcElec_eta->SetAddress(&McElec_eta);
	TBranch* bMcElec_phi = t->GetBranch("RndmMcElec_phi");
	bMcElec_phi->SetAddress(&McElec_phi);
	cout << "moooooooooooooo" << endl;
	TBranch* bMcNu_pt = t->GetBranch("McNu_pt");
	bMcNu_pt->SetAddress(&McNu_pt);
	TBranch* bMcNu_eta = t->GetBranch("McNu_eta");
	bMcNu_eta->SetAddress(&McNu_eta);
	TBranch* bMcNu_phi = t->GetBranch("McNu_phi");
	bMcNu_phi->SetAddress(&McNu_phi);
	cout << "moooooooooooooo" << endl;
	TBranch* bMcElecNu_dPhi = t->GetBranch("McLeptons_dPhi");
	bMcElecNu_dPhi->SetAddress(&McElecNu_dPhi);
	TBranch* bMcElecNu_dEta = t->GetBranch("McLeptons_dEta");
	bMcElecNu_dEta->SetAddress(&McElecNu_dEta);
	TBranch* bMcElecNu_dR = t->GetBranch("McLeptons_dR");
	bMcElecNu_dR->SetAddress(&McElecNu_dR);
	TBranch* bGen_MEt = t->GetBranch("genMEt");
	bGen_MEt->SetAddress(&GenMEt_W);
	TBranch* bCalo_MEt = t->GetBranch("caloMEt");
	bCalo_MEt->SetAddress(&CaloMEt_W);
	TBranch* bTc_MEt = t->GetBranch("tcMEt");
	bTc_MEt->SetAddress(&TcMEt_W);
	TBranch* bPf_MEt = t->GetBranch("pfMEt");
	bPf_MEt->SetAddress(&PfMEt_W);
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
		h_McElec_pt->Fill(McElec_pt);
		h_McElec_eta->Fill(McElec_eta);
		h_McNu_pt->Fill(McNu_pt);
		h_McNu_eta->Fill(McNu_eta);
		h_McElecNu_dPhi->Fill(McElecNu_dPhi);
		h_McElecNu_dEta->Fill(McElecNu_dEta);
		h_McElecNu_dR->Fill(McElecNu_dR);
		h_McW_m->Fill(McW_m);
		h_McW_pt->Fill(McW_pt);
		h_McW_y->Fill(McW_y);
		//h_elec_q->Fill(elec_q[0]);
		h_elec_pt->Fill(elec_pt[0]);
		h_elec_eta->Fill(elec_eta[0]);
		h_elec_ecaliso->Fill(elec_ecaliso[0]);
		h_elec_hcaliso->Fill(elec_hcaliso[0]);
		h_elec_trackiso->Fill(elec_trackiso[0]);
		h_elec_dEta->Fill(elec_dEta[0]);
		h_elec_dPhi->Fill(elec_dPhi[0]);
		h_elec_sIhIh->Fill(elec_sIhIh[0]);
		//h_elec_eop->Fill(elec_eop[0]);
		//h_elec_hoe->Fill(elec_hoe[0]);
		//h_elec_pin->Fill(elec_pin[0]);
		//h_elec_pout->Fill(elec_pout[0]);
		//h_elec_e1x5->Fill(elec_e1x5[0]);
		//h_elec_e2x5->Fill(elec_e2x5[0]);
		//h_elec_e5x5->Fill(elec_e5x5[0]);
		if(elec_pt[0] > cPt)
		{
			bool pass_e_cuts = false;
			bool pass_trkiso_cut = false;
			bool inBarrel = false;
			bool inEndcap = false;
			if(fabs(elec_eta[0]) < 1.4442)
			{
				pass_e_cuts = (elec_sIhIh[0] < csIhIh_EB); 
				pass_trkiso_cut = (elec_trackiso[0] < cTrackiso_EB);
				inBarrel = true;
			}else if(fabs(elec_eta[0]) < 2.5)
			{	
				pass_e_cuts = (elec_sIhIh[0] < csIhIh_EE); 
				pass_trkiso_cut = (elec_trackiso[0] < cTrackiso_EE);
				inEndcap = true;
			}
			if(pass_e_cuts)
			{
				if(pass_trkiso_cut)
				{ 
					h_WGenMEt_pass->Fill(GenMEt_W); 
					h_WCaloMEt_pass->Fill(CaloMEt_W); 
					h_WTcMEt_pass->Fill(TcMEt_W); 
					h_WPfMEt_pass->Fill(PfMEt_W); 
					if(inBarrel) 
					{
						h_WGenMEt_pass_EB->Fill(GenMEt_W); 
						h_WCaloMEt_pass_EB->Fill(CaloMEt_W); 
						h_WTcMEt_pass_EB->Fill(TcMEt_W); 
						h_WPfMEt_pass_EB->Fill(PfMEt_W); 
						aaa++;
						if(fabs(McNu_eta) < 2.5) 
						{
							h_mcWGenMEtin_pass_EB->Fill(GenMEt_W);
							h_mcWCaloMEtin_pass_EB->Fill(CaloMEt_W);
							h_mcWTcMEtin_pass_EB->Fill(TcMEt_W);
							h_mcWPfMEtin_pass_EB->Fill(PfMEt_W);
						}else{ 
							h_mcWGenMEtout_pass_EB->Fill(GenMEt_W); 
							h_mcWCaloMEtout_pass_EB->Fill(CaloMEt_W); 
							h_mcWTcMEtout_pass_EB->Fill(TcMEt_W); 
							h_mcWPfMEtout_pass_EB->Fill(PfMEt_W); 
						} 
					}
					if(inEndcap) 
					{ 
						h_WGenMEt_pass_EE->Fill(GenMEt_W);
						h_WCaloMEt_pass_EE->Fill(CaloMEt_W);
						h_WTcMEt_pass_EE->Fill(TcMEt_W);
						h_WPfMEt_pass_EE->Fill(PfMEt_W);
						bbb++;
						if(fabs(McNu_eta) < 2.5) 
						{
							h_mcWGenMEtin_pass_EE->Fill(GenMEt_W);
							h_mcWCaloMEtin_pass_EE->Fill(CaloMEt_W);
							h_mcWTcMEtin_pass_EE->Fill(TcMEt_W);
							h_mcWPfMEtin_pass_EE->Fill(PfMEt_W);
						}else{ 
							h_mcWGenMEtout_pass_EE->Fill(GenMEt_W); 
							h_mcWCaloMEtout_pass_EE->Fill(CaloMEt_W); 
							h_mcWTcMEtout_pass_EE->Fill(TcMEt_W); 
							h_mcWPfMEtout_pass_EE->Fill(PfMEt_W); 
						} 
					}
				}else
				{
					h_WGenMEt_fail->Fill(GenMEt_W); 
					h_WCaloMEt_fail->Fill(CaloMEt_W); 
					h_WTcMEt_fail->Fill(TcMEt_W); 
					h_WPfMEt_fail->Fill(PfMEt_W); 
					if(inBarrel) 
					{ 
						h_WGenMEt_fail_EB->Fill(GenMEt_W);
						h_WCaloMEt_fail_EB->Fill(CaloMEt_W);
						h_WTcMEt_fail_EB->Fill(TcMEt_W);
						h_WPfMEt_fail_EB->Fill(PfMEt_W);
						ccc++;
						if(fabs(McNu_eta) < 2.5) 
						{
							h_mcWGenMEtin_fail_EB->Fill(GenMEt_W);
							h_mcWCaloMEtin_fail_EB->Fill(CaloMEt_W);
							h_mcWTcMEtin_fail_EB->Fill(TcMEt_W);
							h_mcWPfMEtin_fail_EB->Fill(PfMEt_W);
						}else{ 
							h_mcWGenMEtout_fail_EB->Fill(GenMEt_W); 
							h_mcWCaloMEtout_fail_EB->Fill(CaloMEt_W); 
							h_mcWTcMEtout_fail_EB->Fill(TcMEt_W); 
							h_mcWPfMEtout_fail_EB->Fill(PfMEt_W); 
						} 
					}
					if(inEndcap) 
					{ 
						h_WGenMEt_fail_EE->Fill(GenMEt_W);
						h_WCaloMEt_fail_EE->Fill(CaloMEt_W);
						h_WTcMEt_fail_EE->Fill(TcMEt_W);
						h_WPfMEt_fail_EE->Fill(PfMEt_W);
						ddd++;
						if(fabs(McNu_eta) < 2.5) 
						{
							h_mcWGenMEtin_fail_EE->Fill(GenMEt_W);
							h_mcWCaloMEtin_fail_EE->Fill(CaloMEt_W);
							h_mcWTcMEtin_fail_EE->Fill(TcMEt_W);
							h_mcWPfMEtin_fail_EE->Fill(PfMEt_W);
						}else{ 
							h_mcWGenMEtout_fail_EE->Fill(GenMEt_W); 
							h_mcWCaloMEtout_fail_EE->Fill(CaloMEt_W); 
							h_mcWTcMEtout_fail_EE->Fill(TcMEt_W); 
							h_mcWPfMEtout_fail_EE->Fill(PfMEt_W); 
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

	int nTags, nProbes;
	int nTagMatched, nProbeMatched;
	double GenMEt, CaloMEt, PfMEt, TcMEt;
	double GenMEtphi, CaloMEtphi, PfMEtphi, TcMEtphi;
	double McZ_m, McZ_pt, McZ_y;
	double McZ_rescM, McZ_rescPt, McZ_rescY;
	double McTag_pt[4], McTag_eta[4];
	double McTag_rescPt[4], McTag_rescEta[4]; 
	double McProbe_pt[4], McProbe_eta[4];
	double McProbe_rescPt[4], McProbe_rescEta[4]; 
	double McTagProbe_dPhi[4], McTagProbe_dEta[4], McTagProbe_dR[4];
	int tag_q[4];
	double tag_pt[4], tag_eta[4], tag_phi[4];
	double tag_rescPt[4], tag_rescEta[4], tag_rescPhi[4];
	double tag_trackiso[4], tag_ecaliso[4], tag_hcaliso[4];
	double tag_sIhIh[4], tag_dPhi[4], tag_dEta[4];
	double tag_eop[4], tag_hoe[4], tag_pin[4], tag_pout[4];
	double tag_e1x5[4], tag_e2x5[4], tag_e5x5[4];
	int probe_q[4];
	double probe_pt[4], probe_eta[4], probe_phi[4];
	double probe_rescPt[4], probe_rescEta[4], probe_rescPhi[4];
	double probe_trackiso[4], probe_ecaliso[4], probe_hcaliso[4];
	double probe_sIhIh[4], probe_dPhi[4], probe_dEta[4];
	double probe_eop[4], probe_hoe[4], probe_pin[4], probe_pout[4];
	double probe_e1x5[4], probe_e2x5[4], probe_e5x5[4];
	double Z_m[4], Z_pt[4], Z_y[4], Z_phi[4];
	double Z_rescM[4], Z_rescPt[4], Z_rescY[4], Z_rescPhi[4];
	int sc_nClus[4];
	double sc_E[4], sc_rawE[4], sc_fEtaCorr[4], sc_fBremCorr[4];
	double ErsatzGenMEt[4], ErsatzCaloMEt[4], ErsatzPfMEt[4], ErsatzTcMEt[4], ErsatzT1MEt[4];
	//double ErsatzGenMEtPhi[4], ErsatzCaloMEtPhi[4], ErsatzPfMEtPhi[4], ErsatzTcMEtPhi[4];
	//double ErsatzGenMt[4], ErsatzCaloMt[4], ErsatzPfMt[4], ErsatzTcMt[4];
	double dpt_pf_gsf[4];
	double mesc[4];
	
	TBranch* bnTags = t->GetBranch("nTags");
	bnTags->SetAddress(&nTags);
	TBranch* bnProbes = t->GetBranch("nProbes");
	bnProbes->SetAddress(&nProbes);
	TBranch* bnTagMatched = t->GetBranch("nTagMatched");
	bnTagMatched->SetAddress(&nTagMatched);
	TBranch* bnProbeMatched = t->GetBranch("nProbeMatched");
	bnProbeMatched->SetAddress(&nProbeMatched);
	TBranch* bdpt_pf_gsf = t->GetBranch("dpt_pf_gsf");
	bdpt_pf_gsf->SetAddress(&dpt_pf_gsf);
	cout << "moooooooooooooo" << endl;
	//MET
	TBranch* bGenMEt = t->GetBranch("GenMEt");
	bGenMEt->SetAddress(&GenMEt);
	TBranch* bCaloMEt = t->GetBranch("CaloMEt");
	bCaloMEt->SetAddress(&CaloMEt);
	TBranch* bPfMEt = t->GetBranch("PfMEt");
	bPfMEt->SetAddress(&PfMEt);
	TBranch* bTcMEt = t->GetBranch("TcMEt");
	bTcMEt->SetAddress(&TcMEt);
	cout << "moooooooooooooo" << endl;
	//MET phi
	TBranch* bGenMEtphi = t->GetBranch("GenMEtphi");
	bGenMEtphi->SetAddress(&GenMEtphi);
	TBranch* bCaloMEtphi = t->GetBranch("CaloMEtphi");
	bCaloMEtphi->SetAddress(&CaloMEtphi);
	TBranch* bPfMEtphi = t->GetBranch("PfMEtphi");
	bPfMEtphi->SetAddress(&PfMEtphi);
	TBranch* bTcMEtphi = t->GetBranch("TcMEtphi");
	bTcMEtphi->SetAddress(&TcMEtphi);
	cout << "moooooooooooooo" << endl;
	//Mc particles
	TBranch* bMcZ_m = t->GetBranch("McZ_m");
	bMcZ_m->SetAddress(&McZ_m);
	TBranch* bMcZ_pt = t->GetBranch("McZ_Pt");
	bMcZ_pt->SetAddress(&McZ_pt);
	TBranch* bMcZ_y = t->GetBranch("McZ_y");
	bMcZ_y->SetAddress(&McZ_y);
	TBranch* bMcZ_rescM = t->GetBranch("McZ_rescM");
	bMcZ_rescM->SetAddress(&McZ_rescM);
	TBranch* bMcZ_rescPt = t->GetBranch("McZ_rescPt");
	bMcZ_rescPt->SetAddress(&McZ_rescPt);
	TBranch* bMcZ_rescY = t->GetBranch("McZ_rescY");
	bMcZ_rescY->SetAddress(&McZ_rescY);
	cout << "moooooooooooooo" << endl;
	TBranch* bMcTag_pt = t->GetBranch("McElec_pt");
	bMcTag_pt->SetAddress(&McTag_pt);
	TBranch* bMcTag_eta = t->GetBranch("McElec_eta");
	bMcTag_eta->SetAddress(&McTag_eta);
	TBranch* bMcTag_rescPt = t->GetBranch("McElec_rescPt");
	bMcTag_rescPt->SetAddress(&McTag_rescPt);
	TBranch* bMcTag_rescEta = t->GetBranch("McElec_rescEta");
	bMcTag_rescEta->SetAddress(&McTag_rescEta);
	cout << "moooooooooooooo" << endl;
	TBranch* bMcProbe_pt = t->GetBranch("McProbe_pt");
	bMcProbe_pt->SetAddress(&McProbe_pt);
	TBranch* bMcProbe_eta = t->GetBranch("McProbe_eta");
	bMcProbe_eta->SetAddress(&McProbe_eta);
	TBranch* bMcProbe_rescPt = t->GetBranch("McProbe_rescPt");
	bMcProbe_rescPt->SetAddress(&McProbe_rescPt);
	TBranch* bMcProbe_rescEta = t->GetBranch("McProbe_rescEta");
	bMcProbe_rescEta->SetAddress(&McProbe_rescEta);
	cout << "moooooooooooooo" << endl;
	TBranch* bMcTagProbe_dPhi = t->GetBranch("McElecProbe_dPhi");
	bMcTagProbe_dPhi->SetAddress(&McTagProbe_dPhi); 
	TBranch* bMcTagProbe_dEta = t->GetBranch("McElecProbe_dEta");
	bMcTagProbe_dEta->SetAddress(&McTagProbe_dEta); 
	TBranch* bMcTagProbe_dR = t->GetBranch("McElecProbe_dR");
	bMcTagProbe_dR->SetAddress(&McTagProbe_dR); 

	//Z boson properties
	TBranch* bZ_m = t->GetBranch("Z_m");
	bZ_m->SetAddress(&Z_m);
	TBranch* bZ_pt = t->GetBranch("Z_pt");
	bZ_pt->SetAddress(&Z_pt);
	TBranch* bZ_y = t->GetBranch("Z_y");
	bZ_y->SetAddress(&Z_y);
	//TBranch* bZ_eta = t->GetBranch("Z_eta");
	//bZ_eta->SetAddress(&Z_eta);
	TBranch* bZ_phi = t->GetBranch("Z_phi");
	bZ_phi->SetAddress(&Z_phi);
	TBranch* bZ_rescM = t->GetBranch("Z_rescM");
	bZ_rescM->SetAddress(&Z_rescM);
	TBranch* bZ_rescPt = t->GetBranch("Z_rescPt");
	bZ_rescPt->SetAddress(&Z_rescPt);
	TBranch* bZ_rescY = t->GetBranch("Z_rescY");
	bZ_rescY->SetAddress(&Z_rescY);
	//TBranch* bZ_rescEta = t->GetBranch("Z_rescEta");
	//bZ_rescEta->SetAddress(&Z_rescEta);
	TBranch* bZ_rescPhi = t->GetBranch("Z_rescPhi");
	bZ_rescPhi->SetAddress(&Z_rescPhi);
	cout << "moooooooooooooo" << endl;
	//tag properties
	TBranch* bTag_q = t->GetBranch("tag_q");
	bTag_q->SetAddress(&tag_q);
	TBranch* bTag_pt = t->GetBranch("tag_pt");
	bTag_pt->SetAddress(&tag_pt);
	TBranch* bTag_eta = t->GetBranch("tag_eta");
	bTag_eta->SetAddress(&tag_eta);
	TBranch* bTag_phi = t->GetBranch("tag_phi");
	bTag_phi->SetAddress(&tag_phi);
	TBranch* bTag_rescPt = t->GetBranch("tag_rescPt");
	bTag_rescPt->SetAddress(&tag_rescPt);
	TBranch* bTag_rescEta = t->GetBranch("tag_rescEta");
	bTag_rescEta->SetAddress(&tag_rescEta);
	TBranch* bTag_rescPhi = t->GetBranch("tag_rescPhi");
	bTag_rescPhi->SetAddress(&tag_rescPhi);
	TBranch* bTag_sIhIh = t->GetBranch("tag_sIhIh");
	bTag_sIhIh->SetAddress(&tag_sIhIh);
	TBranch* bTag_dPhi = t->GetBranch("tag_dPhiIn");
	bTag_dPhi->SetAddress(&tag_dPhi);
	TBranch* bTag_dEta = t->GetBranch("tag_dEtaIn");
	bTag_dEta->SetAddress(&tag_dEta);
	TBranch* bTag_tIso = t->GetBranch("tag_trckIso");
	bTag_tIso->SetAddress(&tag_trackiso);
	TBranch* bTag_eIso = t->GetBranch("tag_ecalIso");
	bTag_eIso->SetAddress(&tag_ecaliso);
	TBranch* bTag_hIso = t->GetBranch("tag_hcalIso");
	bTag_hIso->SetAddress(&tag_hcaliso);
	TBranch* bTag_pin = t->GetBranch("tag_pin");
	bTag_pin->SetAddress(&tag_pin);
	TBranch* bTag_pout = t->GetBranch("tag_pout");
	bTag_pout->SetAddress(&tag_pout);
	TBranch* bTag_eop = t->GetBranch("tag_eop");
	bTag_eop->SetAddress(&tag_eop);
	TBranch* bTag_hoe = t->GetBranch("tag_hoe");
	bTag_hoe->SetAddress(&tag_hoe);
	TBranch* bTag_e1x5 = t->GetBranch("tag_e1x5Max");
	bTag_e1x5->SetAddress(&tag_e1x5);
	TBranch* bTag_e2x5 = t->GetBranch("tag_e2x5Max");
	bTag_e2x5->SetAddress(&tag_e2x5);
	TBranch* bTag_e5x5 = t->GetBranch("tag_e5x5");
	bTag_e5x5->SetAddress(&tag_e5x5);
	//probe properties
	TBranch* bProbe_q = t->GetBranch("probe_q");
	bProbe_q->SetAddress(&probe_q);
	TBranch* bProbe_pt = t->GetBranch("probe_pt");
	bProbe_pt->SetAddress(&probe_pt);
	TBranch* bProbe_eta = t->GetBranch("probe_eta");
	bProbe_eta->SetAddress(&probe_eta);
	TBranch* bProbe_phi = t->GetBranch("probe_phi");
	bProbe_phi->SetAddress(&probe_phi);
	TBranch* bProbe_sIhIh = t->GetBranch("probe_sIhIh");
	bProbe_sIhIh->SetAddress(&probe_sIhIh);
	TBranch* bProbe_dPhi = t->GetBranch("probe_dPhiIn");
	bProbe_dPhi->SetAddress(&probe_dPhi);
	TBranch* bProbe_dEta = t->GetBranch("probe_dEtaIn");
	bProbe_dEta->SetAddress(&probe_dEta);
	TBranch* bProbe_tIso = t->GetBranch("probe_trckIso");
	bProbe_tIso->SetAddress(&probe_trackiso);
	TBranch* bProbe_eIso = t->GetBranch("probe_ecalIso");
	bProbe_eIso->SetAddress(&probe_ecaliso);
	TBranch* bProbe_hIso = t->GetBranch("probe_hcalIso");
	bProbe_hIso->SetAddress(&probe_hcaliso);
	TBranch* bProbe_pin = t->GetBranch("probe_pin");
	bProbe_pin->SetAddress(&probe_pin);
	TBranch* bProbe_pout = t->GetBranch("probe_pout");
	bProbe_pout->SetAddress(&probe_pout);
	TBranch* bProbe_eop = t->GetBranch("probe_eop");
	bProbe_eop->SetAddress(&probe_eop);
	TBranch* bProbe_hoe = t->GetBranch("probe_hoe");
	bProbe_hoe->SetAddress(&probe_hoe);
	TBranch* bProbe_e1x5 = t->GetBranch("probe_e1x5Max");
	bProbe_e1x5->SetAddress(&probe_e1x5);
	TBranch* bProbe_e2x5 = t->GetBranch("probe_e2x5Max");
	bProbe_e2x5->SetAddress(&probe_e2x5);
	TBranch* bProbe_e5x5 = t->GetBranch("probe_e5x5");
	bProbe_e5x5->SetAddress(&probe_e5x5);
	//TBranch* bProbe_HCAL = t->GetBranch("probe_HcalE015");
	//bProbe_HCAL->SetAddress(&probe_HCAL);
	//TBranch* bProbe_HCALEt = t->GetBranch("probe_HcalEt015");
	//bProbe_HCALEt->SetAddress(&probe_HCALEt);
	//Sc energy
	cout << "moooooooooooooo" << endl;
	//Ersatz MEt results
	TBranch* bErsatzGenMEt = t->GetBranch("ErsatzGenMEt");
	bErsatzGenMEt->SetAddress(&ErsatzGenMEt);
	TBranch* bErsatzCaloMEt = t->GetBranch("ErsatzCaloMEt");
	bErsatzCaloMEt->SetAddress(&ErsatzCaloMEt);
	TBranch* bErsatzPfMEt = t->GetBranch("ErsatzPfMEt");
	bErsatzPfMEt->SetAddress(&ErsatzPfMEt);
	TBranch* bErsatzTcMEt = t->GetBranch("ErsatzTcMEt");
	bErsatzTcMEt->SetAddress(&ErsatzTcMEt);
	cout << "moooooooooooooo" << endl;
	TBranch* bMesc = t->GetBranch("Ersatz_Mesc");
	bMesc->SetAddress(&mesc); 
	cout << "Set up Branches" << endl;

	aaa=0, bbb=0, ccc=0, ddd=0;
	for(int i=0; i < nEntries; ++i)
	{
		if(i%100000 == 0) cout <<"Processing event "<< i << endl;
		t->GetEntry(i);
		for(int j = 0; j < nProbes; ++j)
		{ 
			h_McTag_pt->Fill(McTag_pt[j]);
			h_McTag_rescPt->Fill(McTag_rescPt[j]);
			h_McTag_eta->Fill(McTag_eta[j]);
			h_McTag_rescEta->Fill(McTag_rescEta[j]);
			h_McProbe_pt->Fill(McProbe_pt[j]);
			h_McProbe_rescPt->Fill(McProbe_rescPt[j]);
			h_McProbe_eta->Fill(McProbe_eta[j]);
			h_McProbe_rescEta->Fill(McProbe_rescEta[j]);
			h_McTagProbe_dPhi->Fill(McTagProbe_dPhi[j]);
			h_McTagProbe_dEta->Fill(McTagProbe_dEta[j]);
			h_McTagProbe_dR->Fill(McTagProbe_dR[j]);
			h_McZ_m->Fill(McZ_m);
			h_McZ_rescM->Fill(McZ_rescM);
			h_McZ_pt->Fill(McZ_pt);
			h_McZ_rescPt->Fill(McZ_rescPt);
			h_McZ_y->Fill(McZ_y);
			h_McZ_rescY->Fill(McZ_rescY);
			h_tag_q->Fill(tag_q[j]);
			h_tag_pt->Fill(tag_pt[j]);
			h_tag_rescPt->Fill(tag_rescPt[j]);
			h_tag_eta->Fill(tag_eta[j]);
			h_tag_rescEta->Fill(tag_rescEta[j]);
			h_tag_ecaliso->Fill(tag_ecaliso[j]);
			h_tag_hcaliso->Fill(tag_hcaliso[j]);
			h_tag_trackiso->Fill(tag_trackiso[j]);
			h_tag_dEta->Fill(tag_dEta[j]);
			h_tag_dPhi->Fill(tag_dPhi[j]);
			h_tag_sIhIh->Fill(tag_sIhIh[j]);
			//h_tag_eop->Fill(tag_eop[j]);
			//h_tag_hoe->Fill(tag_hoe[j]);
			//h_tag_pin->Fill(tag_pin[j]);
			//h_tag_pout->Fill(tag_pout[j]);
			//h_tag_e1x5->Fill(tag_e1x5[j]);
			//h_tag_e2x5->Fill(tag_e2x5[j]);
			//h_tag_e5x5->Fill(tag_e5x5[j]);
			//h_Z_pt->Fill(Z_pt[j]);
			//h_Z_rescPt->Fill(Z_rescPt[j]);
			//h_Z_y->Fill(Z_y[j]);
			//h_Z_rescY->Fill(Z_rescY[j]);
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
                                //if(fabs(mesc[j]-91.1876) < 21.1876)
				//{
					bool pass_e_cuts = false;
					bool pass_trkiso_cut = false;
					bool inBarrel = false;
					bool inEndcap = false;
					if(fabs(tag_eta[j])<1.4442)
					{
						pass_e_cuts = (tag_ecaliso[j] < cECALiso_EB && tag_hcaliso[j] < cHCALiso_EB
								&& tag_sIhIh[j] < csIhIh_EB && tag_dPhi[j] < cDeltaPhi_EB
								&& tag_dEta[j] < cDeltaEta_EB);
						pass_trkiso_cut = (tag_trackiso[j] < cTrackiso_EB);
						inBarrel = true;
					}else if(fabs(tag_eta[j] < 2.5))
					{
						pass_e_cuts = (tag_ecaliso[j] < cECALiso_EE && tag_hcaliso[j] < cHCALiso_EE
								&& tag_sIhIh[j] < csIhIh_EE && tag_dPhi[j] < cDeltaPhi_EE
								&& tag_dEta[j] < cDeltaEta_EE);
						pass_trkiso_cut = (tag_trackiso[j] < cTrackiso_EE);
						inEndcap = true;
					}
					if(pass_e_cuts)
					{
						bool pass_probe_cuts = false;
						double f1x5 = probe_e1x5[j]/probe_e5x5[j];
						double f2x5 = probe_e2x5[j]/probe_e5x5[j];
						if(fabs(probe_eta[j]) < 1.4442)
						{
							pass_probe_cuts = (/*probe_HCAL[j] < cHCAL && (*/f1x5 > cf1x5 || f2x5 > cf2x5//)
									/*&& probe_elec[j] == celecmatch*/);
						}else if(fabs(probe_eta[j] < 2.5)){
							pass_probe_cuts = (/*probe_HCALEt[j] < cHCALEt &&*/ probe_sIhIh[j] < cnusIhIh
									/*&& probe_elec[j] == celecmatch*/);
						}
						if(pass_probe_cuts)
						{
							int EtaInt = int((probe_eta[j] + 3.)/0.01739);
							double weight = eventweight/nueff[EtaInt];
							if(pass_trkiso_cut)
							{
								h_ErsatzGenMEt_pass->Fill(ErsatzGenMEt[j], weight);
								h_ErsatzCaloMEt_pass->Fill(ErsatzCaloMEt[j], weight);
								h_ErsatzTcMEt_pass->Fill(ErsatzTcMEt[j], weight);
								h_ErsatzPfMEt_pass->Fill(ErsatzPfMEt[j], weight);
								h_nMatched->Fill(nTagMatched, weight);
								h_dpt_pf_gsf->Fill(dpt_pf_gsf[j], weight);
								if(inBarrel) 
								{
									aaa++;	
									h_ErsatzGenMEt_pass_EB->Fill(ErsatzGenMEt[j], weight);
									h_ErsatzCaloMEt_pass_EB->Fill(ErsatzCaloMEt[j], weight);
									h_ErsatzTcMEt_pass_EB->Fill(ErsatzTcMEt[j], weight);
									h_ErsatzPfMEt_pass_EB->Fill(ErsatzPfMEt[j], weight);
									h_nMatched_EB->Fill(nTagMatched, weight);
									h_dpt_pf_gsf_EB->Fill(dpt_pf_gsf[j], weight);
								}
								if(inEndcap)
								{
									bbb++;
									h_ErsatzGenMEt_pass_EE->Fill(ErsatzGenMEt[j], weight);
									h_ErsatzCaloMEt_pass_EE->Fill(ErsatzCaloMEt[j], weight);
									h_ErsatzTcMEt_pass_EE->Fill(ErsatzTcMEt[j], weight);
									h_ErsatzPfMEt_pass_EE->Fill(ErsatzPfMEt[j], weight);
									h_nMatched_EE->Fill(nTagMatched, weight);
									h_dpt_pf_gsf_EE->Fill(dpt_pf_gsf[j], weight);
								}	
							}else{
								h_ErsatzGenMEt_fail->Fill(ErsatzGenMEt[j], weight);
								h_ErsatzCaloMEt_fail->Fill(ErsatzCaloMEt[j], weight);
								h_ErsatzTcMEt_fail->Fill(ErsatzTcMEt[j], weight);
								h_ErsatzPfMEt_fail->Fill(ErsatzPfMEt[j], weight);
								if(inBarrel)
								{
									ccc++;
									h_ErsatzGenMEt_fail_EB->Fill(ErsatzGenMEt[j], weight);
									h_ErsatzCaloMEt_fail_EB->Fill(ErsatzCaloMEt[j], weight);
									h_ErsatzTcMEt_fail_EB->Fill(ErsatzTcMEt[j], weight);
									h_ErsatzPfMEt_fail_EB->Fill(ErsatzPfMEt[j], weight);
								}
								if(inEndcap)
								{
									ddd++;
									h_ErsatzGenMEt_fail_EE->Fill(ErsatzGenMEt[j], weight);
									h_ErsatzCaloMEt_fail_EE->Fill(ErsatzCaloMEt[j], weight);
									h_ErsatzTcMEt_fail_EE->Fill(ErsatzTcMEt[j], weight);
									h_ErsatzPfMEt_fail_EE->Fill(ErsatzPfMEt[j], weight);
								}	
							}
						}		
					}
				//}
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

	TCanvas* c_McBoson_m = new TCanvas("McBoson_m", "", 800, 600);
	c_McBoson_m->cd();
	h_McZ_m->Scale(1./h_McZ_m->Integral());
	h_McZ_m->SetLineColor(2);
	h_McZ_m->SetStats(kFALSE);
	h_McZ_m->SetTitle(kFALSE);
	h_McZ_m->GetYaxis()->SetTitleOffset(1.2);
	h_McZ_m->Draw();
	h_McW_m->Scale(1./h_McW_m->Integral());
	h_McW_m->SetLineColor(4);
	h_McW_m->Draw("same");

	TLegend* legend_e = new TLegend(0.8, 0.7, 0.99, 0.99);
	legend_e->AddEntry(h_McZ_m, "Z #rightarrow ee", "l");
	legend_e->AddEntry(h_McW_m, "W #rightarrow e#nu", "l");

	legend_e->Draw();
	c_McBoson_m->SaveAs("McBoson_m.png");
	delete c_McBoson_m;

	TCanvas* c_McBoson_pt = new TCanvas("McBoson_pt", "", 800, 600);
	c_McBoson_pt->cd();
	h_McZ_pt->Scale(1./h_McZ_pt->Integral());
	h_McZ_pt->SetLineColor(2);
	h_McZ_pt->SetStats(kFALSE);
	h_McZ_pt->SetTitle(kFALSE);
	h_McZ_pt->GetYaxis()->SetTitleOffset(1.2);
	h_McZ_pt->Draw();
	h_McW_pt->Scale(1./h_McW_pt->Integral());
	h_McW_pt->SetLineColor(4);
	h_McW_pt->Draw("same");
	legend_e->Draw();
	c_McBoson_pt->SaveAs("McBoson_pt.png");
	delete c_McBoson_pt;

	TCanvas* c_McBoson_y = new TCanvas("McBoson_y", "", 800, 600);
	c_McBoson_y->cd();
	h_McZ_y->Scale(1./h_McZ_y->Integral());
	h_McZ_y->SetLineColor(2);
	h_McZ_y->SetStats(kFALSE);
	h_McZ_y->SetTitle(kFALSE);
	h_McZ_y->GetYaxis()->SetTitleOffset(1.2);
	h_McZ_y->Draw();
	h_McW_y->Scale(1./h_McW_y->Integral());
	h_McW_y->SetLineColor(4);
	h_McW_y->Draw("same");
	legend_e->Draw();
	c_McBoson_y->SaveAs("McBoson_y.png");
	delete c_McBoson_y;

	TCanvas* c_McBoson_rescM = new TCanvas("McBoson_rescM", "", 800, 600);
	c_McBoson_rescM->cd();
	h_McZ_rescM->Scale(1./h_McZ_rescM->Integral());
	h_McZ_rescM->SetLineColor(2);
	h_McZ_rescM->SetStats(kFALSE);
	h_McZ_rescM->SetTitle(kFALSE);
	h_McZ_rescM->GetYaxis()->SetTitleOffset(1.2);
	h_McZ_rescM->Draw();
	h_McW_m->Scale(1./h_McW_m->Integral());
	h_McW_m->SetLineColor(4);
	h_McW_m->Draw("same");
	legend_e->Draw();
	c_McBoson_rescM->SaveAs("McBoson_rescM.png");
	delete c_McBoson_rescM;

	TCanvas* c_McBoson_rescPt = new TCanvas("McBoson_rescPt", "", 800, 600);
	c_McBoson_rescPt->cd();
	h_McZ_rescPt->Scale(1./h_McZ_rescPt->Integral());
	h_McZ_rescPt->SetLineColor(2);
	h_McZ_rescPt->SetStats(kFALSE);
	h_McZ_rescPt->SetTitle(kFALSE);
	h_McZ_rescPt->GetYaxis()->SetTitleOffset(1.2);
	h_McZ_rescPt->Draw();
	h_McW_pt->Scale(1./h_McW_pt->Integral());
	h_McW_pt->SetLineColor(4);
	h_McW_pt->Draw("same");
	legend_e->Draw();
	c_McBoson_rescPt->SaveAs("McBoson_rescPt.png");
	delete c_McBoson_rescPt;

	TCanvas* c_McBoson_rescY = new TCanvas("McBoson_rescY", "", 800, 600);
	c_McBoson_rescY->cd();
	h_McZ_rescY->Scale(1./h_McZ_rescY->Integral());
	h_McZ_rescY->SetLineColor(2);
	h_McZ_rescY->SetStats(kFALSE);
	h_McZ_rescY->SetTitle(kFALSE);
	h_McZ_rescY->GetYaxis()->SetTitleOffset(1.2);
	h_McZ_rescY->Draw();
	h_McW_y->Scale(1./h_McW_y->Integral());
	h_McW_y->SetLineColor(4);
	h_McW_y->Draw("same");
	legend_e->Draw();
	c_McBoson_rescY->SaveAs("McBoson_rescY.png");
	delete c_McBoson_rescY;

	TCanvas* c_McElec_pt = new TCanvas("McElec_pt", "", 800, 600);
	c_McElec_pt->cd();
	h_McTag_pt->Scale(1./h_McTag_pt->Integral());
	h_McTag_pt->SetLineColor(2);
	h_McTag_pt->SetStats(kFALSE);
	h_McTag_pt->SetTitle(kFALSE);
	h_McTag_pt->GetYaxis()->SetTitleOffset(1.2);
	h_McTag_pt->Draw();
	h_McElec_pt->Scale(1./h_McElec_pt->Integral());
	h_McElec_pt->SetLineColor(4);
	h_McElec_pt->Draw("same");
	legend_e->Draw();
	c_McElec_pt->SaveAs("McElec_pt.png");
	delete c_McElec_pt;

	TCanvas* c_McElec_eta = new TCanvas("McElec_eta", "", 800, 600);
	c_McElec_eta->cd();
	h_McTag_eta->Scale(1./h_McTag_eta->Integral());
	h_McTag_eta->SetLineColor(2);
	h_McTag_eta->SetStats(kFALSE);
	h_McTag_eta->SetTitle(kFALSE);
	h_McTag_eta->GetYaxis()->SetTitleOffset(1.2);
	h_McTag_eta->Draw();
	h_McElec_eta->Scale(1./h_McElec_eta->Integral());
	h_McElec_eta->SetLineColor(4);
	h_McElec_eta->Draw("same");
	legend_e->Draw();
	c_McElec_eta->SaveAs("McElec_eta.png");
	delete c_McElec_eta;

	TCanvas* c_McElec_rescPt = new TCanvas("McElec_rescPt", "", 800, 600);
	c_McElec_rescPt->cd();
	h_McTag_rescPt->Scale(1./h_McTag_rescPt->Integral());
	h_McTag_rescPt->SetLineColor(2);
	h_McTag_rescPt->SetStats(kFALSE);
	h_McTag_rescPt->SetTitle(kFALSE);
	h_McTag_rescPt->GetYaxis()->SetTitleOffset(1.2);
	h_McTag_rescPt->Draw();
	h_McElec_pt->Scale(1./h_McElec_pt->Integral());
	h_McElec_pt->SetLineColor(4);
	h_McElec_pt->Draw("same");
	legend_e->Draw();
	c_McElec_rescPt->SaveAs("McElec_rescPt.png");
	delete c_McElec_rescPt;

	TCanvas* c_McElec_rescEta = new TCanvas("McElec_rescEta", "", 800, 600);
	c_McElec_rescEta->cd();
	h_McTag_rescEta->Scale(1./h_McTag_rescEta->Integral());
	h_McTag_rescEta->SetLineColor(2);
	h_McTag_rescEta->SetStats(kFALSE);
	h_McTag_rescEta->SetTitle(kFALSE);
	h_McTag_rescEta->GetYaxis()->SetTitleOffset(1.2);
	h_McTag_rescEta->Draw();
	h_McElec_eta->Scale(1./h_McElec_eta->Integral());
	h_McElec_eta->SetLineColor(4);
	h_McElec_eta->Draw("same");
	legend_e->Draw();
	c_McElec_rescEta->SaveAs("McElec_rescEta.png");
	delete c_McElec_rescEta;

	TCanvas* c_McNu_pt = new TCanvas("McNu_pt", "", 800, 600);
	c_McNu_pt->cd();
	h_McProbe_pt->Scale(1./h_McProbe_pt->Integral());
	h_McProbe_pt->SetLineColor(2);
	h_McProbe_pt->SetStats(kFALSE);
	h_McProbe_pt->SetTitle(kFALSE);
	h_McProbe_pt->GetYaxis()->SetTitleOffset(1.2);
	h_McProbe_pt->Draw();
	h_McNu_pt->Scale(1./h_McNu_pt->Integral());
	h_McNu_pt->SetLineColor(4);
	h_McNu_pt->Draw("same");
	legend_e->Draw();
	c_McNu_pt->SaveAs("McNu_pt.png");
	delete c_McNu_pt;

	TCanvas* c_McNu_eta = new TCanvas("McNu_eta", "", 800, 600);
	c_McNu_eta->cd();
	h_McProbe_eta->Scale(1./h_McProbe_eta->Integral());
	h_McProbe_eta->SetLineColor(2);
	h_McProbe_eta->SetStats(kFALSE);
	h_McProbe_eta->SetTitle(kFALSE);
	h_McProbe_eta->GetYaxis()->SetTitleOffset(1.2);
	h_McProbe_eta->Draw();
	h_McNu_eta->Scale(1./h_McNu_eta->Integral());
	h_McNu_eta->SetLineColor(4);
	h_McNu_eta->Draw("same");
	legend_e->Draw();
	c_McNu_eta->SaveAs("McNu_eta.png");
	delete c_McNu_eta;

	TCanvas* c_McNu_rescPt = new TCanvas("McNu_rescPt", "", 800, 600);
	c_McNu_rescPt->cd();
	h_McProbe_rescPt->Scale(1./h_McProbe_rescPt->Integral());
	h_McProbe_rescPt->SetLineColor(2);
	h_McProbe_rescPt->SetStats(kFALSE);
	h_McProbe_rescPt->SetTitle(kFALSE);
	h_McProbe_rescPt->GetYaxis()->SetTitleOffset(1.2);
	h_McProbe_rescPt->Draw();
	h_McNu_pt->Scale(1./h_McNu_pt->Integral());
	h_McNu_pt->SetLineColor(4);
	h_McNu_pt->Draw("same");
	legend_e->Draw();
	c_McNu_rescPt->SaveAs("McNu_rescPt.png");
	delete c_McNu_rescPt;

	TCanvas* c_McNu_rescEta = new TCanvas("McNu_rescEta", "", 800, 600);
	c_McNu_rescEta->cd();
	h_McProbe_rescEta->Scale(1./h_McProbe_rescEta->Integral());
	h_McProbe_rescEta->SetLineColor(2);
	h_McProbe_rescEta->SetStats(kFALSE);
	h_McProbe_rescEta->SetTitle(kFALSE);
	h_McProbe_rescEta->GetYaxis()->SetTitleOffset(1.2);
	h_McProbe_rescEta->Draw();
	h_McNu_eta->Scale(1./h_McNu_eta->Integral());
	h_McNu_eta->SetLineColor(4);
	h_McNu_eta->Draw("same");
	legend_e->Draw();
	c_McNu_rescEta->SaveAs("McNu_rescEta.png");
	delete c_McNu_rescEta;

	TCanvas* c_McLeptons_dPhi = new TCanvas("McLeptons_dPhi", "", 800, 600);
	c_McLeptons_dPhi->cd();
	h_McTagProbe_dPhi->Scale(1./h_McTagProbe_dPhi->Integral());
	h_McTagProbe_dPhi->SetLineColor(2);
	h_McTagProbe_dPhi->SetStats(kFALSE);
	h_McTagProbe_dPhi->SetTitle(kFALSE);
	h_McTagProbe_dPhi->GetYaxis()->SetTitleOffset(1.2);
	h_McTagProbe_dPhi->Draw();
	h_McElecNu_dPhi->Scale(1./h_McElecNu_dPhi->Integral());
	h_McElecNu_dPhi->SetLineColor(4);
	h_McElecNu_dPhi->Draw("same");
	legend_e->Draw();
	c_McLeptons_dPhi->SaveAs("McLeptons_dPhi.png");
	delete c_McLeptons_dPhi;

	TCanvas* c_McLeptons_dEta = new TCanvas("McLeptons_dEta", "", 800, 600);
	c_McLeptons_dEta->cd();
	h_McTagProbe_dEta->Scale(1./h_McTagProbe_dEta->Integral());
	h_McTagProbe_dEta->SetLineColor(2);
	h_McTagProbe_dEta->SetStats(kFALSE);
	h_McTagProbe_dEta->SetTitle(kFALSE);
	h_McTagProbe_dEta->GetYaxis()->SetTitleOffset(1.2);
	h_McTagProbe_dEta->Draw();
	h_McElecNu_dEta->Scale(1./h_McElecNu_dEta->Integral());
	h_McElecNu_dEta->SetLineColor(4);
	h_McElecNu_dEta->Draw("same");
	legend_e->Draw();
	c_McLeptons_dEta->SaveAs("McLeptons_dEta.png");
	delete c_McLeptons_dEta;

	TCanvas* c_McLeptons_dR = new TCanvas("McLeptons_dR", "", 800, 600);
	c_McLeptons_dR->cd();
	h_McTagProbe_dR->Scale(1./h_McTagProbe_dR->Integral());
	h_McTagProbe_dR->SetLineColor(2);
	h_McTagProbe_dR->SetStats(kFALSE);
	h_McTagProbe_dR->SetTitle(kFALSE);
	h_McTagProbe_dR->GetYaxis()->SetTitleOffset(1.2);
	h_McTagProbe_dR->Draw();
	h_McElecNu_dR->Scale(1./h_McElecNu_dR->Integral());
	h_McElecNu_dR->SetLineColor(4);
	h_McElecNu_dR->Draw("same");
	legend_e->Draw();
	c_McLeptons_dR->SaveAs("McLeptons_dR.png");
	delete c_McLeptons_dR;

/*
	TCanvas* c_elec_q = new TCanvas("elec_q", "", 800, 600);
	c_elec_q->cd();
	h_tag_q->Scale(1./h_tag_q->Integral());
	h_tag_q->SetLineColor(2);
	h_tag_q->SetStats(kFALSE);
	h_tag_q->SetTitle(kFALSE);
	h_tag_q->GetYaxis()->SetTitleOffset(1.2);
	h_tag_q->Draw();
	h_elec_q->Scale(1./h_elec_q->Integral());
	h_elec_q->SetLineColor(4);
	h_elec_q->Draw("same");

	TLegend* legend_e = new TLegend(0.8, 0.7, 0.99, 0.99);
	legend_e->AddEntry(h_tag_q, "Z -> ee", "l");
	legend_e->AddEntry(h_elec_q, "W -> e#nu", "l");

	legend_e->Draw();
	c_elec_q->SaveAs("elec_q.png");
	delete c_elec_q;
*/
	TCanvas* c_elec_pt = new TCanvas("elec_pt", "", 800, 600);
	c_elec_pt->cd();
	h_tag_pt->Scale(1./h_tag_pt->Integral());
	h_tag_pt->SetLineColor(2);
	h_tag_pt->SetStats(kFALSE);
	h_tag_pt->SetTitle(kFALSE);
	h_tag_pt->GetYaxis()->SetTitleOffset(1.2);
	h_tag_pt->Draw();
	h_elec_pt->Scale(1./h_elec_pt->Integral());
	h_elec_pt->SetLineColor(4);
	h_elec_pt->Draw("same");
	legend_e->Draw();
	c_elec_pt->SaveAs("elec_pt.png");
	delete c_elec_pt;

	TCanvas* c_elec_eta = new TCanvas("elec_eta", "", 800, 600);
	c_elec_eta->cd();
	h_tag_eta->Scale(1./h_tag_eta->Integral());
	h_tag_eta->SetLineColor(2);
	h_tag_eta->SetStats(kFALSE);
	h_tag_eta->SetTitle(kFALSE);
	h_tag_eta->GetYaxis()->SetTitleOffset(1.2);
	h_tag_eta->Draw();
	h_elec_eta->Scale(1./h_elec_eta->Integral());
	h_elec_eta->SetLineColor(4);
	h_elec_eta->Draw("same");
	legend_e->Draw();
	c_elec_eta->SaveAs("elec_eta.png");
	delete c_elec_eta;

	TCanvas* c_elec_rescPt = new TCanvas("elec_rescPt", "", 800, 600);
	c_elec_rescPt->cd();
	h_tag_rescPt->Scale(1./h_tag_rescPt->Integral());
	h_tag_rescPt->SetLineColor(2);
	h_tag_rescPt->SetStats(kFALSE);
	h_tag_rescPt->SetTitle(kFALSE);
	h_tag_rescPt->GetYaxis()->SetTitleOffset(1.2);
	h_tag_rescPt->Draw();
	h_elec_pt->Scale(1./h_elec_pt->Integral());
	h_elec_pt->SetLineColor(4);
	h_elec_pt->Draw("same");
	legend_e->Draw();
	c_elec_rescPt->SaveAs("elec_rescPt.png");
	delete c_elec_rescPt;

	TCanvas* c_elec_rescEta = new TCanvas("elec_rescEta", "", 800, 600);
	c_elec_rescEta->cd();
	h_tag_rescEta->Scale(1./h_tag_rescEta->Integral());
	h_tag_rescEta->SetLineColor(2);
	h_tag_rescEta->SetStats(kFALSE);
	h_tag_rescEta->SetTitle(kFALSE);
	h_tag_rescEta->GetYaxis()->SetTitleOffset(1.2);
	h_tag_rescEta->Draw();
	h_elec_eta->Scale(1./h_elec_eta->Integral());
	h_elec_eta->SetLineColor(4);
	h_elec_eta->Draw("same");
	legend_e->Draw();
	c_elec_rescEta->SaveAs("elec_rescEta.png");
	delete c_elec_rescEta;

	TCanvas* c_elec_ecaliso = new TCanvas("elec_ecaliso", "", 800, 600);
	c_elec_ecaliso->cd();
	h_tag_ecaliso->Scale(1./h_tag_ecaliso->Integral());
	h_tag_ecaliso->SetLineColor(2);
	h_tag_ecaliso->SetStats(kFALSE);
	h_tag_ecaliso->SetTitle(kFALSE);
	h_tag_ecaliso->GetYaxis()->SetTitleOffset(1.2);
	h_tag_ecaliso->Draw();
	h_elec_ecaliso->Scale(1./h_elec_ecaliso->Integral());
	h_elec_ecaliso->SetLineColor(4);
	h_elec_ecaliso->Draw("same");
	legend_e->Draw();
	c_elec_ecaliso->SaveAs("elec_ecaliso.png");
	delete c_elec_ecaliso;

	TCanvas* c_elec_hcaliso = new TCanvas("elec_hcaliso", "", 800, 600);
	c_elec_hcaliso->cd();
	h_tag_hcaliso->Scale(1./h_tag_hcaliso->Integral());
	h_tag_hcaliso->SetLineColor(2);
	h_tag_hcaliso->SetStats(kFALSE);
	h_tag_hcaliso->SetTitle(kFALSE);
	h_tag_hcaliso->GetYaxis()->SetTitleOffset(1.2);
	h_tag_hcaliso->Draw();
	h_elec_hcaliso->Scale(1./h_elec_hcaliso->Integral());
	h_elec_hcaliso->SetLineColor(4);
	h_elec_hcaliso->Draw("same");
	legend_e->Draw();
	c_elec_hcaliso->SaveAs("elec_hcaliso.png");
	delete c_elec_hcaliso;

	TCanvas* c_elec_trackiso = new TCanvas("elec_trackiso", "", 800, 600);
	c_elec_trackiso->cd();
	h_tag_trackiso->Scale(1./h_tag_trackiso->Integral());
	h_tag_trackiso->SetLineColor(2);
	h_tag_trackiso->SetStats(kFALSE);
	h_tag_trackiso->SetTitle(kFALSE);
	h_tag_trackiso->GetYaxis()->SetTitleOffset(1.2);
	h_tag_trackiso->Draw();
	h_elec_trackiso->Scale(1./h_elec_trackiso->Integral());
	h_elec_trackiso->SetLineColor(4);
	h_elec_trackiso->Draw("same");
	legend_e->Draw();
	c_elec_trackiso->SaveAs("elec_trackiso.png");
	delete c_elec_trackiso;

	TCanvas* c_elec_dEtaIn = new TCanvas("elec_dEtaIn", "", 800, 600);
	c_elec_dEtaIn->cd();
	h_tag_dEta->Scale(1./h_tag_dEta->Integral());
	h_tag_dEta->SetLineColor(2);
	h_tag_dEta->SetStats(kFALSE);
	h_tag_dEta->SetTitle(kFALSE);
	h_tag_dEta->GetYaxis()->SetTitleOffset(1.2);
	h_tag_dEta->Draw();
	h_elec_dEta->Scale(1./h_elec_dEta->Integral());
	h_elec_dEta->SetLineColor(4);
	h_elec_dEta->Draw("same");
	legend_e->Draw();
	c_elec_dEtaIn->SaveAs("elec_dEtaIn.png");
	delete c_elec_dEtaIn;

	TCanvas* c_elec_dPhiIn = new TCanvas("elec_dPhiIn", "", 800, 600);
	c_elec_dPhiIn->cd();
	h_tag_dPhi->Scale(1./h_tag_dPhi->Integral());
	h_tag_dPhi->SetLineColor(2);
	h_tag_dPhi->SetStats(kFALSE);
	h_tag_dPhi->SetTitle(kFALSE);
	h_tag_dPhi->GetYaxis()->SetTitleOffset(1.2);
	h_tag_dPhi->Draw();
	h_elec_dPhi->Scale(1./h_elec_dPhi->Integral());
	h_elec_dPhi->SetLineColor(4);
	h_elec_dPhi->Draw("same");
	legend_e->Draw();
	c_elec_dPhiIn->SaveAs("elec_dPhiIn.png");
	delete c_elec_dPhiIn;

	TCanvas* c_elec_sIhIh = new TCanvas("elec_sIhIh", "", 800, 600);
	c_elec_sIhIh->cd();
	h_tag_sIhIh->Scale(1./h_tag_sIhIh->Integral());
	h_tag_sIhIh->SetLineColor(2);
	h_tag_sIhIh->SetStats(kFALSE);
	h_tag_sIhIh->SetTitle(kFALSE);
	h_tag_sIhIh->GetYaxis()->SetTitleOffset(1.2);
	h_tag_sIhIh->Draw();
	h_elec_sIhIh->Scale(1./h_elec_sIhIh->Integral());
	h_elec_sIhIh->SetLineColor(4);
	h_elec_sIhIh->Draw("same");
	legend_e->Draw();
	c_elec_sIhIh->SaveAs("elec_sIhIh.png");
	delete c_elec_sIhIh;

	TCanvas* c_nMatched = new TCanvas("nMatched", "", 800, 600);
	c_nMatched->cd();
	h_nMatched->Draw();
	c_nMatched->SaveAs("nMatched.png");
	delete c_nMatched;

	TCanvas* c_nMatched_EB = new TCanvas("nMatched_EB", "", 800, 600);
	c_nMatched_EB->cd();
	h_nMatched_EB->Draw();
	c_nMatched_EB->SaveAs("nMatched_EB.png");
	delete c_nMatched_EB;

	TCanvas* c_nMatched_EE = new TCanvas("nMatched_EE", "", 800, 600);
	c_nMatched_EE->cd();
	h_nMatched_EE->Draw();
	c_nMatched_EE->SaveAs("nMatched_EE.png");
	delete c_nMatched_EE;

	TCanvas* c_dpt_pf_gsf = new TCanvas("dpt_pf_gsf", "", 800, 600);
	c_dpt_pf_gsf->cd();
	h_dpt_pf_gsf->Draw();
	c_dpt_pf_gsf->SaveAs("dpt_pf_gsf.png");
	delete c_dpt_pf_gsf;

	TCanvas* c_dpt_pf_gsf_EB = new TCanvas("dpt_pf_gsf_EB", "", 800, 600);
	c_dpt_pf_gsf_EB->cd();
	h_dpt_pf_gsf_EB->Draw();
	c_dpt_pf_gsf_EB->SaveAs("dpt_pf_gsf_EB.png");
	delete c_dpt_pf_gsf_EB;

	TCanvas* c_dpt_pf_gsf_EE = new TCanvas("dpt_pf_gsf_EE", "", 800, 600);
	c_dpt_pf_gsf_EE->cd();
	h_dpt_pf_gsf_EE->Draw();
	c_dpt_pf_gsf_EE->SaveAs("dpt_pf_gsf_EE.png");
	delete c_dpt_pf_gsf_EE;

	cout << "Apply acceptance correction ..." << endl;	
	
	for (int i=1; i<101; i++)
	{
		if(h_mcWCaloMEtin_pass_EB->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_caloMEt_pass_EB->SetBinContent(i, 1. + h_mcWCaloMEtout_pass_EB->GetBinContent(i)/h_mcWCaloMEtin_pass_EB->GetBinContent(i)); 
		}else{
			h_acceptance_correction_caloMEt_pass_EB->SetBinContent(i, 1.);
		}
		if(h_mcWCaloMEtin_pass_EE->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_caloMEt_pass_EE->SetBinContent(i, 1. + h_mcWCaloMEtout_pass_EE->GetBinContent(i)/h_mcWCaloMEtin_pass_EE->GetBinContent(i)); 
		}else{
			h_acceptance_correction_caloMEt_pass_EE->SetBinContent(i, 1.);
		}
		if(h_mcWCaloMEtin_fail_EB->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_caloMEt_fail_EB->SetBinContent(i, 1. + h_mcWCaloMEtout_fail_EB->GetBinContent(i)/h_mcWCaloMEtin_fail_EB->GetBinContent(i)); 
		}else{
			h_acceptance_correction_caloMEt_fail_EB->SetBinContent(i, 1.);
		}
		if(h_mcWCaloMEtin_fail_EE->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_caloMEt_fail_EE->SetBinContent(i, 1. + h_mcWCaloMEtout_fail_EE->GetBinContent(i)/h_mcWCaloMEtin_fail_EE->GetBinContent(i)); 
		}else{
			h_acceptance_correction_caloMEt_fail_EE->SetBinContent(i, 1.);
		}

		if(h_mcWTcMEtin_pass_EB->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_tcMEt_pass_EB->SetBinContent(i, 1. + h_mcWTcMEtout_pass_EB->GetBinContent(i)/h_mcWTcMEtin_pass_EB->GetBinContent(i)); 
		}else{
			h_acceptance_correction_tcMEt_pass_EB->SetBinContent(i, 1.);
		}
		if(h_mcWTcMEtin_pass_EE->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_tcMEt_pass_EE->SetBinContent(i, 1. + h_mcWTcMEtout_pass_EE->GetBinContent(i)/h_mcWTcMEtin_pass_EE->GetBinContent(i)); 
		}else{
			h_acceptance_correction_tcMEt_pass_EE->SetBinContent(i, 1.);
		}
		if(h_mcWTcMEtin_fail_EB->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_tcMEt_fail_EB->SetBinContent(i, 1. + h_mcWTcMEtout_fail_EB->GetBinContent(i)/h_mcWTcMEtin_fail_EB->GetBinContent(i)); 
		}else{
			h_acceptance_correction_tcMEt_fail_EB->SetBinContent(i, 1.);
		}
		if(h_mcWTcMEtin_fail_EE->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_tcMEt_fail_EE->SetBinContent(i, 1. + h_mcWTcMEtout_fail_EE->GetBinContent(i)/h_mcWTcMEtin_fail_EE->GetBinContent(i)); 
		}else{
			h_acceptance_correction_tcMEt_fail_EE->SetBinContent(i, 1.);
		}

		if(h_mcWGenMEtin_pass_EB->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_genMEt_pass_EB->SetBinContent(i, 1. + h_mcWGenMEtout_pass_EB->GetBinContent(i)/h_mcWGenMEtin_pass_EB->GetBinContent(i)); 
		}else{
			h_acceptance_correction_genMEt_pass_EB->SetBinContent(i, 1.);
		}
		if(h_mcWGenMEtin_pass_EE->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_genMEt_pass_EE->SetBinContent(i, 1. + h_mcWGenMEtout_pass_EE->GetBinContent(i)/h_mcWGenMEtin_pass_EE->GetBinContent(i)); 
		}else{
			h_acceptance_correction_genMEt_pass_EE->SetBinContent(i, 1.);
		}
		if(h_mcWGenMEtin_fail_EB->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_genMEt_fail_EB->SetBinContent(i, 1. + h_mcWGenMEtout_fail_EB->GetBinContent(i)/h_mcWGenMEtin_fail_EB->GetBinContent(i)); 
		}else{
			h_acceptance_correction_genMEt_fail_EB->SetBinContent(i, 1.);
		}
		if(h_mcWGenMEtin_fail_EE->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_genMEt_fail_EE->SetBinContent(i, 1. + h_mcWGenMEtout_fail_EE->GetBinContent(i)/h_mcWGenMEtin_fail_EE->GetBinContent(i)); 
		}else{
			h_acceptance_correction_genMEt_fail_EE->SetBinContent(i, 1.);
		}

		if(h_mcWPfMEtin_pass_EB->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_pfMEt_pass_EB->SetBinContent(i, 1. + h_mcWPfMEtout_pass_EB->GetBinContent(i)/h_mcWPfMEtin_pass_EB->GetBinContent(i)); 
		}else{
			h_acceptance_correction_pfMEt_pass_EB->SetBinContent(i, 1.);
		}
		if(h_mcWPfMEtin_pass_EE->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_pfMEt_pass_EE->SetBinContent(i, 1. + h_mcWPfMEtout_pass_EE->GetBinContent(i)/h_mcWPfMEtin_pass_EE->GetBinContent(i)); 
		}else{
			h_acceptance_correction_pfMEt_pass_EE->SetBinContent(i, 1.);
		}
		if(h_mcWPfMEtin_fail_EB->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_pfMEt_fail_EB->SetBinContent(i, 1. + h_mcWPfMEtout_fail_EB->GetBinContent(i)/h_mcWPfMEtin_fail_EB->GetBinContent(i)); 
		}else{
			h_acceptance_correction_pfMEt_fail_EB->SetBinContent(i, 1.);
		}
		if(h_mcWPfMEtin_fail_EE->GetBinContent(i) != 0.) 
		{
			h_acceptance_correction_pfMEt_fail_EE->SetBinContent(i, 1. + h_mcWPfMEtout_fail_EE->GetBinContent(i)/h_mcWPfMEtin_fail_EE->GetBinContent(i)); 
		}else{
			h_acceptance_correction_pfMEt_fail_EE->SetBinContent(i, 1.);
		}

	}

	h_ErsatzGenMEt_pass_EB->Multiply(h_ErsatzGenMEt_pass_EB, h_acceptance_correction_genMEt_pass_EB);
	h_ErsatzCaloMEt_pass_EB->Multiply(h_ErsatzCaloMEt_pass_EB, h_acceptance_correction_caloMEt_pass_EB);
	h_ErsatzTcMEt_pass_EB->Multiply(h_ErsatzTcMEt_pass_EB, h_acceptance_correction_tcMEt_pass_EB);
	h_ErsatzPfMEt_pass_EB->Multiply(h_ErsatzPfMEt_pass_EB, h_acceptance_correction_pfMEt_pass_EB);

	h_ErsatzGenMEt_pass_EE->Multiply(h_ErsatzGenMEt_pass_EE, h_acceptance_correction_genMEt_pass_EE);
	h_ErsatzCaloMEt_pass_EE->Multiply(h_ErsatzCaloMEt_pass_EE, h_acceptance_correction_caloMEt_pass_EE);
	h_ErsatzTcMEt_pass_EE->Multiply(h_ErsatzTcMEt_pass_EE, h_acceptance_correction_tcMEt_pass_EE);
	h_ErsatzPfMEt_pass_EE->Multiply(h_ErsatzPfMEt_pass_EE, h_acceptance_correction_pfMEt_pass_EE);

	h_ErsatzGenMEt_fail_EB->Multiply(h_ErsatzGenMEt_fail_EB, h_acceptance_correction_genMEt_fail_EB);
	h_ErsatzCaloMEt_fail_EB->Multiply(h_ErsatzCaloMEt_fail_EB, h_acceptance_correction_caloMEt_fail_EB);
	h_ErsatzTcMEt_fail_EB->Multiply(h_ErsatzTcMEt_fail_EB, h_acceptance_correction_tcMEt_fail_EB);
	h_ErsatzPfMEt_fail_EB->Multiply(h_ErsatzPfMEt_fail_EB, h_acceptance_correction_pfMEt_fail_EB);

	h_ErsatzGenMEt_fail_EE->Multiply(h_ErsatzGenMEt_fail_EE, h_acceptance_correction_genMEt_fail_EE);
	h_ErsatzCaloMEt_fail_EE->Multiply(h_ErsatzCaloMEt_fail_EE, h_acceptance_correction_caloMEt_fail_EE);
	h_ErsatzTcMEt_fail_EE->Multiply(h_ErsatzTcMEt_fail_EE, h_acceptance_correction_tcMEt_fail_EE);
	h_ErsatzPfMEt_fail_EE->Multiply(h_ErsatzPfMEt_fail_EE, h_acceptance_correction_pfMEt_fail_EE);

	h_ErsatzGenMEt_pass->Add(h_ErsatzGenMEt_pass_EB, h_ErsatzGenMEt_pass_EE);
	h_ErsatzCaloMEt_pass->Add(h_ErsatzCaloMEt_pass_EB, h_ErsatzCaloMEt_pass_EE);
	h_ErsatzTcMEt_pass->Add(h_ErsatzTcMEt_pass_EB, h_ErsatzTcMEt_pass_EE); 
	h_ErsatzPfMEt_pass->Add(h_ErsatzPfMEt_pass_EB, h_ErsatzPfMEt_pass_EE);

	h_ErsatzGenMEt_fail->Add(h_ErsatzGenMEt_fail_EB, h_ErsatzGenMEt_fail_EE);
	h_ErsatzCaloMEt_fail->Add(h_ErsatzCaloMEt_fail_EB, h_ErsatzCaloMEt_fail_EE);
	h_ErsatzTcMEt_fail->Add(h_ErsatzTcMEt_fail_EB, h_ErsatzTcMEt_fail_EE); 
	h_ErsatzPfMEt_fail->Add(h_ErsatzPfMEt_fail_EB, h_ErsatzPfMEt_fail_EE);

	//Fill Ersatz MET histogram with the type of MET you want
	for(int i=1; i<101; i++)
	{
		h_ErsatzMEt_pass_EB->SetBinContent(i, h_ErsatzCaloMEt_pass_EB->GetBinContent(i));
		h_ErsatzMEt_pass_EE->SetBinContent(i, h_ErsatzCaloMEt_pass_EE->GetBinContent(i));
		h_ErsatzMEt_fail_EB->SetBinContent(i, h_ErsatzCaloMEt_fail_EB->GetBinContent(i));
		h_ErsatzMEt_fail_EE->SetBinContent(i, h_ErsatzCaloMEt_fail_EE->GetBinContent(i));
		h_ErsatzMEt_pass->SetBinContent(i, h_ErsatzCaloMEt_pass->GetBinContent(i));
		h_ErsatzMEt_fail->SetBinContent(i, h_ErsatzCaloMEt_fail->GetBinContent(i));
	}
        
	TCanvas* c_ErsatzCaloMEt_corr_pass_EB =  new TCanvas("ErsatzCaloMEt_corr_pass_EB", "", 800, 600);	
	c_ErsatzCaloMEt_corr_pass_EB->cd();
	//h_ErsatzCaloMEt_pass_EB->Scale(1./h_ErsatzCaloMEt_pass_EB->Integral());
	h_ErsatzCaloMEt_pass_EB->SetLineColor(2);
	h_ErsatzCaloMEt_pass_EB->SetStats(kFALSE);
	h_ErsatzCaloMEt_pass_EB->SetTitle(kFALSE);
	h_ErsatzCaloMEt_pass_EB->GetYaxis()->SetTitleOffset(1.2);
	h_ErsatzCaloMEt_pass_EB->Draw();
	h_WCaloMEt_pass_EB->Scale(h_ErsatzCaloMEt_pass_EB->Integral()/h_WCaloMEt_pass_EB->Integral());
	h_WCaloMEt_pass_EB->SetLineColor(4);
	h_WCaloMEt_pass_EB->Draw("same");

	TLegend* legend = new TLegend(0.8, 0.7, 0.99, 0.99);
	legend->AddEntry(h_ErsatzCaloMEt_pass_EB, "Ersatz MET", "l");
	legend->AddEntry(h_WCaloMEt_pass_EB, "W MET", "l");

	legend->Draw();
	c_ErsatzCaloMEt_corr_pass_EB->SaveAs("ErsatzCaloMEt_corr_pass_EB.png");
	delete c_ErsatzCaloMEt_corr_pass_EB;

	TCanvas* c_ErsatzCaloMEt_corr_pass_EE =  new TCanvas("ErsatzCaloMEt_corr_pass_EE", "", 800, 600);	
	c_ErsatzCaloMEt_corr_pass_EE->cd();
	//h_ErsatzCaloMEt_pass_EE->Scale(1./h_ErsatzCaloMEt_pass_EE->Integral());
	h_ErsatzCaloMEt_pass_EE->SetLineColor(2);
	h_ErsatzCaloMEt_pass_EE->SetStats(kFALSE);
	h_ErsatzCaloMEt_pass_EE->SetTitle(kFALSE);
	h_ErsatzCaloMEt_pass_EE->GetYaxis()->SetTitleOffset(1.2);
	h_ErsatzCaloMEt_pass_EE->Draw();
	h_WCaloMEt_pass_EE->Scale(h_ErsatzCaloMEt_pass_EE->Integral()/h_WCaloMEt_pass_EE->Integral());
	h_WCaloMEt_pass_EE->SetLineColor(4);
	h_WCaloMEt_pass_EE->Draw("same");
	legend->Draw();
	c_ErsatzCaloMEt_corr_pass_EE->SaveAs("ErsatzCaloMEt_corr_pass_EE.png");
	delete c_ErsatzCaloMEt_corr_pass_EE;

	TCanvas* c_ErsatzCaloMEt_corr_pass =  new TCanvas("ErsatzCaloMEt_corr_pass", "", 800, 600);	
	c_ErsatzCaloMEt_corr_pass->cd();
	//h_ErsatzCaloMEt_pass->Scale(1./h_ErsatzCaloMEt_pass->Integral());
	h_ErsatzCaloMEt_pass->SetLineColor(2);
	h_ErsatzCaloMEt_pass->SetStats(kFALSE);
	h_ErsatzCaloMEt_pass->SetTitle(kFALSE);
	h_ErsatzCaloMEt_pass->GetYaxis()->SetTitleOffset(1.2);
	h_ErsatzCaloMEt_pass->Draw();
	h_WCaloMEt_pass->Scale(h_ErsatzCaloMEt_pass->Integral()/h_WCaloMEt_pass->Integral());
	h_WCaloMEt_pass->SetLineColor(4);
	h_WCaloMEt_pass->Draw("same");
	legend->Draw();
	c_ErsatzCaloMEt_corr_pass->SaveAs("ErsatzCaloMEt_corr_pass.png");
	delete c_ErsatzCaloMEt_corr_pass;

	TCanvas* c_ErsatzTcMEt_corr_pass_EB =  new TCanvas("ErsatzTcMEt_corr_pass_EB", "", 800, 600);	
	c_ErsatzTcMEt_corr_pass_EB->cd();
	//h_ErsatzTcMEt_pass_EB->Scale(1./h_ErsatzTcMEt_pass_EB->Integral());
	h_ErsatzTcMEt_pass_EB->SetLineColor(2);
	h_ErsatzTcMEt_pass_EB->SetStats(kFALSE);
	h_ErsatzTcMEt_pass_EB->SetTitle(kFALSE);
	h_ErsatzTcMEt_pass_EB->GetYaxis()->SetTitleOffset(1.2);
	h_ErsatzTcMEt_pass_EB->Draw();
	h_WTcMEt_pass_EB->Scale(h_ErsatzTcMEt_pass_EB->Integral()/h_WTcMEt_pass_EB->Integral());
	h_WTcMEt_pass_EB->SetLineColor(4);
	h_WTcMEt_pass_EB->Draw("same");
	legend->Draw();
	c_ErsatzTcMEt_corr_pass_EB->SaveAs("ErsatzTcMEt_corr_pass_EB.png");
	delete c_ErsatzTcMEt_corr_pass_EB;

	TCanvas* c_ErsatzTcMEt_corr_pass_EE =  new TCanvas("ErsatzTcMEt_corr_pass_EE", "", 800, 600);	
	c_ErsatzTcMEt_corr_pass_EE->cd();
	//h_ErsatzTcMEt_pass_EE->Scale(1./h_ErsatzTcMEt_pass_EE->Integral());
	h_ErsatzTcMEt_pass_EE->SetLineColor(2);
	h_ErsatzTcMEt_pass_EE->SetStats(kFALSE);
	h_ErsatzTcMEt_pass_EE->SetTitle(kFALSE);
	h_ErsatzTcMEt_pass_EE->GetYaxis()->SetTitleOffset(1.2);
	h_ErsatzTcMEt_pass_EE->Draw();
	h_WTcMEt_pass_EE->Scale(h_ErsatzTcMEt_pass_EE->Integral()/h_WTcMEt_pass_EE->Integral());
	h_WTcMEt_pass_EE->SetLineColor(4);
	h_WTcMEt_pass_EE->Draw("same");
	legend->Draw();
	c_ErsatzTcMEt_corr_pass_EE->SaveAs("ErsatzTcMEt_corr_pass_EE.png");
	delete c_ErsatzTcMEt_corr_pass_EE;

	TCanvas* c_ErsatzTcMEt_corr_pass =  new TCanvas("ErsatzTcMEt_corr_pass", "", 800, 600);	
	c_ErsatzTcMEt_corr_pass->cd();
	//h_ErsatzTcMEt_pass->Scale(1./h_ErsatzTcMEt_pass->Integral());
	h_ErsatzTcMEt_pass->SetLineColor(2);
	h_ErsatzTcMEt_pass->SetStats(kFALSE);
	h_ErsatzTcMEt_pass->SetTitle(kFALSE);
	h_ErsatzTcMEt_pass->GetYaxis()->SetTitleOffset(1.2);
	h_ErsatzTcMEt_pass->Draw();
	h_WTcMEt_pass->Scale(h_ErsatzTcMEt_pass->Integral()/h_WTcMEt_pass->Integral());
	h_WTcMEt_pass->SetLineColor(4);
	h_WTcMEt_pass->Draw("same");
	legend->Draw();
	c_ErsatzTcMEt_corr_pass->SaveAs("ErsatzTcMEt_corr_pass.png");
	delete c_ErsatzTcMEt_corr_pass;

	TCanvas* c_ErsatzGenMEt_corr_pass_EB =  new TCanvas("ErsatzGenMEt_corr_pass_EB", "", 800, 600);	
	c_ErsatzGenMEt_corr_pass_EB->cd();
	//h_ErsatzGenMEt_pass_EB->Scale(1./h_ErsatzGenMEt_pass_EB->Integral());
	h_ErsatzGenMEt_pass_EB->SetLineColor(2);
	h_ErsatzGenMEt_pass_EB->SetStats(kFALSE);
	h_ErsatzGenMEt_pass_EB->SetTitle(kFALSE);
	h_ErsatzGenMEt_pass_EB->GetYaxis()->SetTitleOffset(1.2);
	h_ErsatzGenMEt_pass_EB->Draw();
	h_WGenMEt_pass_EB->Scale(h_ErsatzGenMEt_pass_EB->Integral()/h_WGenMEt_pass_EB->Integral());
	h_WGenMEt_pass_EB->SetLineColor(4);
	h_WGenMEt_pass_EB->Draw("same");
	legend->Draw();
	c_ErsatzGenMEt_corr_pass_EB->SaveAs("ErsatzGenMEt_corr_pass_EB.png");
	delete c_ErsatzGenMEt_corr_pass_EB;

	TCanvas* c_ErsatzGenMEt_corr_pass_EE =  new TCanvas("ErsatzGenMEt_corr_pass_EE", "", 800, 600);	
	c_ErsatzGenMEt_corr_pass_EE->cd();
	//h_ErsatzGenMEt_pass_EE->Scale(1./h_ErsatzGenMEt_pass_EE->Integral());
	h_ErsatzGenMEt_pass_EE->SetLineColor(2);
	h_ErsatzGenMEt_pass_EE->SetStats(kFALSE);
	h_ErsatzGenMEt_pass_EE->SetTitle(kFALSE);
	h_ErsatzGenMEt_pass_EE->GetYaxis()->SetTitleOffset(1.2);
	h_ErsatzGenMEt_pass_EE->Draw();
	h_WGenMEt_pass_EE->Scale(h_ErsatzGenMEt_pass_EE->Integral()/h_WGenMEt_pass_EE->Integral());
	h_WGenMEt_pass_EE->SetLineColor(4);
	h_WGenMEt_pass_EE->Draw("same");
	legend->Draw();
	c_ErsatzGenMEt_corr_pass_EE->SaveAs("ErsatzGenMEt_corr_pass_EE.png");
	delete c_ErsatzGenMEt_corr_pass_EE;

	TCanvas* c_ErsatzGenMEt_corr_pass =  new TCanvas("ErsatzGenMEt_corr_pass", "", 800, 600);	
	c_ErsatzGenMEt_corr_pass->cd();
	//h_ErsatzGenMEt_pass->Scale(1./h_ErsatzGenMEt_pass->Integral());
	h_ErsatzGenMEt_pass->SetLineColor(2);
	h_ErsatzGenMEt_pass->SetStats(kFALSE);
	h_ErsatzGenMEt_pass->SetTitle(kFALSE);
	h_ErsatzGenMEt_pass->GetYaxis()->SetTitleOffset(1.2);
	h_ErsatzGenMEt_pass->Draw();
	h_WGenMEt_pass->Scale(h_ErsatzGenMEt_pass->Integral()/h_WGenMEt_pass->Integral());
	h_WGenMEt_pass->SetLineColor(4);
	h_WGenMEt_pass->Draw("same");
	legend->Draw();
	c_ErsatzGenMEt_corr_pass->SaveAs("ErsatzGenMEt_corr_pass.png");
	delete c_ErsatzGenMEt_corr_pass;

	TCanvas* c_ErsatzPfMEt_corr_pass_EB =  new TCanvas("ErsatzPfMEt_corr_pass_EB", "", 800, 600);	
	c_ErsatzPfMEt_corr_pass_EB->cd();
	//h_ErsatzPfMEt_pass_EB->Scale(1./h_ErsatzPfMEt_pass_EB->Integral());
	h_ErsatzPfMEt_pass_EB->SetLineColor(2);
	h_ErsatzPfMEt_pass_EB->SetStats(kFALSE);
	h_ErsatzPfMEt_pass_EB->SetTitle(kFALSE);
	h_ErsatzPfMEt_pass_EB->GetYaxis()->SetTitleOffset(1.2);
	h_ErsatzPfMEt_pass_EB->Draw();
	h_WPfMEt_pass_EB->Scale(h_ErsatzPfMEt_pass_EB->Integral()/h_WPfMEt_pass_EB->Integral());
	h_WPfMEt_pass_EB->SetLineColor(4);
	h_WPfMEt_pass_EB->Draw("same");
	legend->Draw();
	c_ErsatzPfMEt_corr_pass_EB->SaveAs("ErsatzPfMEt_corr_pass_EB.png");
	delete c_ErsatzPfMEt_corr_pass_EB;

	TCanvas* c_ErsatzPfMEt_corr_pass_EE =  new TCanvas("ErsatzPfMEt_corr_pass_EE", "", 800, 600);	
	c_ErsatzPfMEt_corr_pass_EE->cd();
	//h_ErsatzPfMEt_pass_EE->Scale(1./h_ErsatzPfMEt_pass_EE->Integral());
	h_ErsatzPfMEt_pass_EE->SetLineColor(2);
	h_ErsatzPfMEt_pass_EE->SetStats(kFALSE);
	h_ErsatzPfMEt_pass_EE->SetTitle(kFALSE);
	h_ErsatzPfMEt_pass_EE->GetYaxis()->SetTitleOffset(1.2);
	h_ErsatzPfMEt_pass_EE->Draw();
	h_WPfMEt_pass_EE->Scale(h_ErsatzPfMEt_pass_EE->Integral()/h_WPfMEt_pass_EE->Integral());
	h_WPfMEt_pass_EE->SetLineColor(4);
	h_WPfMEt_pass_EE->Draw("same");
	legend->Draw();
	c_ErsatzPfMEt_corr_pass_EE->SaveAs("ErsatzPfMEt_corr_pass_EE.png");
	delete c_ErsatzPfMEt_corr_pass_EE;

	TCanvas* c_ErsatzPfMEt_corr_pass =  new TCanvas("ErsatzPfMEt_corr_pass", "", 800, 600);	
	c_ErsatzPfMEt_corr_pass->cd();
	//h_ErsatzPfMEt_pass->Scale(1./h_ErsatzPfMEt_pass->Integral());
	h_ErsatzPfMEt_pass->SetLineColor(2);
	h_ErsatzPfMEt_pass->SetStats(kFALSE);
	h_ErsatzPfMEt_pass->SetTitle(kFALSE);
	h_ErsatzPfMEt_pass->GetYaxis()->SetTitleOffset(1.2);
	h_ErsatzPfMEt_pass->Draw();
	h_WPfMEt_pass->Scale(h_ErsatzPfMEt_pass->Integral()/h_WPfMEt_pass->Integral());
	h_WPfMEt_pass->SetLineColor(4);
	h_WPfMEt_pass->Draw("same");
	legend->Draw();
	c_ErsatzPfMEt_corr_pass->SaveAs("ErsatzPfMEt_corr_pass.png");
	delete c_ErsatzPfMEt_corr_pass;

	cout << "Calculating f ..." << endl;
	cout << "MET cut = " << int(cMEt) << endl;
	cout << "nPass = " << h_ErsatzMEt_pass->Integral(1,100) << endl;
	cout << "nFail = " << h_ErsatzMEt_fail->Integral(1,100) << endl;

	double N_pass_EB = h_ErsatzMEt_pass_EB->Integral(1,100);
	double A_EB = h_ErsatzMEt_pass_EB->Integral(int(cMEt)+1,100); 
	double B_EB = h_ErsatzMEt_pass_EB->Integral(1,int(cMEt));
	double N_pass_EE = h_ErsatzMEt_pass_EE->Integral(1,100);
	double A_EE = h_ErsatzMEt_pass_EE->Integral(int(cMEt)+1,100); 
	double B_EE = h_ErsatzMEt_pass_EE->Integral(1,int(cMEt));
	double N_fail_EB = h_ErsatzMEt_fail_EB->Integral(1,100);
	double D_EB = h_ErsatzMEt_fail_EB->Integral(int(cMEt)+1,100); 
	double C_EB = h_ErsatzMEt_fail_EB->Integral(1,int(cMEt));
	double N_fail_EE = h_ErsatzMEt_fail_EE->Integral(1,100);
	double D_EE = h_ErsatzMEt_fail_EE->Integral(int(cMEt)+1,100); 
	double C_EE = h_ErsatzMEt_fail_EE->Integral(1,int(cMEt));

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
