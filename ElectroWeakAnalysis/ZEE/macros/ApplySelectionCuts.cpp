#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TString.h"
#include <cmath>
#include <iostream>
#include <fstream>
//#include <map>

using namespace std;

//double deltaR(double eta1, double phi1, double eta2, double phi2);
//double deltaPhi(double phi1, double phi2);
void ApplySelectionCuts(TString InFileName)
{
	// Declare electron cut value variables	
	double cMEt, cPt;
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
	
	// Import cut values from config file
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
/*
	cuts >> cPt;
	cuts >> cECALiso_EB;
	cuts >> cECALiso_EE;
	cuts >> cHCALiso_EB;
	cuts >> cHCALiso_EE;
	cuts >> cTrackiso_EB;
	cuts >> cTrackiso_EE;
	cuts >> cDeltaEta_EB;
	cuts >> cDeltaEta_EE;
	cuts >> cDeltaPhi_EB;
	cuts >> cDeltaPhi_EE;
	cuts >> csIhIh_EB;
	cuts >> csIhIh_EE;
	cuts >> cHCAL;
	cuts >> cf1x5;
	cuts >> cf2x5;
	cuts >> celecmatch;
	cuts >> cnusIhIh;
	cuts >> cHCALEt;
*/
	cuts.close();
	
	cout << "Got Cut Values" << endl;
	// Import probe selection efficiency weights
/*
	double nueff[345];
	ifstream weightsin("EtaWeights.txt");

	for(int eta=0; eta < 345; ++eta)
	{
		double weight;
		weightsin >> weight; 
		nueff[eta] = weight;
	}
*/	
 	int nTags, nErNu; 
	int nMcElecs_Zmum, nMcElecs_Final;
	int nRecHitsInStrip[4], nRecHitsInCone[4];
	
	double Z_pt[4];

	double elec_q[4], elec_pt[4], elec_eta[4], elec_phi[4]; 
	double elec_rescPt[4], elec_rescEta[4];//, elec_rescPhi[4]; 
	
	double /*probe_q[4], */probe_pt[4], probe_eta[4], probe_phi[4];
	double probe_rescPt[4], probe_rescEta[4];//, probe_rescPhi[4];
	
	double ErsatzV1MEt[4], ErsatzV1Mt[4], ErsatzV1MEtphi[4];
	double ErsatzV1aMEt[4], ErsatzV1aMt[4], ErsatzV1aMEtphi[4];
	double ErsatzV1bMEt[4], ErsatzV1bMt[4], ErsatzV1bMEtphi[4];
	double ErsatzV1cMEt[4]/*, ErsatzV1cMt[4]*/, ErsatzV1cMEtphi[4];
	double ErsatzV2MEt[4], ErsatzV2Mt[4], ErsatzV2MEtphi[4];
	double caloMEt;
	double mesc[4], rescMesc[4];
	
	double elec_trkIso[4], elec_EcalIso[4], elec_HcalIso[4];
	double elec_sIhIh[4], elec_dPhi[4], elec_dEta[4];
	
	double ernu_e1x5[4], ernu_e2x5[4], ernu_e5x5[4], ernu_sIhIh[4];
	double ernu_HcalEt015[4], ernu_HcalE015[4], ernu_trkIso[4];
	
	double ernu_E[4], ernu_rawE[4], ernu_unclusE[4], ernu_d_McE_ScE[4]; 
	double ernu_fEtaCorrE[4], ernu_fBremCorrE[4], ernu_AddCorrE[4];
	int ernu_nClusters[4];
	
	double McZ_m[4], McZ_pt[4]/*, McZ_eta[4]*/, McZ_phi[4];
	double McZ_rescM[4];//, McZ_rescPt[4], McZ_rescEta[4], McZ_rescPhi[4];
	double McElec_pt[4], McElec_eta[4];//, McElec_phi[4]; 
	double McElec_rescPt[4], McElec_rescEta[4];//, McElec_rescPhi[4]; 
	double McErNu_pt[4], McErNu_eta[4], McErNu_phi[4]; 
	double McErNu_rescPt[4], McErNu_rescEta[4], McErNu_rescPhi[4]; 
	double McElecErNu_dPhi[4], McElecErNu_dR[4];
	int ernu_McMatch[4];
	
	cout << "Declared Variables" << endl;
	//TString OutFileName = "/tmp/rnandi/"+InFileName+"_Selected";
	//TFile *outfile = TFile::Open(OutFileName+".root", "recreate");

	TString OutFileName_Pass_EB = "/tmp/rnandi/"+InFileName+"_Selected_Pass_EB";
	TFile *outfile_pass_EB = TFile::Open(OutFileName_Pass_EB+".root", "recreate");
	
	cout << "Opened outfile_Pass_EB" << endl;

	TH1I* h_nTags_Pass_EB = new TH1I("nTags", "Number of Tags;Number of Tags;Arbitrary Units", 4, 0, 4);
	TH1I* h_nErNu_Pass_EB = new TH1I("nErNu", "Number of Ersatz Neutrinos;Number of Ersatz Neutrinos;Arbitrary Units", 2, 0, 2);
	TH1I* h_nMcElecs_Zmum_Pass_EB = new TH1I("nMcElecs_Zmum", "Number of Monte Carlo Electrons with Z as Mother;Number of MC Electrons;Arbitrary Units", 4, 0, 4);
	TH1I* h_nMcElecs_Final_Pass_EB = new TH1I("nMcElecs_Final", "Number of Final State Monte Carlo Electrons;Number of MC Electrons;Arbitrary Units", 4, 0, 4);
	TH1I* h_nRecHitsInStrip_Pass_EB = new TH1I("nRecHitsInStrip", ";;Arbitrary Units", 20, 0, 20);
	TH1I* h_nRecHitsInCone_Pass_EB = new TH1I("nRecHitsInCone", ";;Arbitrary Units", 20, 0, 20);

	TH1F* h_Z_pt_Pass_EB = new TH1F("Z_pt", "Z p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);

	TH1F* h_elec_q_Pass_EB = new TH1F("elec_q", "Electron Charge;q;Arbitrary Units", 3, -1.5, 1.5);
	TH1F* h_elec_pt_Pass_EB = new TH1F("elec_pt", "Electron p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_elec_eta_Pass_EB = new TH1F("elec_eta", "Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_elec_phi_Pass_EB = new TH1F("elec_phi", "Electron #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_elec_sIhIh_Pass_EB = new TH1F("elec_sIhIh", "Electron #sigma_{i#etai#eta};#sigma_{i#etai#eta};Arbitary Units", 100, 0., 0.05);
	TH1F* h_elec_dEta_Pass_EB = new TH1F("elec_dEta", "Electron #Delta#eta;#Delta#eta;Arbitrary Units", 100, 0., 0.02);
	TH1F* h_elec_dPhi_Pass_EB = new TH1F("elec_dPhi", "Electron #Delta#phi;#Delta#phi;Arbitrary Units", 100, 0., 0.1);
	TH1F* h_elec_TrkIso_Pass_EB = new TH1F("elec_TrkIso", "Electon Track Isolation;Track Isolation #Sigma p_{T}^{tracks} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_EcalIso_Pass_EB = new TH1F("elec_EcalIso", "Electron ECAL Isolation;ECAL Isolation #Sigma E_{T}^{ECAL} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_HcalIso_Pass_EB = new TH1F("elec_HcalIso", "Electron HCAL Isolation;HCAL Isolation #Sigma E_{T}^{HCAL} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_rescPt_Pass_EB = new TH1F("elec_rescPt", "Electron Rescaled p_{T};Rescaled p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_elec_rescEta_Pass_EB = new TH1F("elec_rescEta", "Electron Rescaled #eta;Rescaled #eta;Arbitrary Units", 100, -3., 3.);

//	TH1F* h_ErNu_q_Pass_EB = new TH1F("h_ErNu_q", "Ersatz Neutrino Charge;q;Arbitrary Units", 3, -1.5, 1.5);
	TH1F* h_ErNu_pt_Pass_EB = new TH1F("ErNu_pt", "Ersatz Neutrino p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_eta_Pass_EB = new TH1F("ErNu_eta", "Ersatz Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_ErNu_phi_Pass_EB = new TH1F("ErNu_phi", "Ersatz Neutrino #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_ErNu_rescPt_Pass_EB = new TH1F("ErNu_rescPt", "Ersatz Neutrino Rescaled p_{T};Rescaled p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_rescEta_Pass_EB = new TH1F("ErNu_rescEta", "Ersatz Neutrino Rescaled #eta;Rescaled #eta", 100, -3., 3.);
	
	TH1F* h_ErNu_sIhIh_Pass_EB = new TH1F("ErNu_sIhIh", "Ersatz Neutrino #sigma_{i#etai#eta};#sigma_{i#etai#eta};Arbitrary Units", 100, 0., 0.05);
//	TH1F* h_ErNu_dEta_Pass_EB = new TH1F("ErNu_dEta", "Ersatz Neutrino #Delta#eta;#Delta#eta;Arbitrary Units", 100, 0., 0.02);
//	TH1F* h_ErNu_dPhi_Pass_EB = new TH1F("ErNu_dPhi", "Ersatz Neutrino #Delta#phi;#Delta#phi;Arbitrary Units", 100, 0., 0.1);
	TH1F* h_ErNu_TrkIso_Pass_EB = new TH1F("ErNu_TrkIso", "Ersatz Neutrino Track Isolation;Track Isolation #Sigma p_{T}^{tracks} / GeV;Arbitrary Units", 100, 0., 10.);
//	TH1F* h_ErNu_EcalIso_Pass_EB = new TH1F("ErNu_EcalIso", "Ersatz Neutrino ECAL Isolation;ECAL Isolation #Sigma E_{T}^{ECAL} / GeV;Arbitrary Units", 100, 0., 10.);
//	TH1F* h_ErNu_HcalIso_Pass_EB = new TH1F("ErNu_HcalIso", "Ersatz Neutrino HCAL Isolation;HCAL Isolation #Sigma E_{T}^{HCAL} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_ErNu_e1x5Max_Pass_EB = new TH1F("ErNu_e1x5Max", "Ersatz Neutrino Maximum Energy in 1x5 Array of Crystals;e1x5Max / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_e2x5Max_Pass_EB = new TH1F("ErNu_e2x5Max", "Ersatz Neutrino Maximum Energy in 2x5 Array of Crystals;e2x5Max / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_e5x5_Pass_EB = new TH1F("ErNu_e5x5", "Ersatz Neutrino Energy in 5x5 Array of Crystals;e5x5 / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_HcalE015_Pass_EB = new TH1F("ErNu_HcalE015", "Ersatz Neutrino HCAL Energy in a 0.15 Cone;HCAL Energy / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_ErNu_HcalEt015_Pass_EB = new TH1F("ErNu_HcalEt015", "Ersatz Neutrino HCAL E_{T} in a 0.15 Cone;HCAL E_{T};Arbitrary Units", 100, 0., 10.);
	
//	TH1F* h_rechitE_Pass_EB = new TH1F("h_rechitE", "", 100, , ); // What is this
	TH1F* h_ErNu_E_Pass_EB = new TH1F("ErNu_E", "Ersatz Neutrino Energy;Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_rawE_Pass_EB = new TH1F("ErNu_rawE", "Ersatz Neutrino Raw Energy;Raw Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_unclusE_Pass_EB = new TH1F("ErNu_unclusE", "Ersatz Neutrino Unclustered ECAL Energy;Unclustered Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_fEtaCorrE_Pass_EB = new TH1F("ErNu_fEtaCorrE", "Ersatz Neutrino fEta Corrected Energy;Corrected Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_fBremCorrE_Pass_EB = new TH1F("ErNu_fBremCorrE", "Ersatz Neutrino fBrem Corrected Energy;Correcteed Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_AddCorrE_Pass_EB = new TH1F("ErNu_AddCorrE", "Ersatz Neutrino Add Corrected Energy;Corrected Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1I* h_ErNu_nClusters_Pass_EB = new TH1I("ErNu_nClusters", "Ersatz Neutrino Number of Clusters;Number of Clusters;Arbitrary Units", 10, 0, 10);
	
//	TH1F* h_ErsatzV1MEt_Pass_EB = new TH1F("ErsatzV1MEt", "ErsatzV1MEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1Mt_Pass_EB = new TH1F("ErsatzV1Mt", "ErsatzV1Mt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1MEtphi_Pass_EB = new TH1F("ErsatzV1MEtPhi", "ErsatzV1MEtPhi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1aMEt_Pass_EB = new TH1F("ErsatzV1aMEt", "ErsatzV1aMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1aMt_Pass_EB = new TH1F("ErsatzV1aMt", "ErsatzV1aMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1aMEtphi_Pass_EB = new TH1F("ErsatzV1aMEtphi", "ErsatzV1aMEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1bMEt_Pass_EB = new TH1F("ErsatzV1bMEt", "ErsatzV1bMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1bMt_Pass_EB = new TH1F("ErsatzV1bMt", "ErsatzV1bMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1bMEtphi_Pass_EB = new TH1F("ErsatzV1bMEtphi", "ErsatzV1bMEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1cMEt_Pass_EB = new TH1F("ErsatzV1cMEt", "ErsatzV1cMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1cMt_Pass_EB = new TH1F("ErsatzV1cMt", "ErsatzV1cMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1cMEtphi_Pass_EB = new TH1F("ErsatzV1cMEtphi", "ErsatzV1cMEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV2MEt_Pass_EB = new TH1F("ErsatzV1aMEt", "ErsatzV1aMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV2Mt_Pass_EB = new TH1F("ErsatzV1aMt", "ErsatzV1aMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV2MEtphi_Pass_EB = new TH1F("ErsatzV2MEtphi", "ErsatzV2MEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_RecoCaloMEt_Pass_EB = new TH1F("RecoCaloMEt", "Calometric #slashE_{T};#slashE_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1Mesc_Pass_EB = new TH1F("ErsatzV1Mesc", "Invariant Mass;Invariant Mass / GeV;Arbitrary Units", 100, 41., 141.);
	TH1F* h_ErsatzV1rescMesc_Pass_EB = new TH1F("ErsatzV1rescMesc", "Rescaled Invariant Mass;Rescaled Invariant Mass / GeV;Arbitrary Units", 100, 41., 141.);
	
	TH1F* h_McZ_M_Pass_EB = new TH1F("McZ_M", "Monte Carlo Z Mass;Mass / GeV; Arbitrary Units", 100, 41., 141.);
	TH1F* h_McZ_rescM_Pass_EB = new TH1F("McZ_rescM", "Monte Carlo Rescaled Z Mass;Mass / GeV;Arbitrary Units", 100, 41., 141.);
	TH1F* h_McZ_Pt_Pass_EB = new TH1F("McZ_Pt", "Monte Carlo Z p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_McZ_rescPt_Pass_EB = new TH1F("McZ_rescPt", "Monte Carlo Rescaled Z p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_McZ_Eta_Pass_EB = new TH1F("McZ_eta", "Monte Carlo Z #eta;#eta;Arbitrary Units", 100, -3., 3.);
//	TH1F* h_McZ_rescEta_Pass_EB = new TH1F("McZ_rescEta", "Monte Carlo Rescaled Z #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McZ_Phi_Pass_EB = new TH1F("McZ_phi", "Monte Carlo Z #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
//	TH1F* h_McZ_rescPhi_Pass_EB = new TH1F("McZ_rescPhi", "Monte Carlo Z Rescaled #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	
	TH1F* h_McElec_Pt_Pass_EB = new TH1F("McElec_Pt", "Monte Carlo Electron p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_McElec_rescPt_Pass_EB = new TH1F("McElec_rescPt", "Monte Carlo Rescaled Electron p_{T};Rescaled p_{T};Arbitrary Units", 100, 0., 100.);
	TH1F* h_McElec_Eta_Pass_EB = new TH1F("McElec_Eta", "Monte Carlo Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McElec_rescEta_Pass_EB = new TH1F("McElec_rescEta", "Monte Carlo Rescaled Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
//	TH1F* h_McElec_Phi_Pass_EB = new TH1F("McElec_Phi", "Monte Carlo Electron #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
//	TH1F* h_McElec_rescPhi_Pass_EB = new TH1F("McElec_rescPhi", "Monte Carlo Rescaled Electron #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	
	TH1F* h_McErNu_Pt_Pass_EB = new TH1F("McErNu_Pt", "Monte Carlo Ersatz Neutrino p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_McErNu_rescPt_Pass_EB = new TH1F("McErNu_rescPt", "Monte Carlo Rescaled Ersatz Neutrino p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_McErNu_Eta_Pass_EB = new TH1F("McErNu_Eta", "Monte Carlo Ersatz Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McErNu_rescEta_Pass_EB = new TH1F("McErNu_rescEta", "Monte Carlo Rescaled Ersatz Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McErNu_Phi_Pass_EB = new TH1F("McErNu_Phi", "Monte Carlo Ersatz Neutrino #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_McErNu_rescPhi_Pass_EB = new TH1F("McErNu_rescPhi", "Monte Carlo Rescaled Ersatz Neutrino #phi;Rescaled #phi;Arbitrary Units", 100, 0., 3.1416);
	
	TH1F* h_McElecErNu_dPhi_Pass_EB = new TH1F("McElecErNu_dPhi", "Monte Carlo #Delta#phi between Electrons;#Delta#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_McElecErNu_dR_Pass_EB = new TH1F("McElecErNu_dR", "Monte Carlo #DeltaR between Electrons;MC Match;Arbitrary Units", 100, 0., 3.1416);
	TH1I* h_McMatch_Pass_EB = new TH1I("McMatch", "Monte Carlo Match of Ersatz Neutrino to an Electron;MC Match;Arbitrary Units", 2, 0, 2);

	cout << "Declared histograms h_xxx_Pass_EB" << endl;
//	TH1F* h_EtaInt_Pass_EB = new TH1F("EtaInt", "", 345, 0, 345);
//	TH1F* h_EtaWeights_Pass_EB = new TH1F("EtaWeights", "", 40, 0., 2.);

	TString OutFileName_Pass_EE = "/tmp/rnandi/"+InFileName+"_Selected_Pass_EE";
	TFile *outfile_pass_EE = TFile::Open(OutFileName_Pass_EE+".root", "recreate");
	cout << "Opened outfile_Pass_EE" << endl;
	
	TH1I* h_nTags_Pass_EE = new TH1I("nTags", "Number of Tags;Number of Tags;Arbitrary Units", 4, 0, 4);
	TH1I* h_nErNu_Pass_EE = new TH1I("nErNu", "Number of Ersatz Neutrinos;Number of Ersatz Neutrinos;Arbitrary Units", 2, 0, 2);
	TH1I* h_nMcElecs_Zmum_Pass_EE = new TH1I("nMcElecs_Zmum", "Number of Monte Carlo Electrons with Z as Mother;Number of MC Electrons;Arbitrary Units", 4, 0, 4);
	TH1I* h_nMcElecs_Final_Pass_EE = new TH1I("nMcElecs_Final", "Number of Final State Monte Carlo Electrons;Number of MC Electrons;Arbitrary Units", 4, 0, 4);
	TH1I* h_nRecHitsInStrip_Pass_EE = new TH1I("nRecHitsInStrip", ";;Arbitrary Units", 20, 0, 20);
	TH1I* h_nRecHitsInCone_Pass_EE = new TH1I("nRecHitsInCone", ";;Arbitrary Units", 20, 0, 20);

	TH1F* h_Z_pt_Pass_EE = new TH1F("Z_pt", "Z p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);

	TH1F* h_elec_q_Pass_EE = new TH1F("elec_q", "Electron Charge;q;Arbitrary Units", 3, -1.5, 1.5);
	TH1F* h_elec_pt_Pass_EE = new TH1F("elec_pt", "Electron p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_elec_eta_Pass_EE = new TH1F("elec_eta", "Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_elec_phi_Pass_EE = new TH1F("elec_phi", "Electron #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_elec_sIhIh_Pass_EE = new TH1F("elec_sIhIh", "Electron #sigma_{i#etai#eta};#sigma_{i#etai#eta};Arbitary Units", 100, 0., 0.05);
	TH1F* h_elec_dEta_Pass_EE = new TH1F("elec_dEta", "Electron #Delta#eta;#Delta#eta;Arbitrary Units", 100, 0., 0.02);
	TH1F* h_elec_dPhi_Pass_EE = new TH1F("elec_dPhi", "Electron #Delta#phi;#Delta#phi;Arbitrary Units", 100, 0., 0.1);
	TH1F* h_elec_TrkIso_Pass_EE = new TH1F("elec_TrkIso", "Electon Track Isolation;Track Isolation #Sigma p_{T}^{tracks} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_EcalIso_Pass_EE = new TH1F("elec_EcalIso", "Electron ECAL Isolation;ECAL Isolation #Sigma E_{T}^{ECAL} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_HcalIso_Pass_EE = new TH1F("elec_HcalIso", "Electron HCAL Isolation;HCAL Isolation #Sigma E_{T}^{HCAL} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_rescPt_Pass_EE = new TH1F("elec_rescPt", "Electron Rescaled p_{T};Rescaled p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_elec_rescEta_Pass_EE = new TH1F("elec_rescEta", "Electron Rescaled #eta;Rescaled #eta;Arbitrary Units", 100, -3., 3.);

//	TH1F* h_ErNu_q_Pass_EE = new TH1F("h_ErNu_q", "Ersatz Neutrino Charge;q;Arbitrary Units", 3, -1.5, 1.5);
	TH1F* h_ErNu_pt_Pass_EE = new TH1F("ErNu_pt", "Ersatz Neutrino p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_eta_Pass_EE = new TH1F("ErNu_eta", "Ersatz Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_ErNu_phi_Pass_EE = new TH1F("ErNu_phi", "Ersatz Neutrino #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_ErNu_rescPt_Pass_EE = new TH1F("ErNu_rescPt", "Ersatz Neutrino Rescaled p_{T};Rescaled p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_rescEta_Pass_EE = new TH1F("ErNu_rescEta", "Ersatz Neutrino Rescaled #eta;Rescaled #eta", 100, -3., 3.);
	
	TH1F* h_ErNu_sIhIh_Pass_EE = new TH1F("ErNu_sIhIh", "Ersatz Neutrino #sigma_{i#etai#eta};#sigma_{i#etai#eta};Arbitrary Units", 100, 0., 0.05);
//	TH1F* h_ErNu_dEta_Pass_EE = new TH1F("ErNu_dEta", "Ersatz Neutrino #Delta#eta;#Delta#eta;Arbitrary Units", 100, 0., 0.02);
//	TH1F* h_ErNu_dPhi_Pass_EE = new TH1F("ErNu_dPhi", "Ersatz Neutrino #Delta#phi;#Delta#phi;Arbitrary Units", 100, 0., 0.1);
	TH1F* h_ErNu_TrkIso_Pass_EE = new TH1F("ErNu_TrkIso", "Ersatz Neutrino Track Isolation;Track Isolation #Sigma p_{T}^{tracks} / GeV;Arbitrary Units", 100, 0., 10.);
//	TH1F* h_ErNu_EcalIso_Pass_EE = new TH1F("ErNu_EcalIso", "Ersatz Neutrino ECAL Isolation;ECAL Isolation #Sigma E_{T}^{ECAL} / GeV;Arbitrary Units", 100, 0., 10.);
//	TH1F* h_ErNu_HcalIso_Pass_EE = new TH1F("ErNu_HcalIso", "Ersatz Neutrino HCAL Isolation;HCAL Isolation #Sigma E_{T}^{HCAL} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_ErNu_e1x5Max_Pass_EE = new TH1F("ErNu_e1x5Max", "Ersatz Neutrino Maximum Energy in 1x5 Array of Crystals;e1x5Max / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_e2x5Max_Pass_EE = new TH1F("ErNu_e2x5Max", "Ersatz Neutrino Maximum Energy in 2x5 Array of Crystals;e2x5Max / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_e5x5_Pass_EE = new TH1F("ErNu_e5x5", "Ersatz Neutrino Energy in 5x5 Array of Crystals;e5x5 / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_HcalE015_Pass_EE = new TH1F("ErNu_HcalE015", "Ersatz Neutrino HCAL Energy in a 0.15 Cone;HCAL Energy / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_ErNu_HcalEt015_Pass_EE = new TH1F("ErNu_HcalEt015", "Ersatz Neutrino HCAL E_{T} in a 0.15 Cone;HCAL E_{T};Arbitrary Units", 100, 0., 10.);
	
//	TH1F* h_rechitE_Pass_EE = new TH1F("h_rechitE", "", 100, , ); // What is this
	TH1F* h_ErNu_E_Pass_EE = new TH1F("ErNu_E", "Ersatz Neutrino Energy;Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_rawE_Pass_EE = new TH1F("ErNu_rawE", "Ersatz Neutrino Raw Energy;Raw Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_unclusE_Pass_EE = new TH1F("ErNu_unclusE", "Ersatz Neutrino Unclustered ECAL Energy;Unclustered Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_fEtaCorrE_Pass_EE = new TH1F("ErNu_fEtaCorrE", "Ersatz Neutrino fEta Corrected Energy;Corrected Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_fBremCorrE_Pass_EE = new TH1F("ErNu_fBremCorrE", "Ersatz Neutrino fBrem Corrected Energy;Correcteed Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_AddCorrE_Pass_EE = new TH1F("ErNu_AddCorrE", "Ersatz Neutrino Add Corrected Energy;Corrected Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1I* h_ErNu_nClusters_Pass_EE = new TH1I("ErNu_nClusters", "Ersatz Neutrino Number of Clusters;Number of Clusters;Arbitrary Units", 10, 0, 10);
	
//	TH1F* h_ErsatzV1MEt_Pass_EE = new TH1F("ErsatzV1MEt", "ErsatzV1MEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1Mt_Pass_EE = new TH1F("ErsatzV1Mt", "ErsatzV1Mt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1MEtphi_Pass_EE = new TH1F("ErsatzV1MEtPhi", "ErsatzV1MEtPhi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1aMEt_Pass_EE = new TH1F("ErsatzV1aMEt", "ErsatzV1aMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1aMt_Pass_EE = new TH1F("ErsatzV1aMt", "ErsatzV1aMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1aMEtphi_Pass_EE = new TH1F("ErsatzV1aMEtphi", "ErsatzV1aMEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1bMEt_Pass_EE = new TH1F("ErsatzV1bMEt", "ErsatzV1bMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1bMt_Pass_EE = new TH1F("ErsatzV1bMt", "ErsatzV1bMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1bMEtphi_Pass_EE = new TH1F("ErsatzV1bMEtphi", "ErsatzV1bMEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1cMEt_Pass_EE = new TH1F("ErsatzV1cMEt", "ErsatzV1cMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1cMt_Pass_EE = new TH1F("ErsatzV1cMt", "ErsatzV1cMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1cMEtphi_Pass_EE = new TH1F("ErsatzV1cMEtphi", "ErsatzV1cMEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV2MEt_Pass_EE = new TH1F("ErsatzV1aMEt", "ErsatzV1aMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV2Mt_Pass_EE = new TH1F("ErsatzV1aMt", "ErsatzV1aMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV2MEtphi_Pass_EE = new TH1F("ErsatzV2MEtphi", "ErsatzV2MEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_RecoCaloMEt_Pass_EE = new TH1F("RecoCaloMEt", "Calometric #slashE_{T};#slashE_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1Mesc_Pass_EE = new TH1F("ErsatzV1Mesc", "Invariant Mass;Invariant Mass / GeV;Arbitrary Units", 100, 41., 141.);
	TH1F* h_ErsatzV1rescMesc_Pass_EE = new TH1F("ErsatzV1rescMesc", "Rescaled Invariant Mass;Rescaled Invariant Mass / GeV;Arbitrary Units", 100, 41., 141.);
	
	TH1F* h_McZ_M_Pass_EE = new TH1F("McZ_M", "Monte Carlo Z Mass;Mass / GeV; Arbitrary Units", 100, 41., 141.);
	TH1F* h_McZ_rescM_Pass_EE = new TH1F("McZ_rescM", "Monte Carlo Rescaled Z Mass;Mass / GeV;Arbitrary Units", 100, 41., 141.);
	TH1F* h_McZ_Pt_Pass_EE = new TH1F("McZ_Pt", "Monte Carlo Z p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_McZ_rescPt_Pass_EE = new TH1F("McZ_rescPt", "Monte Carlo Rescaled Z p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_McZ_Eta_Pass_EE = new TH1F("McZ_eta", "Monte Carlo Z #eta;#eta;Arbitrary Units", 100, -3., 3.);
//	TH1F* h_McZ_rescEta_Pass_EE = new TH1F("McZ_rescEta", "Monte Carlo Rescaled Z #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McZ_Phi_Pass_EE = new TH1F("McZ_phi", "Monte Carlo Z #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
//	TH1F* h_McZ_rescPhi_Pass_EE = new TH1F("McZ_rescPhi", "Monte Carlo Z Rescaled #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	
	TH1F* h_McElec_Pt_Pass_EE = new TH1F("McElec_Pt", "Monte Carlo Electron p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_McElec_rescPt_Pass_EE = new TH1F("McElec_rescPt", "Monte Carlo Rescaled Electron p_{T};Rescaled p_{T};Arbitrary Units", 100, 0., 100.);
	TH1F* h_McElec_Eta_Pass_EE = new TH1F("McElec_Eta", "Monte Carlo Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McElec_rescEta_Pass_EE = new TH1F("McElec_rescEta", "Monte Carlo Rescaled Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
//	TH1F* h_McElec_Phi_Pass_EE = new TH1F("McElec_Phi", "Monte Carlo Electron #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
//	TH1F* h_McElec_rescPhi_Pass_EE = new TH1F("McElec_rescPhi", "Monte Carlo Rescaled Electron #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	
	TH1F* h_McErNu_Pt_Pass_EE = new TH1F("McErNu_Pt", "Monte Carlo Ersatz Neutrino p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_McErNu_rescPt_Pass_EE = new TH1F("McErNu_rescPt", "Monte Carlo Rescaled Ersatz Neutrino p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_McErNu_Eta_Pass_EE = new TH1F("McErNu_Eta", "Monte Carlo Ersatz Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McErNu_rescEta_Pass_EE = new TH1F("McErNu_rescEta", "Monte Carlo Rescaled Ersatz Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McErNu_Phi_Pass_EE = new TH1F("McErNu_Phi", "Monte Carlo Ersatz Neutrino #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_McErNu_rescPhi_Pass_EE = new TH1F("McErNu_rescPhi", "Monte Carlo Rescaled Ersatz Neutrino #phi;Rescaled #phi;Arbitrary Units", 100, 0., 3.1416);
	
	TH1F* h_McElecErNu_dPhi_Pass_EE = new TH1F("McElecErNu_dPhi", "Monte Carlo #Delta#phi between Electrons;#Delta#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_McElecErNu_dR_Pass_EE = new TH1F("McElecErNu_dR", "Monte Carlo #DeltaR between Electrons;MC Match;Arbitrary Units", 100, 0., 3.1416);
	TH1I* h_McMatch_Pass_EE = new TH1I("McMatch", "Monte Carlo Match of Ersatz Neutrino to an Electron;MC Match;Arbitrary Units", 2, 0, 2);

	cout << "Declared histograms h_xxx_Pass_EE" << endl;
//	TH1F* h_EtaInt_Pass_EE = new TH1F("EtaInt", "", 345, 0, 345);
//	TH1F* h_EtaWeights_Pass_EE = new TH1F("EtaWeights", "", 40, 0., 2.);
	
	TString OutFileName_Fail_EB = "/tmp/rnandi/"+InFileName+"_Selected_Fail_EB";
	TFile *outfile_fail_EB = TFile::Open(OutFileName_Fail_EB+".root", "recreate");
	cout << "Opened outfile_Fail_EB" << endl;

	TH1I* h_nTags_Fail_EB = new TH1I("nTags", "Number of Tags;Number of Tags;Arbitrary Units", 4, 0, 4);
	TH1I* h_nErNu_Fail_EB = new TH1I("nErNu", "Number of Ersatz Neutrinos;Number of Ersatz Neutrinos;Arbitrary Units", 2, 0, 2);
	TH1I* h_nMcElecs_Zmum_Fail_EB = new TH1I("nMcElecs_Zmum", "Number of Monte Carlo Electrons with Z as Mother;Number of MC Electrons;Arbitrary Units", 4, 0, 4);
	TH1I* h_nMcElecs_Final_Fail_EB = new TH1I("nMcElecs_Final", "Number of Final State Monte Carlo Electrons;Number of MC Electrons;Arbitrary Units", 4, 0, 4);
	TH1I* h_nRecHitsInStrip_Fail_EB = new TH1I("nRecHitsInStrip", ";;Arbitrary Units", 20, 0, 20);
	TH1I* h_nRecHitsInCone_Fail_EB = new TH1I("nRecHitsInCone", ";;Arbitrary Units", 20, 0, 20);

	TH1F* h_Z_pt_Fail_EB = new TH1F("Z_pt", "Z p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);

	TH1F* h_elec_q_Fail_EB = new TH1F("elec_q", "Electron Charge;q;Arbitrary Units", 3, -1.5, 1.5);
	TH1F* h_elec_pt_Fail_EB = new TH1F("elec_pt", "Electron p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_elec_eta_Fail_EB = new TH1F("elec_eta", "Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_elec_phi_Fail_EB = new TH1F("elec_phi", "Electron #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_elec_sIhIh_Fail_EB = new TH1F("elec_sIhIh", "Electron #sigma_{i#etai#eta};#sigma_{i#etai#eta};Arbitary Units", 100, 0., 0.05);
	TH1F* h_elec_dEta_Fail_EB = new TH1F("elec_dEta", "Electron #Delta#eta;#Delta#eta;Arbitrary Units", 100, 0., 0.02);
	TH1F* h_elec_dPhi_Fail_EB = new TH1F("elec_dPhi", "Electron #Delta#phi;#Delta#phi;Arbitrary Units", 100, 0., 0.1);
	TH1F* h_elec_TrkIso_Fail_EB = new TH1F("elec_TrkIso", "Electon Track Isolation;Track Isolation #Sigma p_{T}^{tracks} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_EcalIso_Fail_EB = new TH1F("elec_EcalIso", "Electron ECAL Isolation;ECAL Isolation #Sigma E_{T}^{ECAL} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_HcalIso_Fail_EB = new TH1F("elec_HcalIso", "Electron HCAL Isolation;HCAL Isolation #Sigma E_{T}^{HCAL} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_rescPt_Fail_EB = new TH1F("elec_rescPt", "Electron Rescaled p_{T};Rescaled p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_elec_rescEta_Fail_EB = new TH1F("elec_rescEta", "Electron Rescaled #eta;Rescaled #eta;Arbitrary Units", 100, -3., 3.);

//	TH1F* h_ErNu_q_Fail_EB = new TH1F("h_ErNu_q", "Ersatz Neutrino Charge;q;Arbitrary Units", 3, -1.5, 1.5);
	TH1F* h_ErNu_pt_Fail_EB = new TH1F("ErNu_pt", "Ersatz Neutrino p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_eta_Fail_EB = new TH1F("ErNu_eta", "Ersatz Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_ErNu_phi_Fail_EB = new TH1F("ErNu_phi", "Ersatz Neutrino #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_ErNu_rescPt_Fail_EB = new TH1F("ErNu_rescPt", "Ersatz Neutrino Rescaled p_{T};Rescaled p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_rescEta_Fail_EB = new TH1F("ErNu_rescEta", "Ersatz Neutrino Rescaled #eta;Rescaled #eta", 100, -3., 3.);
	
	TH1F* h_ErNu_sIhIh_Fail_EB = new TH1F("ErNu_sIhIh", "Ersatz Neutrino #sigma_{i#etai#eta};#sigma_{i#etai#eta};Arbitrary Units", 100, 0., 0.05);
//	TH1F* h_ErNu_dEta_Fail_EB = new TH1F("ErNu_dEta", "Ersatz Neutrino #Delta#eta;#Delta#eta;Arbitrary Units", 100, 0., 0.02);
//	TH1F* h_ErNu_dPhi_Fail_EB = new TH1F("ErNu_dPhi", "Ersatz Neutrino #Delta#phi;#Delta#phi;Arbitrary Units", 100, 0., 0.1);
	TH1F* h_ErNu_TrkIso_Fail_EB = new TH1F("ErNu_TrkIso", "Ersatz Neutrino Track Isolation;Track Isolation #Sigma p_{T}^{tracks} / GeV;Arbitrary Units", 100, 0., 10.);
//	TH1F* h_ErNu_EcalIso_Fail_EB = new TH1F("ErNu_EcalIso", "Ersatz Neutrino ECAL Isolation;ECAL Isolation #Sigma E_{T}^{ECAL} / GeV;Arbitrary Units", 100, 0., 10.);
//	TH1F* h_ErNu_HcalIso_Fail_EB = new TH1F("ErNu_HcalIso", "Ersatz Neutrino HCAL Isolation;HCAL Isolation #Sigma E_{T}^{HCAL} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_ErNu_e1x5Max_Fail_EB = new TH1F("ErNu_e1x5Max", "Ersatz Neutrino Maximum Energy in 1x5 Array of Crystals;e1x5Max / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_e2x5Max_Fail_EB = new TH1F("ErNu_e2x5Max", "Ersatz Neutrino Maximum Energy in 2x5 Array of Crystals;e2x5Max / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_e5x5_Fail_EB = new TH1F("ErNu_e5x5", "Ersatz Neutrino Energy in 5x5 Array of Crystals;e5x5 / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_HcalE015_Fail_EB = new TH1F("ErNu_HcalE015", "Ersatz Neutrino HCAL Energy in a 0.15 Cone;HCAL Energy / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_ErNu_HcalEt015_Fail_EB = new TH1F("ErNu_HcalEt015", "Ersatz Neutrino HCAL E_{T} in a 0.15 Cone;HCAL E_{T};Arbitrary Units", 100, 0., 10.);
	
//	TH1F* h_rechitE_Fail_EB = new TH1F("h_rechitE", "", 100, , ); // What is this
	TH1F* h_ErNu_E_Fail_EB = new TH1F("ErNu_E", "Ersatz Neutrino Energy;Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_rawE_Fail_EB = new TH1F("ErNu_rawE", "Ersatz Neutrino Raw Energy;Raw Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_unclusE_Fail_EB = new TH1F("ErNu_unclusE", "Ersatz Neutrino Unclustered ECAL Energy;Unclustered Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_fEtaCorrE_Fail_EB = new TH1F("ErNu_fEtaCorrE", "Ersatz Neutrino fEta Corrected Energy;Corrected Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_fBremCorrE_Fail_EB = new TH1F("ErNu_fBremCorrE", "Ersatz Neutrino fBrem Corrected Energy;Correcteed Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_AddCorrE_Fail_EB = new TH1F("ErNu_AddCorrE", "Ersatz Neutrino Add Corrected Energy;Corrected Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1I* h_ErNu_nClusters_Fail_EB = new TH1I("ErNu_nClusters", "Ersatz Neutrino Number of Clusters;Number of Clusters;Arbitrary Units", 10, 0, 10);
	
//	TH1F* h_ErsatzV1MEt_Fail_EB = new TH1F("ErsatzV1MEt", "ErsatzV1MEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1Mt_Fail_EB = new TH1F("ErsatzV1Mt", "ErsatzV1Mt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1MEtphi_Fail_EB = new TH1F("ErsatzV1MEtPhi", "ErsatzV1MEtPhi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1aMEt_Fail_EB = new TH1F("ErsatzV1aMEt", "ErsatzV1aMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1aMt_Fail_EB = new TH1F("ErsatzV1aMt", "ErsatzV1aMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1aMEtphi_Fail_EB = new TH1F("ErsatzV1aMEtphi", "ErsatzV1aMEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1bMEt_Fail_EB = new TH1F("ErsatzV1bMEt", "ErsatzV1bMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1bMt_Fail_EB = new TH1F("ErsatzV1bMt", "ErsatzV1bMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1bMEtphi_Fail_EB = new TH1F("ErsatzV1bMEtphi", "ErsatzV1bMEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1cMEt_Fail_EB = new TH1F("ErsatzV1cMEt", "ErsatzV1cMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1cMt_Fail_EB = new TH1F("ErsatzV1cMt", "ErsatzV1cMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1cMEtphi_Fail_EB = new TH1F("ErsatzV1cMEtphi", "ErsatzV1cMEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV2MEt_Fail_EB = new TH1F("ErsatzV1aMEt", "ErsatzV1aMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV2Mt_Fail_EB = new TH1F("ErsatzV1aMt", "ErsatzV1aMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV2MEtphi_Fail_EB = new TH1F("ErsatzV2MEtphi", "ErsatzV2MEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_RecoCaloMEt_Fail_EB = new TH1F("RecoCaloMEt", "Calometric #slashE_{T};#slashE_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1Mesc_Fail_EB = new TH1F("ErsatzV1Mesc", "Invariant Mass;Invariant Mass / GeV;Arbitrary Units", 100, 41., 141.);
	TH1F* h_ErsatzV1rescMesc_Fail_EB = new TH1F("ErsatzV1rescMesc", "Rescaled Invariant Mass;Rescaled Invariant Mass / GeV;Arbitrary Units", 100, 41., 141.);
	
	TH1F* h_McZ_M_Fail_EB = new TH1F("McZ_M", "Monte Carlo Z Mass;Mass / GeV; Arbitrary Units", 100, 41., 141.);
	TH1F* h_McZ_rescM_Fail_EB = new TH1F("McZ_rescM", "Monte Carlo Rescaled Z Mass;Mass / GeV;Arbitrary Units", 100, 41., 141.);
	TH1F* h_McZ_Pt_Fail_EB = new TH1F("McZ_Pt", "Monte Carlo Z p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_McZ_rescPt_Fail_EB = new TH1F("McZ_rescPt", "Monte Carlo Rescaled Z p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_McZ_Eta_Fail_EB = new TH1F("McZ_eta", "Monte Carlo Z #eta;#eta;Arbitrary Units", 100, -3., 3.);
//	TH1F* h_McZ_rescEta_Fail_EB = new TH1F("McZ_rescEta", "Monte Carlo Rescaled Z #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McZ_Phi_Fail_EB = new TH1F("McZ_phi", "Monte Carlo Z #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
//	TH1F* h_McZ_rescPhi_Fail_EB = new TH1F("McZ_rescPhi", "Monte Carlo Z Rescaled #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	
	TH1F* h_McElec_Pt_Fail_EB = new TH1F("McElec_Pt", "Monte Carlo Electron p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_McElec_rescPt_Fail_EB = new TH1F("McElec_rescPt", "Monte Carlo Rescaled Electron p_{T};Rescaled p_{T};Arbitrary Units", 100, 0., 100.);
	TH1F* h_McElec_Eta_Fail_EB = new TH1F("McElec_Eta", "Monte Carlo Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McElec_rescEta_Fail_EB = new TH1F("McElec_rescEta", "Monte Carlo Rescaled Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
//	TH1F* h_McElec_Phi_Fail_EB = new TH1F("McElec_Phi", "Monte Carlo Electron #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
//	TH1F* h_McElec_rescPhi_Fail_EB = new TH1F("McElec_rescPhi", "Monte Carlo Rescaled Electron #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	
	TH1F* h_McErNu_Pt_Fail_EB = new TH1F("McErNu_Pt", "Monte Carlo Ersatz Neutrino p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_McErNu_rescPt_Fail_EB = new TH1F("McErNu_rescPt", "Monte Carlo Rescaled Ersatz Neutrino p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_McErNu_Eta_Fail_EB = new TH1F("McErNu_Eta", "Monte Carlo Ersatz Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McErNu_rescEta_Fail_EB = new TH1F("McErNu_rescEta", "Monte Carlo Rescaled Ersatz Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McErNu_Phi_Fail_EB = new TH1F("McErNu_Phi", "Monte Carlo Ersatz Neutrino #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_McErNu_rescPhi_Fail_EB = new TH1F("McErNu_rescPhi", "Monte Carlo Rescaled Ersatz Neutrino #phi;Rescaled #phi;Arbitrary Units", 100, 0., 3.1416);
	
	TH1F* h_McElecErNu_dPhi_Fail_EB = new TH1F("McElecErNu_dPhi", "Monte Carlo #Delta#phi between Electrons;#Delta#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_McElecErNu_dR_Fail_EB = new TH1F("McElecErNu_dR", "Monte Carlo #DeltaR between Electrons;MC Match;Arbitrary Units", 100, 0., 3.1416);
	TH1I* h_McMatch_Fail_EB = new TH1I("McMatch", "Monte Carlo Match of Ersatz Neutrino to an Electron;MC Match;Arbitrary Units", 2, 0, 2);

//	TH1F* h_EtaInt_Fail_EB = new TH1F("EtaInt", "", 345, 0, 345);
//	TH1F* h_EtaWeights_Fail_EB = new TH1F("EtaWeights", "", 40, 0., 2.);
	cout << "Declared histograms h_xxx_Fail_EB" << endl;

	TString OutFileName_Fail_EE = "/tmp/rnandi/"+InFileName+"_Selected_Fail_EE";
	TFile *outfile_fail_EE = TFile::Open(OutFileName_Fail_EE+".root", "recreate");
	cout << "Opened outfile_Fail_EE" << endl;
	
	TH1I* h_nTags_Fail_EE = new TH1I("nTags", "Number of Tags;Number of Tags;Arbitrary Units", 4, 0, 4);
	TH1I* h_nErNu_Fail_EE = new TH1I("nErNu", "Number of Ersatz Neutrinos;Number of Ersatz Neutrinos;Arbitrary Units", 2, 0, 2);
	TH1I* h_nMcElecs_Zmum_Fail_EE = new TH1I("nMcElecs_Zmum", "Number of Monte Carlo Electrons with Z as Mother;Number of MC Electrons;Arbitrary Units", 4, 0, 4);
	TH1I* h_nMcElecs_Final_Fail_EE = new TH1I("nMcElecs_Final", "Number of Final State Monte Carlo Electrons;Number of MC Electrons;Arbitrary Units", 4, 0, 4);
	TH1I* h_nRecHitsInStrip_Fail_EE = new TH1I("nRecHitsInStrip", ";;Arbitrary Units", 20, 0, 20);
	TH1I* h_nRecHitsInCone_Fail_EE = new TH1I("nRecHitsInCone", ";;Arbitrary Units", 20, 0, 20);

	TH1F* h_Z_pt_Fail_EE = new TH1F("Z_pt", "Z p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);

	TH1F* h_elec_q_Fail_EE = new TH1F("elec_q", "Electron Charge;q;Arbitrary Units", 3, -1.5, 1.5);
	TH1F* h_elec_pt_Fail_EE = new TH1F("elec_pt", "Electron p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_elec_eta_Fail_EE = new TH1F("elec_eta", "Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_elec_phi_Fail_EE = new TH1F("elec_phi", "Electron #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_elec_sIhIh_Fail_EE = new TH1F("elec_sIhIh", "Electron #sigma_{i#etai#eta};#sigma_{i#etai#eta};Arbitary Units", 100, 0., 0.05);
	TH1F* h_elec_dEta_Fail_EE = new TH1F("elec_dEta", "Electron #Delta#eta;#Delta#eta;Arbitrary Units", 100, 0., 0.02);
	TH1F* h_elec_dPhi_Fail_EE = new TH1F("elec_dPhi", "Electron #Delta#phi;#Delta#phi;Arbitrary Units", 100, 0., 0.1);
	TH1F* h_elec_TrkIso_Fail_EE = new TH1F("elec_TrkIso", "Electon Track Isolation;Track Isolation #Sigma p_{T}^{tracks} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_EcalIso_Fail_EE = new TH1F("elec_EcalIso", "Electron ECAL Isolation;ECAL Isolation #Sigma E_{T}^{ECAL} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_HcalIso_Fail_EE = new TH1F("elec_HcalIso", "Electron HCAL Isolation;HCAL Isolation #Sigma E_{T}^{HCAL} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_elec_rescPt_Fail_EE = new TH1F("elec_rescPt", "Electron Rescaled p_{T};Rescaled p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_elec_rescEta_Fail_EE = new TH1F("elec_rescEta", "Electron Rescaled #eta;Rescaled #eta;Arbitrary Units", 100, -3., 3.);

//	TH1F* h_ErNu_q_Fail_EE = new TH1F("h_ErNu_q", "Ersatz Neutrino Charge;q;Arbitrary Units", 3, -1.5, 1.5);
	TH1F* h_ErNu_pt_Fail_EE = new TH1F("ErNu_pt", "Ersatz Neutrino p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_eta_Fail_EE = new TH1F("ErNu_eta", "Ersatz Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_ErNu_phi_Fail_EE = new TH1F("ErNu_phi", "Ersatz Neutrino #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_ErNu_rescPt_Fail_EE = new TH1F("ErNu_rescPt", "Ersatz Neutrino Rescaled p_{T};Rescaled p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_rescEta_Fail_EE = new TH1F("ErNu_rescEta", "Ersatz Neutrino Rescaled #eta;Rescaled #eta", 100, -3., 3.);
	
	TH1F* h_ErNu_sIhIh_Fail_EE = new TH1F("ErNu_sIhIh", "Ersatz Neutrino #sigma_{i#etai#eta};#sigma_{i#etai#eta};Arbitrary Units", 100, 0., 0.05);
//	TH1F* h_ErNu_dEta_Fail_EE = new TH1F("ErNu_dEta", "Ersatz Neutrino #Delta#eta;#Delta#eta;Arbitrary Units", 100, 0., 0.02);
//	TH1F* h_ErNu_dPhi_Fail_EE = new TH1F("ErNu_dPhi", "Ersatz Neutrino #Delta#phi;#Delta#phi;Arbitrary Units", 100, 0., 0.1);
	TH1F* h_ErNu_TrkIso_Fail_EE = new TH1F("ErNu_TrkIso", "Ersatz Neutrino Track Isolation;Track Isolation #Sigma p_{T}^{tracks} / GeV;Arbitrary Units", 100, 0., 10.);
//	TH1F* h_ErNu_EcalIso_Fail_EE = new TH1F("ErNu_EcalIso", "Ersatz Neutrino ECAL Isolation;ECAL Isolation #Sigma E_{T}^{ECAL} / GeV;Arbitrary Units", 100, 0., 10.);
//	TH1F* h_ErNu_HcalIso_Fail_EE = new TH1F("ErNu_HcalIso", "Ersatz Neutrino HCAL Isolation;HCAL Isolation #Sigma E_{T}^{HCAL} / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_ErNu_e1x5Max_Fail_EE = new TH1F("ErNu_e1x5Max", "Ersatz Neutrino Maximum Energy in 1x5 Array of Crystals;e1x5Max / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_e2x5Max_Fail_EE = new TH1F("ErNu_e2x5Max", "Ersatz Neutrino Maximum Energy in 2x5 Array of Crystals;e2x5Max / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_e5x5_Fail_EE = new TH1F("ErNu_e5x5", "Ersatz Neutrino Energy in 5x5 Array of Crystals;e5x5 / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_HcalE015_Fail_EE = new TH1F("ErNu_HcalE015", "Ersatz Neutrino HCAL Energy in a 0.15 Cone;HCAL Energy / GeV;Arbitrary Units", 100, 0., 10.);
	TH1F* h_ErNu_HcalEt015_Fail_EE = new TH1F("ErNu_HcalEt015", "Ersatz Neutrino HCAL E_{T} in a 0.15 Cone;HCAL E_{T};Arbitrary Units", 100, 0., 10.);
	
//	TH1F* h_rechitE_Fail_EE = new TH1F("h_rechitE", "", 100, , ); // What is this
	TH1F* h_ErNu_E_Fail_EE = new TH1F("ErNu_E", "Ersatz Neutrino Energy;Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_rawE_Fail_EE = new TH1F("ErNu_rawE", "Ersatz Neutrino Raw Energy;Raw Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_unclusE_Fail_EE = new TH1F("ErNu_unclusE", "Ersatz Neutrino Unclustered ECAL Energy;Unclustered Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_fEtaCorrE_Fail_EE = new TH1F("ErNu_fEtaCorrE", "Ersatz Neutrino fEta Corrected Energy;Corrected Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_fBremCorrE_Fail_EE = new TH1F("ErNu_fBremCorrE", "Ersatz Neutrino fBrem Corrected Energy;Correcteed Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErNu_AddCorrE_Fail_EE = new TH1F("ErNu_AddCorrE", "Ersatz Neutrino Add Corrected Energy;Corrected Energy / GeV;Arbitrary Units", 100, 0., 100.);
	TH1I* h_ErNu_nClusters_Fail_EE = new TH1I("ErNu_nClusters", "Ersatz Neutrino Number of Clusters;Number of Clusters;Arbitrary Units", 10, 0, 10);
	
//	TH1F* h_ErsatzV1MEt_Fail_EE = new TH1F("ErsatzV1MEt", "ErsatzV1MEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1Mt_Fail_EE = new TH1F("ErsatzV1Mt", "ErsatzV1Mt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1MEtphi_Fail_EE = new TH1F("ErsatzV1MEtPhi", "ErsatzV1MEtPhi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1aMEt_Fail_EE = new TH1F("ErsatzV1aMEt", "ErsatzV1aMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1aMt_Fail_EE = new TH1F("ErsatzV1aMt", "ErsatzV1aMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1aMEtphi_Fail_EE = new TH1F("ErsatzV1aMEtphi", "ErsatzV1aMEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1bMEt_Fail_EE = new TH1F("ErsatzV1bMEt", "ErsatzV1bMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1bMt_Fail_EE = new TH1F("ErsatzV1bMt", "ErsatzV1bMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1bMEtphi_Fail_EE = new TH1F("ErsatzV1bMEtphi", "ErsatzV1bMEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1cMEt_Fail_EE = new TH1F("ErsatzV1cMEt", "ErsatzV1cMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1cMt_Fail_EE = new TH1F("ErsatzV1cMt", "ErsatzV1cMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV1cMEtphi_Fail_EE = new TH1F("ErsatzV1cMEtphi", "ErsatzV1cMEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV2MEt_Fail_EE = new TH1F("ErsatzV1aMEt", "ErsatzV1aMEt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV2Mt_Fail_EE = new TH1F("ErsatzV1aMt", "ErsatzV1aMt;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_ErsatzV2MEtphi_Fail_EE = new TH1F("ErsatzV2MEtphi", "ErsatzV2MEtphi;#slash{E_{T}} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_RecoCaloMEt_Fail_EE = new TH1F("RecoCaloMEt", "Calometric #slashE_{T};#slashE_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_ErsatzV1Mesc_Fail_EE = new TH1F("ErsatzV1Mesc", "Invariant Mass;Invariant Mass / GeV;Arbitrary Units", 100, 41., 141.);
	TH1F* h_ErsatzV1rescMesc_Fail_EE = new TH1F("ErsatzV1rescMesc", "Rescaled Invariant Mass;Rescaled Invariant Mass / GeV;Arbitrary Units", 100, 41., 141.);
	
	TH1F* h_McZ_M_Fail_EE = new TH1F("McZ_M", "Monte Carlo Z Mass;Mass / GeV; Arbitrary Units", 100, 41., 141.);
	TH1F* h_McZ_rescM_Fail_EE = new TH1F("McZ_rescM", "Monte Carlo Rescaled Z Mass;Mass / GeV;Arbitrary Units", 100, 41., 141.);
	TH1F* h_McZ_Pt_Fail_EE = new TH1F("McZ_Pt", "Monte Carlo Z p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_McZ_rescPt_Fail_EE = new TH1F("McZ_rescPt", "Monte Carlo Rescaled Z p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
//	TH1F* h_McZ_Eta_Fail_EE = new TH1F("McZ_eta", "Monte Carlo Z #eta;#eta;Arbitrary Units", 100, -3., 3.);
//	TH1F* h_McZ_rescEta_Fail_EE = new TH1F("McZ_rescEta", "Monte Carlo Rescaled Z #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McZ_Phi_Fail_EE = new TH1F("McZ_phi", "Monte Carlo Z #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
//	TH1F* h_McZ_rescPhi_Fail_EE = new TH1F("McZ_rescPhi", "Monte Carlo Z Rescaled #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	
	TH1F* h_McElec_Pt_Fail_EE = new TH1F("McElec_Pt", "Monte Carlo Electron p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_McElec_rescPt_Fail_EE = new TH1F("McElec_rescPt", "Monte Carlo Rescaled Electron p_{T};Rescaled p_{T};Arbitrary Units", 100, 0., 100.);
	TH1F* h_McElec_Eta_Fail_EE = new TH1F("McElec_Eta", "Monte Carlo Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McElec_rescEta_Fail_EE = new TH1F("McElec_rescEta", "Monte Carlo Rescaled Electron #eta;#eta;Arbitrary Units", 100, -3., 3.);
//	TH1F* h_McElec_Phi_Fail_EE = new TH1F("McElec_Phi", "Monte Carlo Electron #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
//	TH1F* h_McElec_rescPhi_Fail_EE = new TH1F("McElec_rescPhi", "Monte Carlo Rescaled Electron #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	
	TH1F* h_McErNu_Pt_Fail_EE = new TH1F("McErNu_Pt", "Monte Carlo Ersatz Neutrino p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_McErNu_rescPt_Fail_EE = new TH1F("McErNu_rescPt", "Monte Carlo Rescaled Ersatz Neutrino p_{T};p_{T} / GeV;Arbitrary Units", 100, 0., 100.);
	TH1F* h_McErNu_Eta_Fail_EE = new TH1F("McErNu_Eta", "Monte Carlo Ersatz Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McErNu_rescEta_Fail_EE = new TH1F("McErNu_rescEta", "Monte Carlo Rescaled Ersatz Neutrino #eta;#eta;Arbitrary Units", 100, -3., 3.);
	TH1F* h_McErNu_Phi_Fail_EE = new TH1F("McErNu_Phi", "Monte Carlo Ersatz Neutrino #phi;#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_McErNu_rescPhi_Fail_EE = new TH1F("McErNu_rescPhi", "Monte Carlo Rescaled Ersatz Neutrino #phi;Rescaled #phi;Arbitrary Units", 100, 0., 3.1416);
	
	TH1F* h_McElecErNu_dPhi_Fail_EE = new TH1F("McElecErNu_dPhi", "Monte Carlo #Delta#phi between Electrons;#Delta#phi;Arbitrary Units", 100, 0., 3.1416);
	TH1F* h_McElecErNu_dR_Fail_EE = new TH1F("McElecErNu_dR", "Monte Carlo #DeltaR between Electrons;MC Match;Arbitrary Units", 100, 0., 3.1416);
	TH1I* h_McMatch_Fail_EE = new TH1I("McMatch", "Monte Carlo Match of Ersatz Neutrino to an Electron;MC Match;Arbitrary Units", 2, 0, 2);

//	TH1F* h_EtaInt_Fail_EE = new TH1F("EtaInt", "", 345, 0, 345);
//	TH1F* h_EtaWeights_Fail_EE = new TH1F("EtaWeights", "", 40, 0., 2.);
	cout << "Declared histograms h_xxx_Fail_EE" << endl;
	
	//TTree *t_pass_EB = new TTree(Pass_EB, "Pass_tIso_EB");
	//TTree *t_pass_EE = new TTree(Pass_EE, "Pass_tIso_EE");
	//TTree *t_fail_EB = new TTree(Fail_EB, "Fail_tIso_EB");
	//TTree *t_fail_EE = new TTree(Fail_EB, "Fail_tIso_EE");
	//TTree *t_pass_trackiso_EB = t->CloneTree(0);
	//TTree *t_pass_trackiso_EE = t->CloneTree(0);
	//TTree *t_fail_trackiso_EE = t->CloneTree(0);
	//TTree *t_fail_trackiso_EE = t->CloneTree(0);
	
	//TString OutFileNameFail = "/tmp/rnandi/"+InFileName+"_Selected_Fail";
	//TFile *outfile_fail = TFile::Open(OutFileNameFail+".root", "recreate");
	
	//TTree *t_fail_trackiso = t->CloneTree(0);
	//t_fail_trackiso->SetTitle("Fail_tIso");
	
	TFile *file = TFile::Open("/tmp/rnandi/"+InFileName+".root");

        TTree *t = (TTree*) file->Get("ErsatzMEt/ErsatzMEt");
	std::cout<< "Got Tree " << t->GetTitle() << std::endl;
/*	
	int nErNu;
	double tag_pt[4], tag_eta[4], tag_phi[4], probe_pt[4], probe_eta[4], probe_phi[4];
	double ErsatzV1bMEt[4];
	double elec_trckIso[4], elec_ECALIso[4], elec_HCALIso[4];
	double elec_sigIhIh[4], elec_dPhi[4], elec_dEta[4];
	double tag_rescPt[4], mesc[4];
	double nu_e1x5[4], nu_e2x5[4], nu_e5x5[4], nu_sigIhIh[4];
	double nu_HCALEt[4], nu_HCAL[4], nu_trckIso[4];
	double caloMEt;
	int nu_elec;
*/
	cout << "Setting up branches." << endl;
	
	TBranch* bNum_Tags = t->GetBranch("nTags");
	bNum_Tags->SetAddress(&nTags);
	TBranch* bNum_ErNu = t->GetBranch("nProbes");
	bNum_ErNu->SetAddress(&nErNu);
	TBranch* bNum_McElecs_Zmum = t->GetBranch("McElec_nZmum");
	bNum_McElecs_Zmum->SetAddress(&nMcElecs_Zmum);
	TBranch* bNum_McElecs_Final = t->GetBranch("McElec_nFinal");
	bNum_McElecs_Final->SetAddress(&nMcElecs_Final);
	TBranch* bNum_RecHitsInStrip = t->GetBranch("nRecHitsInStrip");
	bNum_RecHitsInStrip->SetAddress(&nRecHitsInStrip);
	TBranch* bNum_RecHitsInCone = t->GetBranch("nRecHitsInCone");
	bNum_RecHitsInCone->SetAddress(&nRecHitsInCone);
	
	//Z properties
	TBranch* bZ_pt = t->GetBranch("Z_pt");
	bZ_pt->SetAddress(&Z_pt);

	//Selected electron properties
	TBranch* bTag_q = t->GetBranch("tag_q");
	bTag_q->SetAddress(&elec_q);
	TBranch* bTag_pt = t->GetBranch("tag_pt");
	bTag_pt->SetAddress(&elec_pt);
	TBranch* bTag_eta = t->GetBranch("tag_eta");
	bTag_eta->SetAddress(&elec_eta);
	TBranch* bTag_phi = t->GetBranch("tag_phi");
	bTag_phi->SetAddress(&elec_phi);
	TBranch* bTag_rescPt = t->GetBranch("tag_rescPt");
	bTag_rescPt->SetAddress(&elec_rescPt);
	TBranch* bTag_rescEta = t->GetBranch("tag_rescEta");
	bTag_rescEta->SetAddress(&elec_rescEta);
//	TBranch* bTag_rescPhi = t->GetBranch("tag_rescPhi");
//	bTag_rescPhi->SetAddress(&elec_rescPhi);
	
	TBranch* bTag_sIhIh = t->GetBranch("tag_sIhIh");
	bTag_sIhIh->SetAddress(&elec_sIhIh);
	TBranch* bTag_dPhi = t->GetBranch("tag_dPhiIn");
	bTag_dPhi->SetAddress(&elec_dPhi);
	TBranch* bTag_dEta = t->GetBranch("tag_dEtaIn");
	bTag_dEta->SetAddress(&elec_dEta);
	TBranch* bTag_tIso = t->GetBranch("tag_isoTrack");
	bTag_tIso->SetAddress(&elec_trkIso);
	TBranch* bTag_eIso = t->GetBranch("tag_isoEcal");
	bTag_eIso->SetAddress(&elec_EcalIso);
	TBranch* bTag_hIso = t->GetBranch("tag_isoHcal");
	bTag_hIso->SetAddress(&elec_HcalIso);

	//ersatz neutrino properties
//	TBranch* bProbe_q = t->GetBranch("probe_q");
//	bProbe_q->SetAddress(&probe_q);
	TBranch* bProbe_pt = t->GetBranch("probe_pt");
	bProbe_pt->SetAddress(&probe_pt);
	TBranch* bProbe_eta = t->GetBranch("probe_eta");
	bProbe_eta->SetAddress(&probe_eta);
	TBranch* bProbe_phi = t->GetBranch("probe_phi");
	bProbe_phi->SetAddress(&probe_phi);
	TBranch* bProbe_rescPt = t->GetBranch("probe_rescPt");
	bProbe_rescPt->SetAddress(&probe_rescPt);
	TBranch* bProbe_rescEta = t->GetBranch("probe_rescEta");
	bProbe_rescEta->SetAddress(&probe_rescEta);
//	TBranch* bProbe_rescPhi = t->GetBranch("probe_rescPhi");
//	bProbe_rescPhi->SetAddress(&probe_rescPhi);
	TBranch* bProbe_trckIso = t->GetBranch("probe_isoTrack");
	bProbe_trckIso->SetAddress(&ernu_trkIso);
//	TBranch* bProbe_ECALIso = t->GetBranch("probe_isoECAL");
//	bProbe_ECALIso->SetAddress(&ernu_ECALIso);
//	TBranch* bProbe_HCALIso = t->GetBranch("probe_isoHCAL");
//	bProbe_HCALIso->SetAddress(&ernu_HCALIso);
	TBranch* bProbe_sIhIh = t->GetBranch("probe_sIhIh");
	bProbe_sIhIh->SetAddress(&ernu_sIhIh);
//	TBranch* bProbe_DeltaEta = t->GetBranch("probe_DeltaEta");
//	bProbe_DeltaEta->SetAddress(&ernu_DeltaEta);
//	TBranch* bProbe_DeltaPhiIso = t->GetBranch("probe_DeltaPhi");
//	bProbe_DeltaPhi->SetAddress(&ernu_DeltaPhi);
	TBranch* bProbe_e1x5 = t->GetBranch("probe_e1x5Max");
	bProbe_e1x5->SetAddress(&ernu_e1x5);
	TBranch* bProbe_e2x5 = t->GetBranch("probe_e2x5Max");
	bProbe_e2x5->SetAddress(&ernu_e2x5);
	TBranch* bProbe_e5x5 = t->GetBranch("probe_e5x5");
	bProbe_e5x5->SetAddress(&ernu_e5x5);
	TBranch* bProbe_HcalE015 = t->GetBranch("probe_HcalE015");
	bProbe_HcalE015->SetAddress(&ernu_HcalE015);
	TBranch* bProbe_HcalEt015 = t->GetBranch("probe_HcalEt015");
	bProbe_HcalEt015->SetAddress(&ernu_HcalEt015);

	//Energy Correction results
	TBranch* bProbe_E = t->GetBranch("probe_E");
	bProbe_E->SetAddress(&ernu_E);
	TBranch* bProbe_rawE = t->GetBranch("probe_rawE");
	bProbe_rawE->SetAddress(&ernu_rawE);
	TBranch* bProbe_unclusE = t->GetBranch("probe_UnclusEcalE");
	bProbe_unclusE->SetAddress(&ernu_unclusE);
	TBranch* bProbe_fEtaCorrE = t->GetBranch("probe_fEtaCorrE");
	bProbe_fEtaCorrE->SetAddress(&ernu_fEtaCorrE);
	TBranch* bProbe_fBremCorrE = t->GetBranch("probe_fBremCorrE");
	bProbe_fBremCorrE->SetAddress(&ernu_fBremCorrE);
	TBranch* bProbe_AddCorrE = t->GetBranch("probe_EAdd");
	bProbe_AddCorrE->SetAddress(&ernu_AddCorrE);
	TBranch* bProbe_d_MCE_SCE = t->GetBranch("probe_d_MCE_SCE");
	bProbe_d_MCE_SCE->SetAddress(&ernu_d_McE_ScE);
	TBranch* bProbe_nClus = t->GetBranch("probe_nClus");
	bProbe_nClus->SetAddress(&ernu_nClusters);
	
	//Ersatz MEt results
	TBranch* bErsatzV1_MEt = t->GetBranch("ErsatzV1CaloMEt");
	bErsatzV1_MEt->SetAddress(&ErsatzV1MEt);
	TBranch* bErsatzV1_Mt = t->GetBranch("ErsatzV1CaloMt");
	bErsatzV1_Mt->SetAddress(&ErsatzV1Mt);
	TBranch* bErsatzV1_MEtphi = t->GetBranch("ErsatzV1CaloMEtPhi");
	bErsatzV1_MEtphi->SetAddress(&ErsatzV1MEtphi);
	
	TBranch* bErsatzV1a_MEt = t->GetBranch("ErsatzV1aCaloMEt");
	bErsatzV1a_MEt->SetAddress(&ErsatzV1aMEt);
	TBranch* bErsatzV1a_Mt = t->GetBranch("ErsatzV1aCaloMt");
	bErsatzV1a_Mt->SetAddress(&ErsatzV1aMt);
	TBranch* bErsatzV1a_MEtphi = t->GetBranch("ErsatzV1aCaloMEtPhi");
	bErsatzV1a_MEtphi->SetAddress(&ErsatzV1aMEtphi);
	
	TBranch* bErsatzV1b_MEt = t->GetBranch("ErsatzV1bCaloMEt");
	bErsatzV1b_MEt->SetAddress(&ErsatzV1bMEt);
	TBranch* bErsatzV1b_Mt = t->GetBranch("ErsatzV1bCaloMt");
	bErsatzV1b_Mt->SetAddress(&ErsatzV1bMt);
	TBranch* bErsatzV1b_MEtphi = t->GetBranch("ErsatzV1bCaloMEtPhi");
	bErsatzV1b_MEtphi->SetAddress(&ErsatzV1bMEtphi);
	
	TBranch* bErsatzV1c_MEt = t->GetBranch("ErsatzV1cCaloMEt");
	bErsatzV1c_MEt->SetAddress(&ErsatzV1cMEt);
	//TBranch* bErsatzV1c_Mt = t->GetBranch("ErsatzV1cCaloMt");
	//bErsatzV1c_Mt->SetAddress(&ErsatzV1cMt);
	TBranch* bErsatzV1c_MEtphi = t->GetBranch("ErsatzV1cCaloMEtPhi");
	bErsatzV1c_MEtphi->SetAddress(&ErsatzV1cMEtphi);
	
	TBranch* bErsatzV2_MEt = t->GetBranch("ErsatzV2CaloMEt");
	bErsatzV2_MEt->SetAddress(&ErsatzV2MEt);
	TBranch* bErsatzV2_Mt = t->GetBranch("ErsatzV2CaloMt");
	bErsatzV2_Mt->SetAddress(&ErsatzV2Mt);
	TBranch* bErsatzV2_MEtphi = t->GetBranch("ErsatzV2CaloMEtPhi");
	bErsatzV2_MEtphi->SetAddress(&ErsatzV2MEtphi);
	
	TBranch* bMesc = t->GetBranch("ErsatzV1_Mesc");
	bMesc->SetAddress(&mesc); 
	TBranch* brescMesc = t->GetBranch("ErsatzV1_rescMesc");
	brescMesc->SetAddress(&rescMesc); 
	TBranch* bCaloMEt = t->GetBranch("recoCaloMEt");
        bCaloMEt->SetAddress(&caloMEt);

	TBranch* bMcZ_m = t->GetBranch("McZ_m");
	bMcZ_m->SetAddress(&McZ_m);
	TBranch* bMcZ_pt = t->GetBranch("McZ_Pt");
	bMcZ_pt->SetAddress(&McZ_pt);
//	TBranch* bMcZ_eta = t->GetBranch("McZ_eta");
//	bMcZ_eta->SetAddress(&McZ_eta);
	TBranch* bMcZ_phi = t->GetBranch("McZ_Phi");
	bMcZ_phi->SetAddress(&McZ_phi);
	TBranch* bMcZ_rescM = t->GetBranch("McZ_rescM");
	bMcZ_rescM->SetAddress(&McZ_rescM);
//	TBranch* bMcZ_rescPt = t->GetBranch("McZ_rescPt");
//	bMcZ_rescPt->SetAddress(&McZ_rescPt);
//	TBranch* bMcZ_rescEta = t->GetBranch("McZ_rescEta");
//	bMcZ_rescEta->SetAddress(&McZ_rescEta);
//	TBranch* bMcZ_rescPhi = t->GetBranch("McZ_rescPhi");
//	bMcZ_rescPhi->SetAddress(&McZ_rescPhi);

	TBranch* bMcElec_pt = t->GetBranch("McElec_pt");
	bMcElec_pt->SetAddress(&McElec_pt);
	TBranch* bMcElec_eta = t->GetBranch("McElec_eta");
	bMcElec_eta->SetAddress(&McElec_eta);
//	TBranch* bMcElec_phi = t->GetBranch("McElec_phi");
//	bMcElec_phi->SetAddress(&McElec_phi);
	TBranch* bMcElec_rescPt = t->GetBranch("McElec_rescPt");
	bMcElec_rescPt->SetAddress(&McElec_rescPt);
	TBranch* bMcElec_rescEta = t->GetBranch("McElec_rescEta");
	bMcElec_rescEta->SetAddress(&McElec_rescEta);
//	TBranch* bMcElec_rescPhi = t->GetBranch("McElec_rescPhi");
//	bMcElec_rescPhi->SetAddress(&McElec_rescPhi);

	TBranch* bMcErNu_pt = t->GetBranch("McProbe_pt");
	bMcErNu_pt->SetAddress(&McErNu_pt);
	TBranch* bMcErNu_eta = t->GetBranch("McProbe_eta");
	bMcErNu_eta->SetAddress(&McErNu_eta);
	TBranch* bMcErNu_phi = t->GetBranch("McProbe_phi");
	bMcErNu_phi->SetAddress(&McErNu_phi);
	TBranch* bMcErNu_rescPt = t->GetBranch("McProbe_rescPt");
	bMcErNu_rescPt->SetAddress(&McErNu_rescPt);
	TBranch* bMcErNu_rescEta = t->GetBranch("McProbe_rescEta");
	bMcErNu_rescEta->SetAddress(&McErNu_rescEta);
	TBranch* bMcErNu_rescPhi = t->GetBranch("McProbe_rescPhi");
	bMcErNu_rescPhi->SetAddress(&McErNu_rescPhi);

	TBranch* bMcElecErNu_dPhi = t->GetBranch("McElecProbe_dPhi");
	bMcElecErNu_dPhi->SetAddress(&McElecErNu_dPhi);
	TBranch* bMcElecErNu_dR = t->GetBranch("McElecProbe_dR");
	bMcElecErNu_dR->SetAddress(&McElecErNu_dR);
//	TBranch* bProbe_elec = t->GetBranch("probe_elecMatch");
//	bProbe_elec->SetAddress(&ernu_McMatch);
	
	//TString OutFileName = "/tmp/rnandi/"+InFileName+"_Selected";
	//TFile *outfile = TFile::Open(OutFileName+".root", "recreate");
	
	//TString OutFileName = "Zee_Histograms.root";
	//TFile *outfile = TFile::Open("/tmp/rnandi/Zee_Histograms.root", "recreate");

/*
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
	bTag_tIso->SetAddress(&elec_trckIso);
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
*/
	long nentries = t->GetEntries();
	std::cout <<"Processing "<< nentries << std::endl;
	for(long i = 0; i<nentries;++i)
	{
		if(i%10000 == 0) std::cout <<"Processing event "<< i <<"."<< std::endl;
		t->GetEntry(i);
		for(int j = 0; j < nErNu; ++j)
		{ 
			bool passEtCut = false;
			/*
			if(process == "Zee" || process == "BCtoE_30to80" || process == "BCtoE_80to170"){
				if(tag_rescPt[j] > cPt) passEtCut = true;
			}else{
				if(tag_pt[j] > (91.188/80.398)*cPt) passEtCut = true;
			}
			*/
			if(elec_rescPt[j] > cPt) passEtCut = true;
			if(passEtCut)
			{
                                if(fabs(mesc[j]-91.1876) < 21.1876)
				{
					bool pass_e_cuts = false;
					bool inBarrel = false;
					double cTrackiso;
					if(fabs(elec_eta[j])<1.4442)
					{
						pass_e_cuts = (elec_EcalIso[j] < cECALiso_EB && elec_HcalIso[j] < cHCALiso_EB
								&& elec_sIhIh[j] < csIhIh_EB && elec_dPhi[j] < cDeltaPhi_EB
								&& elec_dEta[j] < cDeltaEta_EB);
						cTrackiso = cTrackiso_EB;
						inBarrel = true;
					}else if(fabs(elec_eta[j] < 2.5))
					{
						pass_e_cuts = (elec_EcalIso[j] < cECALiso_EE && elec_HcalIso[j] < cHCALiso_EE
								&& elec_sIhIh[j] < csIhIh_EE && elec_dPhi[j] < cDeltaPhi_EE
								&& elec_dEta[j] < cDeltaEta_EE);
						cTrackiso = cTrackiso_EE;
					}
					if(pass_e_cuts)
					{
						bool pass_nu_cuts = false;
						double f1x5 = ernu_e1x5[j]/ernu_e5x5[j];
						double f2x5 = ernu_e2x5[j]/ernu_e5x5[j];
						if(fabs(probe_eta[j]) < 1.4442)
						{
							pass_nu_cuts = (ernu_HcalE015[j] < cHCAL && (f1x5 > cf1x5 || f2x5 > cf2x5)
									/*&& ernu_McMatch[j] == celecmatch*/);
						}else if(fabs(probe_eta[j] < 2.5)){
							pass_nu_cuts = (ernu_HcalEt015[j] < cHCALEt && ernu_sIhIh[j] < cnusIhIh
									/*&& ernu_McMatch[j] == celecmatch*/);
						}
						if(pass_nu_cuts)
						{
							if(elec_trkIso[j] < cTrackiso)
							{
								if(inBarrel)
								{
									h_nTags_Pass_EB->Fill(nTags);
									h_nErNu_Pass_EB->Fill(nErNu);
									h_nMcElecs_Zmum_Pass_EB->Fill(nMcElecs_Zmum);
									h_nMcElecs_Final_Pass_EB->Fill(nMcElecs_Final);
									h_RecoCaloMEt_Pass_EB->Fill(caloMEt);

									h_nRecHitsInStrip_Pass_EB->Fill(nRecHitsInStrip[j]);
									h_nRecHitsInCone_Pass_EB->Fill(nRecHitsInCone[j]);

									h_Z_pt_Pass_EB->Fill(Z_pt[j]);

									h_elec_q_Pass_EB->Fill(elec_q[j]);
									h_elec_pt_Pass_EB->Fill(elec_pt[j]);
									h_elec_eta_Pass_EB->Fill(elec_eta[j]);
									h_elec_phi_Pass_EB->Fill(elec_phi[j]);
									h_elec_sIhIh_Pass_EB->Fill(elec_sIhIh[j]);
									h_elec_dPhi_Pass_EB->Fill(elec_dPhi[j]);
									h_elec_dEta_Pass_EB->Fill(elec_dEta[j]);
									h_elec_TrkIso_Pass_EB->Fill(elec_trkIso[j]);
									h_elec_EcalIso_Pass_EB->Fill(elec_EcalIso[j]);
									h_elec_HcalIso_Pass_EB->Fill(elec_HcalIso[j]);
									h_elec_rescPt_Pass_EB->Fill(elec_rescPt[j]);
									h_elec_rescEta_Pass_EB->Fill(elec_rescEta[j]);
		
//									h_ErNu_q_Pass_EB->Fill(probe_q[j]);
									h_ErNu_pt_Pass_EB->Fill(probe_pt[j]);
									h_ErNu_eta_Pass_EB->Fill(probe_eta[j]);
									h_ErNu_phi_Pass_EB->Fill(probe_phi[j]);
									h_ErNu_rescPt_Pass_EB->Fill(probe_rescPt[j]);
									h_ErNu_rescEta_Pass_EB->Fill(probe_rescEta[j]);
		
									h_ErNu_sIhIh_Pass_EB->Fill(ernu_sIhIh[j]);
//									h_ErNu_dPhi_Pass_EB->Fill(ernu_dPhi[j]);
//									h_ErNu_dEta_Pass_EB->Fill(ernu_dEta[j]);
									h_ErNu_TrkIso_Pass_EB->Fill(ernu_trkIso[j]);
//									h_ErNu_EcalIso_Pass_EB->Fill(ernu_EcalIso[j]);
//									h_ErNu_HcalIso_Pass_EB->Fill(ernu_HcalIso[j]);
									h_ErNu_e1x5Max_Pass_EB->Fill(ernu_e1x5[j]);
									h_ErNu_e2x5Max_Pass_EB->Fill(ernu_e2x5[j]);
									h_ErNu_e5x5_Pass_EB->Fill(ernu_e5x5[j]);
									h_ErNu_HcalE015_Pass_EB->Fill(ernu_HcalE015[j]);
									h_ErNu_HcalEt015_Pass_EB->Fill(ernu_HcalEt015[j]);

									h_ErNu_E_Pass_EB->Fill(ernu_E[j]);
									h_ErNu_rawE_Pass_EB->Fill(ernu_rawE[j]);
									h_ErNu_unclusE_Pass_EB->Fill(ernu_unclusE[j]);
									h_ErNu_fEtaCorrE_Pass_EB->Fill(ernu_fEtaCorrE[j]);
									h_ErNu_fBremCorrE_Pass_EB->Fill(ernu_fBremCorrE[j]);
									h_ErNu_AddCorrE_Pass_EB->Fill(ernu_AddCorrE[j]);
									h_ErNu_nClusters_Pass_EB->Fill(ernu_nClusters[j]);

//									h_ErsatzV1MEt_Pass_EB->Fill(ErsatzV1MEt[j]);
//									h_ErsatzV1Mt_Pass_EB->Fill(ErsatzV1Mt[j]);
//									h_ErsatzV1MEtphi_Pass_EB->Fill(ErsatzV1MEtphi[j]);
//									h_ErsatzV1aMEt_Pass_EB->Fill(ErsatzV1aMEt[j]);
//									h_ErsatzV1aMt_Pass_EB->Fill(ErsatzV1aMt[j]);
//									h_ErsatzV1aMEtphi_Pass_EB->Fill(ErsatzV1aMEtphi[j]);
									h_ErsatzV1bMEt_Pass_EB->Fill(ErsatzV1bMEt[j]);
									h_ErsatzV1bMt_Pass_EB->Fill(ErsatzV1bMt[j]);
									h_ErsatzV1bMEtphi_Pass_EB->Fill(ErsatzV1bMEtphi[j]);
//									h_ErsatzV1cMEt_Pass_EB->Fill(ErsatzV1cMEt[j]);
//									h_ErsatzV1cMt_Pass_EB->Fill(ErsatzV1cMt[j]);
//									h_ErsatzV1cMEtphi_Pass_EB->Fill(ErsatzV1cMEtphi[j]);
//									h_ErsatzV2MEt_Pass_EB->Fill(ErsatzV2MEt[j]);
//									h_ErsatzV2Mt_Pass_EB->Fill(ErsatzV2Mt[j]);
//									h_ErsatzV2MEtphi_Pass_EB->Fill(ErsatzV2MEtphi[j]);
									h_ErsatzV1Mesc_Pass_EB->Fill(mesc[j]);
									h_ErsatzV1rescMesc_Pass_EB->Fill(rescMesc[j]);

									h_McZ_M_Pass_EB->Fill(McZ_m[j]);
									h_McZ_rescM_Pass_EB->Fill(McZ_rescM[j]);
									h_McZ_Pt_Pass_EB->Fill(McZ_pt[j]);
//									h_McZ_rescPt_Pass_EB->Fill(McZ_rescPt[j]);
//									h_McZ_Eta_Pass_EB->Fill(McZ_eta[j]);
//									h_McZ_rescEta_Pass_EB->Fill(McZ_rescEta[j]);
									h_McZ_Phi_Pass_EB->Fill(McZ_phi[j]);
//									h_McZ_rescPhi_Pass_EB->Fill(McZ_rescPhi[j]);

									h_McElec_Pt_Pass_EB->Fill(McElec_pt[j]);
									h_McElec_rescPt_Pass_EB->Fill(McElec_rescPt[j]);
									h_McElec_Eta_Pass_EB->Fill(McElec_eta[j]);
									h_McElec_rescEta_Pass_EB->Fill(McElec_rescEta[j]);
//									h_McElec_Phi_Pass_EB->Fill(McElec_phi[j]);
//									h_McElec_rescPhi_Pass_EB->Fill(McElec_rescPhi[j]);
		
									h_McErNu_Pt_Pass_EB->Fill(McErNu_pt[j]);
									h_McErNu_rescPt_Pass_EB->Fill(McErNu_rescPt[j]);
									h_McErNu_Eta_Pass_EB->Fill(McErNu_eta[j]);
									h_McErNu_rescEta_Pass_EB->Fill(McErNu_rescEta[j]);
									h_McErNu_Phi_Pass_EB->Fill(McErNu_phi[j]);
									h_McErNu_rescPhi_Pass_EB->Fill(McErNu_rescPhi[j]);

									h_McElecErNu_dPhi_Pass_EB->Fill(McElecErNu_dPhi[j]);
									h_McElecErNu_dR_Pass_EB->Fill(McElecErNu_dR[j]);
									h_McMatch_Pass_EB->Fill(ernu_McMatch[j]);
								}else{
									h_nTags_Pass_EE->Fill(nTags);
									h_nErNu_Pass_EE->Fill(nErNu);
									h_nMcElecs_Zmum_Pass_EE->Fill(nMcElecs_Zmum);
									h_nMcElecs_Final_Pass_EE->Fill(nMcElecs_Final);
									h_RecoCaloMEt_Pass_EE->Fill(caloMEt);

									h_nRecHitsInStrip_Pass_EE->Fill(nRecHitsInStrip[j]);
									h_nRecHitsInCone_Pass_EE->Fill(nRecHitsInCone[j]);

									h_Z_pt_Pass_EE->Fill(Z_pt[j]);

									h_elec_q_Pass_EE->Fill(elec_q[j]);
									h_elec_pt_Pass_EE->Fill(elec_pt[j]);
									h_elec_eta_Pass_EE->Fill(elec_eta[j]);
									h_elec_phi_Pass_EE->Fill(elec_phi[j]);
									h_elec_sIhIh_Pass_EE->Fill(elec_sIhIh[j]);
									h_elec_dPhi_Pass_EE->Fill(elec_dPhi[j]);
									h_elec_dEta_Pass_EE->Fill(elec_dEta[j]);
									h_elec_TrkIso_Pass_EE->Fill(elec_trkIso[j]);
									h_elec_EcalIso_Pass_EE->Fill(elec_EcalIso[j]);
									h_elec_HcalIso_Pass_EE->Fill(elec_HcalIso[j]);
									h_elec_rescPt_Pass_EE->Fill(elec_rescPt[j]);
									h_elec_rescEta_Pass_EE->Fill(elec_rescEta[j]);
		
//									h_ErNu_q_Pass_EE->Fill(probe_q[j]);
									h_ErNu_pt_Pass_EE->Fill(probe_pt[j]);
									h_ErNu_eta_Pass_EE->Fill(probe_eta[j]);
									h_ErNu_phi_Pass_EE->Fill(probe_phi[j]);
									h_ErNu_rescPt_Pass_EE->Fill(probe_rescPt[j]);
									h_ErNu_rescEta_Pass_EE->Fill(probe_rescEta[j]);
		
									h_ErNu_sIhIh_Pass_EE->Fill(ernu_sIhIh[j]);
//									h_ErNu_dPhi_Pass_EE->Fill(ernu_dPhi[j]);
//									h_ErNu_dEta_Pass_EE->Fill(ernu_dEta[j]);
									h_ErNu_TrkIso_Pass_EE->Fill(ernu_trkIso[j]);
//									h_ErNu_EcalIso_Pass_EE->Fill(ernu_EcalIso[j]);
//									h_ErNu_HcalIso_Pass_EE->Fill(ernu_HcalIso[j]);
									h_ErNu_e1x5Max_Pass_EE->Fill(ernu_e1x5[j]);
									h_ErNu_e2x5Max_Pass_EE->Fill(ernu_e2x5[j]);
									h_ErNu_e5x5_Pass_EE->Fill(ernu_e5x5[j]);
									h_ErNu_HcalE015_Pass_EE->Fill(ernu_HcalE015[j]);
									h_ErNu_HcalEt015_Pass_EE->Fill(ernu_HcalEt015[j]);

									h_ErNu_E_Pass_EE->Fill(ernu_E[j]);
									h_ErNu_rawE_Pass_EE->Fill(ernu_rawE[j]);
									h_ErNu_unclusE_Pass_EE->Fill(ernu_unclusE[j]);
									h_ErNu_fEtaCorrE_Pass_EE->Fill(ernu_fEtaCorrE[j]);
									h_ErNu_fBremCorrE_Pass_EE->Fill(ernu_fBremCorrE[j]);
									h_ErNu_AddCorrE_Pass_EE->Fill(ernu_AddCorrE[j]);
									h_ErNu_nClusters_Pass_EE->Fill(ernu_nClusters[j]);

//									h_ErsatzV1MEt_Pass_EE->Fill(ErsatzV1MEt[j]);
//									h_ErsatzV1Mt_Pass_EE->Fill(ErsatzV1Mt[j]);
//									h_ErsatzV1MEtphi_Pass_EE->Fill(ErsatzV1MEtphi[j]);
//									h_ErsatzV1aMEt_Pass_EE->Fill(ErsatzV1aMEt[j]);
//									h_ErsatzV1aMt_Pass_EE->Fill(ErsatzV1aMt[j]);
//									h_ErsatzV1aMEtphi_Pass_EE->Fill(ErsatzV1aMEtphi[j]);
									h_ErsatzV1bMEt_Pass_EE->Fill(ErsatzV1bMEt[j]);
									h_ErsatzV1bMt_Pass_EE->Fill(ErsatzV1bMt[j]);
									h_ErsatzV1bMEtphi_Pass_EE->Fill(ErsatzV1bMEtphi[j]);
//									h_ErsatzV1cMEt_Pass_EE->Fill(ErsatzV1cMEt[j]);
//									h_ErsatzV1cMt_Pass_EE->Fill(ErsatzV1cMt[j]);
//									h_ErsatzV1cMEtphi_Pass_EE->Fill(ErsatzV1cMEtphi[j]);
//									h_ErsatzV2MEt_Pass_EE->Fill(ErsatzV2MEt[j]);
//									h_ErsatzV2Mt_Pass_EE->Fill(ErsatzV2Mt[j]);
//									h_ErsatzV2MEtphi_Pass_EE->Fill(ErsatzV2MEtphi[j]);
									h_ErsatzV1Mesc_Pass_EE->Fill(mesc[j]);
									h_ErsatzV1rescMesc_Pass_EE->Fill(rescMesc[j]);

									h_McZ_M_Pass_EE->Fill(McZ_m[j]);
									h_McZ_rescM_Pass_EE->Fill(McZ_rescM[j]);
									h_McZ_Pt_Pass_EE->Fill(McZ_pt[j]);
//									h_McZ_rescPt_Pass_EE->Fill(McZ_rescPt[j]);
//									h_McZ_Eta_Pass_EE->Fill(McZ_eta[j]);
//									h_McZ_rescEta_Pass_EE->Fill(McZ_rescEta[j]);
									h_McZ_Phi_Pass_EE->Fill(McZ_phi[j]);
//									h_McZ_rescPhi_Pass_EE->Fill(McZ_rescPhi[j]);

									h_McElec_Pt_Pass_EE->Fill(McElec_pt[j]);
									h_McElec_rescPt_Pass_EE->Fill(McElec_rescPt[j]);
									h_McElec_Eta_Pass_EE->Fill(McElec_eta[j]);
									h_McElec_rescEta_Pass_EE->Fill(McElec_rescEta[j]);
//									h_McElec_Phi_Pass_EE->Fill(McElec_phi[j]);
//									h_McElec_rescPhi_Pass_EE->Fill(McElec_rescPhi[j]);
		
									h_McErNu_Pt_Pass_EE->Fill(McErNu_pt[j]);
									h_McErNu_rescPt_Pass_EE->Fill(McErNu_rescPt[j]);
									h_McErNu_Eta_Pass_EE->Fill(McErNu_eta[j]);
									h_McErNu_rescEta_Pass_EE->Fill(McErNu_rescEta[j]);
									h_McErNu_Phi_Pass_EE->Fill(McErNu_phi[j]);
									h_McErNu_rescPhi_Pass_EE->Fill(McErNu_rescPhi[j]);

									h_McElecErNu_dPhi_Pass_EE->Fill(McElecErNu_dPhi[j]);
									h_McElecErNu_dR_Pass_EE->Fill(McElecErNu_dR[j]);
									h_McMatch_Pass_EE->Fill(ernu_McMatch[j]);
								}
							}else{
								if(inBarrel)
								{
									h_nTags_Fail_EB->Fill(nTags);
									h_nErNu_Fail_EB->Fill(nErNu);
									h_nMcElecs_Zmum_Fail_EB->Fill(nMcElecs_Zmum);
									h_nMcElecs_Final_Fail_EB->Fill(nMcElecs_Final);
									h_RecoCaloMEt_Fail_EB->Fill(caloMEt);

									h_nRecHitsInStrip_Fail_EB->Fill(nRecHitsInStrip[j]);
									h_nRecHitsInCone_Fail_EB->Fill(nRecHitsInCone[j]);

									h_Z_pt_Fail_EB->Fill(Z_pt[j]);

									h_elec_q_Fail_EB->Fill(elec_q[j]);
									h_elec_pt_Fail_EB->Fill(elec_pt[j]);
									h_elec_eta_Fail_EB->Fill(elec_eta[j]);
									h_elec_phi_Fail_EB->Fill(elec_phi[j]);
									h_elec_sIhIh_Fail_EB->Fill(elec_sIhIh[j]);
									h_elec_dPhi_Fail_EB->Fill(elec_dPhi[j]);
									h_elec_dEta_Fail_EB->Fill(elec_dEta[j]);
									h_elec_TrkIso_Fail_EB->Fill(elec_trkIso[j]);
									h_elec_EcalIso_Fail_EB->Fill(elec_EcalIso[j]);
									h_elec_HcalIso_Fail_EB->Fill(elec_HcalIso[j]);
									h_elec_rescPt_Fail_EB->Fill(elec_rescPt[j]);
									h_elec_rescEta_Fail_EB->Fill(elec_rescEta[j]);
		
//									h_ErNu_q_Fail_EB->Fill(probe_q[j]);
									h_ErNu_pt_Fail_EB->Fill(probe_pt[j]);
									h_ErNu_eta_Fail_EB->Fill(probe_eta[j]);
									h_ErNu_phi_Fail_EB->Fill(probe_phi[j]);
									h_ErNu_rescPt_Fail_EB->Fill(probe_rescPt[j]);
									h_ErNu_rescEta_Fail_EB->Fill(probe_rescEta[j]);
		
									h_ErNu_sIhIh_Fail_EB->Fill(ernu_sIhIh[j]);
//									h_ErNu_dPhi_Fail_EB->Fill(ernu_dPhi[j]);
//									h_ErNu_dEta_Fail_EB->Fill(ernu_dEta[j]);
									h_ErNu_TrkIso_Fail_EB->Fill(ernu_trkIso[j]);
//									h_ErNu_EcalIso_Fail_EB->Fill(ernu_EcalIso[j]);
//									h_ErNu_HcalIso_Fail_EB->Fill(ernu_HcalIso[j]);
									h_ErNu_e1x5Max_Fail_EB->Fill(ernu_e1x5[j]);
									h_ErNu_e2x5Max_Fail_EB->Fill(ernu_e2x5[j]);
									h_ErNu_e5x5_Fail_EB->Fill(ernu_e5x5[j]);
									h_ErNu_HcalE015_Fail_EB->Fill(ernu_HcalE015[j]);
									h_ErNu_HcalEt015_Fail_EB->Fill(ernu_HcalEt015[j]);

									h_ErNu_E_Fail_EB->Fill(ernu_E[j]);
									h_ErNu_rawE_Fail_EB->Fill(ernu_rawE[j]);
									h_ErNu_unclusE_Fail_EB->Fill(ernu_unclusE[j]);
									h_ErNu_fEtaCorrE_Fail_EB->Fill(ernu_fEtaCorrE[j]);
									h_ErNu_fBremCorrE_Fail_EB->Fill(ernu_fBremCorrE[j]);
									h_ErNu_AddCorrE_Fail_EB->Fill(ernu_AddCorrE[j]);
									h_ErNu_nClusters_Fail_EB->Fill(ernu_nClusters[j]);

//									h_ErsatzV1MEt_Fail_EB->Fill(ErsatzV1MEt[j]);
//									h_ErsatzV1Mt_Fail_EB->Fill(ErsatzV1Mt[j]);
//									h_ErsatzV1MEtphi_Fail_EB->Fill(ErsatzV1MEtphi[j]);
//									h_ErsatzV1aMEt_Fail_EB->Fill(ErsatzV1aMEt[j]);
//									h_ErsatzV1aMt_Fail_EB->Fill(ErsatzV1aMt[j]);
//									h_ErsatzV1aMEtphi_Fail_EB->Fill(ErsatzV1aMEtphi[j]);
									h_ErsatzV1bMEt_Fail_EB->Fill(ErsatzV1bMEt[j]);
									h_ErsatzV1bMt_Fail_EB->Fill(ErsatzV1bMt[j]);
									h_ErsatzV1bMEtphi_Fail_EB->Fill(ErsatzV1bMEtphi[j]);
//									h_ErsatzV1cMEt_Fail_EB->Fill(ErsatzV1cMEt[j]);
//									h_ErsatzV1cMt_Fail_EB->Fill(ErsatzV1cMt[j]);
//									h_ErsatzV1cMEtphi_Fail_EB->Fill(ErsatzV1cMEtphi[j]);
//									h_ErsatzV2MEt_Fail_EB->Fill(ErsatzV2MEt[j]);
//									h_ErsatzV2Mt_Fail_EB->Fill(ErsatzV2Mt[j]);
//									h_ErsatzV2MEtphi_Fail_EB->Fill(ErsatzV2MEtphi[j]);
									h_ErsatzV1Mesc_Fail_EB->Fill(mesc[j]);
									h_ErsatzV1rescMesc_Fail_EB->Fill(rescMesc[j]);

									h_McZ_M_Fail_EB->Fill(McZ_m[j]);
									h_McZ_rescM_Fail_EB->Fill(McZ_rescM[j]);
									h_McZ_Pt_Fail_EB->Fill(McZ_pt[j]);
//									h_McZ_rescPt_Fail_EB->Fill(McZ_rescPt[j]);
//									h_McZ_Eta_Fail_EB->Fill(McZ_eta[j]);
//									h_McZ_rescEta_Fail_EB->Fill(McZ_rescEta[j]);
									h_McZ_Phi_Fail_EB->Fill(McZ_phi[j]);
//									h_McZ_rescPhi_Fail_EB->Fill(McZ_rescPhi[j]);

									h_McElec_Pt_Fail_EB->Fill(McElec_pt[j]);
									h_McElec_rescPt_Fail_EB->Fill(McElec_rescPt[j]);
									h_McElec_Eta_Fail_EB->Fill(McElec_eta[j]);
									h_McElec_rescEta_Fail_EB->Fill(McElec_rescEta[j]);
//									h_McElec_Phi_Fail_EB->Fill(McElec_phi[j]);
//									h_McElec_rescPhi_Fail_EB->Fill(McElec_rescPhi[j]);
		
									h_McErNu_Pt_Fail_EB->Fill(McErNu_pt[j]);
									h_McErNu_rescPt_Fail_EB->Fill(McErNu_rescPt[j]);
									h_McErNu_Eta_Fail_EB->Fill(McErNu_eta[j]);
									h_McErNu_rescEta_Fail_EB->Fill(McErNu_rescEta[j]);
									h_McErNu_Phi_Fail_EB->Fill(McErNu_phi[j]);
									h_McErNu_rescPhi_Fail_EB->Fill(McErNu_rescPhi[j]);

									h_McElecErNu_dPhi_Fail_EB->Fill(McElecErNu_dPhi[j]);
									h_McElecErNu_dR_Fail_EB->Fill(McElecErNu_dR[j]);
									h_McMatch_Fail_EB->Fill(ernu_McMatch[j]);
								}else{
									h_nTags_Fail_EE->Fill(nTags);
									h_nErNu_Fail_EE->Fill(nErNu);
									h_nMcElecs_Zmum_Fail_EE->Fill(nMcElecs_Zmum);
									h_nMcElecs_Final_Fail_EE->Fill(nMcElecs_Final);
									h_RecoCaloMEt_Fail_EE->Fill(caloMEt);

									h_nRecHitsInStrip_Fail_EE->Fill(nRecHitsInStrip[j]);
									h_nRecHitsInCone_Fail_EE->Fill(nRecHitsInCone[j]);

									h_Z_pt_Fail_EE->Fill(Z_pt[j]);

									h_elec_q_Fail_EE->Fill(elec_q[j]);
									h_elec_pt_Fail_EE->Fill(elec_pt[j]);
									h_elec_eta_Fail_EE->Fill(elec_eta[j]);
									h_elec_phi_Fail_EE->Fill(elec_phi[j]);
									h_elec_sIhIh_Fail_EE->Fill(elec_sIhIh[j]);
									h_elec_dPhi_Fail_EE->Fill(elec_dPhi[j]);
									h_elec_dEta_Fail_EE->Fill(elec_dEta[j]);
									h_elec_TrkIso_Fail_EE->Fill(elec_trkIso[j]);
									h_elec_EcalIso_Fail_EE->Fill(elec_EcalIso[j]);
									h_elec_HcalIso_Fail_EE->Fill(elec_HcalIso[j]);
									h_elec_rescPt_Fail_EE->Fill(elec_rescPt[j]);
									h_elec_rescEta_Fail_EE->Fill(elec_rescEta[j]);
		
//									h_ErNu_q_Fail_EE->Fill(probe_q[j]);
									h_ErNu_pt_Fail_EE->Fill(probe_pt[j]);
									h_ErNu_eta_Fail_EE->Fill(probe_eta[j]);
									h_ErNu_phi_Fail_EE->Fill(probe_phi[j]);
									h_ErNu_rescPt_Fail_EE->Fill(probe_rescPt[j]);
									h_ErNu_rescEta_Fail_EE->Fill(probe_rescEta[j]);
		
									h_ErNu_sIhIh_Fail_EE->Fill(ernu_sIhIh[j]);
//									h_ErNu_dPhi_Fail_EE->Fill(ernu_dPhi[j]);
//									h_ErNu_dEta_Fail_EE->Fill(ernu_dEta[j]);
									h_ErNu_TrkIso_Fail_EE->Fill(ernu_trkIso[j]);
//									h_ErNu_EcalIso_Fail_EE->Fill(ernu_EcalIso[j]);
//									h_ErNu_HcalIso_Fail_EE->Fill(ernu_HcalIso[j]);
									h_ErNu_e1x5Max_Fail_EE->Fill(ernu_e1x5[j]);
									h_ErNu_e2x5Max_Fail_EE->Fill(ernu_e2x5[j]);
									h_ErNu_e5x5_Fail_EE->Fill(ernu_e5x5[j]);
									h_ErNu_HcalE015_Fail_EE->Fill(ernu_HcalE015[j]);
									h_ErNu_HcalEt015_Fail_EE->Fill(ernu_HcalEt015[j]);

									h_ErNu_E_Fail_EE->Fill(ernu_E[j]);
									h_ErNu_rawE_Fail_EE->Fill(ernu_rawE[j]);
									h_ErNu_unclusE_Fail_EE->Fill(ernu_unclusE[j]);
									h_ErNu_fEtaCorrE_Fail_EE->Fill(ernu_fEtaCorrE[j]);
									h_ErNu_fBremCorrE_Fail_EE->Fill(ernu_fBremCorrE[j]);
									h_ErNu_AddCorrE_Fail_EE->Fill(ernu_AddCorrE[j]);
									h_ErNu_nClusters_Fail_EE->Fill(ernu_nClusters[j]);

//									h_ErsatzV1MEt_Fail_EE->Fill(ErsatzV1MEt[j]);
//									h_ErsatzV1Mt_Fail_EE->Fill(ErsatzV1Mt[j]);
//									h_ErsatzV1MEtphi_Fail_EE->Fill(ErsatzV1MEtphi[j]);
//									h_ErsatzV1aMEt_Fail_EE->Fill(ErsatzV1aMEt[j]);
//									h_ErsatzV1aMt_Fail_EE->Fill(ErsatzV1aMt[j]);
//									h_ErsatzV1aMEtphi_Fail_EE->Fill(ErsatzV1aMEtphi[j]);
									h_ErsatzV1bMEt_Fail_EE->Fill(ErsatzV1bMEt[j]);
									h_ErsatzV1bMt_Fail_EE->Fill(ErsatzV1bMt[j]);
									h_ErsatzV1bMEtphi_Fail_EE->Fill(ErsatzV1bMEtphi[j]);
//									h_ErsatzV1cMEt_Fail_EE->Fill(ErsatzV1cMEt[j]);
//									h_ErsatzV1cMt_Fail_EE->Fill(ErsatzV1cMt[j]);
//									h_ErsatzV1cMEtphi_Fail_EE->Fill(ErsatzV1cMEtphi[j]);
//									h_ErsatzV2MEt_Fail_EE->Fill(ErsatzV2MEt[j]);
//									h_ErsatzV2Mt_Fail_EE->Fill(ErsatzV2Mt[j]);
//									h_ErsatzV2MEtphi_Fail_EE->Fill(ErsatzV2MEtphi[j]);
									h_ErsatzV1Mesc_Fail_EE->Fill(mesc[j]);
									h_ErsatzV1rescMesc_Fail_EE->Fill(rescMesc[j]);

									h_McZ_M_Fail_EE->Fill(McZ_m[j]);
									h_McZ_rescM_Fail_EE->Fill(McZ_rescM[j]);
									h_McZ_Pt_Fail_EE->Fill(McZ_pt[j]);
//									h_McZ_rescPt_Fail_EE->Fill(McZ_rescPt[j]);
//									h_McZ_Eta_Fail_EE->Fill(McZ_eta[j]);
//									h_McZ_rescEta_Fail_EE->Fill(McZ_rescEta[j]);
									h_McZ_Phi_Fail_EE->Fill(McZ_phi[j]);
//									h_McZ_rescPhi_Fail_EE->Fill(McZ_rescPhi[j]);

									h_McElec_Pt_Fail_EE->Fill(McElec_pt[j]);
									h_McElec_rescPt_Fail_EE->Fill(McElec_rescPt[j]);
									h_McElec_Eta_Fail_EE->Fill(McElec_eta[j]);
									h_McElec_rescEta_Fail_EE->Fill(McElec_rescEta[j]);
//									h_McElec_Phi_Fail_EE->Fill(McElec_phi[j]);
//									h_McElec_rescPhi_Fail_EE->Fill(McElec_rescPhi[j]);
		
									h_McErNu_Pt_Fail_EE->Fill(McErNu_pt[j]);
									h_McErNu_rescPt_Fail_EE->Fill(McErNu_rescPt[j]);
									h_McErNu_Eta_Fail_EE->Fill(McErNu_eta[j]);
									h_McErNu_rescEta_Fail_EE->Fill(McErNu_rescEta[j]);
									h_McErNu_Phi_Fail_EE->Fill(McErNu_phi[j]);
									h_McErNu_rescPhi_Fail_EE->Fill(McErNu_rescPhi[j]);

									h_McElecErNu_dPhi_Fail_EE->Fill(McElecErNu_dPhi[j]);
									h_McElecErNu_dR_Fail_EE->Fill(McElecErNu_dR[j]);
									h_McMatch_Fail_EE->Fill(ernu_McMatch[j]);
								}
							}
						}		
					}
				}
			}
		}
	}
	//outfile->Write();
	//outfile->Close();
	outfile_pass_EB->Write();
	outfile_pass_EB->Close();
	outfile_pass_EE->Write();
	outfile_pass_EE->Close();
	outfile_fail_EB->Write();
	outfile_fail_EB->Close();
	outfile_fail_EE->Write();
	outfile_fail_EE->Close();
}
/*
double deltaPhi(double phi1, double phi2)
{
     double result = phi1 - phi2;
     while (result > acos(-1.)) result -= 2*acos(-1.);
     while (result <= -acos(-1.)) result += 2*acos(-1.);
     return result;
}

double deltaR(double eta1, double phi1, double eta2, double phi2)
{
	double dEta = eta1-eta2;
	double dPhi = deltaPhi(phi1, phi2);
	double dR = sqrt(dEta*dEta+dPhi*dPhi);
	return dR;
}
*/
