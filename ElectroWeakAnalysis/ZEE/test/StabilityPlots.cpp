#include "TFile.h"
#include "TGraphErrors.h"
#include "TVectorT.h"
#include "TH1.h"
#include "TLegend.h"
#include "TCanvas.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

using namespace std;

void StabilityPlots()
{
	ofstream CaloMET("CaloMET.txt");
	ofstream TcMET("TcMET.txt");
	//ofstream T1MET("T1MET.txt");
	ofstream PfMET("PfMET.txt");
	CaloMET << "MET cut\tftrue\tf\tdf\teff\tdeff\ta\tb\tnpass\tf'true\tf'\tdf'\teff'\tdeff'\td\tc\tnfail" << endl;
	TcMET << "MET cut\tftrue\tf\tdf\teff\tdeff\ta\tb\tnpass\tf'true\tf'\tdf'\teff'\tdeff'\td\tc\tnfail" << endl;
	//T1MET << "MET cut\tftrue\tf\tdf\teff\tdeff\ta\tb\tnpass\tf'true\tf'\tdf'\teff'\tdeff'\td\tc\tnfail" << endl;
	PfMET << "MET cut\tftrue\tf\tdf\teff\tdeff\ta\tb\tnpass\tf'true\tf'\tdf'\teff'\tdeff'\td\tc\tnfail" << endl;
	cout << setprecision(3);
	CaloMET << setprecision(3);
	TcMET << setprecision(3);
	//T1MET << setprecision(3);
	PfMET << setprecision(3);

	TVectorT<double> metcut(11), metcuterror(11);
	TVectorT<double> f_calo_ersatz(11), df_calo_ersatz(11), eff_calo_ersatz(11), deff_calo_ersatz(11), deltaf_calo_ersatz(11);
	TVectorT<double> fprime_calo_ersatz(11), dfprime_calo_ersatz(11), effprime_calo_ersatz(11), deffprime_calo_ersatz(11), deltafprime_calo_ersatz(11);
	TVectorT<double> f_tc_ersatz(11), df_tc_ersatz(11), eff_tc_ersatz(11), deff_tc_ersatz(11), deltaf_tc_ersatz(11);
	TVectorT<double> fprime_tc_ersatz(11), dfprime_tc_ersatz(11), effprime_tc_ersatz(11), deffprime_tc_ersatz(11), deltafprime_tc_ersatz(11);
	//TVectorT<double> f_t1_ersatz(11), df_t1_ersatz(11), eff_t1_ersatz(11), deff_t1_ersatz(11), deltaf_t1_ersatz(11);
	//TVectorT<double> fprime_t1_ersatz(11), dfprime_t1_ersatz(11), effprime_t1_ersatz(11), deffprime_t1_ersatz(11), deltafprime_t1_ersatz(11);
	TVectorT<double> f_pf_ersatz(11), df_pf_ersatz(11), eff_pf_ersatz(11), deff_pf_ersatz(11), deltaf_pf_ersatz(11);
	TVectorT<double> fprime_pf_ersatz(11), dfprime_pf_ersatz(11), effprime_pf_ersatz(11), deffprime_pf_ersatz(11), deltafprime_pf_ersatz(11);

	TFile* file = TFile::Open("ZeePlots.root"); 

	TH1F* h_ErsatzCaloMEt_pass = (TH1F*) file->Get("ErsatzMEt_pass");
	TH1F* h_ErsatzCaloMEt_fail = (TH1F*) file->Get("ErsatzMEt_fail");
	TH1F* h_WCaloMEt_pass = (TH1F*) file->Get("WCaloMEt_pass");
	TH1F* h_WCaloMEt_fail = (TH1F*) file->Get("WCaloMEt_fail");
	TH1F* h_ErsatzTcMEt_pass = (TH1F*) file->Get("ErsatzTcMEt_pass");
	TH1F* h_ErsatzTcMEt_fail = (TH1F*) file->Get("ErsatzTcMEt_fail");
	TH1F* h_WTcMEt_pass = (TH1F*) file->Get("WTcMEt_pass");
	TH1F* h_WTcMEt_fail = (TH1F*) file->Get("WTcMEt_fail");
	//TH1F* h_ErsatzT1MEt_pass = (TH1F*) file->Get("ErsatzT1MEt_pass");
	//TH1F* h_ErsatzT1MEt_fail = (TH1F*) file->Get("ErsatzT1MEt_fail");
	//TH1F* h_WT1MEt_pass = (TH1F*) file->Get("WT1MEt_pass");
	//TH1F* h_WT1MEt_fail = (TH1F*) file->Get("WT1MEt_fail");
	TH1F* h_ErsatzPfMEt_pass = (TH1F*) file->Get("ErsatzPfMEt_pass");
	TH1F* h_ErsatzPfMEt_fail = (TH1F*) file->Get("ErsatzPfMEt_fail");
	TH1F* h_WPfMEt_pass = (TH1F*) file->Get("WPfMEt_pass");
	TH1F* h_WPfMEt_fail = (TH1F*) file->Get("WPfMEt_fail");

	for(int i=0; i<102; i++)
	{
		cout << h_WCaloMEt_pass->GetBinContent(i) << endl;
	}
	cout << "sum = " << h_WCaloMEt_pass->Integral(21,100) << endl; 
	for(int i=0; i<102; i++)
	{
		cout << h_WCaloMEt_fail->GetBinContent(i) << endl;
	}
	cout << "sum = " << h_WCaloMEt_fail->Integral(21,100) << endl; 

	int i = 0;
	double a_calo_ersatz, a_calo_true;
	double a_tc_ersatz, a_tc_true;
	//double a_t1_ersatz, a_t1_true;
	double a_pf_ersatz, a_pf_true;
	double b_calo_ersatz, b_calo_true;
	double b_tc_ersatz, b_tc_true;
	//double b_t1_ersatz, b_t1_true;
	double b_pf_ersatz, b_pf_true;
	double c_calo_ersatz, c_calo_true;
	double c_tc_ersatz, c_tc_true;
	//double c_t1_ersatz, c_t1_true;
	double c_pf_ersatz, c_pf_true;
	double d_calo_ersatz, d_calo_true;
	double d_tc_ersatz, d_tc_true;
	//double d_t1_ersatz, d_t1_true;
	double d_pf_ersatz, d_pf_true;
	cout << "mooooooooo" << endl;
	for(int cMET=20; cMET<41; cMET+=2)
	{
		a_calo_ersatz = h_ErsatzCaloMEt_pass->Integral(cMET+1, 100); // Problem Here
		a_tc_ersatz = h_ErsatzTcMEt_pass->Integral(cMET+1, 100);
		//a_t1_ersatz = h_ErsatzT1MEt_pass->Integral(cMET+1, 100);
		a_pf_ersatz = h_ErsatzPfMEt_pass->Integral(cMET+1, 100);
	cout << "mooooooooo" << endl;
		a_calo_true = h_WCaloMEt_pass->Integral(cMET+1, 100);
		a_tc_true = h_WTcMEt_pass->Integral(cMET+1, 100);
		//a_t1_true = h_WT1MEt_pass->Integral(cMET+1, 100);
		a_pf_true = h_WPfMEt_pass->Integral(cMET+1, 100);
	cout << "mooooooooo" << endl;
		b_calo_ersatz = h_ErsatzCaloMEt_pass->Integral(1, cMET);
		b_tc_ersatz = h_ErsatzTcMEt_pass->Integral(1, cMET);
		//b_t1_ersatz = h_ErsatzT1MEt_pass->Integral(1, cMET);
		b_pf_ersatz = h_ErsatzPfMEt_pass->Integral(1, cMET);
	cout << "mooooooooo" << endl;
		b_calo_true = h_WCaloMEt_pass->Integral(1, cMET);
		b_tc_true = h_WTcMEt_pass->Integral(1, cMET);
		//b_t1_true = h_WT1MEt_pass->Integral(1, cMET);
		b_pf_true = h_WPfMEt_pass->Integral(1, cMET);
	cout << "mooooooooo" << endl;
		c_calo_ersatz = h_ErsatzCaloMEt_fail->Integral(1, cMET);
		c_tc_ersatz = h_ErsatzTcMEt_fail->Integral(1, cMET);
		//c_t1_ersatz = h_ErsatzT1MEt_fail->Integral(1, cMET);
		c_pf_ersatz = h_ErsatzPfMEt_fail->Integral(1, cMET);
	cout << "mooooooooo" << endl;
		c_calo_true = h_WCaloMEt_fail->Integral(1, cMET);
		c_tc_true = h_WTcMEt_fail->Integral(1, cMET);
		//c_t1_true = h_WT1MEt_fail->Integral(1, cMET);
		c_pf_true = h_WPfMEt_fail->Integral(1, cMET);
	cout << "mooooooooo" << endl;
		d_calo_ersatz = h_ErsatzCaloMEt_fail->Integral(cMET+1, 100);
		d_tc_ersatz = h_ErsatzTcMEt_fail->Integral(cMET+1, 100);
		//d_t1_ersatz = h_ErsatzT1MEt_fail->Integral(cMET+1, 100);
		d_pf_ersatz = h_ErsatzPfMEt_fail->Integral(cMET+1, 100);
	cout << "mooooooooo" << endl;
		d_calo_true = h_WCaloMEt_fail->Integral(cMET+1, 100);
		d_tc_true = h_WTcMEt_fail->Integral(cMET+1, 100);
		//d_t1_true = h_WT1MEt_fail->Integral(cMET+1, 100);
		d_pf_true = h_WPfMEt_fail->Integral(cMET+1, 100);
	cout << "mooooooooo" << endl;

		metcut[i] = 1.0*cMET;
		metcuterror[i] = 0.0;

		eff_calo_ersatz[i] = a_calo_ersatz/(a_calo_ersatz + b_calo_ersatz);
		deff_calo_ersatz[i] = sqrt(eff_calo_ersatz[i]*(1. - eff_calo_ersatz[i])/(a_calo_ersatz + b_calo_ersatz));
		f_calo_ersatz[i] = a_calo_ersatz/b_calo_ersatz;
		df_calo_ersatz[i] = deff_calo_ersatz[i]/((1. - eff_calo_ersatz[i])*(1. - eff_calo_ersatz[i]));
		deltaf_calo_ersatz[i] = a_calo_ersatz/b_calo_ersatz - a_calo_true/b_calo_true;
		effprime_calo_ersatz[i] = d_calo_ersatz/(d_calo_ersatz + c_calo_ersatz);
		deffprime_calo_ersatz[i] = sqrt(effprime_calo_ersatz[i]*(1. - effprime_calo_ersatz[i])/(d_calo_ersatz + c_calo_ersatz));
		fprime_calo_ersatz[i] = d_calo_ersatz/c_calo_ersatz;
		dfprime_calo_ersatz[i] = deffprime_calo_ersatz[i]/((1. - effprime_calo_ersatz[i])*(1. - effprime_calo_ersatz[i]));
		deltafprime_calo_ersatz[i] = d_calo_ersatz/c_calo_ersatz - d_calo_true/c_calo_true;

		eff_tc_ersatz[i] = a_tc_ersatz/(a_tc_ersatz + b_tc_ersatz);
		deff_tc_ersatz[i] = sqrt(eff_tc_ersatz[i]*(1. - eff_tc_ersatz[i])/(a_tc_ersatz + b_tc_ersatz));
		f_tc_ersatz[i] = a_tc_ersatz/b_tc_ersatz;
		df_tc_ersatz[i] = deff_tc_ersatz[i]/((1. - eff_tc_ersatz[i])*(1. - eff_tc_ersatz[i]));
		deltaf_tc_ersatz[i] = a_tc_ersatz/b_tc_ersatz - a_tc_true/b_tc_true;
		effprime_tc_ersatz[i] = d_tc_ersatz/(d_tc_ersatz + c_tc_ersatz);
		deffprime_tc_ersatz[i] = sqrt(effprime_tc_ersatz[i]*(1. - effprime_tc_ersatz[i])/(d_tc_ersatz + c_tc_ersatz));
		fprime_tc_ersatz[i] = d_tc_ersatz/c_tc_ersatz;
		dfprime_tc_ersatz[i] = deffprime_tc_ersatz[i]/((1. - effprime_tc_ersatz[i])*(1. - effprime_tc_ersatz[i]));
		deltafprime_tc_ersatz[i] = d_tc_ersatz/c_tc_ersatz - d_tc_true/c_tc_true;
/*
		eff_t1_ersatz[i] = a_t1_ersatz/(a_t1_ersatz + b_t1_ersatz);
		deff_t1_ersatz[i] = sqrt(eff_t1_ersatz[i]*(1. - eff_t1_ersatz[i])/(a_t1_ersatz + b_t1_ersatz));
		f_t1_ersatz[i] = a_t1_ersatz/b_t1_ersatz;
		df_t1_ersatz[i] = deff_t1_ersatz[i]/((1. - eff_t1_ersatz[i])*(1. - eff_t1_ersatz[i]));
		deltaf_t1_ersatz[i] = a_t1_ersatz/b_t1_ersatz - a_t1_true/b_t1_true;
		effprime_t1_ersatz[i] = d_t1_ersatz/(d_t1_ersatz + c_t1_ersatz);
		deffprime_t1_ersatz[i] = sqrt(effprime_t1_ersatz[i]*(1. - effprime_t1_ersatz[i])/(d_t1_ersatz + c_t1_ersatz));
		fprime_t1_ersatz[i] = d_t1_ersatz/c_t1_ersatz;
		dfprime_t1_ersatz[i] = deffprime_t1_ersatz[i]/((1. - effprime_t1_ersatz[i])*(1. - effprime_t1_ersatz[i]));
		deltafprime_t1_ersatz[i] = d_t1_ersatz/c_t1_ersatz - d_t1_true/c_t1_true;
*/
		eff_pf_ersatz[i] = a_pf_ersatz/(a_pf_ersatz + b_pf_ersatz);
		deff_pf_ersatz[i] = sqrt(eff_pf_ersatz[i]*(1. - eff_pf_ersatz[i])/(a_pf_ersatz + b_pf_ersatz));
		f_pf_ersatz[i] = a_pf_ersatz/b_pf_ersatz;
		df_pf_ersatz[i] = deff_pf_ersatz[i]/((1. - eff_pf_ersatz[i])*(1. - eff_pf_ersatz[i]));
		deltaf_pf_ersatz[i] = a_pf_ersatz/b_pf_ersatz - a_pf_true/b_pf_true;
		effprime_pf_ersatz[i] = d_pf_ersatz/(d_pf_ersatz + c_pf_ersatz);
		deffprime_pf_ersatz[i] = sqrt(effprime_pf_ersatz[i]*(1. - effprime_pf_ersatz[i])/(d_pf_ersatz + c_pf_ersatz));
		fprime_pf_ersatz[i] = d_pf_ersatz/c_pf_ersatz;
		dfprime_pf_ersatz[i] = deffprime_pf_ersatz[i]/((1. - effprime_pf_ersatz[i])*(1. - effprime_pf_ersatz[i]));
		deltafprime_pf_ersatz[i] = d_pf_ersatz/c_pf_ersatz - d_pf_true/c_pf_true;

		cout << "CaloMEt " << cMET << endl;
		cout << "MET cut\tftrue\tf\tdf\teff\tdeff\ta\tb\tnpass\tf'true\tf'\tdf'\teff'\tdeff'\td\tc\tnfail" << endl;
		cout << cMET << "\t" 
		<< a_calo_true/b_calo_true << "\t" 
		<< f_calo_ersatz[i] << "\t" << df_calo_ersatz[i] << "\t"
		<< eff_calo_ersatz[i] << "\t" << deff_calo_ersatz[i] << "\t" 
		<< a_calo_ersatz << "\t" << b_calo_ersatz << "\t" << a_calo_ersatz + b_calo_ersatz << "\t"
		<< d_calo_true/c_calo_true << "\t"
		<< fprime_calo_ersatz[i] << "\t" << dfprime_calo_ersatz[i] << "\t"
		<< effprime_calo_ersatz[i] << "\t" << deffprime_calo_ersatz[i] << "\t" 
		<< d_calo_ersatz << "\t" << c_calo_ersatz << "\t" << d_calo_ersatz + c_calo_ersatz << endl; 
		cout << cMET << "\t" 
		<< a_tc_true/b_tc_true << "\t" 
		<< f_tc_ersatz[i] << "\t" << df_tc_ersatz[i] << "\t"
		<< eff_tc_ersatz[i] << "\t" << deff_tc_ersatz[i] << "\t" 
		<< a_tc_ersatz << "\t" << b_tc_ersatz << "\t" << a_tc_ersatz + b_tc_ersatz << "\t"
		<< d_tc_true/c_tc_true << "\t"
		<< fprime_tc_ersatz[i] << "\t" << dfprime_tc_ersatz[i] << "\t"
		<< effprime_tc_ersatz[i] << "\t" << deffprime_tc_ersatz[i] << "\t" 
		<< d_tc_ersatz << "\t" << c_tc_ersatz << "\t" << d_tc_ersatz + c_tc_ersatz << endl; 
/*		cout << cMET << "\t" 
		<< a_t1_true/b_t1_true << "\t" 
		<< f_t1_ersatz[i] << "\t" << df_t1_ersatz[i] << "\t"
		<< eff_t1_ersatz[i] << "\t" << deff_t1_ersatz[i] << "\t" 
		<< a_t1_ersatz << "\t" << b_t1_ersatz << "\t" << a_t1_ersatz + b_t1_ersatz << "\t"
		<< d_t1_true/c_t1_true << "\t"
		<< fprime_t1_ersatz[i] << "\t" << dfprime_t1_ersatz[i] << "\t"
		<< effprime_t1_ersatz[i] << "\t" << deffprime_t1_ersatz[i] << "\t" 
		<< d_t1_ersatz << "\t" << c_t1_ersatz << "\t" << d_t1_ersatz + c_t1_ersatz << endl; */
		cout << cMET << "\t" 
		<< a_pf_true/b_pf_true << "\t" 
		<< f_pf_ersatz[i] << "\t" << df_pf_ersatz[i] << "\t"
		<< eff_pf_ersatz[i] << "\t" << deff_pf_ersatz[i] << "\t" 
		<< a_pf_ersatz << "\t" << b_pf_ersatz << "\t" << a_pf_ersatz + b_pf_ersatz << "\t"
		<< d_pf_true/c_pf_true << "\t"
		<< fprime_pf_ersatz[i] << "\t" << dfprime_pf_ersatz[i] << "\t"
		<< effprime_pf_ersatz[i] << "\t" << deffprime_pf_ersatz[i] << "\t" 
		<< d_pf_ersatz << "\t" << c_pf_ersatz << "\t" << d_pf_ersatz + c_pf_ersatz << endl; 

		CaloMET << cMET << "\t" 
		<< a_calo_true/b_calo_true << "\t" 
		<< f_calo_ersatz[i] << "\t" << df_calo_ersatz[i] << "\t"
		<< eff_calo_ersatz[i] << "\t" << deff_calo_ersatz[i] << "\t" 
		<< a_calo_ersatz << "\t" << b_calo_ersatz << "\t" << a_calo_ersatz + b_calo_ersatz << "\t"
		<< d_calo_true/c_calo_true << "\t"
		<< fprime_calo_ersatz[i] << "\t" << dfprime_calo_ersatz[i] << "\t"
		<< effprime_calo_ersatz[i] << "\t" << deffprime_calo_ersatz[i] << "\t" 
		<< d_calo_ersatz << "\t" << c_calo_ersatz << "\t" << d_calo_ersatz + c_calo_ersatz << endl; 
		TcMET << cMET << "\t" 
		<< a_tc_true/b_tc_true << "\t" 
		<< f_tc_ersatz[i] << "\t" << df_tc_ersatz[i] << "\t"
		<< eff_tc_ersatz[i] << "\t" << deff_tc_ersatz[i] << "\t" 
		<< a_tc_ersatz << "\t" << b_tc_ersatz << "\t" << a_tc_ersatz + b_tc_ersatz << "\t"
		<< d_tc_true/c_tc_true << "\t"
		<< fprime_tc_ersatz[i] << "\t" << dfprime_tc_ersatz[i] << "\t"
		<< effprime_tc_ersatz[i] << "\t" << deffprime_tc_ersatz[i] << "\t" 
		<< d_tc_ersatz << "\t" << c_tc_ersatz << "\t" << d_tc_ersatz + c_tc_ersatz << endl; 
/*		T1MET << cMET << "\t" 
		<< a_t1_true/b_t1_true << "\t" 
		<< f_t1_ersatz[i] << "\t" << df_t1_ersatz[i] << "\t"
		<< eff_t1_ersatz[i] << "\t" << deff_t1_ersatz[i] << "\t" 
		<< a_t1_ersatz << "\t" << b_t1_ersatz << "\t" << a_t1_ersatz + b_t1_ersatz << "\t"
		<< d_t1_true/c_t1_true << "\t"
		<< fprime_t1_ersatz[i] << "\t" << dfprime_t1_ersatz[i] << "\t"
		<< effprime_t1_ersatz[i] << "\t" << deffprime_t1_ersatz[i] << "\t" 
		<< d_t1_ersatz << "\t" << c_t1_ersatz << "\t" << d_t1_ersatz + c_t1_ersatz << endl;*/ 
		PfMET << cMET << "\t" 
		<< a_pf_true/b_pf_true << "\t" 
		<< f_pf_ersatz[i] << "\t" << df_pf_ersatz[i] << "\t"
		<< eff_pf_ersatz[i] << "\t" << deff_pf_ersatz[i] << "\t" 
		<< a_pf_ersatz << "\t" << b_pf_ersatz << "\t" << a_pf_ersatz + b_pf_ersatz << "\t"
		<< d_pf_true/c_pf_true << "\t"
		<< fprime_pf_ersatz[i] << "\t" << dfprime_pf_ersatz[i] << "\t"
		<< effprime_pf_ersatz[i] << "\t" << deffprime_pf_ersatz[i] << "\t" 
		<< d_pf_ersatz << "\t" << c_pf_ersatz << "\t" << d_pf_ersatz + c_pf_ersatz << endl; 

		i++;
	}		

	TCanvas* c_calo_f = new TCanvas("calo_f", "", 800, 600);
	TGraphErrors* g_calo_f = new TGraphErrors(metcut, f_calo_ersatz, metcuterror, df_calo_ersatz);
	g_calo_f->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_calo_f->GetYaxis()->SetTitle("f_{ersatz}");
	g_calo_f->Draw("ap");
	c_calo_f->SaveAs("calo_f.png");
	delete g_calo_f;
	delete c_calo_f;

	TCanvas* c_calo_eff = new TCanvas("calo_eff", "", 800, 600);
	TGraphErrors* g_calo_eff = new TGraphErrors(metcut, eff_calo_ersatz, metcuterror, deff_calo_ersatz);
	g_calo_eff->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_calo_eff->GetYaxis()->SetTitle("#epsilon_{ersatz}");
	g_calo_eff->Draw("ap");
	c_calo_eff->SaveAs("calo_eff.png");
	delete g_calo_eff;
	delete c_calo_eff;

	TCanvas* c_calo_deltaf = new TCanvas("calo_deltaf", "", 800, 600);
	TGraphErrors* g_calo_deltaf = new TGraphErrors(metcut, deltaf_calo_ersatz, metcuterror, df_calo_ersatz);
	g_calo_deltaf->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_calo_deltaf->GetYaxis()->SetTitle("f_{ersatz} - f_{W#rightarrowe#nu}");
	g_calo_deltaf->Draw("ap");
	c_calo_deltaf->SaveAs("calo_deltaf.png");
	delete g_calo_deltaf;
	delete c_calo_deltaf;

	TCanvas* c_calo_fprime = new TCanvas("calo_fprime", "", 800, 600);
	TGraphErrors* g_calo_fprime = new TGraphErrors(metcut, fprime_calo_ersatz, metcuterror, dfprime_calo_ersatz);
	g_calo_fprime->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_calo_fprime->GetYaxis()->SetTitle("f'_{ersatz}");
	g_calo_fprime->Draw("ap");
	c_calo_fprime->SaveAs("calo_fprime.png");
	delete g_calo_fprime;
	delete c_calo_fprime;

	TCanvas* c_calo_effprime = new TCanvas("calo_effprime", "", 800, 600);
	TGraphErrors* g_calo_effprime = new TGraphErrors(metcut, effprime_calo_ersatz, metcuterror, deffprime_calo_ersatz);
	g_calo_effprime->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_calo_effprime->GetYaxis()->SetTitle("#epsilon'_{ersatz}");
	g_calo_effprime->Draw("ap");
	c_calo_effprime->SaveAs("calo_effprime.png");
	delete g_calo_effprime;
	delete c_calo_effprime;

	TCanvas* c_calo_deltafprime = new TCanvas("calo_deltafprime", "", 800, 600);
	TGraphErrors* g_calo_deltafprime = new TGraphErrors(metcut, deltafprime_calo_ersatz, metcuterror, dfprime_calo_ersatz);
	g_calo_deltafprime->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_calo_deltafprime->GetYaxis()->SetTitle("f'_{ersatz} - f'_{W#rightarrowe#nu}");
	g_calo_deltafprime->Draw("ap");
	c_calo_deltafprime->SaveAs("calo_deltafprime.png");
	delete g_calo_deltafprime;
	delete c_calo_deltafprime;

	TCanvas* c_tc_f = new TCanvas("tc_f", "", 800, 600);
	TGraphErrors* g_tc_f = new TGraphErrors(metcut, f_tc_ersatz, metcuterror, df_tc_ersatz);
	g_tc_f->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_tc_f->GetYaxis()->SetTitle("f_{ersatz}");
	g_tc_f->Draw("ap");
	c_tc_f->SaveAs("tc_f.png");
	delete g_tc_f;
	delete c_tc_f;

	TCanvas* c_tc_eff = new TCanvas("tc_eff", "", 800, 600);
	TGraphErrors* g_tc_eff = new TGraphErrors(metcut, eff_tc_ersatz, metcuterror, deff_tc_ersatz);
	g_tc_eff->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_tc_eff->GetYaxis()->SetTitle("#epsilon_{ersatz}");
	g_tc_eff->Draw("ap");
	c_tc_eff->SaveAs("tc_eff.png");
	delete g_tc_eff;
	delete c_tc_eff;

	TCanvas* c_tc_deltaf = new TCanvas("tc_deltaf", "", 800, 600);
	TGraphErrors* g_tc_deltaf = new TGraphErrors(metcut, deltaf_tc_ersatz, metcuterror, df_tc_ersatz);
	g_tc_deltaf->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_tc_deltaf->GetYaxis()->SetTitle("f_{ersatz} - f_{W#rightarrowe#nu}");
	g_tc_deltaf->Draw("ap");
	c_tc_deltaf->SaveAs("tc_deltaf.png");
	delete g_tc_deltaf;
	delete c_tc_deltaf;

	TCanvas* c_tc_fprime = new TCanvas("tc_fprime", "", 800, 600);
	TGraphErrors* g_tc_fprime = new TGraphErrors(metcut, fprime_tc_ersatz, metcuterror, dfprime_tc_ersatz);
	g_tc_fprime->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_tc_fprime->GetYaxis()->SetTitle("f'_{ersatz}");
	g_tc_fprime->Draw("ap");
	c_tc_fprime->SaveAs("tc_fprime.png");
	delete g_tc_fprime;
	delete c_tc_fprime;

	TCanvas* c_tc_effprime = new TCanvas("tc_effprime", "", 800, 600);
	TGraphErrors* g_tc_effprime = new TGraphErrors(metcut, effprime_tc_ersatz, metcuterror, deffprime_tc_ersatz);
	g_tc_effprime->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_tc_effprime->GetYaxis()->SetTitle("#epsilon'_{ersatz}");
	g_tc_effprime->Draw("ap");
	c_tc_effprime->SaveAs("tc_effprime.png");
	delete g_tc_effprime;
	delete c_tc_effprime;

	TCanvas* c_tc_deltafprime = new TCanvas("tc_deltafprime", "", 800, 600);
	TGraphErrors* g_tc_deltafprime = new TGraphErrors(metcut, deltafprime_tc_ersatz, metcuterror, dfprime_tc_ersatz);
	g_tc_deltafprime->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_tc_deltafprime->GetYaxis()->SetTitle("f'_{ersatz} - f'_{W#rightarrowe#nu}");
	g_tc_deltafprime->Draw("ap");
	c_tc_deltafprime->SaveAs("tc_deltafprime.png");
	delete g_tc_deltafprime;
	delete c_tc_deltafprime;
/*
	TCanvas* c_t1_f = new TCanvas("t1_f", "", 800, 600);
	TGraphErrors* g_t1_f = new TGraphErrors(metcut, f_t1_ersatz, metcuterror, df_t1_ersatz);
	g_t1_f->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_t1_f->GetYaxis()->SetTitle("f_{ersatz}");
	g_t1_f->Draw("ap");
	c_t1_f->SaveAs("t1_f.png");
	delete g_t1_f;
	delete c_t1_f;

	TCanvas* c_t1_eff = new TCanvas("t1_eff", "", 800, 600);
	TGraphErrors* g_t1_eff = new TGraphErrors(metcut, eff_t1_ersatz, metcuterror, deff_t1_ersatz);
	g_t1_eff->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_t1_eff->GetYaxis()->SetTitle("#epsilon_{ersatz}");
	g_t1_eff->Draw("ap");
	c_t1_eff->SaveAs("t1_eff.png");
	delete g_t1_eff;
	delete c_t1_eff;

	TCanvas* c_t1_deltaf = new TCanvas("t1_deltaf", "", 800, 600);
	TGraphErrors* g_t1_deltaf = new TGraphErrors(metcut, deltaf_t1_ersatz, metcuterror, df_t1_ersatz);
	g_t1_deltaf->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_t1_deltaf->GetYaxis()->SetTitle("f_{ersatz} - f_{W#rightarrowe#nu}");
	g_t1_deltaf->Draw("ap");
	c_t1_deltaf->SaveAs("t1_deltaf.png");
	delete g_t1_deltaf;
	delete c_t1_deltaf;

	TCanvas* c_t1_fprime = new TCanvas("t1_fprime", "", 800, 600);
	TGraphErrors* g_t1_fprime = new TGraphErrors(metcut, fprime_t1_ersatz, metcuterror, dfprime_t1_ersatz);
	g_t1_fprime->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_t1_fprime->GetYaxis()->SetTitle("f'_{ersatz}");
	g_t1_fprime->Draw("ap");
	c_t1_fprime->SaveAs("t1_fprime.png");
	delete g_t1_fprime;
	delete c_t1_fprime;

	TCanvas* c_t1_effprime = new TCanvas("t1_effprime", "", 800, 600);
	TGraphErrors* g_t1_effprime = new TGraphErrors(metcut, effprime_t1_ersatz, metcuterror, deffprime_t1_ersatz);
	g_t1_effprime->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_t1_effprime->GetYaxis()->SetTitle("#epsilon'_{ersatz}");
	g_t1_effprime->Draw("ap");
	c_t1_effprime->SaveAs("t1_effprime.png");
	delete g_t1_effprime;
	delete c_t1_effprime;

	TCanvas* c_t1_deltafprime = new TCanvas("t1_deltafprime", "", 800, 600);
	TGraphErrors* g_t1_deltafprime = new TGraphErrors(metcut, deltafprime_t1_ersatz, metcuterror, dfprime_t1_ersatz);
	g_t1_deltafprime->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_t1_deltafprime->GetYaxis()->SetTitle("f'_{ersatz} - f'_{W#rightarrowe#nu}");
	g_t1_deltafprime->Draw("ap");
	c_t1_deltafprime->SaveAs("t1_deltafprime.png");
	delete g_t1_deltafprime;
	delete c_t1_deltafprime;
*/
	TCanvas* c_pf_f = new TCanvas("pf_f", "", 800, 600);
	TGraphErrors* g_pf_f = new TGraphErrors(metcut, f_pf_ersatz, metcuterror, df_pf_ersatz);
	g_pf_f->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_pf_f->GetYaxis()->SetTitle("f_{ersatz}");
	g_pf_f->Draw("ap");
	c_pf_f->SaveAs("pf_f.png");
	delete g_pf_f;
	delete c_pf_f;

	TCanvas* c_pf_eff = new TCanvas("pf_eff", "", 800, 600);
	TGraphErrors* g_pf_eff = new TGraphErrors(metcut, eff_pf_ersatz, metcuterror, deff_pf_ersatz);
	g_pf_eff->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_pf_eff->GetYaxis()->SetTitle("#epsilon_{ersatz}");
	g_pf_eff->Draw("ap");
	c_pf_eff->SaveAs("pf_eff.png");
	delete g_pf_eff;
	delete c_pf_eff;

	TCanvas* c_pf_deltaf = new TCanvas("pf_deltaf", "", 800, 600);
	TGraphErrors* g_pf_deltaf = new TGraphErrors(metcut, deltaf_pf_ersatz, metcuterror, df_pf_ersatz);
	g_pf_deltaf->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_pf_deltaf->GetYaxis()->SetTitle("f_{ersatz} - f_{W#rightarrowe#nu}");
	g_pf_deltaf->Draw("ap");
	c_pf_deltaf->SaveAs("pf_deltaf.png");
	delete g_pf_deltaf;
	delete c_pf_deltaf;

	TCanvas* c_pf_fprime = new TCanvas("pf_fprime", "", 800, 600);
	TGraphErrors* g_pf_fprime = new TGraphErrors(metcut, fprime_pf_ersatz, metcuterror, dfprime_pf_ersatz);
	g_pf_fprime->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_pf_fprime->GetYaxis()->SetTitle("f'_{ersatz}");
	g_pf_fprime->Draw("ap");
	c_pf_fprime->SaveAs("pf_fprime.png");
	delete g_pf_fprime;
	delete c_pf_fprime;

	TCanvas* c_pf_effprime = new TCanvas("pf_effprime", "", 800, 600);
	TGraphErrors* g_pf_effprime = new TGraphErrors(metcut, effprime_pf_ersatz, metcuterror, deffprime_pf_ersatz);
	g_pf_effprime->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_pf_effprime->GetYaxis()->SetTitle("#epsilon'_{ersatz}");
	g_pf_effprime->Draw("ap");
	c_pf_effprime->SaveAs("pf_effprime.png");
	delete g_pf_effprime;
	delete c_pf_effprime;

	TCanvas* c_pf_deltafprime = new TCanvas("pf_deltafprime", "", 800, 600);
	TGraphErrors* g_pf_deltafprime = new TGraphErrors(metcut, deltafprime_pf_ersatz, metcuterror, dfprime_pf_ersatz);
	g_pf_deltafprime->GetXaxis()->SetTitle("#slash{E}_{T} cut");
	g_pf_deltafprime->GetYaxis()->SetTitle("f'_{ersatz} - f'_{W#rightarrowe#nu}");
	g_pf_deltafprime->Draw("ap");
	c_pf_deltafprime->SaveAs("pf_deltafprime.png");
	delete g_pf_deltafprime;
	delete c_pf_deltafprime;

	file->Close(); 
	CaloMET.close();
	TcMET.close();
	//T1MET.close();
	PfMET.close();
}
