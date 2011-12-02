#include <TROOT.h>
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TString.h>
#include <TF1.h>
#include "TProfile2D.h"
#include <TH1.h>
#include <TProfile.h>
#include <TLegend.h>
#include <TObjArray.h>
#include <TText.h>
#include <TVectorT.h>
#include <TFile.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TVirtualFitter.h>
#include <TFitterMinuit.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include <stdio.h>
#include <iostream>


static bool sortPairs(pair<int, double>  i , pair<int, double>  j){
 return( i.second > j.second   );
}

using namespace std;
typedef   unsigned int size;

// Macro for fitting rates of single paths and of the whole DS
// The input files are the output root files created by OHltRateEff
// In particular the 2 histos 
// totalPerLS
// individualPerLS
// are used.

void FitRatesPerLS (TString DS="AlphaT", double L=5000, size rebin=30, double lowLumiCutoff=100) {
//L=input target lumi. It will select the input files and the lumi to use for rate extrapolation

// For the Global title:
  gStyle->SetOptTitle(1);
  gStyle->SetTitleFont(42);
  gStyle->SetTitleFontSize(0.08);
// For the axis labels:
	gStyle->SetLabelFont(42, "XYZ");
  gStyle->SetLabelSize(0.06, "XYZ");

  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(0.5);

// fitter magic from roberto

 TVirtualFitter::SetDefaultFitter("Minuit2");
 TVirtualFitter::Fitter(0,50);
 TFitterMinuit* fitter =
 dynamic_cast<TFitterMinuit*>(TVirtualFitter::GetFitter());
 fitter->SetStrategy(2);
 gRandom = new TRandom3();

 	double lumiByLS(int runNumber, int LS);

	// 
	TFile* f[10];
	TH1F* hTotalPerLS     [10];
	TH2F* hIndividualPerLS[10];
	TH2F* hIndividualCountsPerLS[10];
	TH2F* hTotalPSPerLS   [10];
	double vlumiSFperFile [10]; 
	size fileCounter=0;


  if (DS=="AlphaT") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_AlphaT_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_AlphaT_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="DoubleMu") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_DoubleMu_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_DoubleMu_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="EleHadEG12") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_EleHadEG12_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_EleHadEG12_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="HTMHT") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_HTMHT_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_HTMHT_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="MET") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_MET_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_MET_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="MuHad") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_MuHad_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_MuHad_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="MultiJet") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_MultiJet_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_MultiJet_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="PhotonDoubleEle") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_PhotonDoubleEle_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_PhotonDoubleEle_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="PhotonDoublePhoton") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_PhotonDoublePhoton_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_PhotonDoublePhoton_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="PhotonHad") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_PhotonHad_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_PhotonHad_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="PhotonPhoton") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_PhotonPhoton_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_PhotonPhoton_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="RMR") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_RMR_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_RMR_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="SingleMu") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_SingleMu_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_SingleMu_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="Tau") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_Tau_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_Tau_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };
  if (DS=="Jet") {
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_HighPU_r179828_Jet_forHighPU.root"); vlumiSFperFile[fileCounter++]=131.8;
    f[fileCounter] = new TFile("/1TB/hartl/ratesVsLumiHighPileup/root/hltmenu_3e33_r178479_Jet_forHighPU.root"); vlumiSFperFile[fileCounter++]=1;
  };

	gStyle->SetPalette(1);


  // get vector of trigger names
  // for each file, get histograms for rate,counts, prescale
	size nPaths=0;
	vector <TString> vTriggerNames;
	for (size iFile=0; iFile<fileCounter; ++iFile) {
		if (f[iFile] == NULL) {
			printf("Error opening file [%d]. Exiting...\n" ,iFile);
			return;
		}
	// Open DS VS LS rate
		hTotalPerLS     [iFile] = (TH1F*) f[iFile]->Get("totalPerLS");	
	// Open paths VS LS rate
		hIndividualPerLS[iFile]       = (TH2F*) f[iFile]->Get("individualPerLS");
		hIndividualCountsPerLS[iFile] = (TH2F*) f[iFile]->Get("individualCountsPerLS");
		hTotalPSPerLS   [iFile]       = (TH2F*) f[iFile]->Get("totalprescalePerLS");

		if (iFile==0) nPaths = hIndividualPerLS     [iFile]->GetNbinsX();
		if (nPaths != (size)hIndividualPerLS     [iFile]->GetNbinsX()) {
			printf("Number of paths do not match for file [%d]. Exiting...\n" ,iFile);
			return;
		}
    // TODO: CHECK: shouldn't this be applied only once (i.e. for one file)
		for (size iPath=0; iPath<nPaths; ++iPath) {
			vTriggerNames.push_back(TString(hIndividualPerLS[0]->GetXaxis()->GetBinLabel(iPath+1)));
		}
	}

	for (size iPath=0; iPath<nPaths; ++iPath) printf("%02d\t%s\n",iPath+1,vTriggerNames.at(iPath).Data());
	cout << "# of Paths= " << nPaths << endl;


	vector <pair<int, double> > vLumiLSindex;// contains the sorted index Lumi per LS for the whole data. 
	vector <double> vLumi;       // contains the inst Lumi per LS for the whole data. same size of vLS
	vector <double> vLumiRebin;  // contains the inst Lumi per LS for the whole data. same size of vLS

	vector <vector <double> > vvRates;      // contains vectors of path rates per LS, one vector per path.
	vector <vector <double> > vvCounts;     // contains vectors of path rates per LS, one vector per path.
	vector <vector <double> > vvRatesRebin; // contains vectors of path rates per LS, one vector per path.
	vector <vector <double> > vvTotalPrescales;    // contains vectors of L1*HLT PS per LS, one vector per path.

	double lumiByLS_(int runNumber, int LS);
	int runNumber=0, LS=0;
	
	for (size iFile=0; iFile<fileCounter; ++iFile) {//looping over files
		for (int iLS=0; iLS<hIndividualPerLS     [iFile]->GetNbinsY(); ++iLS) {//looping over file LS
		TString tS =        hIndividualPerLS     [iFile]->GetYaxis()->GetBinLabel(iLS+1);

		TObjArray* tokens = tS.Tokenize("-");
		int nTokens = tokens->GetEntries();
		if (nTokens==2) {
			runNumber = (((TObjString*) tokens->First())->GetString()).Atoi();
			LS        = (((TObjString*) tokens->Last()) ->GetString()).Atoi();
		}
		else {
			cout << "Label pattern in bin " << iLS << " not recognized: " << tS << endl;
			cout << "Exiting..." << endl;
			return;
		}
		double Lumi = lumiByLS_( runNumber,  LS)*vlumiSFperFile[iFile];
		vLumi.push_back(Lumi);
		
		}// end looping on file LS
	}//end looping over files


// Fill vectors of PS and Rates
	for (size iPath=0; iPath<nPaths; ++iPath) {//looping over paths
		vector <double> vRates;
		vector <double> vCounts;
		vector <double> vTotalPrescales;
		for (size iFile=0; iFile<fileCounter; ++iFile) {//looping over files

			for (int iLS=0; iLS<hIndividualPerLS     [iFile]->GetNbinsY(); ++iLS) {//looping over file LS
				vRates.push_back         (hIndividualPerLS      [iFile]->GetBinContent(iPath+1,iLS+1));
				vCounts.push_back        (hIndividualCountsPerLS[iFile]->GetBinContent(iPath+1,iLS+1));
				vTotalPrescales.push_back(hTotalPSPerLS         [iFile]->GetBinContent(iPath+1,iLS+1));
			}// end looping on file LS
		}//end looping over files
		vvRates         .push_back(vRates         );
		vvCounts        .push_back(vCounts        );
		vvTotalPrescales.push_back(vTotalPrescales);
	}//end looping over paths

	vector <TGraphErrors*> vGraph;

	for (size iLS=0; iLS<vLumi.size(); ++iLS) {
		vLumiLSindex.push_back(make_pair(iLS,vLumi.at(iLS)));
	}
	sort (vLumiLSindex.begin(), vLumiLSindex.end(), sortPairs); //sorting the vector of lumi. keep trak of the index.

	for (size iPath=0; iPath<nPaths; ++iPath) {//looping over paths

		vector <double> vRatesUnprescaled; //temp vector containing unprescaled rates 
		vector <double> vCounts;           //temp vector containing counts for each bin of the final TGraph (rebinning...)

		vector <double> vRateUncertainty;  //temp vector containing the uncertainties of the unprescaled rates
		vector <double> vXAxisUncertainty;  //dummy

		vector <double> vLumiSorted;       //temp vector containing decreasing inst lumis 
		for (size iLS=0; iLS<vLumiLSindex.size();) {//looping over the entire set of data

 			double lumi = 0;
 			double rate  = 0;
      double count = 0;
 			size iRebin = 0;
 			for (iRebin=0; iRebin<rebin && iLS<vLumiLSindex.size(); ++iRebin) {
				int idx= vLumiLSindex.at(iLS).first;
 				lumi += vLumiLSindex.at(iLS).second;
        
// 				rate  += vvRates.at(iPath).at(idx)*vvTotalPrescales.at(iPath).at(idx); // TODO : fix me
 				rate  += vvRates.at(iPath).at(idx)*1.0; // since for emulations and strange RMR stuff it doesn't work
 				count += vvCounts.at(iPath).at(idx);
 				++iLS;
 			}
			//std::cout << "iLS = " << iLS << ": vLumiLSindex.at(iLS).second = " << vLumiLSindex.at(iLS).second << endl;
 			lumi /=iRebin;
 			rate  /=iRebin;
			if (vLumiLSindex.at(iLS).second<lowLumiCutoff) break;

      // calculation of the uncertainty of the final unprescaled rate:
      // A) r = g*c  ==>  g = r/c
      // B) u(r) = g*u(c) = (r/c)*u(c) = r/sqrt(c)
			double rateUncertainty = rate / sqrt(count);
			vRatesUnprescaled.push_back(rate);
			vCounts          .push_back(count);
      vXAxisUncertainty.push_back(0.0);
      vRateUncertainty.push_back(rateUncertainty);
			vLumiSorted      .push_back(lumi);
		}//end looping over the entire set of data

		TGraphErrors* g = new TGraphErrors(vLumiSorted.size(), &vLumiSorted[0], &vRatesUnprescaled[0], 0, &vRateUncertainty[0]); //&vXAxisUncertainty[0]
		g->SetTitle(vTriggerNames.at(iPath));
		vGraph.push_back(g);
	}//end looping over paths

	double FitPars[4];

	double p1      (double* x, double* p);
	double p2      (double* x, double* p);
	double p3      (double* x, double* p);
	double LinExpo(double* x, double* p);
	double ExpoExpo(double* x, double* p);

	// Fitting w/ quadratic and cubic
	// User should check the Chi2/Ndof
	TF1* fp1     = new TF1("fp1"   ,p1,0,10000,2);
	TF1* fp2     = new TF1("fp2"   ,p2,0,10000,3);
	TF1* fp3     = new TF1("fp3"   ,p3,0,10000,4);
	TF1* fLinExp = new TF1("fLinExp",LinExpo,0,10000,4);
	TF1* fExpExp = new TF1("fExpExp",ExpoExpo,0,10000,3);

	// Finally. Plot!

 	size nx = TMath::Floor(sqrt(nPaths)); //Choose the proper canvas division
 	size ny = nPaths/nx;
  if (nx*ny < nPaths) ny++;

 	TCanvas* c = new TCanvas("c","c",0,0,1200,1000);
 	c->Divide(nx,ny);

//	if (ix*iy==nPaths);
// 	else if (sqrt(nPaths)*(iy+1)>nPaths) ++iy;
//	else {++ix; ++iy;}// end of canvas division

 	for (size iPath=0; iPath<nPaths; ++iPath)
  {//looping over paths
 		c->cd(iPath+1);
 		vGraph.at(iPath)->SetMarkerSize(1.0);
 		vGraph.at(iPath)->SetMarkerColor(kRed);
 		vGraph.at(iPath)->Draw("ap");
 		vGraph.at(iPath)->Fit("fp1","Q","",1000, 3000); // Fitting only the region 1E33 - 3E33 for the linear trend
 		fp1->GetParameters(FitPars);
// 		cout << "DS quadratic rate extrapolation. Lumi = "<< L << "E30 " 
// 				 << FitPars[0]+FitPars[1]*L+FitPars[2]*L*L                  << endl;
 		fLinExp->SetLineColor(4);
 		fLinExp->FixParameter(0,FitPars[0]);
 		fLinExp->FixParameter(1,FitPars[1]);
 		vGraph.at(iPath)->Fit("fLinExp","R");
 		fLinExp->GetParameters(FitPars);
 		fp1->DrawCopy("same");
		

//  		printf("Linear + Exponential extrapolation. Path: % 40s\tLumi = %2.2lfE30\tRate = %5.1lf\n", vGraph.at(iPath)->GetTitle(), L,fLinExp->Eval(L));
// 		vGraph.at(iPath)->Fit("fExpExp","R");
// 		fExpExp->GetParameters(FitPars);
// 		printf("Cubic rate extrapolation. Path: %s\tLumi = %2.2lfE30\tRate = %5.1lf\n", vGraph.at(iPath)->GetTitle(), L,fExpExp->Eval(L));
 	}
	char text[100];
	sprintf(text,"/1TB/hartl/ratesVsLumiHighPileup/plots/%s.png",DS.Data());
	c->SaveAs(text);

	return;
}

double p1(double* x, double* p) {
	return p[0]+p[1]*x[0];
}
double p2(double* x, double* p) {
	return p[0]+p[1]*x[0]+p[2]*x[0]*x[0];
}
double p3(double* x, double* p) {
	return p[0]+p[1]*x[0]+p[2]*x[0]*x[0]+p[3]*x[0]*x[0]*x[0];
}
double LinExpo(double* x, double* p) {
	return p[0]+p[1]*x[0]+p[2]*TMath::Exp(p[3]*x[0]);
}
double ExpoExpo(double* x, double* p) {
	return p[0]+p[1]*TMath::Exp(p[2]*TMath::Power(x[0],1));
}

double lumiByLS_(int runNumber, int LS) {

	typedef std::map < pair<int, int>, double > mapLS;
	mapLS lumiByLS;


	lumiByLS[make_pair(178479,56)] = 3234.86;
	lumiByLS[make_pair(178479,57)] = 3233.04;
	lumiByLS[make_pair(178479,74)] = 3199.64;
	lumiByLS[make_pair(178479,79)] = 3191.47;
	lumiByLS[make_pair(178479,87)] = 3175.06;
	lumiByLS[make_pair(178479,89)] = 3170.98;
	lumiByLS[make_pair(178479,94)] = 3160.15;
	lumiByLS[make_pair(178479,95)] = 3160.05;
	lumiByLS[make_pair(178479,97)] = 3153.55;
	lumiByLS[make_pair(178479,98)] = 3149.83;
	lumiByLS[make_pair(178479,99)] = 3147.23;
	lumiByLS[make_pair(178479,101)] = 3141.01;
	lumiByLS[make_pair(178479,102)] = 3139.04;
	lumiByLS[make_pair(178479,103)] = 3135.54;
	lumiByLS[make_pair(178479,104)] = 3135.29;
	lumiByLS[make_pair(178479,105)] = 3132.07;
	lumiByLS[make_pair(178479,107)] = 3128.97;
	lumiByLS[make_pair(178479,108)] = 3126.13;
	lumiByLS[make_pair(178479,109)] = 3125.86;
	lumiByLS[make_pair(178479,110)] = 3122.69;
	lumiByLS[make_pair(178479,111)] = 3120.29;
	lumiByLS[make_pair(178479,112)] = 3118.46;
	lumiByLS[make_pair(178479,113)] = 3116.1;
	lumiByLS[make_pair(178479,114)] = 3114.31;
	lumiByLS[make_pair(178479,115)] = 3111.08;
	lumiByLS[make_pair(178479,116)] = 3107.7;
	lumiByLS[make_pair(178479,117)] = 3104.81;
	lumiByLS[make_pair(178479,118)] = 3098;
	lumiByLS[make_pair(178479,119)] = 3095.66;
	lumiByLS[make_pair(178479,120)] = 3093.27;
	lumiByLS[make_pair(178479,121)] = 3090.91;
	lumiByLS[make_pair(178479,122)] = 3088.31;
	lumiByLS[make_pair(178479,123)] = 3085.55;
	lumiByLS[make_pair(178479,124)] = 3083.02;
	lumiByLS[make_pair(178479,125)] = 3080.43;
	lumiByLS[make_pair(178479,126)] = 3077.4;
	lumiByLS[make_pair(178479,127)] = 3075.35;
	lumiByLS[make_pair(178479,128)] = 3073.51;
	lumiByLS[make_pair(178479,129)] = 3070.8;
	lumiByLS[make_pair(178479,130)] = 3068.28;
	lumiByLS[make_pair(178479,131)] = 3065.17;
	lumiByLS[make_pair(178479,132)] = 3061.84;
	lumiByLS[make_pair(178479,134)] = 3050.18;
	lumiByLS[make_pair(178479,135)] = 3036.12;
	lumiByLS[make_pair(178479,136)] = 3018.12;
	lumiByLS[make_pair(178479,137)] = 3092.18;
	lumiByLS[make_pair(178479,139)] = 3101.55;
	lumiByLS[make_pair(178479,140)] = 3099.36;
	lumiByLS[make_pair(178479,141)] = 3097.66;
	lumiByLS[make_pair(178479,143)] = 3093.91;
	lumiByLS[make_pair(178479,144)] = 3091.48;
	lumiByLS[make_pair(178479,145)] = 3089.42;
	lumiByLS[make_pair(178479,146)] = 3088.02;
	lumiByLS[make_pair(178479,147)] = 3085.94;
	lumiByLS[make_pair(178479,148)] = 3084.4;
	lumiByLS[make_pair(178479,149)] = 3082.64;
	lumiByLS[make_pair(178479,150)] = 3080.89;
	lumiByLS[make_pair(178479,151)] = 3079.58;
	lumiByLS[make_pair(178479,152)] = 3076.78;
	lumiByLS[make_pair(178479,153)] = 3075.08;
	lumiByLS[make_pair(178479,154)] = 3073.4;
	lumiByLS[make_pair(178479,155)] = 3072.36;
	lumiByLS[make_pair(178479,156)] = 3071.55;
	lumiByLS[make_pair(178479,157)] = 3069.89;
	lumiByLS[make_pair(178479,158)] = 3068.68;
	lumiByLS[make_pair(178479,159)] = 3066.84;
	lumiByLS[make_pair(178479,160)] = 3065.36;
	lumiByLS[make_pair(178479,161)] = 3063.78;
	lumiByLS[make_pair(178479,162)] = 3062.24;
	lumiByLS[make_pair(178479,163)] = 3060.83;
	lumiByLS[make_pair(178479,164)] = 3059.28;
	lumiByLS[make_pair(178479,165)] = 3058.04;
	lumiByLS[make_pair(178479,166)] = 3056.21;
	lumiByLS[make_pair(178479,167)] = 3054.98;
	lumiByLS[make_pair(178479,168)] = 3053.43;
	lumiByLS[make_pair(178479,169)] = 3051.85;
	lumiByLS[make_pair(178479,170)] = 3050.31;
	lumiByLS[make_pair(178479,171)] = 3048.71;
	lumiByLS[make_pair(178479,172)] = 3047.04;
	lumiByLS[make_pair(178479,173)] = 3045.48;
	lumiByLS[make_pair(178479,174)] = 3044.05;
	lumiByLS[make_pair(178479,175)] = 3042.26;
	lumiByLS[make_pair(178479,176)] = 3040.83;
	lumiByLS[make_pair(178479,177)] = 3038.72;
	lumiByLS[make_pair(178479,178)] = 3036.85;
	lumiByLS[make_pair(178479,179)] = 3034.94;
	lumiByLS[make_pair(178479,180)] = 3033.31;
	lumiByLS[make_pair(178479,181)] = 3031.37;
	lumiByLS[make_pair(178479,182)] = 3029.39;
	lumiByLS[make_pair(178479,183)] = 3027.9;
	lumiByLS[make_pair(178479,184)] = 3026.17;
	lumiByLS[make_pair(178479,185)] = 3024.29;
	lumiByLS[make_pair(178479,186)] = 3022.47;
	lumiByLS[make_pair(178479,187)] = 3020.92;
	lumiByLS[make_pair(178479,188)] = 3019.3;
	lumiByLS[make_pair(178479,189)] = 3017.32;
	lumiByLS[make_pair(178479,193)] = 3010.03;
	lumiByLS[make_pair(178479,200)] = 2997.32;
	lumiByLS[make_pair(178479,201)] = 2995.91;
	lumiByLS[make_pair(178479,202)] = 2994.05;
	lumiByLS[make_pair(178479,203)] = 2992.32;
	lumiByLS[make_pair(178479,204)] = 2991.7;
	lumiByLS[make_pair(178479,205)] = 2989.88;
	lumiByLS[make_pair(178479,206)] = 2988.59;
	lumiByLS[make_pair(178479,207)] = 2986.87;
	lumiByLS[make_pair(178479,208)] = 2985.59;
	lumiByLS[make_pair(178479,209)] = 2984.24;
	lumiByLS[make_pair(178479,210)] = 2982.45;
	lumiByLS[make_pair(178479,211)] = 2980.54;
	lumiByLS[make_pair(178479,212)] = 2978.54;
	lumiByLS[make_pair(178479,213)] = 2977.32;
	lumiByLS[make_pair(178479,216)] = 2971.8;
	lumiByLS[make_pair(178479,217)] = 2969.55;
	lumiByLS[make_pair(178479,218)] = 2968.12;
	lumiByLS[make_pair(178479,219)] = 2965.67;
	lumiByLS[make_pair(178479,220)] = 2963.71;
	lumiByLS[make_pair(178479,221)] = 2962.26;
	lumiByLS[make_pair(178479,222)] = 2959.48;
	lumiByLS[make_pair(178479,223)] = 2957.68;
	lumiByLS[make_pair(178479,224)] = 2955.72;
	lumiByLS[make_pair(178479,226)] = 2952.97;
	lumiByLS[make_pair(178479,228)] = 2949.3;
	lumiByLS[make_pair(178479,229)] = 2947.99;
	lumiByLS[make_pair(178479,230)] = 2946.45;
	lumiByLS[make_pair(178479,231)] = 2944.62;
	lumiByLS[make_pair(178479,232)] = 2942.76;
	lumiByLS[make_pair(178479,233)] = 2941.2;
	lumiByLS[make_pair(178479,236)] = 2935.8;
	lumiByLS[make_pair(178479,237)] = 2933.62;
	lumiByLS[make_pair(178479,238)] = 2931.43;
	lumiByLS[make_pair(178479,240)] = 2928.33;
	lumiByLS[make_pair(178479,241)] = 2926.62;
	lumiByLS[make_pair(178479,242)] = 2924.42;
	lumiByLS[make_pair(178479,243)] = 2921.76;
	lumiByLS[make_pair(178479,244)] = 2919.74;
	lumiByLS[make_pair(178479,245)] = 2918.78;
	lumiByLS[make_pair(178479,246)] = 2916.88;
	lumiByLS[make_pair(178479,247)] = 2914.34;
	lumiByLS[make_pair(178479,248)] = 2912.52;
	lumiByLS[make_pair(178479,249)] = 2911.24;
	lumiByLS[make_pair(178479,250)] = 2908.11;
	lumiByLS[make_pair(178479,251)] = 2907.48;
	lumiByLS[make_pair(178479,252)] = 2905.44;
	lumiByLS[make_pair(178479,253)] = 2903.99;
	lumiByLS[make_pair(178479,254)] = 2902.62;
	lumiByLS[make_pair(178479,255)] = 2899.79;
	lumiByLS[make_pair(178479,256)] = 2898.7;
	lumiByLS[make_pair(178479,257)] = 2896.3;
	lumiByLS[make_pair(178479,258)] = 2894.03;
	lumiByLS[make_pair(178479,259)] = 2892.1;
	lumiByLS[make_pair(178479,260)] = 2891.01;
	lumiByLS[make_pair(178479,261)] = 2889.29;
	lumiByLS[make_pair(178479,262)] = 2887.37;
	lumiByLS[make_pair(178479,263)] = 2886.49;
	lumiByLS[make_pair(178479,264)] = 2884.8;
	lumiByLS[make_pair(178479,265)] = 2883.2;
	lumiByLS[make_pair(178479,266)] = 2881.19;
	lumiByLS[make_pair(178479,267)] = 2879.92;
	lumiByLS[make_pair(178479,268)] = 2877.75;
	lumiByLS[make_pair(178479,269)] = 2876.64;
	lumiByLS[make_pair(178479,270)] = 2874.64;
	lumiByLS[make_pair(178479,271)] = 2872.66;
	lumiByLS[make_pair(178479,272)] = 2870.43;
	lumiByLS[make_pair(178479,273)] = 2869.39;
	lumiByLS[make_pair(178479,274)] = 2866.96;
	lumiByLS[make_pair(178479,275)] = 2866.43;
	lumiByLS[make_pair(178479,277)] = 2866.81;
	lumiByLS[make_pair(178479,278)] = 2866.1;
	lumiByLS[make_pair(178479,279)] = 2866.83;
	lumiByLS[make_pair(178479,280)] = 2865.99;
	lumiByLS[make_pair(178479,281)] = 2865.42;
	lumiByLS[make_pair(178479,282)] = 2865.01;
	lumiByLS[make_pair(178479,283)] = 2865.55;
	lumiByLS[make_pair(178479,284)] = 2863.69;
	lumiByLS[make_pair(178479,285)] = 2863.03;
	lumiByLS[make_pair(178479,286)] = 2861.26;
	lumiByLS[make_pair(178479,287)] = 2860.27;
	lumiByLS[make_pair(178479,288)] = 2858.81;
	lumiByLS[make_pair(178479,289)] = 2857.21;
	lumiByLS[make_pair(178479,290)] = 2855.11;
	lumiByLS[make_pair(178479,291)] = 2854.16;
	lumiByLS[make_pair(178479,292)] = 2852.72;
	lumiByLS[make_pair(178479,293)] = 2850.5;
	lumiByLS[make_pair(178479,294)] = 2848.65;
	lumiByLS[make_pair(178479,295)] = 2848.37;
	lumiByLS[make_pair(178479,296)] = 2846.32;
	lumiByLS[make_pair(178479,297)] = 2844.75;
	lumiByLS[make_pair(178479,298)] = 2842.64;
	lumiByLS[make_pair(178479,299)] = 2841.25;
	lumiByLS[make_pair(178479,300)] = 2840.11;
	lumiByLS[make_pair(178479,301)] = 2838.41;
	lumiByLS[make_pair(178479,302)] = 2836.17;
	lumiByLS[make_pair(178479,303)] = 2833.54;
	lumiByLS[make_pair(178479,304)] = 2831.58;
	lumiByLS[make_pair(178479,305)] = 2828.73;
	lumiByLS[make_pair(178479,306)] = 2826.95;
	lumiByLS[make_pair(178479,307)] = 2823.78;
	lumiByLS[make_pair(178479,308)] = 2822.31;
	lumiByLS[make_pair(178479,309)] = 2819.93;
	lumiByLS[make_pair(178479,310)] = 2818.56;
	lumiByLS[make_pair(178479,311)] = 2816.25;
	lumiByLS[make_pair(178479,312)] = 2813.56;
	lumiByLS[make_pair(178479,313)] = 2813.36;
	lumiByLS[make_pair(178479,314)] = 2811.35;
	lumiByLS[make_pair(178479,315)] = 2808.52;
	lumiByLS[make_pair(178479,316)] = 2809.24;
	lumiByLS[make_pair(178479,317)] = 2808.65;
	lumiByLS[make_pair(178479,318)] = 2808.22;
	lumiByLS[make_pair(178479,319)] = 2806.66;
	lumiByLS[make_pair(178479,320)] = 2806.46;
	lumiByLS[make_pair(178479,321)] = 2805.67;
	lumiByLS[make_pair(178479,322)] = 2803.52;
	lumiByLS[make_pair(178479,323)] = 2803.18;
	lumiByLS[make_pair(178479,324)] = 2803.02;
	lumiByLS[make_pair(178479,325)] = 2801.28;
	lumiByLS[make_pair(178479,326)] = 2799.43;
	lumiByLS[make_pair(178479,327)] = 2796.51;
	lumiByLS[make_pair(178479,329)] = 2794.96;
	lumiByLS[make_pair(178479,330)] = 2792.13;
	lumiByLS[make_pair(178479,331)] = 2791.25;
	lumiByLS[make_pair(178479,332)] = 2790.4;
	lumiByLS[make_pair(178479,333)] = 2792.71;
	lumiByLS[make_pair(178479,334)] = 2790.02;
	lumiByLS[make_pair(178479,335)] = 2787.81;
	lumiByLS[make_pair(178479,336)] = 2786.21;
	lumiByLS[make_pair(178479,337)] = 2783.99;
	lumiByLS[make_pair(178479,338)] = 2782.81;
	lumiByLS[make_pair(178479,339)] = 2780.4;
	lumiByLS[make_pair(178479,340)] = 2778.59;
	lumiByLS[make_pair(178479,341)] = 2776.41;
	lumiByLS[make_pair(178479,342)] = 2774.96;
	lumiByLS[make_pair(178479,343)] = 2773.54;
	lumiByLS[make_pair(178479,344)] = 2771.62;
	lumiByLS[make_pair(178479,345)] = 2770.71;
	lumiByLS[make_pair(178479,346)] = 2769.21;
	lumiByLS[make_pair(178479,347)] = 2767.52;
	lumiByLS[make_pair(178479,348)] = 2765.84;
	lumiByLS[make_pair(178479,349)] = 2764.37;
	lumiByLS[make_pair(178479,350)] = 2761.89;
	lumiByLS[make_pair(178479,351)] = 2760.77;
	lumiByLS[make_pair(178479,352)] = 2759;
	lumiByLS[make_pair(178479,353)] = 2757.7;
	lumiByLS[make_pair(178479,354)] = 2756.89;
	lumiByLS[make_pair(178479,355)] = 2755.65;
	lumiByLS[make_pair(178479,356)] = 2754.29;
	lumiByLS[make_pair(178479,357)] = 2752.85;
	lumiByLS[make_pair(178479,358)] = 2751;
	lumiByLS[make_pair(178479,359)] = 2749.06;
	lumiByLS[make_pair(178479,360)] = 2748.18;
	lumiByLS[make_pair(178479,361)] = 2747.1;
	lumiByLS[make_pair(178479,362)] = 2746.37;
	lumiByLS[make_pair(178479,363)] = 2745.02;
	lumiByLS[make_pair(178479,364)] = 2743.74;
	lumiByLS[make_pair(178479,365)] = 2741.96;
	lumiByLS[make_pair(178479,366)] = 2739.9;
	lumiByLS[make_pair(178479,367)] = 2737.98;
	lumiByLS[make_pair(178479,368)] = 2735.38;
	lumiByLS[make_pair(178479,369)] = 2733.18;
	lumiByLS[make_pair(178479,370)] = 2730.84;
	lumiByLS[make_pair(178479,371)] = 2729.33;
	lumiByLS[make_pair(178479,372)] = 2727.23;
	lumiByLS[make_pair(178479,373)] = 2724.85;
	lumiByLS[make_pair(178479,374)] = 2722.9;
	lumiByLS[make_pair(178479,375)] = 2721.26;
	lumiByLS[make_pair(178479,376)] = 2719.89;
	lumiByLS[make_pair(178479,377)] = 2719.35;
	lumiByLS[make_pair(178479,378)] = 2717.55;
	lumiByLS[make_pair(178479,379)] = 2717.48;
	lumiByLS[make_pair(178479,380)] = 2716.86;
	lumiByLS[make_pair(178479,381)] = 2716.43;
	lumiByLS[make_pair(178479,382)] = 2715.77;
	lumiByLS[make_pair(178479,383)] = 2715.46;
	lumiByLS[make_pair(178479,384)] = 2715.96;
	lumiByLS[make_pair(178479,385)] = 2716.51;
	lumiByLS[make_pair(178479,386)] = 2715.86;
	lumiByLS[make_pair(178479,387)] = 2714.68;
	lumiByLS[make_pair(178479,388)] = 2712.77;
	lumiByLS[make_pair(178479,389)] = 2711.48;
	lumiByLS[make_pair(178479,390)] = 2710.29;
	lumiByLS[make_pair(178479,391)] = 2708.6;
	lumiByLS[make_pair(178479,392)] = 2707.35;
	lumiByLS[make_pair(178479,393)] = 2706.63;
	lumiByLS[make_pair(178479,394)] = 2705.08;
	lumiByLS[make_pair(178479,395)] = 2704.09;
	lumiByLS[make_pair(178479,396)] = 2701.65;
	lumiByLS[make_pair(178479,397)] = 2700.72;
	lumiByLS[make_pair(178479,398)] = 2699.11;
	lumiByLS[make_pair(178479,399)] = 2697.58;
	lumiByLS[make_pair(178479,400)] = 2696.48;
	lumiByLS[make_pair(178479,401)] = 2694.5;
	lumiByLS[make_pair(178479,402)] = 2692.55;
	lumiByLS[make_pair(178479,403)] = 2691.49;
	lumiByLS[make_pair(178479,404)] = 2689.48;
	lumiByLS[make_pair(178479,405)] = 2687.92;
	lumiByLS[make_pair(178479,406)] = 2685.62;
	lumiByLS[make_pair(178479,407)] = 2683.96;
	lumiByLS[make_pair(178479,408)] = 2682.77;
	lumiByLS[make_pair(178479,409)] = 2681.29;
	lumiByLS[make_pair(178479,410)] = 2680.07;
	lumiByLS[make_pair(178479,411)] = 2679;
	lumiByLS[make_pair(178479,412)] = 2678.26;
	lumiByLS[make_pair(178479,413)] = 2677.48;
	lumiByLS[make_pair(178479,414)] = 2676.86;
	lumiByLS[make_pair(178479,415)] = 2676.66;
	lumiByLS[make_pair(178479,416)] = 2676.21;
	lumiByLS[make_pair(178479,417)] = 2675.53;
	lumiByLS[make_pair(178479,418)] = 2675.89;
	lumiByLS[make_pair(178479,419)] = 2675.38;
	lumiByLS[make_pair(178479,420)] = 2674.33;
	lumiByLS[make_pair(178479,421)] = 2674.45;
	lumiByLS[make_pair(178479,422)] = 2673.65;
	lumiByLS[make_pair(178479,423)] = 2673.46;
	lumiByLS[make_pair(178479,424)] = 2672.07;
	lumiByLS[make_pair(178479,425)] = 2671;
	lumiByLS[make_pair(178479,426)] = 2670.1;
	lumiByLS[make_pair(178479,427)] = 2668.76;
	lumiByLS[make_pair(178479,428)] = 2667.14;
	lumiByLS[make_pair(178479,429)] = 2665.19;
	lumiByLS[make_pair(178479,430)] = 2664.22;
	lumiByLS[make_pair(178479,431)] = 2662.03;
	lumiByLS[make_pair(178479,432)] = 2660.15;
	lumiByLS[make_pair(178479,433)] = 2658.71;
	lumiByLS[make_pair(178479,434)] = 2657.76;
	lumiByLS[make_pair(178479,435)] = 2656.25;
	lumiByLS[make_pair(178479,436)] = 2654.29;
	lumiByLS[make_pair(178479,437)] = 2653.23;
	lumiByLS[make_pair(178479,438)] = 2651.36;
	lumiByLS[make_pair(178479,439)] = 2650.16;
	lumiByLS[make_pair(178479,440)] = 2648.44;
	lumiByLS[make_pair(178479,441)] = 2646.68;
	lumiByLS[make_pair(178479,442)] = 2645.52;
	lumiByLS[make_pair(178479,443)] = 2643.61;
	lumiByLS[make_pair(178479,444)] = 2640.74;
	lumiByLS[make_pair(178479,445)] = 2639.85;
	lumiByLS[make_pair(178479,446)] = 2638.15;
	lumiByLS[make_pair(178479,447)] = 2636.67;
	lumiByLS[make_pair(178479,448)] = 2634.83;
	lumiByLS[make_pair(178479,449)] = 2633.03;
	lumiByLS[make_pair(178479,450)] = 2630.75;
	lumiByLS[make_pair(178479,451)] = 2628.82;
	lumiByLS[make_pair(178479,452)] = 2627.28;
	lumiByLS[make_pair(178479,453)] = 2625.92;
	lumiByLS[make_pair(178479,454)] = 2624.7;
	lumiByLS[make_pair(178479,455)] = 2622.98;
	lumiByLS[make_pair(178479,456)] = 2622.03;
	lumiByLS[make_pair(178479,457)] = 2620.37;
	lumiByLS[make_pair(178479,458)] = 2620.25;
	lumiByLS[make_pair(178479,459)] = 2618.72;
	lumiByLS[make_pair(178479,460)] = 2618.96;
	lumiByLS[make_pair(178479,461)] = 2619.31;
	lumiByLS[make_pair(178479,462)] = 2619.01;
	lumiByLS[make_pair(178479,463)] = 2618.27;
	lumiByLS[make_pair(178479,464)] = 2618.31;
	lumiByLS[make_pair(178479,465)] = 2617.06;
	lumiByLS[make_pair(178479,466)] = 2616.42;
	lumiByLS[make_pair(178479,467)] = 2614.79;
	lumiByLS[make_pair(178479,468)] = 2612.93;
	lumiByLS[make_pair(178479,469)] = 2611.6;
	lumiByLS[make_pair(178479,470)] = 2610.55;
	lumiByLS[make_pair(178479,471)] = 2609.23;
	lumiByLS[make_pair(178479,472)] = 2607.58;
	lumiByLS[make_pair(178479,473)] = 2606.33;
	lumiByLS[make_pair(178479,474)] = 2605.09;
	lumiByLS[make_pair(178479,475)] = 2603.93;
	lumiByLS[make_pair(178479,476)] = 2603.28;
	lumiByLS[make_pair(178479,477)] = 2602.55;
	lumiByLS[make_pair(178479,478)] = 2601.85;
	lumiByLS[make_pair(178479,479)] = 2602.11;
	lumiByLS[make_pair(178479,480)] = 2601.25;
	lumiByLS[make_pair(178479,481)] = 2600.65;
	lumiByLS[make_pair(178479,482)] = 2599.8;
	lumiByLS[make_pair(178479,483)] = 2599.48;
	lumiByLS[make_pair(178479,484)] = 2598.88;
	lumiByLS[make_pair(178479,485)] = 2597.48;
	lumiByLS[make_pair(178479,486)] = 2596.59;
	lumiByLS[make_pair(178479,487)] = 2595.07;
	lumiByLS[make_pair(178479,488)] = 2593.28;
	lumiByLS[make_pair(178479,489)] = 2591.46;
	lumiByLS[make_pair(178479,490)] = 2589.96;
	lumiByLS[make_pair(178479,492)] = 2584.04;
	lumiByLS[make_pair(178479,493)] = 2583.66;
	lumiByLS[make_pair(178479,494)] = 2582.77;
	lumiByLS[make_pair(178479,495)] = 2581.46;
	lumiByLS[make_pair(178479,496)] = 2579.93;
	lumiByLS[make_pair(178479,497)] = 2578.33;
	lumiByLS[make_pair(178479,498)] = 2575.84;
	lumiByLS[make_pair(178479,499)] = 2573.61;
	lumiByLS[make_pair(178479,500)] = 2573.05;
	lumiByLS[make_pair(178479,501)] = 2571.47;
	lumiByLS[make_pair(178479,502)] = 2570.44;
	lumiByLS[make_pair(178479,503)] = 2569.35;
	lumiByLS[make_pair(178479,504)] = 2568.34;
	lumiByLS[make_pair(178479,505)] = 2569.27;
	lumiByLS[make_pair(178479,506)] = 2567.41;
	lumiByLS[make_pair(178479,507)] = 2567.28;
	lumiByLS[make_pair(178479,508)] = 2565.88;
	lumiByLS[make_pair(178479,509)] = 2565.08;
	lumiByLS[make_pair(178479,510)] = 2564.31;
	lumiByLS[make_pair(178479,511)] = 2562.73;
	lumiByLS[make_pair(178479,512)] = 2561.85;
	lumiByLS[make_pair(178479,513)] = 2559.94;
	lumiByLS[make_pair(178479,514)] = 2557.98;
	lumiByLS[make_pair(178479,515)] = 2556.87;
	lumiByLS[make_pair(178479,516)] = 2555.83;
	lumiByLS[make_pair(178479,517)] = 2554.53;
	lumiByLS[make_pair(178479,518)] = 2552.71;
	lumiByLS[make_pair(178479,519)] = 2550.76;
	lumiByLS[make_pair(178479,520)] = 2549.84;
	lumiByLS[make_pair(178479,521)] = 2548.76;
	lumiByLS[make_pair(178479,522)] = 2548.21;
	lumiByLS[make_pair(178479,523)] = 2547.76;
	lumiByLS[make_pair(178479,528)] = 2543.27;
	lumiByLS[make_pair(178479,529)] = 2541.86;
	lumiByLS[make_pair(178479,530)] = 2540.26;
	lumiByLS[make_pair(178479,531)] = 2538.82;
	lumiByLS[make_pair(178479,532)] = 2537.9;
	lumiByLS[make_pair(178479,533)] = 2537.94;
	lumiByLS[make_pair(178479,534)] = 2536.02;
	lumiByLS[make_pair(178479,535)] = 2535.48;
	lumiByLS[make_pair(178479,536)] = 2534.22;
	lumiByLS[make_pair(178479,537)] = 2533.81;
	lumiByLS[make_pair(178479,538)] = 2532.59;
	lumiByLS[make_pair(178479,539)] = 2531.29;
	lumiByLS[make_pair(178479,540)] = 2529.89;
	lumiByLS[make_pair(178479,541)] = 2527.95;
	lumiByLS[make_pair(178479,542)] = 2526.65;
	lumiByLS[make_pair(178479,543)] = 2525.39;
	lumiByLS[make_pair(178479,544)] = 2524.44;
	lumiByLS[make_pair(178479,545)] = 2521.83;
	lumiByLS[make_pair(178479,546)] = 2521.25;
	lumiByLS[make_pair(178479,547)] = 2520.24;
	lumiByLS[make_pair(178479,548)] = 2518.68;
	lumiByLS[make_pair(178479,549)] = 2516.8;
	lumiByLS[make_pair(178479,550)] = 2515.42;
	lumiByLS[make_pair(178479,551)] = 2514.3;
	lumiByLS[make_pair(178479,552)] = 2513.19;
	lumiByLS[make_pair(178479,553)] = 2512.74;
	lumiByLS[make_pair(178479,554)] = 2512.26;
	lumiByLS[make_pair(178479,555)] = 2511.46;
	lumiByLS[make_pair(178479,556)] = 2511.33;
	lumiByLS[make_pair(178479,557)] = 2510.61;
	lumiByLS[make_pair(178479,558)] = 2510.66;
	lumiByLS[make_pair(178479,559)] = 2510.95;
	lumiByLS[make_pair(178479,560)] = 2510.32;
	lumiByLS[make_pair(178479,561)] = 2508.96;
	lumiByLS[make_pair(178479,562)] = 2507.76;
	lumiByLS[make_pair(178479,563)] = 2506.42;
	lumiByLS[make_pair(178479,564)] = 2505.28;
	lumiByLS[make_pair(178479,565)] = 2503.74;
	lumiByLS[make_pair(178479,566)] = 2502.05;
	lumiByLS[make_pair(178479,567)] = 2500.53;
	lumiByLS[make_pair(178479,568)] = 2499.65;
	lumiByLS[make_pair(178479,569)] = 2498.27;
	lumiByLS[make_pair(178479,570)] = 2495.97;
	lumiByLS[make_pair(178479,571)] = 2495.19;
	lumiByLS[make_pair(178479,572)] = 2493.67;
	lumiByLS[make_pair(178479,573)] = 2492.21;
	lumiByLS[make_pair(178479,574)] = 2490.81;
	lumiByLS[make_pair(178479,575)] = 2489.42;
	lumiByLS[make_pair(178479,577)] = 2486.63;
	lumiByLS[make_pair(178479,578)] = 2485.79;
	lumiByLS[make_pair(178479,579)] = 2485.52;
	lumiByLS[make_pair(178479,580)] = 2484.6;
	lumiByLS[make_pair(178479,581)] = 2483.27;
	lumiByLS[make_pair(178479,582)] = 2482.22;
	lumiByLS[make_pair(178479,583)] = 2481.1;
	lumiByLS[make_pair(178479,584)] = 2480.13;
	lumiByLS[make_pair(178479,585)] = 2480.4;
	lumiByLS[make_pair(178479,586)] = 2476;
	lumiByLS[make_pair(178479,587)] = 2476.2;
	lumiByLS[make_pair(178479,588)] = 2457.45;
	lumiByLS[make_pair(178479,589)] = 2440.4;
	lumiByLS[make_pair(178479,590)] = 2476.04;
	lumiByLS[make_pair(178479,591)] = 2491.58;
	lumiByLS[make_pair(178479,592)] = 2490.63;
	lumiByLS[make_pair(178479,593)] = 2489.65;
	lumiByLS[make_pair(178479,594)] = 2488.71;
	lumiByLS[make_pair(178479,595)] = 2487.66;
	lumiByLS[make_pair(178479,596)] = 2486.72;
	lumiByLS[make_pair(178479,597)] = 2485.32;
	lumiByLS[make_pair(178479,598)] = 2484.06;
	lumiByLS[make_pair(178479,599)] = 2483.14;
	lumiByLS[make_pair(178479,600)] = 2481.81;
	lumiByLS[make_pair(178479,601)] = 2480.52;
	lumiByLS[make_pair(178479,602)] = 2479.17;
	lumiByLS[make_pair(178479,603)] = 2478.15;
	lumiByLS[make_pair(178479,604)] = 2477.02;
	lumiByLS[make_pair(178479,605)] = 2476.28;
	lumiByLS[make_pair(178479,606)] = 2475.11;
	lumiByLS[make_pair(178479,607)] = 2474.18;
	lumiByLS[make_pair(178479,608)] = 2472.79;
	lumiByLS[make_pair(178479,609)] = 2471.64;
	lumiByLS[make_pair(178479,610)] = 2470.54;
	lumiByLS[make_pair(178479,611)] = 2469.55;
	lumiByLS[make_pair(178479,612)] = 2468.45;
	lumiByLS[make_pair(178479,613)] = 2467.32;
	lumiByLS[make_pair(178479,614)] = 2466.41;
	lumiByLS[make_pair(178479,615)] = 2465.21;
	lumiByLS[make_pair(178479,616)] = 2463.96;
	lumiByLS[make_pair(178479,617)] = 2462.81;
	lumiByLS[make_pair(178479,618)] = 2462.06;
	lumiByLS[make_pair(178479,619)] = 2460.59;
	lumiByLS[make_pair(178479,620)] = 2459.48;
	lumiByLS[make_pair(178479,621)] = 2458.36;
	lumiByLS[make_pair(178479,622)] = 2457.02;
	lumiByLS[make_pair(178479,623)] = 2456.02;
	lumiByLS[make_pair(178479,624)] = 2455.04;
	lumiByLS[make_pair(178479,625)] = 2453.86;
	lumiByLS[make_pair(178479,626)] = 2452.8;
	lumiByLS[make_pair(178479,627)] = 2451.63;
	lumiByLS[make_pair(178479,628)] = 2450.52;
	lumiByLS[make_pair(178479,629)] = 2449.36;
	lumiByLS[make_pair(178479,630)] = 2448.5;
	lumiByLS[make_pair(178479,631)] = 2447.51;
	lumiByLS[make_pair(178479,632)] = 2446.59;
	lumiByLS[make_pair(178479,633)] = 2445.25;
	lumiByLS[make_pair(178479,634)] = 2444.09;
	lumiByLS[make_pair(178479,635)] = 2443.51;
	lumiByLS[make_pair(178479,636)] = 2442.29;
	lumiByLS[make_pair(178479,637)] = 2441.02;
	lumiByLS[make_pair(178479,638)] = 2439.97;
	lumiByLS[make_pair(178479,639)] = 2438.87;
	lumiByLS[make_pair(178479,640)] = 2437.47;
	lumiByLS[make_pair(178479,641)] = 2436.19;
	lumiByLS[make_pair(178479,642)] = 2434.82;
	lumiByLS[make_pair(178479,643)] = 2433.25;
	lumiByLS[make_pair(178479,644)] = 2432.29;
	lumiByLS[make_pair(178479,645)] = 2431.32;
	lumiByLS[make_pair(178479,646)] = 2430.64;
	lumiByLS[make_pair(178479,647)] = 2429.29;
	lumiByLS[make_pair(178479,648)] = 2428.46;
	lumiByLS[make_pair(178479,649)] = 2427.5;
	lumiByLS[make_pair(178479,650)] = 2426.35;
	lumiByLS[make_pair(178479,651)] = 2425.63;
	lumiByLS[make_pair(178479,652)] = 2425.04;
	lumiByLS[make_pair(178479,653)] = 2423.51;
	lumiByLS[make_pair(178479,654)] = 2422.47;
	lumiByLS[make_pair(178479,655)] = 2421.27;
	lumiByLS[make_pair(178479,656)] = 2420.09;
	lumiByLS[make_pair(178479,657)] = 2418.99;
	lumiByLS[make_pair(178479,658)] = 2417.9;
	lumiByLS[make_pair(178479,659)] = 2416.61;
	lumiByLS[make_pair(178479,660)] = 2415.68;
	lumiByLS[make_pair(178479,661)] = 2414.8;
	lumiByLS[make_pair(178479,662)] = 2413.92;
	lumiByLS[make_pair(178479,663)] = 2413.19;
	lumiByLS[make_pair(178479,664)] = 2411.98;
	lumiByLS[make_pair(178479,665)] = 2411.19;
	lumiByLS[make_pair(178479,666)] = 2410.41;
	lumiByLS[make_pair(178479,667)] = 2409.3;
	lumiByLS[make_pair(178479,668)] = 2408.39;
	lumiByLS[make_pair(178479,669)] = 2407.29;
	lumiByLS[make_pair(178479,670)] = 2406.01;
	lumiByLS[make_pair(178479,671)] = 2405.14;
	lumiByLS[make_pair(178479,672)] = 2404.09;
	lumiByLS[make_pair(178479,673)] = 2403.54;
	lumiByLS[make_pair(178479,674)] = 2402.47;
	lumiByLS[make_pair(178479,675)] = 2401.37;
	lumiByLS[make_pair(178479,676)] = 2400.07;
	lumiByLS[make_pair(178479,677)] = 2398.94;
	lumiByLS[make_pair(178479,678)] = 2397.83;
	lumiByLS[make_pair(178479,679)] = 2396.56;
	lumiByLS[make_pair(178479,680)] = 2395.64;
	lumiByLS[make_pair(178479,681)] = 2394.57;
	lumiByLS[make_pair(178479,682)] = 2393.46;
	lumiByLS[make_pair(178479,683)] = 2392.41;
	lumiByLS[make_pair(178479,684)] = 2391.11;
	lumiByLS[make_pair(178479,685)] = 2390.19;
	lumiByLS[make_pair(178479,686)] = 2389.07;
	lumiByLS[make_pair(178479,687)] = 2388.08;
	lumiByLS[make_pair(178479,688)] = 2387.03;
	lumiByLS[make_pair(178479,689)] = 2386.11;
	lumiByLS[make_pair(178479,690)] = 2385.07;
	lumiByLS[make_pair(178479,691)] = 2384.08;
	lumiByLS[make_pair(178479,692)] = 2383.29;
	lumiByLS[make_pair(178479,693)] = 2382.47;
	lumiByLS[make_pair(178479,694)] = 2381.54;
	lumiByLS[make_pair(178479,695)] = 2380.43;
	lumiByLS[make_pair(178479,696)] = 2379.43;
	lumiByLS[make_pair(178479,697)] = 2378.21;
	lumiByLS[make_pair(178479,698)] = 2377.12;
	lumiByLS[make_pair(178479,699)] = 2376.15;
	lumiByLS[make_pair(178479,700)] = 2375.13;
	lumiByLS[make_pair(178479,701)] = 2373.87;
	lumiByLS[make_pair(178479,702)] = 2372.86;
	lumiByLS[make_pair(178479,703)] = 2371.53;
	lumiByLS[make_pair(178479,704)] = 2370.53;
	lumiByLS[make_pair(178479,705)] = 2369.59;
	lumiByLS[make_pair(178479,706)] = 2368.56;
	lumiByLS[make_pair(178479,707)] = 2367.41;
	lumiByLS[make_pair(178479,708)] = 2366.71;
	lumiByLS[make_pair(178479,709)] = 2366.94;
	lumiByLS[make_pair(178479,710)] = 2365.75;
	lumiByLS[make_pair(178479,711)] = 2364.68;
	lumiByLS[make_pair(178479,712)] = 2363.22;
	lumiByLS[make_pair(178479,713)] = 2361.68;
	lumiByLS[make_pair(178479,714)] = 2360.39;
	lumiByLS[make_pair(178479,715)] = 2359.1;
	lumiByLS[make_pair(178479,716)] = 2358;
	lumiByLS[make_pair(178479,717)] = 2356.78;
	lumiByLS[make_pair(178479,718)] = 2355.56;
	lumiByLS[make_pair(178479,719)] = 2354.57;
	lumiByLS[make_pair(178479,720)] = 2353.4;
	lumiByLS[make_pair(178479,721)] = 2352.39;
	lumiByLS[make_pair(178479,722)] = 2351.18;
	lumiByLS[make_pair(178479,723)] = 2350.09;
	lumiByLS[make_pair(178479,724)] = 2346.59;
	lumiByLS[make_pair(178479,725)] = 2346.71;
	lumiByLS[make_pair(178479,726)] = 2345.6;
	lumiByLS[make_pair(178479,727)] = 2345.83;
	lumiByLS[make_pair(178479,728)] = 2344.82;
	lumiByLS[make_pair(178479,729)] = 2343.74;
	lumiByLS[make_pair(178479,730)] = 2342.82;
	lumiByLS[make_pair(178479,731)] = 2341.61;
	lumiByLS[make_pair(178479,732)] = 2340.88;
	lumiByLS[make_pair(178479,733)] = 2339.85;
	lumiByLS[make_pair(178479,734)] = 2338.7;
	lumiByLS[make_pair(178479,735)] = 2337.65;
	lumiByLS[make_pair(178479,736)] = 2336.43;
	lumiByLS[make_pair(178479,737)] = 2335.77;
	lumiByLS[make_pair(178479,738)] = 2334.84;
	lumiByLS[make_pair(178479,739)] = 2333.46;
	lumiByLS[make_pair(178479,740)] = 2332.53;
	lumiByLS[make_pair(178479,741)] = 2331.07;
	lumiByLS[make_pair(178479,742)] = 2329.67;
	lumiByLS[make_pair(178479,743)] = 2328.65;
	lumiByLS[make_pair(178479,744)] = 2327.41;
	lumiByLS[make_pair(178479,745)] = 2326;
	lumiByLS[make_pair(178479,746)] = 2324.85;
	lumiByLS[make_pair(178479,747)] = 2323.86;
	lumiByLS[make_pair(178479,748)] = 2323.2;
	lumiByLS[make_pair(178479,749)] = 2322.38;
	lumiByLS[make_pair(178479,750)] = 2321.47;
	lumiByLS[make_pair(178479,751)] = 2320.59;
	lumiByLS[make_pair(178479,752)] = 2319.74;
	lumiByLS[make_pair(178479,753)] = 2319.11;
	lumiByLS[make_pair(178479,754)] = 2318.5;
	lumiByLS[make_pair(178479,755)] = 2318.86;
	lumiByLS[make_pair(178479,756)] = 2307.63;
	lumiByLS[make_pair(178479,757)] = 2298.33;
	lumiByLS[make_pair(178479,758)] = 2310.04;
	lumiByLS[make_pair(178479,759)] = 2313.3;
	lumiByLS[make_pair(178479,760)] = 2312.15;
	lumiByLS[make_pair(178479,761)] = 2310.34;
	lumiByLS[make_pair(178479,762)] = 2310.1;
	lumiByLS[make_pair(178479,763)] = 2308.92;
	lumiByLS[make_pair(178479,764)] = 2287.96;
	lumiByLS[make_pair(178479,765)] = 2295.68;
	lumiByLS[make_pair(178479,766)] = 2308.76;
	lumiByLS[make_pair(178479,767)] = 2282.27;
	lumiByLS[make_pair(178479,768)] = 2318.8;
	lumiByLS[make_pair(178479,769)] = 2318.11;
	lumiByLS[make_pair(178479,770)] = 2317.3;
	lumiByLS[make_pair(178479,771)] = 2316.56;
	lumiByLS[make_pair(178479,772)] = 2315.62;
	lumiByLS[make_pair(178479,773)] = 2314.85;
	lumiByLS[make_pair(178479,774)] = 2313.93;
	lumiByLS[make_pair(178479,775)] = 2312.99;
	lumiByLS[make_pair(178479,776)] = 2312.17;
	lumiByLS[make_pair(178479,777)] = 2311.21;
	lumiByLS[make_pair(178479,778)] = 2310.3;
	lumiByLS[make_pair(178479,779)] = 2309.27;
	lumiByLS[make_pair(178479,780)] = 2308.41;
	lumiByLS[make_pair(178479,781)] = 2307.48;
	lumiByLS[make_pair(178479,782)] = 2306.59;
	lumiByLS[make_pair(178479,783)] = 2305.78;
	lumiByLS[make_pair(178479,784)] = 2305;
	lumiByLS[make_pair(178479,785)] = 2304.2;
	lumiByLS[make_pair(178479,786)] = 2303.22;
	lumiByLS[make_pair(178479,787)] = 2302.29;
	lumiByLS[make_pair(178479,788)] = 2301.38;
	lumiByLS[make_pair(178479,789)] = 2300.38;
	lumiByLS[make_pair(178479,790)] = 2299.46;
	lumiByLS[make_pair(178479,791)] = 2298.52;
	lumiByLS[make_pair(178479,792)] = 2297.45;
	lumiByLS[make_pair(178479,793)] = 2296.57;
	lumiByLS[make_pair(178479,794)] = 2295.5;
	lumiByLS[make_pair(178479,795)] = 2294.54;
	lumiByLS[make_pair(178479,796)] = 2293.64;
	lumiByLS[make_pair(178479,797)] = 2292.77;
	lumiByLS[make_pair(178479,798)] = 2291.72;
	lumiByLS[make_pair(178479,799)] = 2290.88;
	lumiByLS[make_pair(178479,800)] = 2289.88;
	lumiByLS[make_pair(178479,801)] = 2288.85;
	lumiByLS[make_pair(178479,802)] = 2287.75;
	lumiByLS[make_pair(178479,803)] = 2286.78;
	lumiByLS[make_pair(178479,804)] = 2285.87;
	lumiByLS[make_pair(178479,805)] = 2284.98;
	lumiByLS[make_pair(178479,806)] = 2284.07;
	lumiByLS[make_pair(178479,807)] = 2282.85;
	lumiByLS[make_pair(178479,808)] = 2281.85;
	lumiByLS[make_pair(178479,809)] = 2280.81;
	lumiByLS[make_pair(178479,810)] = 2279.59;
	lumiByLS[make_pair(178479,811)] = 2278.3;
	lumiByLS[make_pair(178479,812)] = 2277.17;
	lumiByLS[make_pair(178479,813)] = 2275.92;
	lumiByLS[make_pair(178479,814)] = 2274.82;
	lumiByLS[make_pair(178479,815)] = 2273.61;
	lumiByLS[make_pair(178479,816)] = 2272.51;
	lumiByLS[make_pair(178479,817)] = 2271.53;
	lumiByLS[make_pair(178479,818)] = 2270.54;
	lumiByLS[make_pair(178479,819)] = 2269.66;
	lumiByLS[make_pair(178479,820)] = 2268.45;
	lumiByLS[make_pair(178479,821)] = 2267.46;
	lumiByLS[make_pair(178479,822)] = 2266.86;
	lumiByLS[make_pair(178479,823)] = 2266.23;
	lumiByLS[make_pair(178479,824)] = 2265.37;
	lumiByLS[make_pair(178479,825)] = 2264.74;
	lumiByLS[make_pair(178479,826)] = 2263.92;
	lumiByLS[make_pair(178479,827)] = 2263.13;
	lumiByLS[make_pair(178479,828)] = 2262.48;
	lumiByLS[make_pair(178479,829)] = 2261.6;
	lumiByLS[make_pair(178479,830)] = 2260.86;
	lumiByLS[make_pair(178479,831)] = 2260.01;
	lumiByLS[make_pair(178479,832)] = 2259.15;
	lumiByLS[make_pair(178479,833)] = 2258.22;
	lumiByLS[make_pair(178479,834)] = 2257.49;
	lumiByLS[make_pair(178479,835)] = 2256.56;
	lumiByLS[make_pair(178479,836)] = 2255.73;
	lumiByLS[make_pair(178479,837)] = 2254.92;
	lumiByLS[make_pair(178479,838)] = 2254.15;
	lumiByLS[make_pair(178479,839)] = 2253.38;
	lumiByLS[make_pair(178479,840)] = 2252.5;
	lumiByLS[make_pair(178479,841)] = 2251.71;
	lumiByLS[make_pair(178479,842)] = 2250.86;
	lumiByLS[make_pair(178479,843)] = 2250.21;
	lumiByLS[make_pair(178479,844)] = 2249.29;
	lumiByLS[make_pair(178479,845)] = 2248.58;
	lumiByLS[make_pair(178479,846)] = 2247.82;
	lumiByLS[make_pair(178479,847)] = 2246.98;
	lumiByLS[make_pair(178479,848)] = 2246.08;
	lumiByLS[make_pair(178479,849)] = 2245.5;
	lumiByLS[make_pair(178479,850)] = 2244.55;
	lumiByLS[make_pair(178479,851)] = 2243.62;
	lumiByLS[make_pair(178479,852)] = 2242.63;
	lumiByLS[make_pair(178479,853)] = 2241.07;
	lumiByLS[make_pair(178479,854)] = 2240.84;
	lumiByLS[make_pair(178479,855)] = 2239.88;
	lumiByLS[make_pair(178479,856)] = 2239.13;
	lumiByLS[make_pair(178479,857)] = 2238.07;
	lumiByLS[make_pair(178479,858)] = 2237.19;
	lumiByLS[make_pair(178479,859)] = 2236.32;
	lumiByLS[make_pair(178479,860)] = 2235.47;
	lumiByLS[make_pair(178479,861)] = 2234.82;
	lumiByLS[make_pair(178479,862)] = 2234;
	lumiByLS[make_pair(178479,863)] = 2233.09;
	lumiByLS[make_pair(178479,864)] = 2232.15;
	lumiByLS[make_pair(178479,865)] = 2231.47;
	lumiByLS[make_pair(178479,866)] = 2230.36;
	lumiByLS[make_pair(178479,867)] = 2229.44;
	lumiByLS[make_pair(178479,868)] = 2228.68;
	lumiByLS[make_pair(178479,869)] = 2227.79;
	lumiByLS[make_pair(178479,870)] = 2226.78;
	lumiByLS[make_pair(178479,871)] = 2225.98;
	lumiByLS[make_pair(178479,872)] = 2224.8;
	lumiByLS[make_pair(178479,873)] = 2223.65;
	lumiByLS[make_pair(178479,874)] = 2222.83;
	lumiByLS[make_pair(178479,875)] = 2221.82;
	lumiByLS[make_pair(178479,876)] = 2221.02;
	lumiByLS[make_pair(178479,877)] = 2220.18;
	lumiByLS[make_pair(178479,878)] = 2219.36;
	lumiByLS[make_pair(178479,879)] = 2218.5;
	lumiByLS[make_pair(178479,880)] = 2217.57;
	lumiByLS[make_pair(178479,881)] = 2216.94;
	lumiByLS[make_pair(178479,882)] = 2216.09;
	lumiByLS[make_pair(178479,883)] = 2215.31;
	lumiByLS[make_pair(178479,884)] = 2214.7;
	lumiByLS[make_pair(178479,885)] = 2213.86;
	lumiByLS[make_pair(178479,886)] = 2212.81;
	lumiByLS[make_pair(178479,887)] = 2211.41;
	lumiByLS[make_pair(178479,888)] = 2211.14;
	lumiByLS[make_pair(178479,889)] = 2210.27;
	lumiByLS[make_pair(178479,890)] = 2209.13;
	lumiByLS[make_pair(178479,891)] = 2208.37;
	lumiByLS[make_pair(178479,892)] = 2207.37;
	lumiByLS[make_pair(178479,893)] = 2206.3;
	lumiByLS[make_pair(178479,894)] = 2205.16;
	lumiByLS[make_pair(178479,895)] = 2204.23;
	lumiByLS[make_pair(178479,896)] = 2203.27;
	lumiByLS[make_pair(178479,897)] = 2202.32;
	lumiByLS[make_pair(178479,898)] = 2201.34;
	lumiByLS[make_pair(178479,899)] = 2200.65;
	lumiByLS[make_pair(178479,900)] = 2199.7;
	lumiByLS[make_pair(178479,901)] = 2198.79;
	lumiByLS[make_pair(178479,902)] = 2183.05;
	lumiByLS[make_pair(178479,903)] = 2190.22;
	lumiByLS[make_pair(178479,904)] = 2183.45;
	lumiByLS[make_pair(178479,905)] = 2195.76;
	lumiByLS[make_pair(178479,906)] = 2194.87;
	lumiByLS[make_pair(178479,907)] = 2194.09;
	lumiByLS[make_pair(178479,908)] = 2193.31;
	lumiByLS[make_pair(178479,909)] = 2192.61;
	lumiByLS[make_pair(178479,910)] = 2191.93;
	lumiByLS[make_pair(178479,911)] = 2191.23;
	lumiByLS[make_pair(178479,912)] = 2190.48;
	lumiByLS[make_pair(178479,913)] = 2189.77;
	lumiByLS[make_pair(178479,914)] = 2189.07;
	lumiByLS[make_pair(178479,915)] = 2188.28;
	lumiByLS[make_pair(178479,916)] = 2187.55;
	lumiByLS[make_pair(178479,917)] = 2186.73;
	lumiByLS[make_pair(178479,918)] = 2185.87;
	lumiByLS[make_pair(178479,919)] = 2185.03;
	lumiByLS[make_pair(178479,920)] = 2184.12;
	lumiByLS[make_pair(178479,921)] = 2183.11;
	lumiByLS[make_pair(178479,922)] = 2182.11;
	lumiByLS[make_pair(178479,923)] = 2181.3;
	lumiByLS[make_pair(178479,924)] = 2180.52;
	lumiByLS[make_pair(178479,925)] = 2179.55;
	lumiByLS[make_pair(178479,926)] = 2178.75;
	lumiByLS[make_pair(178479,927)] = 2177.85;
	lumiByLS[make_pair(178479,928)] = 2177.05;
	lumiByLS[make_pair(178479,929)] = 2176.09;
	lumiByLS[make_pair(178479,930)] = 2175.49;
	lumiByLS[make_pair(178479,931)] = 2174.57;
	lumiByLS[make_pair(178479,932)] = 2173.87;
	lumiByLS[make_pair(178479,933)] = 2173.21;
	lumiByLS[make_pair(178479,934)] = 2172.39;
	lumiByLS[make_pair(178479,935)] = 2171.74;
	lumiByLS[make_pair(178479,936)] = 2170.94;
	lumiByLS[make_pair(178479,937)] = 2170.15;
	lumiByLS[make_pair(178479,938)] = 2169.3;
	lumiByLS[make_pair(178479,939)] = 2168.54;
	lumiByLS[make_pair(178479,940)] = 2167.63;
	lumiByLS[make_pair(178479,941)] = 2166.78;
	lumiByLS[make_pair(178479,942)] = 2166.02;
	lumiByLS[make_pair(178479,943)] = 2165.16;
	lumiByLS[make_pair(178479,944)] = 2164.6;
	lumiByLS[make_pair(178479,945)] = 2163.78;
	lumiByLS[make_pair(178479,946)] = 2163.01;
	lumiByLS[make_pair(178479,947)] = 2162.16;
	lumiByLS[make_pair(178479,948)] = 2161.44;
	lumiByLS[make_pair(178479,949)] = 2160.78;
	lumiByLS[make_pair(178479,950)] = 2159.98;
	lumiByLS[make_pair(178479,951)] = 2159.16;
	lumiByLS[make_pair(178479,952)] = 2158.47;
	lumiByLS[make_pair(178479,953)] = 2157.65;
	lumiByLS[make_pair(178479,954)] = 2156.92;
	lumiByLS[make_pair(178479,955)] = 2155.98;
	lumiByLS[make_pair(178479,956)] = 2155.23;
	lumiByLS[make_pair(178479,957)] = 2154.58;
	lumiByLS[make_pair(178479,958)] = 2153.84;
	lumiByLS[make_pair(178479,959)] = 2153.06;
	lumiByLS[make_pair(178479,960)] = 2152.37;
	lumiByLS[make_pair(178479,961)] = 2151.56;
	lumiByLS[make_pair(178479,962)] = 2150.81;
	lumiByLS[make_pair(178479,963)] = 2150.03;
	lumiByLS[make_pair(178479,964)] = 2149.43;
	lumiByLS[make_pair(178479,965)] = 2148.52;
	lumiByLS[make_pair(178479,966)] = 2147.7;
	lumiByLS[make_pair(178479,967)] = 2146.94;
	lumiByLS[make_pair(178479,968)] = 2146.33;
	lumiByLS[make_pair(178479,969)] = 2145.41;
	lumiByLS[make_pair(178479,970)] = 2144.52;
	lumiByLS[make_pair(178479,971)] = 2143.86;
	lumiByLS[make_pair(178479,972)] = 2143.02;
	lumiByLS[make_pair(178479,973)] = 2142.06;
	lumiByLS[make_pair(178479,974)] = 2141.16;
	lumiByLS[make_pair(178479,975)] = 2140.27;
	lumiByLS[make_pair(178479,976)] = 2139.46;
	lumiByLS[make_pair(178479,977)] = 2138.6;
	lumiByLS[make_pair(178479,978)] = 2137.69;
	lumiByLS[make_pair(178479,979)] = 2136.87;
	lumiByLS[make_pair(178479,980)] = 2135.93;
	lumiByLS[make_pair(178479,981)] = 2135.22;
	lumiByLS[make_pair(178479,982)] = 2134.32;
	lumiByLS[make_pair(178479,983)] = 2133.68;
	lumiByLS[make_pair(178479,984)] = 2132.75;
	lumiByLS[make_pair(178479,985)] = 2132.05;
	lumiByLS[make_pair(178479,986)] = 2131.16;
	lumiByLS[make_pair(178479,987)] = 2130.58;
	lumiByLS[make_pair(178479,988)] = 2129.74;
	lumiByLS[make_pair(178479,989)] = 2129.02;
	lumiByLS[make_pair(178479,990)] = 2128.13;
	lumiByLS[make_pair(178479,991)] = 2127.35;
	lumiByLS[make_pair(178479,992)] = 2126.53;
	lumiByLS[make_pair(178479,993)] = 2125.82;
	lumiByLS[make_pair(178479,994)] = 2124.96;
	lumiByLS[make_pair(178479,995)] = 2123.92;
	lumiByLS[make_pair(178479,996)] = 2122.88;
	lumiByLS[make_pair(178479,997)] = 2122;
	lumiByLS[make_pair(178479,998)] = 2121.17;
	lumiByLS[make_pair(178479,999)] = 2120.27;
	lumiByLS[make_pair(178479,1000)] = 2119.47;
	lumiByLS[make_pair(178479,1001)] = 826.439;
	lumiByLS[make_pair(178479,1002)] = 1.12961;

	lumiByLS[make_pair(179828,2)] = 54.7679 ;
	lumiByLS[make_pair(179828,3)] = 54.7545 ;
	lumiByLS[make_pair(179828,4)] = 54.7517 ;
	lumiByLS[make_pair(179828,5)] = 54.6902 ;
	lumiByLS[make_pair(179828,6)] = 54.6495 ;
	lumiByLS[make_pair(179828,7)] = 54.6528 ;
	lumiByLS[make_pair(179828,8)] = 54.6589 ;
	lumiByLS[make_pair(179828,9)] = 54.6330 ;
	lumiByLS[make_pair(179828,10)] = 54.6405 ;
	lumiByLS[make_pair(179828,11)] = 54.5495 ;
	lumiByLS[make_pair(179828,12)] = 54.5409 ;
	lumiByLS[make_pair(179828,13)] = 54.5161 ;
	lumiByLS[make_pair(179828,14)] = 54.5075 ;
	lumiByLS[make_pair(179828,15)] = 54.4970 ;
	lumiByLS[make_pair(179828,16)] = 54.4747 ;
	lumiByLS[make_pair(179828,17)] = 54.3852 ;
	lumiByLS[make_pair(179828,18)] = 54.3597 ;
	lumiByLS[make_pair(179828,19)] = 54.3280 ;
	lumiByLS[make_pair(179828,20)] = 54.2806 ;
	lumiByLS[make_pair(179828,21)] = 54.2385 ;
	lumiByLS[make_pair(179828,22)] = 54.2137 ;
	lumiByLS[make_pair(179828,23)] = 54.1893 ;
	lumiByLS[make_pair(179828,24)] = 54.1781 ;
	lumiByLS[make_pair(179828,25)] = 54.1224 ;
	lumiByLS[make_pair(179828,26)] = 54.0704 ;
	lumiByLS[make_pair(179828,27)] = 54.0254 ;
	lumiByLS[make_pair(179828,28)] = 54.0172 ;
	lumiByLS[make_pair(179828,29)] = 53.9870 ;
	lumiByLS[make_pair(179828,30)] = 53.9576 ;
	lumiByLS[make_pair(179828,31)] = 53.9270 ;
	lumiByLS[make_pair(179828,32)] = 53.8398 ;
	lumiByLS[make_pair(179828,33)] = 53.8606 ;
	lumiByLS[make_pair(179828,34)] = 53.7935 ;
	lumiByLS[make_pair(179828,35)] = 53.8050 ;
	lumiByLS[make_pair(179828,36)] = 53.7691 ;
	lumiByLS[make_pair(179828,37)] = 53.7116 ;
	lumiByLS[make_pair(179828,38)] = 53.7124 ;
	lumiByLS[make_pair(179828,39)] = 53.6485 ;
	lumiByLS[make_pair(179828,40)] = 53.6406 ;
	lumiByLS[make_pair(179828,41)] = 53.5882 ;
	lumiByLS[make_pair(179828,42)] = 53.5580 ;
	lumiByLS[make_pair(179828,43)] = 53.5100 ;
	lumiByLS[make_pair(179828,44)] = 53.4587 ;
	lumiByLS[make_pair(179828,45)] = 53.4483 ;
	lumiByLS[make_pair(179828,46)] = 53.4185 ;
	lumiByLS[make_pair(179828,47)] = 53.3880 ;
	lumiByLS[make_pair(179828,48)] = 53.3306 ;
	lumiByLS[make_pair(179828,49)] = 53.2980 ;
	lumiByLS[make_pair(179828,50)] = 53.2933 ;
	lumiByLS[make_pair(179828,51)] = 53.2880 ;
	lumiByLS[make_pair(179828,52)] = 53.2450 ;
	lumiByLS[make_pair(179828,53)] = 53.1442 ;
	lumiByLS[make_pair(179828,54)] = 53.1180 ;
	lumiByLS[make_pair(179828,55)] = 53.1366 ;
	lumiByLS[make_pair(179828,56)] = 53.0632 ;
	lumiByLS[make_pair(179828,57)] = 53.0316 ;
	lumiByLS[make_pair(179828,58)] = 52.9904 ;
	lumiByLS[make_pair(179828,59)] = 52.9603 ;
	lumiByLS[make_pair(179828,60)] = 53.0133 ;
	lumiByLS[make_pair(179828,61)] = 52.9944 ;
	lumiByLS[make_pair(179828,62)] = 52.9116 ;
	lumiByLS[make_pair(179828,63)] = 52.8536 ;
	lumiByLS[make_pair(179828,64)] = 52.7901 ;
	lumiByLS[make_pair(179828,65)] = 52.7768 ;
	lumiByLS[make_pair(179828,66)] = 52.7501 ;
	lumiByLS[make_pair(179828,67)] = 52.7246 ;
	lumiByLS[make_pair(179828,68)] = 52.6834 ;
	lumiByLS[make_pair(179828,69)] = 52.6637 ;
	lumiByLS[make_pair(179828,70)] = 52.5899 ;
	lumiByLS[make_pair(179828,71)] = 52.6104 ;
	lumiByLS[make_pair(179828,72)] = 52.5563 ;
	lumiByLS[make_pair(179828,73)] = 52.4965 ;
	lumiByLS[make_pair(179828,74)] = 52.4220 ;
	lumiByLS[make_pair(179828,75)] = 52.4339 ;
	lumiByLS[make_pair(179828,76)] = 52.3998 ;
	lumiByLS[make_pair(179828,77)] = 52.5098 ;
	lumiByLS[make_pair(179828,78)] = 52.4535 ;
	lumiByLS[make_pair(179828,79)] = 52.4181 ;
	lumiByLS[make_pair(179828,80)] = 52.3289 ;
	lumiByLS[make_pair(179828,81)] = 52.3114 ;
	lumiByLS[make_pair(179828,82)] = 52.3089 ;
	lumiByLS[make_pair(179828,83)] = 52.3458 ;
	lumiByLS[make_pair(179828,84)] = 52.3580 ;
	lumiByLS[make_pair(179828,85)] = 52.2488 ;
	lumiByLS[make_pair(179828,86)] = 52.2499 ;
	lumiByLS[make_pair(179828,87)] = 52.2381 ;
	lumiByLS[make_pair(179828,88)] = 52.2688 ;
	lumiByLS[make_pair(179828,89)] = 52.2607 ;
	lumiByLS[make_pair(179828,90)] = 52.2234 ;
	lumiByLS[make_pair(179828,91)] = 52.2195 ;
	lumiByLS[make_pair(179828,92)] = 52.2202 ;
	lumiByLS[make_pair(179828,93)] = 52.2277 ;
	lumiByLS[make_pair(179828,94)] = 52.2252 ;
	lumiByLS[make_pair(179828,95)] = 52.2101 ;
	lumiByLS[make_pair(179828,96)] = 52.1676 ;
	lumiByLS[make_pair(179828,97)] = 52.1586 ;
	lumiByLS[make_pair(179828,98)] = 52.1172 ;
	lumiByLS[make_pair(179828,99)] = 52.1039 ;
	lumiByLS[make_pair(179828,100)] = 52.0502 ;
	lumiByLS[make_pair(179828,101)] = 52.0542 ;
	lumiByLS[make_pair(179828,102)] = 52.0198 ;
	lumiByLS[make_pair(179828,103)] = 51.9662 ;
	lumiByLS[make_pair(179828,104)] = 51.9322 ;
	lumiByLS[make_pair(179828,105)] = 51.9057 ;
	lumiByLS[make_pair(179828,106)] = 51.8410 ;
	lumiByLS[make_pair(179828,107)] = 51.8238 ;
	lumiByLS[make_pair(179828,108)] = 51.7784 ;
	lumiByLS[make_pair(179828,109)] = 51.7709 ;
	lumiByLS[make_pair(179828,110)] = 51.7455 ;
	lumiByLS[make_pair(179828,111)] = 51.6970 ;
	lumiByLS[make_pair(179828,112)] = 51.6508 ;
	lumiByLS[make_pair(179828,113)] = 51.6165 ;
	lumiByLS[make_pair(179828,114)] = 51.5772 ;
	lumiByLS[make_pair(179828,115)] = 51.5197 ;
	lumiByLS[make_pair(179828,116)] = 51.4693 ;
	lumiByLS[make_pair(179828,117)] = 51.5018 ;
	lumiByLS[make_pair(179828,118)] = 51.4497 ;
	lumiByLS[make_pair(179828,119)] = 51.4028 ;
	lumiByLS[make_pair(179828,120)] = 51.4118 ;
	lumiByLS[make_pair(179828,121)] = 51.3885 ;
	lumiByLS[make_pair(179828,122)] = 51.3499 ;
	lumiByLS[make_pair(179828,123)] = 51.3736 ;
	lumiByLS[make_pair(179828,124)] = 51.3156 ;
	lumiByLS[make_pair(179828,125)] = 51.3342 ;
	lumiByLS[make_pair(179828,126)] = 51.2671 ;
	lumiByLS[make_pair(179828,127)] = 51.3357 ;
	lumiByLS[make_pair(179828,128)] = 51.2935 ;
	lumiByLS[make_pair(179828,129)] = 51.3100 ;
	lumiByLS[make_pair(179828,130)] = 51.3178 ;
	lumiByLS[make_pair(179828,131)] = 51.2642 ;
	lumiByLS[make_pair(179828,132)] = 51.2392 ;
	lumiByLS[make_pair(179828,133)] = 51.2414 ;
	lumiByLS[make_pair(179828,134)] = 51.2231 ;
	lumiByLS[make_pair(179828,135)] = 51.1359 ;
	lumiByLS[make_pair(179828,136)] = 51.1099 ;
	lumiByLS[make_pair(179828,137)] = 51.1317 ;
	lumiByLS[make_pair(179828,138)] = 51.0974 ;
	lumiByLS[make_pair(179828,139)] = 51.0581 ;
	lumiByLS[make_pair(179828,140)] = 50.9832 ;
	lumiByLS[make_pair(179828,141)] = 50.9560 ;
	lumiByLS[make_pair(179828,142)] = 51.0360 ;
	lumiByLS[make_pair(179828,143)] = 51.0199 ;
	lumiByLS[make_pair(179828,144)] = 50.9303 ;
	lumiByLS[make_pair(179828,145)] = 50.8878 ;
	lumiByLS[make_pair(179828,146)] = 50.8339 ;
	lumiByLS[make_pair(179828,147)] = 50.8603 ;
	lumiByLS[make_pair(179828,148)] = 50.8289 ;
	lumiByLS[make_pair(179828,149)] = 50.7754 ;
	lumiByLS[make_pair(179828,150)] = 50.7835 ;
	lumiByLS[make_pair(179828,151)] = 50.7625 ;
	lumiByLS[make_pair(179828,152)] = 50.7468 ;
	lumiByLS[make_pair(179828,153)] = 50.7011 ;
	lumiByLS[make_pair(179828,154)] = 50.6411 ;
	lumiByLS[make_pair(179828,155)] = 50.5994 ;
	lumiByLS[make_pair(179828,156)] = 50.6094 ;
	lumiByLS[make_pair(179828,157)] = 50.5880 ;
	lumiByLS[make_pair(179828,158)] = 50.5976 ;
	lumiByLS[make_pair(179828,159)] = 50.5384 ;
	lumiByLS[make_pair(179828,160)] = 50.4988 ;
	lumiByLS[make_pair(179828,161)] = 50.4035 ;
	lumiByLS[make_pair(179828,162)] = 50.3653 ;
	lumiByLS[make_pair(179828,163)] = 50.3571 ;
	lumiByLS[make_pair(179828,164)] = 50.4031 ;
	lumiByLS[make_pair(179828,165)] = 50.3382 ;
	lumiByLS[make_pair(179828,167)] = 50.2380 ;
	lumiByLS[make_pair(179828,172)] = 50.1306 ;
	lumiByLS[make_pair(179828,173)] = 50.1053 ;
	lumiByLS[make_pair(179828,174)] = 50.0974 ;
	lumiByLS[make_pair(179828,175)] = 50.0661 ;
	lumiByLS[make_pair(179828,176)] = 50.0026 ;
	lumiByLS[make_pair(179828,177)] = 49.9966 ;
	lumiByLS[make_pair(179828,178)] = 46.9718 ;
	lumiByLS[make_pair(179828,179)] = 39.5286 ;
	lumiByLS[make_pair(179828,180)] = 39.5203 ;
	lumiByLS[make_pair(179828,181)] = 39.5891 ;
	lumiByLS[make_pair(179828,182)] = 39.6120 ;
	lumiByLS[make_pair(179828,183)] = 39.6568 ;
	lumiByLS[make_pair(179828,184)] = 39.6403 ;
	lumiByLS[make_pair(179828,185)] = 39.6650 ;
	lumiByLS[make_pair(179828,186)] = 39.6453 ;
	lumiByLS[make_pair(179828,187)] = 39.6589 ;
	lumiByLS[make_pair(179828,188)] = 39.6497 ;
	lumiByLS[make_pair(179828,189)] = 39.6192 ;
	lumiByLS[make_pair(179828,190)] = 35.1804 ;
	lumiByLS[make_pair(179828,191)] = 29.3999 ;
	lumiByLS[make_pair(179828,192)] = 29.4256 ;
	lumiByLS[make_pair(179828,193)] = 29.4260 ;
	lumiByLS[make_pair(179828,194)] = 29.4412 ;
	lumiByLS[make_pair(179828,195)] = 29.4234 ;
	lumiByLS[make_pair(179828,196)] = 29.4799 ;
	lumiByLS[make_pair(179828,197)] = 29.4814 ;
	lumiByLS[make_pair(179828,198)] = 29.4844 ;
	lumiByLS[make_pair(179828,199)] = 29.5060 ;
	lumiByLS[make_pair(179828,200)] = 29.5168 ;
	lumiByLS[make_pair(179828,201)] = 29.5045 ;
	lumiByLS[make_pair(179828,202)] = 26.6365 ;
	lumiByLS[make_pair(179828,203)] = 25.8239 ;
	lumiByLS[make_pair(179828,204)] = 23.2267 ;
	lumiByLS[make_pair(179828,205)] = 22.3233 ;
	lumiByLS[make_pair(179828,206)] = 22.3440 ;
	lumiByLS[make_pair(179828,207)] = 22.3674 ;
	lumiByLS[make_pair(179828,208)] = 22.3365 ;
	lumiByLS[make_pair(179828,209)] = 22.3767 ;
	lumiByLS[make_pair(179828,210)] = 22.3490 ;
	lumiByLS[make_pair(179828,211)] = 22.3911 ;
	lumiByLS[make_pair(179828,212)] = 22.4172 ;
	lumiByLS[make_pair(179828,213)] = 22.3880 ;
	lumiByLS[make_pair(179828,214)] = 21.6298 ;
	lumiByLS[make_pair(179828,215)] = 19.4799 ;
	lumiByLS[make_pair(179828,216)] = 16.7757 ;
	lumiByLS[make_pair(179828,217)] = 16.6903 ;
	lumiByLS[make_pair(179828,218)] = 16.7128 ;
	lumiByLS[make_pair(179828,219)] = 16.6976 ;
	lumiByLS[make_pair(179828,220)] = 16.6776 ;
	lumiByLS[make_pair(179828,221)] = 16.6939 ;
	lumiByLS[make_pair(179828,222)] = 16.6714 ;
	lumiByLS[make_pair(179828,223)] = 16.6420 ;
	lumiByLS[make_pair(179828,224)] = 16.6506 ;
	lumiByLS[make_pair(179828,225)] = 16.6372 ;
	lumiByLS[make_pair(179828,226)] = 16.6539 ;
	lumiByLS[make_pair(179828,227)] = 14.9830 ;
	lumiByLS[make_pair(179828,228)] = 12.2951 ;
	lumiByLS[make_pair(179828,229)] = 11.3243 ;
	lumiByLS[make_pair(179828,230)] = 10.5401 ;
	lumiByLS[make_pair(179828,231)] = 10.1644 ;
	lumiByLS[make_pair(179828,232)] = 10.0039 ;
	lumiByLS[make_pair(179828,233)] = 10.0405 ;
	lumiByLS[make_pair(179828,234)] = 10.0305 ;
	lumiByLS[make_pair(179828,235)] = 10.0597 ;
	lumiByLS[make_pair(179828,236)] = 10.0645 ;
	lumiByLS[make_pair(179828,237)] = 10.0602 ;
	lumiByLS[make_pair(179828,238)] = 10.0794 ;
	lumiByLS[make_pair(179828,239)] = 10.0916 ;
	lumiByLS[make_pair(179828,240)] = 33.3675 ;
	lumiByLS[make_pair(179828,241)] = 47.7500 ;
	lumiByLS[make_pair(179828,242)] = 47.9247 ;
	lumiByLS[make_pair(179828,243)] = 47.8770 ;
	lumiByLS[make_pair(179828,244)] = 47.8657 ;
	lumiByLS[make_pair(179828,245)] = 46.5679 ;
	lumiByLS[make_pair(179828,246)] = 45.9938 ;
	lumiByLS[make_pair(179828,247)] = 45.7114 ;
	lumiByLS[make_pair(179828,248)] = 47.8778 ;
	lumiByLS[make_pair(179828,249)] = 47.6610 ;
	lumiByLS[make_pair(179828,250)] = 47.6101 ;
	lumiByLS[make_pair(179828,252)] = 47.6030 ;
	lumiByLS[make_pair(179828,253)] = 47.4778 ;
	lumiByLS[make_pair(179828,254)] = 47.4261 ;
	lumiByLS[make_pair(179828,255)] = 47.3778 ;
	lumiByLS[make_pair(179828,256)] = 47.3949 ;
	lumiByLS[make_pair(179828,257)] = 47.3525 ;
	lumiByLS[make_pair(179828,258)] = 47.3368 ;
	lumiByLS[make_pair(179828,259)] = 47.2643 ;
	lumiByLS[make_pair(179828,260)] = 47.3191 ;
	lumiByLS[make_pair(179828,261)] = 47.3447 ;
	lumiByLS[make_pair(179828,262)] = 47.2945 ;
	lumiByLS[make_pair(179828,263)] = 47.2639 ;
	lumiByLS[make_pair(179828,264)] = 47.3105 ;
	lumiByLS[make_pair(179828,265)] = 47.2998 ;
	lumiByLS[make_pair(179828,266)] = 47.3318 ;
	lumiByLS[make_pair(179828,267)] = 47.2909 ;
	lumiByLS[make_pair(179828,268)] = 47.2600 ;
	lumiByLS[make_pair(179828,269)] = 47.2614 ;
	lumiByLS[make_pair(179828,270)] = 47.2379 ;
	lumiByLS[make_pair(179828,271)] = 47.2657 ;
	lumiByLS[make_pair(179828,272)] = 47.2401 ;
	lumiByLS[make_pair(179828,273)] = 47.2454 ;
	lumiByLS[make_pair(179828,274)] = 47.1646 ;
	lumiByLS[make_pair(179828,275)] = 47.1831 ;
	lumiByLS[make_pair(179828,276)] = 47.1916 ;
	lumiByLS[make_pair(179828,277)] = 47.1155 ;
	lumiByLS[make_pair(179828,278)] = 47.0821 ;
	lumiByLS[make_pair(179828,279)] = 47.0302 ;
	lumiByLS[make_pair(179828,280)] = 47.0344 ;
	lumiByLS[make_pair(179828,281)] = 47.0426 ;
	lumiByLS[make_pair(179828,282)] = 47.0030 ;
	lumiByLS[make_pair(179828,283)] = 46.9889 ;
	lumiByLS[make_pair(179828,284)] = 46.9397 ;
	lumiByLS[make_pair(179828,285)] = 46.9163 ;
	lumiByLS[make_pair(179828,286)] = 46.9526 ;
	lumiByLS[make_pair(179828,287)] = 46.9344 ;
	lumiByLS[make_pair(179828,288)] = 46.8038 ;
	lumiByLS[make_pair(179828,289)] = 46.7939 ;
	lumiByLS[make_pair(179828,290)] = 46.8174 ;
	lumiByLS[make_pair(179828,291)] = 46.8053 ;
	lumiByLS[make_pair(179828,292)] = 46.7622 ;
	lumiByLS[make_pair(179828,293)] = 46.7430 ;
	lumiByLS[make_pair(179828,294)] = 46.7341 ;
	lumiByLS[make_pair(179828,295)] = 46.6747 ;
	lumiByLS[make_pair(179828,296)] = 46.7394 ;
	lumiByLS[make_pair(179828,297)] = 46.7127 ;
	lumiByLS[make_pair(179828,298)] = 46.7146 ;
	lumiByLS[make_pair(179828,299)] = 46.6423 ;
	lumiByLS[make_pair(179828,300)] = 46.5775 ;
	lumiByLS[make_pair(179828,301)] = 46.6117 ;
	lumiByLS[make_pair(179828,302)] = 46.6463 ;
	lumiByLS[make_pair(179828,303)] = 46.5761 ;
	lumiByLS[make_pair(179828,304)] = 46.6281 ;
	lumiByLS[make_pair(179828,305)] = 46.5840 ;
	lumiByLS[make_pair(179828,306)] = 46.4114 ;
	lumiByLS[make_pair(179828,307)] = 46.2635 ;
	lumiByLS[make_pair(179828,308)] = 46.2891 ;
	lumiByLS[make_pair(179828,309)] = 46.3552 ;
	lumiByLS[make_pair(179828,310)] = 46.2830 ;
	lumiByLS[make_pair(179828,311)] = 46.4299 ;
	lumiByLS[make_pair(179828,312)] = 46.3363 ;
	lumiByLS[make_pair(179828,313)] = 46.3805 ;
	lumiByLS[make_pair(179828,314)] = 46.3374 ;
	lumiByLS[make_pair(179828,315)] = 46.3673 ;
	lumiByLS[make_pair(179828,316)] = 46.2844 ;
	lumiByLS[make_pair(179828,317)] = 46.2570 ;
	lumiByLS[make_pair(179828,318)] = 46.2663 ;
	lumiByLS[make_pair(179828,319)] = 46.2624 ;
	lumiByLS[make_pair(179828,320)] = 46.1887 ;
	lumiByLS[make_pair(179828,321)] = 15.2724 ;
	lumiByLS[make_pair(179828,322)] = 6.6818 ;
	lumiByLS[make_pair(179828,323)] = 5.3524 ;
	lumiByLS[make_pair(179828,324)] = 5.3083 ;
	lumiByLS[make_pair(179828,325)] = 5.3179 ;
	lumiByLS[make_pair(179828,326)] = 5.3276 ;
	lumiByLS[make_pair(179828,327)] = 5.3069 ;
	lumiByLS[make_pair(179828,328)] = 5.2885 ;
	lumiByLS[make_pair(179828,329)] = 5.3018 ;
	lumiByLS[make_pair(179828,330)] = 5.3051 ;
	lumiByLS[make_pair(179828,331)] = 5.3253 ;
	lumiByLS[make_pair(179828,332)] = 5.3464 ;
	lumiByLS[make_pair(179828,333)] = 4.9309 ;
	lumiByLS[make_pair(179828,334)] = 4.1088 ;
	lumiByLS[make_pair(179828,335)] = 2.5439 ;
	lumiByLS[make_pair(179828,336)] = 1.7164 ;
	lumiByLS[make_pair(179828,337)] = 1.2803 ;
	lumiByLS[make_pair(179828,338)] = 1.1708 ;
	lumiByLS[make_pair(179828,339)] = 1.1761 ;
	lumiByLS[make_pair(179828,340)] = 1.1785 ;
	lumiByLS[make_pair(179828,341)] = 1.1785 ;
	lumiByLS[make_pair(179828,342)] = 1.1775 ;
	lumiByLS[make_pair(179828,343)] = 1.1785 ;
	lumiByLS[make_pair(179828,344)] = 1.1814 ;
	lumiByLS[make_pair(179828,345)] = 1.1785 ;
	lumiByLS[make_pair(179828,346)] = 1.1795 ;
	lumiByLS[make_pair(179828,347)] = 1.1790 ;
	lumiByLS[make_pair(179828,348)] = 1.1804 ;
	lumiByLS[make_pair(179828,349)] = 1.1785 ;
	lumiByLS[make_pair(179828,350)] = 1.1829 ;
	lumiByLS[make_pair(179828,351)] = 1.1795 ;
	lumiByLS[make_pair(179828,352)] = 1.1780 ;
	lumiByLS[make_pair(179828,353)] = 1.1761 ;
	lumiByLS[make_pair(179828,354)] = 1.1722 ;
	lumiByLS[make_pair(179828,355)] = 1.1780 ;
	lumiByLS[make_pair(179828,356)] = 1.1833 ;
	lumiByLS[make_pair(179828,357)] = 1.1795 ;
	lumiByLS[make_pair(179828,358)] = 1.1761 ;
	lumiByLS[make_pair(179828,359)] = 1.1795 ;
	lumiByLS[make_pair(179828,360)] = 1.1819 ;
	lumiByLS[make_pair(179828,361)] = 1.1804 ;
	lumiByLS[make_pair(179828,362)] = 1.1843 ;
	lumiByLS[make_pair(179828,363)] = 1.1867 ;
	lumiByLS[make_pair(179828,364)] = 1.1829 ;
	lumiByLS[make_pair(179828,365)] = 1.3126 ;
	lumiByLS[make_pair(179828,366)] = 38.5215 ;
	lumiByLS[make_pair(179828,367)] = 43.0791 ;
	lumiByLS[make_pair(179828,368)] = 43.8156 ;
	lumiByLS[make_pair(179828,369)] = 43.8185 ;
	lumiByLS[make_pair(179828,370)] = 43.8046 ;
	lumiByLS[make_pair(179828,371)] = 41.9208 ;
	lumiByLS[make_pair(179828,372)] = 43.1493 ;
	lumiByLS[make_pair(179828,373)] = 41.3506 ;
	lumiByLS[make_pair(179828,374)] = 44.1204 ;
	lumiByLS[make_pair(179828,375)] = 43.6871 ;
	lumiByLS[make_pair(179828,376)] = 43.6661 ;
	lumiByLS[make_pair(179828,377)] = 43.9538 ;
	lumiByLS[make_pair(179828,378)] = 43.6985 ;
	lumiByLS[make_pair(179828,379)] = 43.6586 ;
	lumiByLS[make_pair(179828,380)] = 43.7811 ;
	lumiByLS[make_pair(179828,381)] = 43.7822 ;
	lumiByLS[make_pair(179828,382)] = 43.7586 ;
	lumiByLS[make_pair(179828,383)] = 43.7658 ;
	lumiByLS[make_pair(179828,384)] = 43.6974 ;
	lumiByLS[make_pair(179828,385)] = 43.6775 ;
	lumiByLS[make_pair(179828,386)] = 43.6932 ;
	lumiByLS[make_pair(179828,387)] = 43.6480 ;
	lumiByLS[make_pair(179828,388)] = 43.6266 ;
	lumiByLS[make_pair(179828,389)] = 43.6112 ;
	lumiByLS[make_pair(179828,390)] = 43.6505 ;
	lumiByLS[make_pair(179828,391)] = 43.6626 ;
	lumiByLS[make_pair(179828,392)] = 43.6550 ;
	lumiByLS[make_pair(179828,393)] = 43.6646 ;
	lumiByLS[make_pair(179828,394)] = 43.6714 ;
	lumiByLS[make_pair(179828,395)] = 43.6679 ;
	lumiByLS[make_pair(179828,396)] = 43.6490 ;
	lumiByLS[make_pair(179828,397)] = 43.6127 ;
	lumiByLS[make_pair(179828,398)] = 43.5745 ;
	lumiByLS[make_pair(179828,399)] = 43.5750 ;
	lumiByLS[make_pair(179828,400)] = 43.5610 ;
	lumiByLS[make_pair(179828,401)] = 43.5439 ;
	lumiByLS[make_pair(179828,404)] = 43.4471 ;
	lumiByLS[make_pair(179828,405)] = 43.3755 ;
	lumiByLS[make_pair(179828,406)] = 43.3520 ;
	lumiByLS[make_pair(179828,407)] = 43.3374 ;
	lumiByLS[make_pair(179828,408)] = 43.3712 ;


	return lumiByLS[make_pair(runNumber,LS)];
}
