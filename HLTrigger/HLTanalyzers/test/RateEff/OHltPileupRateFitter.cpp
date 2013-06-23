#include <iostream>
#include <iomanip>
#include <fstream>
#include <TMath.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TFile.h>
#include <TString.h>

#include <TF1.h>
#include <TProfile2D.h>
#include <TProfile.h>
#include <TLegend.h>
#include <TObjArray.h>
#include <TText.h>
#include <TVectorT.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TCanvas.h>

#include "OHltRatePrinter.h"
#include "OHltTree.h"
#include "OHltPileupRateFitter.h"

using namespace std;

void OHltPileupRateFitter::fitForPileup(
					OHltConfig *thecfg,
					OHltMenu *themenu,
					vector < vector <float> > tRatePerLS,
					vector<float> tTotalRatePerLS,
					vector<double> tLumiPerLS,
					std::vector< std::vector<int> > tCountPerLS,
					std::vector<int> ttotalCountPerLS,
					TFile *histogramfile)
{
  // Individual rates, total rate, and inst lumi.
  // At this point we've already applied all prescale, normalization, and 
  // linear extrapolation factors to the rates 
  RatePerLS = tRatePerLS;
  totalRatePerLS = tTotalRatePerLS;
  LumiPerLS = tLumiPerLS;
  CountPerLS = tCountPerLS;
  totalCountPerLS = ttotalCountPerLS;

  vector <TGraphErrors*> vGraph;
  int RunLSn = RatePerLS.size();
  int nPaths = themenu->GetTriggerSize();
  double targetLumi = thecfg->iLumi/1E30;
  TString model = thecfg->nonlinearPileupFit;
  double minLumi = 999999999.0;
  double maxLumi = 0.0;

  for (int iPath=0; iPath<nPaths; iPath++)
    {
      vector <double> vRates; //temp vector containing rates
      vector <double> vRateErrors; 
      vector <double> vLumiErrors;

      for (int iLS=0; iLS<RunLSn; iLS++) {//looping over the entire set of data
	double lumi = 0;
	double rate = 0;
	double rateerr = 0;
	double lumierr = 0.0;

	// Inst lumi
	lumi = LumiPerLS[iLS];
	
	// Inst rate. Note here we've already applied the linear scale factor for the target lumi
	// So cheat and uncorrect this back to the actual online rate before fitting 
	rate = (double) (RatePerLS[iLS][iPath]) / (thecfg->lumiScaleFactor);
	rateerr = (double) rate * sqrt(CountPerLS[iLS][iPath]) / (CountPerLS[iLS][iPath]);
	lumierr = 0.0;

	vRates.push_back(rate);
	vRateErrors.push_back(rateerr);
	vLumiErrors.push_back(lumierr);

      }//end looping over the entire set of data
      
      TGraphErrors* g = new TGraphErrors(RunLSn, &LumiPerLS[0], &vRates[0], &vLumiErrors[0], &vRateErrors[0]);
      g->SetTitle(themenu->GetTriggerName(iPath));
      vGraph.push_back(g);
    }//end looping over paths
  
  // Now for total/PD rate
  vector <double> vTotalRate; //temp vector containing rates
  vector <double> vTotalRateError;
  vector <double> vLumiError;
  
  for (int iLS=0; iLS<RunLSn; iLS++) 
    {
      double lumi = 0;
      double rate = 0;
      double rateerr = 0;
      double lumierr = 0;
      
      // Inst lumi
      lumi = LumiPerLS[iLS];

      if(lumi > maxLumi)
	maxLumi = lumi;
      if(lumi < minLumi)
	minLumi = lumi;
      
      // Inst rate. Note here we've already applied the linear scale factor for the target lumi
      // So cheat and uncorrect this back to the actual online rate before fitting
      rate = (double) (totalRatePerLS[iLS]) / (thecfg->lumiScaleFactor);
      rateerr = (double) rate * sqrt(totalCountPerLS[iLS]) / (totalCountPerLS[iLS]);
      lumierr = 0.0;
      
      vTotalRate.push_back(rate);
      vTotalRateError.push_back(rateerr);
      vLumiError.push_back(lumierr);
    } //end looping over the entire set of data
     
  TGraphErrors* vTotalRateGraph = new TGraphErrors(RunLSn, &LumiPerLS[0], &vTotalRate[0], &vLumiError[0], &vTotalRateError[0]);
  vTotalRateGraph->SetTitle("Total rate");

  // Fitting w/ quadratic and cubic
  // User should check the Chi2/Ndof
  TF1* fp1     = new TF1("fp1", model, 0, 7000);

  int ix = TMath::Floor(sqrt(nPaths)); //Choose the proper canvas division
  int iy = ix;
  if (ix*iy==nPaths);
  else if (sqrt(nPaths)*(iy+1)>nPaths) ++iy;
  else {++ix; ++iy;}// end of canvas division
  TCanvas* cIndividualRateFits = new TCanvas("cIndividualRateFits","cIndividualRateFits",0,0,1200,1000);
  cIndividualRateFits->Divide(ix,iy);

  cout.setf(ios::floatfield, ios::fixed);
  cout<<setprecision(3);

  cout << "\n";
  cout << "Pileup corrected Trigger Rates [Hz], using " << model << " fit extrapolation to L=" << targetLumi << ": " << "\n";
  cout << "\t(Warning: always check fit qualities!)" << endl; 
  /*
   cout
     << "         Name                       Indiv.                                 Notes \n";
   cout
     << "----------------------------------------------------------------------------------------------\n";
  */

  // Rate per path
  for (int jPath=0; jPath<nPaths; ++jPath) 
    {//looping over paths
      cIndividualRateFits->cd(jPath+1);
      vGraph.at(jPath)->SetMarkerColor(4);
      vGraph.at(jPath)->SetMarkerStyle(20);
      vGraph.at(jPath)->Draw("ap");
      fp1->SetParLimits(2,0,1000000);
      fp1->SetParLimits(3,0,1000000);
      vGraph.at(jPath)->Fit("fp1","Q","",minLumi, maxLumi); 
      fp1->SetLineColor(2);
      fp1->DrawCopy("same");
      /*
      cout<<setw(50)<<themenu->GetTriggerName(jPath)<<" " <<setw(8)
	  <<setw(8)<<fp1->Eval(targetLumi)<<"  " <<setw(8);
      double pchi2 = TMath::Prob(fp1->GetChisquare(),fp1->GetNDF());
      if(pchi2>0.01)
	cout<<endl;
      else
	if((fp1->GetNDF())>0)
	  cout << " chi2/ndof = " << fp1->GetChisquare() << "/" << fp1->GetNDF() << endl;
	else
	  cout << " chi2/ndof = 0/0" << endl;
      */
    }

  // Total rate
  TCanvas* cTotalRateFit = new TCanvas("cTotalRateFit","cTotalRateFit",0,0,6000,1000);
  vTotalRateGraph->SetMarkerColor(4);
  vTotalRateGraph->SetMarkerStyle(20);
  vTotalRateGraph->Draw("ap");
  fp1->SetParLimits(3,0.000000001,0.1);
  //  vTotalRateGraph->Fit("fp1","Q","",minLumi, maxLumi);
  vTotalRateGraph->Fit("fp1","QR","",minLumi, maxLumi);
  fp1->SetLineColor(2);
  fp1->DrawCopy("same");
  cout << "\n";
  cout << setw(60) << "TOTAL RATE : " << setw(5) << fp1->Eval(targetLumi) << " Hz";

  double pchi2total = TMath::Prob(fp1->GetChisquare(),fp1->GetNDF());
  if(pchi2total>0.01)
    cout<<endl;
  else
    cout << " chi2/ndof = " << fp1->GetChisquare() << "/" << fp1->GetNDF() << endl;

   cout
     << "----------------------------------------------------------------------------------------------\n";

 
  cIndividualRateFits->Write();
  cTotalRateFit->Write();
}

