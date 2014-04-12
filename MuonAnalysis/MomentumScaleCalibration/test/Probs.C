#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TH2D.h>
#include <TF1.h>
#include <TProfile.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <TCanvas.h>
#include <TPostScript.h>
#include <TLine.h>
#include <TPaveText.h>
#include <TPad.h>
#include <TStyle.h>
#include <TMath.h>
#include <TChain.h>
#include <TProfile2D.h>
#include <TLegend.h>
#include <TROOT.h>
#include <TMath.h>

#include <iostream>
#include <algorithm>


//TF1 * GL = new TF1 ("GL", 
//	    "0.5/TMath::PI*[0]/(pow(x-[1],2)+pow(0.5*[0],2))*exp(-0.5*pow((x-[2])/[3],2))/([3]*sqrt(2*TMath::PI))", 
TF1 * GL = new TF1 ("GL", 
		    "0.5/TMath::Pi()*[0]/((x-[1])*(x-[1])+(0.5*[0])*(0.5*[0]))*exp(-0.5*(x-[2])*(x-[2])/([3]*[3]))/([3]*sqrt(2*TMath::Pi()))", 
		    0, 1000);
TF1 * L = new TF1 ("L", "0.5/TMath::Pi()*[0]/(pow(x-[1],2)+pow(0.5*[0],2))", 0, 1000);
void Probs (int nbins) {

  // double Gamma[6] = {2.4952, 0.0000934, 0.000337, 0.000054, 0.000032, 0.000020};
  double Gamma[6] = {2.4952, 0.000020, 0.000032, 0.000054, 0.000337, 0.0000934};
  double Mass[6] = {90.986, 10.3552, 10.0233, 9.4603, 3.68609, 3.0969};
  double ResHalfWidth[6] = {20., 0.5, 0.5, 0.5, 0.2, 0.2};
  double ResMaxSigma[6] = {50., 5., 5., 5., 2., 2.}; 

  double Xmin[6];
  double Xmax[6];
  double Ymin[6];
  double Ymax[6];
  for (int i=0; i<6; i++) {
    Xmin[i] = Mass[i]-ResHalfWidth[i];
    Xmax[i] = Mass[i]+ResHalfWidth[i];
    Ymin[i] = 0.;
    Ymax[i] = ResMaxSigma[i];
  }

  TH2D * I[6];
  char name[20];
  Int_t np = 20000;
  double * x = new double[np];
  double * w = new double[np];

  for (int i=0; i<6; i++) {
    sprintf (name, "GL%d", i);
    I[i] = new TH2D (name, "Gaussian x Lorentz", nbins+1, Xmin[i], Xmax[i], nbins+1, Ymin[i], Ymax[i]);

    std::cout << "Processing resonance " << i << std::endl;
    for (int ix=0; ix<=nbins/2; ix++) {
      double mass = Xmin[i]+(Xmax[i]-Xmin[i])*((double)ix)/(double)nbins;
      double sigma;
      double P;
      for (int iy=0; iy<=nbins; iy++) {
	sigma = Ymin[i]+(Ymax[i]-Ymin[i])*((double)iy)/(double)nbins;
	P = 0.;
	if (iy==0) {
	  np = 10000;
	  sigma = 0.1*(ResMaxSigma[i])/(double)nbins; //////////////////////////// NNBB approximation
	  GL->SetParameters (Gamma[i], Mass[i], mass, sigma);
	  GL->CalcGaussLegendreSamplingPoints (np, x, w, 0.1e-18);
	  P = GL->IntegralFast (np, x, w, Mass[i]-10*Gamma[i], Mass[i]+10*Gamma[i]);
	  std::cout << "For Resonance #" << i << ": mass = " << mass << ", sigma = " << sigma 
	       << ", P = " << P << std::endl;
	} else if (iy<10) {
	  np = 2000;
	  GL->SetParameters (Gamma[i], Mass[i], mass, sigma);
	  GL->CalcGaussLegendreSamplingPoints (np, x, w, 0.1e-16);
	  P = GL->IntegralFast (np, x, w, Mass[i]-10*Gamma[i], Mass[i]+10*Gamma[i]);
	  // P = GL->Integral(Mass[i]-10*Gamma[i], Mass[i]+10*Gamma[i]);
	  std::cout << "For Resonance #" << i << ": mass = " << mass << ", sigma = " << sigma 
	       << ", P = " << P << std::endl;
	} else {
	  GL->SetParameters (Gamma[i], Mass[i], mass, sigma);
	  P = GL->Integral(Mass[i]-10*Gamma[i], Mass[i]+10*Gamma[i]);
	}

	I[i]->SetBinContent(ix+1, iy+1, P);
	if (ix<nbins/2) I[i]->SetBinContent(nbins-ix+1, iy+1, P);
      }
    }
  }

  sprintf (name, "Probs_%d.root", nbins);
  TFile * Out = new TFile (name, "RECREATE");
  Out->cd();
  for (int i=0; i<6; i++) {
    I[i]->Write();
  }
  Out->Close();

}

