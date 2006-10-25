// Macro to fit Peters cal files.
// This uses A3 + A3*tanh(A0*x-A1) from Urs.
// This fits ADC vs VCAL (x-axis=VCAL) - opposite to Peter
// I have tried VCAL vs ADC with ATanH() but it crashes.
// Compress ADC range to 0-255 range to simulate FED data.
// Usage:
// .L PHCalibration.C
//          dir  roc pixel linear
// FitCurve("mtb",0,20,20,false)  - one pixel
//              dir   linear
// FitAllCurves("mtb",false)      - all pixels
// Plots()                 - show some plost

#include <iostream>
#include <TSystem.h>
#include "TF1.h"
#include "TGraph.h"
#include "TH1D.h"
#include "TFile.h"

// Conversion between high and low range 
static const double rangeConversion = 7.;  // 
// Number of VCAL point per range
static const int vcalSteps = 5;
double vcal[2*vcalSteps], vcalLow[2*vcalSteps];
double x[2*vcalSteps], y[2*vcalSteps], xErr[2*vcalSteps], yErr[2*vcalSteps];
int n;

// Number of parameters to fit
static const int nFitParams = 4;

const bool HISTO = true;

TH1D *histoFit[nFitParams], *histoChi;
TH1D *histoFit0[16],*histoFit1[16],*histoFit2[16],*histoFit3[16];

TGraph *graph;
TF1 *phFit, *phLinear;

// Calibration data points per ROC
TH2D *h2d0=0,*h2d1,*h2d2,*h2d3,*h2d4,*h2d5,*h2d6,*h2d7,*h2d8,*h2d9,
  *h2d10,*h2d11,*h2d12,*h2d13,*h2d14,*h2d15;
// Fit distribution 
TH2D *h2dp0, *h2dp1, *h2dp2, *h2dp3, *h2dchis;

//const double xCut = TMath::Pi()/2. - 0.0005;
//const double tanXCut = TMath::Tan(xCut);

//===========================================================================
// Function to fit
Double_t Fitfcn( Double_t *x, Double_t *par) {

  //if (par[0]*x[0] - par[4] > xCut) return tanXCut + (x[0] - (xCut + par[4])/par[0])* 1e8;
  //Double_t y = TMath::Tan(par[0]*x[0] - par[4]) + par[1]*x[0]*x[0]*x[0] + par[5]*x[0]*x[0] + par[2]*x[0] + par[3];

  Double_t y = par[3] + par[2] * TMath::TanH(par[0]*x[0]-par[1]);
  return y;
}
//===========================================================================
// Function to fit
Double_t FitLinear( Double_t *x, Double_t *par) {

  //if (par[0]*x[0] - par[4] > xCut) return tanXCut + (x[0] - (xCut + par[4])/par[0])* 1e8;
  //Double_t y = TMath::Tan(par[0]*x[0] - par[4]) + par[1]*x[0]*x[0]*x[0] + par[5]*x[0]*x[0] + par[2]*x[0] + par[3];

  Double_t y = par[0] + par[1] * x[0];
  return y;
}
//==========================================================================
//String with the function 
const char *Fitfcn() {
  //	return "TMath::Exp(par[1]*x[0] - par[0]) + par[2]*x[0]*x[0]*x[0] + par[3]*x[0]*x[0] + par[4]*x[0] + par[5]";
 //	return "TMath::Tan(par[0]*x[0] - par[4]) + par[1]*x[0]*x[0]*x[0] + par[5]*x[0]*x[0] + par[2]*x[0] + par[3]";
  return "par[3] + par[2] * TMath::TanH(par[0]*x[0] - par[1])";
}
//String with the function 
const char *FitLinear() {
  //	return "TMath::Exp(par[1]*x[0] - par[0]) + par[2]*x[0]*x[0]*x[0] + par[3]*x[0]*x[0] + par[4]*x[0] + par[5]";
 //	return "TMath::Tan(par[0]*x[0] - par[4]) + par[1]*x[0]*x[0]*x[0] + par[5]*x[0]*x[0] + par[2]*x[0] + par[3]";
  return "par[0] + par[1] * x[0]";
}
//===========================================================================
// Transform input ADC data to the 0-255 range
int transformADC(int adc) {
  const int offset = 330.;
  const int gain = 6;
  int temp = (adc + offset)/gain;
  if(temp<0 || temp>255) cout<<" adc wrong "<<temp<<" " <<adc<<endl;
  return temp;
}
//=========================================================================== 
// Init: test histos, fit function, errors
void Initialize(bool linear=false) {
  gROOT->SetStyle("Plain");
  gStyle->SetTitleBorderSize(0);
  gStyle->SetPalette(1,0);
  
  float p0max = 0.01, p1max=2.,p2max=1000.,p3max=1000.;
  float p0min = 0.;
  if(linear) {p0max = 100.; p0min=-100.;}
 
  histoFit[0] = new TH1D("histoFit0", "histoFit0", 100, p0min, p0max);
  histoFit[1] = new TH1D("histoFit1", "histoFit1", 200, 0.0, p1max);
  histoFit[2] = new TH1D("histoFit2", "histoFit2", 100, 0.0, p2max);
  histoFit[3] = new TH1D("histoFit3", "histoFit3", 100, 0.0, p3max);
  histoChi = new TH1D("histoChi", "histoChi", 1000, 0., 10.);

  char hiname[20];
  
  for(int i=0;i<16;i++) {
    sprintf(hiname, "histoFit0_%i", i);
    histoFit0[i] = new TH1D(hiname,hiname, 100, p0min, p0max);
    sprintf(hiname, "histoFit1_%i", i);
    histoFit1[i] = new TH1D(hiname,hiname, 200, 0.0, p1max);
    sprintf(hiname, "histoFit2_%i", i);
    histoFit2[i] = new TH1D(hiname,hiname, 100, 0.0, p2max);
    sprintf(hiname, "histoFit3_%i", i);
    histoFit3[i] = new TH1D(hiname,hiname, 100, 0.0, p3max);
  }  

  if(HISTO) {
    const float xdmin=0., xdmax=260.;
    const int xdbins=130;
    if(h2d0!=0) delete h2d0;
    h2d0 = new TH2D("h2d0", "h2d0", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d1!=0) delete h2d1;
    h2d1 = new TH2D("h2d1", "h2d1", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d2!=0) delete h2d2;
    h2d2 = new TH2D("h2d2", "h2d2", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d3!=0) delete h2d3;
    h2d3 = new TH2D("h2d3", "h2d3", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d4!=0) delete h2d4;
    h2d4 = new TH2D("h2d4", "h2d4", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d5!=0) delete h2d5;
    h2d5 = new TH2D("h2d5", "h2d5", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d6!=0) delete h2d6;
    h2d6 = new TH2D("h2d6", "h2d6", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d7!=0) delete h2d7;
    h2d7 = new TH2D("h2d7", "h2d7", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d8!=0) delete h2d8;
    h2d8 = new TH2D("h2d8", "h2d8", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d9!=0) delete h2d9;
    h2d9 = new TH2D("h2d9", "h2d9", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d10!=0) delete h2d10;
    h2d10 = new TH2D("h2d10", "h2d10", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d11!=0) delete h2d11;
    h2d11 = new TH2D("h2d11", "h2d11", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d12!=0) delete h2d12;
    h2d12 = new TH2D("h2d12", "h2d12", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d13!=0) delete h2d13;
    h2d13 = new TH2D("h2d13", "h2d13", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d14!=0) delete h2d14;
    h2d14 = new TH2D("h2d14", "h2d14", xdbins,xdmin,xdmax, 150, 0., 1500.);
    if(h2d15!=0) delete h2d15;
    h2d15 = new TH2D("h2d15", "h2d15", xdbins,xdmin,xdmax, 150, 0., 1500.);

    if(h2dp0!=0) delete h2dp0;
    h2dp0 = new TH2D("h2dp0", "h2dp0", 100, p0min,p0max, 16, 0., 16.);
    if(h2dp1!=0) delete h2dp1;
    h2dp1 = new TH2D("h2dp1", "h2dp1", 100, 0.,p1max, 16, 0., 16.);
    if(h2dp2!=0) delete h2dp2;
    h2dp2 = new TH2D("h2dp2", "h2dp2", 100, 0.,p2max, 16, 0., 16.);
    if(h2dp3!=0) delete h2dp3;
    h2dp3 = new TH2D("h2dp3", "h2dp3", 100, 0.,p3max, 16, 0., 16.);

    if(h2dchis!=0) delete h2dchis;
    h2dchis = new TH2D("h2dchis", "h2dchis", 200, 0., 2000., 16, 0., 16.);

  }

  // Init fit function
  //phFit = new TF1("phFit", Fitfcn, -400., 1200., nFitParams);
  phFit = new TF1("phFit", Fitfcn, 0., 1600., nFitParams);
  phLinear = new TF1("phLinear", FitLinear, 0., 1600., 2);

  phFit->SetNpx(1000);
  
  // Init errors
  for (int i = 0; i < 2*vcalSteps; i++) {
    xErr[i] = 10.;
    yErr[i] = 2.;
  }
  
  // VCAL points used for calibration
  vcal[0] = 50.;
  vcal[1] = 100.;
  vcal[2] = 150.;
  vcal[3] = 200.;
  vcal[4] = 250.;
  vcal[5] = 30.;
  vcal[6] = 50.;
  vcal[7] = 70.;
  vcal[8] = 90.;
  vcal[9] = 200.;
  // Convert always to low range
  for (int i = 0; i < 2*vcalSteps; i++) {
    vcalLow[i] = vcal[i];
    if (i > (vcalSteps - 1)) vcalLow[i]*=rangeConversion;
  }
}
//====================================================================
// Fit a pixel, linear=false to the full Fit, = true for linear fit
void Fit(bool linear) {
  bool verbose = false;
  
  if (graph) delete graph;
  //graph = new TGraphErrors(n, x, y, xErr, yErr);
  graph = new TGraphErrors(n, y, x, yErr, xErr);
  
  double xmax = 0., xmin = 9999.; 
  double ymax = 0., ymin = 9999.;
  for (int i = 0; i < n; i++) {
    if (x[i] < xmin) xmin = x[i];
    if (x[i] > xmax) xmax = x[i];
    if (y[i] < ymin) ymin = y[i];
    if (y[i] > ymax) ymax = y[i];
  }
  //phFit->SetRange(xmin, xmax);
  if(!linear) phFit->SetRange(ymin, ymax);
  else phLinear->SetRange(ymin, ymax);

  //int upperPoint = vcalSteps+2 - 1;
  //int lowerPoint = vcalSteps/3 - 1;
  //double slope;
  
  //if ( (x[upperPoint]-x[lowerPoint]) != 0 ) slope = (y[upperPoint]-y[lowerPoint])/(x[upperPoint]-x[lowerPoint]);
  //else slope = 0.5;
  
  //phFit->SetParameter(2, slope);
  //phFit->SetParameter(3, y[upperPoint] - slope*x[upperPoint]);
  
  if (n < 7 || linear) {   // Use Linear Fit
    for (int i = 0; i < 2; i++) phFit->ReleaseParameter(i);
    phFit->SetParameter(0, 0.);
    phFit->SetParameter(1, 1.);
    phFit->FixParameter(2, 0.);
    phFit->FixParameter(3, 0.);
    if(n<7 && !linear) cout<<" switch to linear "<<n<<endl;

  } else {    // Use full fit

    for (int i = 0; i < nFitParams; i++) phFit->ReleaseParameter(i);

    //double par0 = (TMath::Pi()/2. - 1.4) / x[n-1];
    // 		printf("par0 %e\n", par0);

//     phFit->SetParameter(0, par0);
//     phFit->SetParameter(1, 5.e-7);
//     phFit->FixParameter(4, -1.4);
//     if (x[upperPoint] != 0.) phFit->SetParameter(5, (y[upperPoint] - (TMath::Tan(phFit->GetParameter(0)*x[upperPoint] - phFit->GetParameter(4)) + phFit->GetParameter(1)*x[upperPoint]*x[upperPoint]*x[upperPoint] + slope*x[upperPoint] + phFit->GetParameter(3)))/(x[upperPoint]*x[upperPoint]));
//     else phFit->SetParameter(5, 0.);

     phFit->SetParameter(0, 0.003);
     phFit->SetParameter(1, 1.);
     phFit->SetParameter(2, 100.);
     phFit->SetParameter(3, 100.);

  }
  
  // 	for (int i = 0; i < 6; i++) 
  // 	{
// 		if (phFit->GetParameter(i)>0) phFit->SetParLimits(i, 0.8*phFit->GetParameter(i), 1.2*phFit->GetParameter(i));
// 		else phFit->SetParLimits(i, 1.2*phFit->GetParameter(i), 0.8*phFit->GetParameter(i));
// 	}
// 	phFit->SetParLimits(1, 0., 1.e-6);
// 	for (int i = 0; i < 6; i++) phFit->FixParameter(i, phFit->GetParameter(i));
			
  if (verbose) printf("par1 %e\n", (TMath::Pi()/2. - 1.4) / y[n-1]);
  if (verbose) printf("par4 %e\n", slope);
  if (verbose) printf("x %e y %e\n", x[upperPoint], y[upperPoint]);
  if (verbose) printf("x %e y %e\n", x[lowerPoint], y[lowerPoint]);	
  if (verbose) printf("par5 %e\n", y[upperPoint] - slope*x[upperPoint]);

  if(linear) {
    if (verbose) graph->Fit("phLinear", "R", "");
    else graph->Fit("phLinear", "RQ", "");
    for (int i = 0; i < 2; i++) 
      {histoFit[i]->Fill(phLinear->GetParameter(i));}
  } else {
    if (verbose) graph->Fit("phFit", "R", "");
    else graph->Fit("phFit", "RQ", "");
    for (int i = 0; i < nFitParams; i++) 
      {histoFit[i]->Fill(phFit->GetParameter(i));}
  }

}
//========================================================================
//Fit all pixels in all ROCs
void FitAllCurves(char *dirName, bool linear = false) {
  FILE *inputFile, *outputFile;
  char fname[1000], string[500];
  int ph[2*vcalSteps], a, b, maxRoc, maxCol, maxRow;
  double chiSquare, maxChiSquare = 0.;
  int countErrors=0;

  Initialize(linear);
  
  for (int chip = 0; chip < 16; chip++) {
    //for (int chip = 0; chip < 1; chip++) {
    printf("Fitting pulse height curves for chip %i\n", chip);
    
    sprintf(fname, "%s/phCalibration_C%i.dat", dirName, chip);
    inputFile = fopen(fname, "r");
    if (!inputFile) {
      printf("!!!!!!!!!  ----> PHCalibration: Could not open file %s to read pulse height calibration\n", fname);
      return;
    }
    
    for (int i = 0; i < 4; i++) fgets(string, 500, inputFile);
    
    //sprintf(fname, "%s/phCalibrationFit_C%i.dat", dirName, chip);
    sprintf(fname, "./phCalibrationFit_C%i.dat", chip);
    outputFile = fopen(fname, "w");
    if (!outputFile) {
      printf("!!!!!!!!!  ----> PHCalibration: Could not open file %s to write the fit results\n", fname);
      return;
    }
    fprintf(outputFile, "Parameters of the vcal vs. pulse height fits\n");
    if(!linear) fprintf(outputFile, "%s\n", Fitfcn());
    else fprintf(outputFile, "%s\n", FitLinear());
    fprintf(outputFile, "\n");
    
    // Loop over columns
    for (int iCol = 0; iCol < 52; iCol++) {
      //for (int iCol = 0; iCol < 1; iCol++) {
      printf("col %i ", iCol);
      for (int iRow = 0; iRow < 80; iRow++) { // Loop over rows
	printf(".");fflush(stdout);
	//printf("col %i row %i\n", iCol, iRow);
	n = 0;
	for (int i = 0; i < 2*vcalSteps; i++) { // Loop over VCALs
	  fscanf(inputFile, "%s", string);
	    
	  if (!linear) {  // Full fit

	    if (strcmp(string, "N/A") == 0);  // invalid point
	    else { // valid data point
	      ph[i] = atoi(string);  // ADC as interger
	      //printf("ph %i vcal %.0f\n", ph[i], vcalLow[i]);
	      //x[n] = (double)ph[i]; 
	      x[n] = (double) transformADC(ph[i]); // transform 
	      y[n] = vcalLow[i];

	      if(HISTO) {
		if(chip==0) h2d0->Fill(x[n],y[n]);
		else if(chip==1) h2d1->Fill(x[n],y[n]);
		else if(chip==2) h2d2->Fill(x[n],y[n]);
		else if(chip==3) h2d3->Fill(x[n],y[n]);
		else if(chip==4) h2d4->Fill(x[n],y[n]);
		else if(chip==5) h2d5->Fill(x[n],y[n]);
		else if(chip==6) h2d6->Fill(x[n],y[n]);
		else if(chip==7) h2d7->Fill(x[n],y[n]);
		else if(chip==8) h2d8->Fill(x[n],y[n]);
		else if(chip==9) h2d9->Fill(x[n],y[n]);
		else if(chip==10) h2d10->Fill(x[n],y[n]);
		else if(chip==11) h2d11->Fill(x[n],y[n]);
		else if(chip==12) h2d12->Fill(x[n],y[n]);
		else if(chip==13) h2d13->Fill(x[n],y[n]);
		else if(chip==14) h2d14->Fill(x[n],y[n]);
		else if(chip==15) h2d15->Fill(x[n],y[n]);
	      }

	      n++;

	    }

	  } else {  // Linear

	    //if ((strcmp(string, "N/A") == 0) || (i < 2) || (i > 2*vcalSteps - 2));
	    if ((strcmp(string, "N/A") == 0) || (i < 1) || (i > 2*vcalSteps - 3));
	    else {
	      ph[i] = atoi(string);
	      //x[n] = (double)ph[i];
	      x[n] = (double) transformADC(ph[i]); // transform 
	      y[n] = vcalLow[i];
	      //printf("ph %i vcal %.0f %f\n", ph[i], vcalLow[i],x[n]);
	      n++;
	    }
	  }
	}
	fscanf(inputFile, "%s %2i %2i", string, &a, &b);  //comment
	
	// Do the Fit
	if (n != 0) {
	  Fit(linear);  // Fit
	  // Check chisq
	  if(!linear) chiSquare = phFit->GetChisquare()/phFit->GetNDF();
	  else chiSquare = phLinear->GetChisquare()/phLinear->GetNDF();
	  histoChi->Fill(chiSquare);
	  if (chiSquare > maxChiSquare) { // Find chisq
	    maxChiSquare = chiSquare;
	    maxRoc = chip;
	    maxCol = iCol;
	    maxRow = iRow;
	  }
	  if (chiSquare > 6.) {
	    countErrors++;
	    cout<<"roc "<<chip<<" col "<<iCol<<" row "<<iRow<<" chisq "
		<<chiSquare<<" errors "<<countErrors<<endl;
	  }
	  //cout<<iCol<<" "<<iRow<<" " <<chiSquare<<endl;
	  // Save results in a file 
	  for (int i = 0; i < nFitParams; i++) {
	    if(!linear) fprintf(outputFile, "%+e ", phFit->GetParameter(i));
	    else if(i<2) fprintf(outputFile, "%+e ", phLinear->GetParameter(i));
	  }

	  if(HISTO) {
            h2dchis->Fill(chiSquare,float(chip));

	    if(!linear) {
	      h2dp0->Fill(phFit->GetParameter(0),float(chip));
	      h2dp1->Fill(phFit->GetParameter(1),float(chip));
	      h2dp2->Fill(phFit->GetParameter(2),float(chip));
	      h2dp3->Fill(phFit->GetParameter(3),float(chip));
	      
	      histoFit0[chip]->Fill(phFit->GetParameter(0));
	      histoFit1[chip]->Fill(phFit->GetParameter(1));
	      histoFit2[chip]->Fill(phFit->GetParameter(2));
	      histoFit3[chip]->Fill(phFit->GetParameter(3));

	    } else {

	      h2dp0->Fill(phLinear->GetParameter(0),float(chip));
	      h2dp1->Fill(phLinear->GetParameter(1),float(chip));
	      
	      histoFit0[chip]->Fill(phLinear->GetParameter(0));
	      histoFit1[chip]->Fill(phLinear->GetParameter(1));

	      //cout<<phLinear->GetParameter(0)<<" "
	      //  <<phLinear->GetParameter(1)<<endl;
;
	    }
          }



	} else {  // there may be dead pixels
	  for (int i = 0; i < nFitParams; i++) {
	    if(!linear || i<2) fprintf(outputFile, "%+e ", 0.);}
	}
	fprintf(outputFile, "    Pix %2i %2i\n", iCol, iRow);
      }
      printf("\n");
    }
    fclose(inputFile);
    fclose(outputFile);
  }
  printf(" %i %i %i Max ChiSquare/NDF %e\n", maxRoc, maxCol, maxRow, maxChiSquare);
}
//=====================================================================
//Fit one pixel only 
void FitCurve(char *dirName, int chip, int col, int row, 
	      bool linear = false) {

  FILE *inputFile, *outputFile;
  char fname[1000], string[1000];
  int ph[2*vcalSteps], a, b;
  double chiSquare;
  
  cout<<"0"<<endl;
  Initialize();
  
  sprintf(fname, "%s/phCalibration_C%i.dat", dirName, chip);
  inputFile = fopen(fname, "r");
  if (!inputFile) {
    printf("!!!!!!!!!  ----> PHCalibration: Could not open file %s to read pulse height calibration\n", fname);
    return;
  }

  for (int i = 0; i < 4; i++) fgets(string, 1000, inputFile);
  for (int i = 0; i < col*80+row; i++) fgets(string, 1000, inputFile);

  n = 0;
  for (int i = 0; i < 2*vcalSteps; i++) {
    fscanf(inputFile, "%s", string);
    if (!linear) {
      if (strcmp(string, "N/A") == 0);
      else {
	ph[i] = atoi(string);
	printf("ph %i vcal %.0f\n", ph[i], vcalLow[i]);
	//x[n] = (double)ph[i];
	x[n] = (double) transformADC(ph[i]);
	y[n] = vcalLow[i];
	n++;
      }
    } else {
      //if ((strcmp(string, "N/A") == 0) || (i < 2) || (i > 2*vcalSteps - 2));
      if ((strcmp(string, "N/A") == 0) || (i < 1) || (i > 2*vcalSteps - 3));
      else {
	ph[i] = atoi(string);
	printf("ph %i vcal %.0f\n", ph[i], vcalLow[i]);
	//x[n] = (double)ph[i];
	x[n] = (double) transformADC(ph[i]);
	y[n] = vcalLow[i];
	n++;
      }
    }
  }
  fscanf(inputFile, "%s %2i %2i", string, &a, &b);  //comment
  
  if (n != 0) {
      Fit(linear);

      if(!linear) chiSquare = phFit->GetChisquare()/phFit->GetNDF();
      else chiSquare = phLinear->GetChisquare()/phLinear->GetNDF();
      printf("chiSquare/NDF %e\n", chiSquare);
      graph->SetTitle("");
      graph->GetYaxis()->SetTitle("Pulse height (ADC units)");
      graph->GetYaxis()->SetRangeUser(0.,260.);
      graph->GetXaxis()->SetTitle("Vcal (DAC units)");
      graph->GetXaxis()->SetTitleOffset(1.2);
      graph->GetXaxis()->SetRangeUser(0., 1700.);
      graph->Draw("A*");

      cout<<" Fit parameter "<<endl;
      for (int i = 0; i < nFitParams; i++) {
	if(!linear) cout<<i<<" "<<phFit->GetParameter(i)<<endl;
	else if(i<2) cout<<i<<" "<<phLinear->GetParameter(i)<<endl;
      }
    }
  else printf("Error: No measured pulse height values for this pixel\n");
  
}
//=========================================================================
// Plots
void Plots() {
  TCanvas *c1 = new TCanvas();
  c1->Divide(3,2);
  
  c1->cd(1);
  histoFit[0]->Draw();
  
  c1->cd(2);
  histoFit[1]->Draw();
  
  c1->cd(3);
  histoFit[2]->Draw();
  
  c1->cd(4);
  histoFit[3]->Draw();
  
  c1->cd(6);
  histoChi->Draw();
}
