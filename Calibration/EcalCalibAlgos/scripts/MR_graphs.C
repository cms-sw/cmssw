// graph residual miscalibration vs eta (barrel)
// graph residual miscalibration vs ring number (barrel)
// graph residual miscalibration vs eta (endcap)
// graph residual miscalibration vs ring number (endcap)


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "TH1F.h"
#include "TCanvas"
#include <string>
#include "TH1D.h"

using std::cout;


// Define macro

void MR_graphs()
{

  //****************open file*************************

  // open file produced by CSA07 calibration exercise 
  // check path !
 
  TFile *inputfile =  TFile::Open ("/home/gozzelin/Desktop/codiceCSA07/PhiSymmetryCalibration_miscal_resid.root");


  if (inputfile->IsOpen()==false)
    {
      cerr << " File doesn't exist " << endl;
      return;
    }


  //*************************************************************
  //-------------------------------BARREL--------------------
  //*************************************************************

  //variables, vectors

  Int_t n = 85; // total barrel ring 

  Double_t eta_EB[85];
  Double_t MR_EB[85];
  Double_t ring_EB[85];

  for (Int_t i = 0; i < n; i++)
    {
      eta_EB[i]= 0.01765 *(i+1);// eta medium of ring (barrel)
      ring_EB[i] = i ;// ring number (barrel)

      // conversion int i in string

      std::stringstream ss_EB;
      std::string str_EB;
      ss_EB << (i+1);
      ss_EB >> str_EB;
      string histoEB = "mr_barl_" + str_EB + ";1";
      
      // cout << histoEB <<endl;
      
      // to catch histogram. Get function has char* argument!!!
      TH1D* mygrafhEB = (TH1D*) inputfile->Get(histoEB.c_str());
      
      MR_EB[i] = mygrafhEB -> GetRMS()*100.0;
      
    }// end for

  // fills graphs

  TGraph *mr_eta_EB = new TGraph(n,eta_EB,MR_EB);

  string titleetaEB = "residual miscalibration(%) versus eta (barrel)";
  mr_eta_EB -> SetTitle (titleetaEB.c_str());
  mr_eta_EB -> GetXaxis() -> SetTitle ("eta (rad)");
  mr_eta_EB -> GetYaxis() -> SetTitle ("residual miscalibration(%)");

  TGraph *mr_ring_EB = new TGraph(n,ring_EB,MR_EB);

  string titleringEB = "residual miscalibration(%) versus ring number (barrel)";
  mr_ring_EB -> SetTitle (titleringEB.c_str());
  mr_ring_EB -> GetXaxis() -> SetTitle (" # ring");
  mr_ring_EB -> GetYaxis() -> SetTitle ("residual miscalibration(%)");
  
  //********************************************************************************
  //------------------------ENDCAP-------------------------------------------------
  //*******************************************************************************

  //variables, vectors

  Int_t m = 39; // total endcap ring 

  // initial eta of ring
  Double_t eta_EE[39]= {1.479,1.51616,1.53608,1.55642,1.57718,1.59707,1.6175,1.63981,1.66264,1.68602,1.70885,1.73235,1.75766,1.78365,1.81034,1.83690,1.86432,1.89352,1.92362,1.95467,1.98613,2.01874,2.05318,2.08887,2.12591,2.16410,2.220393,2.24583,2.28959,2.33536,2.38347,2.43412,2.48748,2.54388,2.60367,2.66795,2.73671,2.80999,2.8891};
 
  Double_t MR_EE[39];
  Double_t ring_EE[39];
 

  for (Int_t k = 0; k < m; k++)
    {

      ring_EE[k] = k;// ring number (endcap)

      // conversion int k in string

      std::stringstream ss_EE;
      std::string str_EE;
      ss_EE << (k+1);
      ss_EE >> str_EE;
      string histoEE = "mr_endc_" + str_EE + ";1";

      // cout << histoEE <<endl;

      // to catch histogram. Get function has char* argument!!!
      TH1D* mygrafhEE = (TH1D*) inputfile->Get(histoEE.c_str());
   
      MR_EE[k] = mygrafhEE -> GetRMS()*100.0;

    }// end for

  // fills graphs

  TGraph *mr_eta_EE = new TGraph(m,eta_EE,MR_EE);

  string titleetaEE = "residual miscalibration(%) versus eta (endcap)";
  mr_eta_EE -> SetTitle (titleetaEE.c_str());
  mr_eta_EE -> GetXaxis() -> SetTitle ("eta (rad)");
  mr_eta_EE -> GetYaxis() -> SetTitle ("residual miscalibration(%)");

  TGraph *mr_ring_EE = new TGraph(m,ring_EE,MR_EE);

  string titleringEE = "residual miscalibration(%) versus ring number (endcap)";
  mr_ring_EE -> SetTitle (titleringEE.c_str());
  mr_ring_EE -> GetXaxis() -> SetTitle (" # ring");
  mr_ring_EE -> GetYaxis() -> SetTitle ("residual miscalibration(%)");
 

  //********************************* screen output *****************************************

  // open graphic window (4 sectors)
  TCanvas *c1 = new TCanvas("residual miscalibration","mr(%)",200,10,600,400);
  c1 -> Divide(2,2);
 
  // draws graphs barrel

  c1 -> cd(1);
  mr_eta_EB -> Draw("A*");

  c1 -> cd(2);
  mr_ring_EB -> Draw("A*");


  // draws graphs endcap

  c1 -> cd(3);
  mr_eta_EE -> Draw("A*");

  c1 -> cd(4);
  mr_ring_EE -> Draw("A*");


  //**************************** root output file **************************************

  // save canvas and graphs

  c1 -> Print("residual miscalibration.gif");
  c1 -> Print("residual miscalibration.eps");

  mr_eta_EB -> SaveAs("MRetaEB.root");
  mr_ring_EB -> SaveAs("MRringEB.root");

  mr_eta_EE -> SaveAs("MRetaEE.root");
  mr_ring_EE -> SaveAs("MRringEE.root");


}//end macro
