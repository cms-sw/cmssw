#include "TMath.h"
#include "TRint.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TF1.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TGaxis.h"
#include <fstream>
#include <iostream>
#include "TMath.h"
#include <string>
#include <map>


void SetPlotStyle();
void mySmallText(Double_t x,Double_t y,Color_t color,const char *text) {
  Double_t tsize=0.044;
  TLatex l;
  l.SetTextSize(tsize); 
  l.SetNDC();
  l.SetTextColor(color);
  l.DrawLatex(x,y,text);
}


void plotstub(TString filename){

  SetPlotStyle();


  TH2F* h_stub_rz = new TH2F("stub_rz",";Stub z position [cm]; Stub r position [cm]",300,0,300,120,0,120);

  ifstream in(filename+".txt");

  int count=0;

  float L1[2] = {999, 0};
  float L2[2] = {999, 0};
  float L3[2] = {999, 0};
  float L4[2] = {999, 0};
  float L5[2] = {999, 0};
  float L6[2] = {999, 0};

  float D1[2] = {999, 0};
  float D2[2] = {999, 0};
  float D3[2] = {999, 0};
  float D4[2] = {999, 0};
  float D5[2] = {999, 0};

  float R_inner[10] = {66.75, 71.8, 77.56, 82.55, 84.87, 89.88, 95.76, 100.78, 102.48, 107.5};
  float R_outer[10] = {65.13, 70.16, 75.6, 80.67, 83.93, 88.97, 94.63, 99.67, 102.46, 107.5};

  float inner[10] = {0,0,0,0,0,0,0,0,0,0};
  float outer[10] = {0,0,0,0,0,0,0,0,0,0};
  int n_inner[10] = {0,0,0,0,0,0,0,0,0,0};
  int n_outer[10] = {0,0,0,0,0,0,0,0,0,0};


  string tmp;

  while (in.good()){

    // read stubs
    in >> tmp;
    
    if (tmp=="Stub:") {
      
      int layer;
      int ladder;
      int module;
      int eventid;
      int simtrk;
      int strip;
      double pt;
      double x;
      double y;
      double z;
      double bend;
      int isPSmodule;
      int isFlipped;

      in >> layer >> ladder >> module >> strip >> eventid >> simtrk >> pt >> x >> y >> z >> bend >> isPSmodule >> isFlipped;

      float r = sqrt(x*x+y*y);
      
      h_stub_rz->Fill(fabs(z),r);

      if (layer==1 && r<L1[0]) L1[0] = r;
      if (layer==1 && r>L1[1]) L1[1] = r;

      if (layer==2 && r<L2[0]) L2[0] = r;
      if (layer==2 && r>L2[1]) L2[1] = r;

      if (layer==3 && r<L3[0]) L3[0] = r;
      if (layer==3 && r>L3[1]) L3[1] = r;

      if (layer==4 && r<L4[0]) L4[0] = r;
      if (layer==4 && r>L4[1]) L4[1] = r;

      if (layer==5 && r<L5[0]) L5[0] = r;
      if (layer==5 && r>L5[1]) L5[1] = r;

      if (layer==6 && r<L6[0]) L6[0] = r;
      if (layer==6 && r>L6[1]) L6[1] = r;
      
      
      float az = fabs(z);
      if ((az>120 && az<145) && az<D1[0]) D1[0] = az;
      if ((az>120 && az<145) && az>D1[1]) D1[1] = az;
      
      if ((az>145 && az<170) && az<D2[0]) D2[0] = az;
      if ((az>145 && az<170) && az>D2[1]) D2[1] = az;
      
      if ((az>170 && az<200) && az<D3[0]) D3[0] = az;
      if ((az>170 && az<200) && az>D3[1]) D3[1] = az;
      
      if ((az>200 && az<250) && az<D4[0]) D4[0] = az;
      if ((az>200 && az<250) && az>D4[1]) D4[1] = az;
	  
      if ((az>250 && az<300) && az<D5[0]) D5[0] = az;
      if ((az>250 && az<300) && az>D5[1]) D5[1] = az;
      
      count++;

      // rDSS values 
      if (az>120 && isPSmodule==0) {

	if (az>120 && az<170) cout << "inner, R = " << r << endl;
	if (az>170 && az<300) cout << "outer, R = " << r << endl;

	if (az>120 && az<170) {
	  for (int id=0; id<10; id++) {
	    if (fabs(R_inner[id]-r)<1.0) {
	      inner[id]+=r;
	      n_inner[id]++;
	    }
	  }
	}
	if (az>170) {
	  for (int id=0; id<10; id++) {
	    if (fabs(R_outer[id]-r)<1.0) {
	      outer[id]+=r;
	      n_outer[id]++;
	    }
	  }
	}

      }//end rDSS values

      //if (az>120 && isPSmodule) cout << "PS R = " << r << endl;

    }
  }

  cout << "Processed: "<<count<<" events"<<endl;


  TCanvas c;
  h_stub_rz->Draw("colz");
  
  c.SaveAs("stub_rz.png");
  c.SaveAs("stub_rz.pdf");
  
  cout << endl;
  cout << "L1, min = " << L1[0] << " max = " << L1[1] << " average R = " << (L1[0]+L1[1])/2 << " cm" << endl;
  cout << "L2, min = " << L2[0] << " max = " << L2[1] << " average R = " << (L2[0]+L2[1])/2 << " cm" << endl;
  cout << "L3, min = " << L3[0] << " max = " << L3[1] << " average R = " << (L3[0]+L3[1])/2 << " cm" << endl;
  cout << "L4, min = " << L4[0] << " max = " << L4[1] << " average R = " << (L4[0]+L4[1])/2 << " cm" << endl;
  cout << "L5, min = " << L5[0] << " max = " << L5[1] << " average R = " << (L5[0]+L5[1])/2 << " cm" << endl;
  cout << "L6, min = " << L6[0] << " max = " << L6[1] << " average R = " << (L6[0]+L6[1])/2 << " cm" << endl;
  cout << endl;
  cout << "D1, min = " << D1[0] << " max = " << D1[1] << " average z = " << (D1[0]+D1[1])/2 << " cm" << endl;
  cout << "D2, min = " << D2[0] << " max = " << D2[1] << " average z = " << (D2[0]+D2[1])/2 << " cm" << endl;
  cout << "D3, min = " << D3[0] << " max = " << D3[1] << " average z = " << (D3[0]+D3[1])/2 << " cm" << endl;
  cout << "D4, min = " << D4[0] << " max = " << D4[1] << " average z = " << (D4[0]+D4[1])/2 << " cm" << endl;
  cout << "D5, min = " << D5[0] << " max = " << D5[1] << " average z = " << (D5[0]+D5[1])/2 << " cm" << endl;
  cout << endl;
 

  cout << "static double rmeanL1 = " << (L1[0]+L1[1])/2 << ";" << endl;
  cout << "static double rmeanL2 = " << (L2[0]+L2[1])/2 << ";" << endl;
  cout << "static double rmeanL3 = " << (L3[0]+L3[1])/2 << ";" << endl;
  cout << "static double rmeanL4 = " << (L4[0]+L4[1])/2 << ";" << endl;
  cout << "static double rmeanL5 = " << (L5[0]+L5[1])/2 << ";" << endl;
  cout << "static double rmeanL6 = " << (L6[0]+L6[1])/2 << ";" << endl;
  cout << endl;
  cout << "static double zmeanD1 = " << (D1[0]+D1[1])/2 << ";" << endl;
  cout << "static double zmeanD2 = " << (D2[0]+D2[1])/2 << ";" << endl;
  cout << "static double zmeanD3 = " << (D3[0]+D3[1])/2 << ";" << endl;
  cout << "static double zmeanD4 = " << (D4[0]+D4[1])/2 << ";" << endl;
  cout << "static double zmeanD5 = " << (D5[0]+D5[1])/2 << ";" << endl;
  cout << endl;

  for (int i=0; i<10; i++) {
    if (n_inner[i]>0) inner[i] = inner[i]/n_inner[i];
    if (n_outer[i]>0) outer[i] = outer[i]/n_outer[i];
  }
  
  cout << "static double rDSS[20] = {";
  for (int i=0; i<10; i++) {
    cout << inner[i] << ", ";
  }
  cout << endl << "			  ";
  for (int i=0; i<10; i++) {
    cout << outer[i];
    if (i<9) cout << ", ";
  }
  cout << "}; " << endl << endl;


}


void SetPlotStyle() {

  // from ATLAS plot style macro

  // use plain black on white colors
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameFillColor(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetStatColor(0);
  gStyle->SetHistLineColor(1);

  gStyle->SetPalette(1);

  // set the paper & margin sizes
  gStyle->SetPaperSize(20,26);
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadRightMargin(0.12);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.16);

  // set title offsets (for axis label)
  gStyle->SetTitleXOffset(1.2);
  gStyle->SetTitleYOffset(1.2);

  // use large fonts
  gStyle->SetTextFont(42);
  gStyle->SetTextSize(0.05);
  gStyle->SetLabelFont(42,"x");
  gStyle->SetTitleFont(42,"x");
  gStyle->SetLabelFont(42,"y");
  gStyle->SetTitleFont(42,"y");
  gStyle->SetLabelFont(42,"z");
  gStyle->SetTitleFont(42,"z");
  gStyle->SetLabelSize(0.05,"x");
  gStyle->SetTitleSize(0.05,"x");
  gStyle->SetLabelSize(0.05,"y");
  gStyle->SetTitleSize(0.05,"y");
  gStyle->SetLabelSize(0.05,"z");
  gStyle->SetTitleSize(0.05,"z");

  // use bold lines and markers
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(1.2);
  gStyle->SetHistLineWidth(2.);
  gStyle->SetLineStyleString(2,"[12 12]");

  // get rid of error bar caps
  gStyle->SetEndErrorSize(0.);

  // do not display any of the standard histogram decorations
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  // put tick marks on top and RHS of plots
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

}
