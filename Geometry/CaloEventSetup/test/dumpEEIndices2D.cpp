#include "TStyle.h"
#include "TH2F.h"
#include "TText.h"
#include "TCanvas.h"
#include "TString.h"
#include "TPRegexp.h"
#include "TObjArray.h"
#include "TObjString.h"

#include <fstream>
#include <iostream>
#include <cstdlib>

int main(int /*argc*/,char** /*argv*/) 
{  
  gStyle->SetOptStat(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(1);
  gStyle->SetOptTitle(0);
  gStyle->SetStatFont(42);
  gStyle->SetTitleFont(22);
  gStyle->SetCanvasColor(10);
  gStyle->SetPadColor(0);
  gStyle->SetLabelFont(42,"x");
  gStyle->SetLabelFont(42,"y");
  gStyle->SetHistFillStyle(1001);
  gStyle->SetHistFillColor(0);
  gStyle->SetHistLineStyle(1);
  gStyle->SetHistLineWidth(3);
  gStyle->SetTitleXOffset(1.1);
  gStyle->SetTitleYOffset(1.15);
  gStyle->SetOptStat(0);
  //gStyle->SetOptStat(00011110);
  gStyle->SetOptFit(0);
  gStyle->SetStatH(0.11); 
  
  TCanvas* myCanvas=new TCanvas("myC","myC",2000,2000); 
  TH2F* reference=new TH2F("reference","reference",300,-160.,160,300,-160.,160.);
  reference->GetXaxis()->SetTitle("X [cm]");
  reference->GetYaxis()->SetTitle("Y [cm]");
  reference->Draw();

  ifstream endcapDump;
  endcapDump.open("ee.C");
  if (!endcapDump)
    {
      std::cout << "ERROR: file ee.C not found" << std::endl;
      exit(-1);
    }
  TText* t=new TText();
  t->SetTextSize(0.004);
  t->SetTextAngle(40);
  int i=0;
  //  char endcapToSelect[1]="+";
  //  char endcapRegexp[10];
  //  sprintf(endcapRegexp,"iz \\%s",endcapToSelect);
  while (!endcapDump.eof())
    {
      char line[256];
      endcapDump.getline(line,256);
      TString aLine(line);
      if (TPRegexp("iz \\-").Match(aLine))
	{
	  TObjArray* tokens=aLine.Tokenize(" ");
	  int ix=(((TObjString*)tokens->At(5))->String()).Atoi();
	  TString sy=((TObjString*)tokens->At(8))->String();
	  TPRegexp("\\)").Substitute(sy,"");
	  int iy = sy.Atoi();
	  endcapDump.getline(line,256);
	  endcapDump.getline(line,256);
	  TString secondLine(line);
	  TObjArray* newTokens=secondLine.Tokenize("(");
	  TObjArray* coordinates=((TObjString*)newTokens->At(3))->String().Tokenize(","); 
	  float x=((TObjString*)coordinates->At(0))->String().Atof();
	  float y=((TObjString*)coordinates->At(1))->String().Atof();
//	  float z=((TObjString*)coordinates->At(2))->String().Atof();
	  char text[10];
	  sprintf(text,"%d,%d",ix,iy);
	  t->DrawText(x,y,text);
	  std::cout << "Volume " << ++i <<  " ix " << ix  <<  " iy " << iy << " Position (" << x << "," << y << ")" << std::endl;   
	}
    }
  myCanvas->SaveAs("eeIndices.eps");
  delete reference;
  delete myCanvas;
  delete t;
  endcapDump.close();
}
