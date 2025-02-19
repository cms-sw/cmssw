#include <TROOT.h>
#include <TSystem.h>
#include <TH1D.h>
#include <THStack.h>
#include <TChain.h>
#include <TTree.h>
#include <TLegend.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TLorentzVector.h>
#include <iostream>
#include <sstream>
#include <fstream.h>
#include <TPostScript.h>
using namespace std;

void NoiseAnalyzer();

void NoiseAnalyzer()
{

  // the input file
  TFile *f1 = new TFile("DTNoiseCalib.root");
  
  // the output files
  TPostScript *ps = new TPostScript("DTNoise.ps", 112);
  ofstream summaryOutput("summaryNoise.txt");
  ofstream detailedOutput("detailedNoise.txt");

  TCanvas* c = new TCanvas("prova", "prova");
  TLegend *leg=new TLegend(0.2,0.75,0.9,0.88);


   for(int W=-2; W<=2; W++){
     std::stringstream wheel; wheel << W;
     for(int Sec=1; Sec<=14; Sec++){
       std::stringstream sector; sector << Sec;
       for(int St=1; St<=4; St++){
	 double counterSomehowNoisyCells=0;
	 double counterNoisyCells=0;
	 bool StationHasData=false;
	 std::stringstream station; station << St;
	 for(int SL=1; SL<=3; SL++){
	   leg->Clear();
	   bool SLhasData=false;
	   bool pageWritten=false;
	   std::stringstream superlayer; superlayer << SL;
	   TString newHistoName="AverageNoise_W"+wheel.str()+"_St"+station.str()+"_Sec"+sector.str()+"_SL"+superlayer.str();
	   for(int L=1; L<=4; L++){
	     std::stringstream layer; layer << L;
	     
	     // find the histo
	     TString histoName="DigiOccupancy_W"+wheel.str()+"_St"+station.str()+"_Sec"+sector.str()+"_SL"+superlayer.str()+"_L"+layer.str();
	     TH1F* h =((TH1F*) f1->Get(histoName));
	     if(h){
	       StationHasData=true;
	       SLhasData=true;
	       //overimpose the plot per SL
	       c->cd();
	       TString legend= "layer_"+layer.str();
	       if(L==1){
		 h->SetTitle(newHistoName);
		 h->SetLineColor(L);
		 h->SetLineWidth(2);
		 h->Draw();
		 leg->AddEntry(h,legend,"L");
	       }
	       else{
		 h->SetLineColor(L);
		 h->SetLineWidth(2);
		 h->Draw("same");
		 leg->AddEntry(h,legend,"L");
	       }
	       //count the numeber of noisy/someHow noisy cells
	       int numBin = h->GetXaxis()->GetNbins();
	       for (int bin=1; bin<=numBin; bin++){
		 if(h->GetBinContent(bin)>100 && h->GetBinContent(bin)<500){
		   counterSomehowNoisyCells++;
		   detailedOutput<<"somehowNoisyCell: W"<<W<<" St"<<St<<" Sec"<<Sec<<" SL"<<SL<<" L"<<L<<" wire"<<bin<<endl;
		 }
		  if(h->GetBinContent(bin)>500){
		    counterNoisyCells++;
		    detailedOutput<<"noisyCell: W"<<W<<" St"<<St<<" Sec"<<Sec<<" SL"<<SL<<" L"<<L<<" wire"<<bin<<endl;
		  }
	       } 
	     }

	    } // loop on layer
	    if(SLhasData && !(pageWritten)){
	      pageWritten=true;
	      leg->Draw("same");
	      gPad->SetLogy();
	      c->Update();
	    }

	 } // loop on SL
	 if(StationHasData){
	   summaryOutput<<" ------------ "<<endl;
	   summaryOutput<<"MB"<<St<<"_W"<<W<<"_Sec"<<Sec<<endl;
	   summaryOutput<<"# of somehow noisy cells: "<<counterSomehowNoisyCells<<endl;
	   summaryOutput<<"# of noisy cells: "<<counterNoisyCells<<endl;
	 }

       } // loop on stations
     } // loop on sectors
   } // loop on wheels

   ps->Close();
}
