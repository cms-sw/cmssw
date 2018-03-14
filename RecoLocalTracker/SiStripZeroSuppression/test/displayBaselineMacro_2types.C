#include "TH1F.h"
#include "TCanvas.h"
#include "TObject.h"
#include "TFile.h"
#include "TPaveStats.h"
#include "TGraphErrors.h"
#include "TGaxis.h"
#include "TROOT.h"
#include "TF1.h"
#include "TLegend.h"
#include "TKey.h"
#include "TClass.h"
#include "iostream"
#include "vector"
#include "math.h"
#include "map"

void  displayBaselineMacro_2types(TString file,const  int limit = 20){

  TFile *f;//, *fo;
  TString BaseDir;
  TString dir[6];
  TString fullPath, title, subDet, genSubDet;
  TCanvas *C;
    C = new TCanvas();
    f = new TFile(file);
//    fo = new TFile(ofile, "RECREATE");
    //BaseDir="DQMData/Results/SpyChannel/";
    dir[0]="baselineAna/ProcessedRawDigis";
    dir[1]="baselineAna/Baseline";
    dir[2]="baselineAna/Clusters";
    dir[3]="baselineAna/RawDigis";
    dir[4]="moddedbaselineAna/Baseline";
    dir[5]="moddedbaselineAna/Clusters";
    f->cd();
//	fo->Write();
//	fo->Close();
    f->cd(dir[0]);
    
    TIter nextkey(gDirectory->GetListOfKeys());
    TKey *key;
	int objcounter=1;
	int histolimit =0;
    while ((key = (TKey*)nextkey())) {
      	if(histolimit< limit){ histolimit++;
          std::cout << histolimit << " " << limit << std::endl;
	  TObject *obj = key->ReadObj();
      

      if ( obj->IsA()->InheritsFrom( "TH1" ) ) {
	
	std::cout << "Found object n: " << objcounter << " Name: " << obj->GetName() << " Title: " << obj->GetTitle()<< std::endl;
	++objcounter;
	//if (strstr(obj->GetTitle(),"470116592")!=NULL)
	//  continue;

	C->Clear();
	TH1F* h = (TH1F*)key->ReadObj();

	//TLegend leg(0.6,0.9,0.8,1,"");
	//leg.AddEntry(h,"VR - Ped - apvCM_{mean}","lep");

        h->SetLineColor(kBlack);
	h->SetLineWidth(1);
	h->SetXTitle("StripNumber");
	h->SetYTitle("Charge (ADC counts)");
	h->Draw("hist p l");
        h->SetStats(0);
        //h->GetYaxis()->SetRangeUser(-300,300);
        //h->GetXaxis()->SetRangeUser(256,512);
	f->cd();
	//f->cd(dir[1]);
	TH1F* hb = (TH1F*) f->Get(dir[1]+"/"+obj->GetName());
	
	if(hb!=0){
	  hb->SetLineWidth(2);
	  hb->SetLineStyle(1);
	  hb->SetLineColor(kRed);
	  //leg.AddEntry(hb,"clusters","lep");
	  hb->Draw("hist p l same");
	}
	
	f->cd();
	//f->cd(dir[1]);
	TH1F* hc = (TH1F*) f->Get(dir[2]+"/"+obj->GetName());
        TH1F* offset = (TH1F*)hc->Clone("offset");	
        offset->Reset();
        for(int i = 0; i<offset->GetSize(); i++) offset->SetBinContent(i,-300);
        hc->Add(offset);
	if(hc!=0){
	  hc->SetLineWidth(1);
	  hc->SetLineStyle(1);
	  hc->SetLineColor(kViolet);
	  //leg.AddEntry(hb,"clusters","lep");
	  hc->Draw("hist p l same");
	}
	TH1F* hd = (TH1F*) f->Get(dir[3]+"/"+obj->GetName());
	
	if(hd!=0){
	  hd->SetLineWidth(1);
	  hd->SetLineStyle(1);
	  hd->SetLineColor(kGray);
	  hd->SetMarkerColor(kGray);
	  //leg.AddEntry(hb,"clusters","lep");
	  hd->Draw("hist p l same");
          h->Draw("hist p l same");
          hb->Draw("hist p l same");
          hc->Draw("hist p l same");
	}

	TH1F* he = (TH1F*) f->Get(dir[4]+"/"+obj->GetName());
	if(he!=0){
	  he->SetLineWidth(2);
	  he->SetLineStyle(2);
          he->SetMarkerSize(-1);
	  he->SetLineColor(kGreen);
	  //leg.AddEntry(hb,"clusters","lep");
	  he->Draw("hist l same");
	}
	
        TH1F* hf = (TH1F*) f->Get(dir[5]+"/"+obj->GetName());
        TH1F* offset2 = (TH1F*)hf->Clone("offset2");	
        offset2->Reset();
        for(int i = 0; i<offset2->GetSize(); i++) offset2->SetBinContent(i,-300);
        hf->Add(offset2);
	if(hf!=0){
	  hf->SetLineWidth(1);
	  hf->SetLineStyle(1);
          hf->SetMarkerSize(-1);
	  hf->SetLineColor(kBlue);
	  //leg.AddEntry(hb,"clusters","lep");
	  hf->Draw("hist l same");
	}
	
	//else
	//  std::cout << "not found " << obj->GetName()<< std::endl;
	//leg.Draw();
    
	
	C->Update();
//	fo->cd();
//	C->Write();
	
	C->SaveAs(TString("img/")+obj->GetName()+TString(".png"));
	
	
      }
    }      
  }
}

void displayClusters(TString file){
  TFile * f = new TFile(file,"read");
  TH1F * h1[5];
  TH1F * h2[5];
  h1[0] = (TH1F*)f->Get("baselineAna/ClusterMult"); 
  h2[0] = (TH1F*)f->Get("moddedbaselineAna/ClusterMult"); 
  h1[1] = (TH1F*)f->Get("baselineAna/ClusterCharge"); 
  h2[1] = (TH1F*)f->Get("moddedbaselineAna/ClusterCharge"); 
  h1[2] = (TH1F*)f->Get("baselineAna/ClusterWidth"); 
  h2[2] = (TH1F*)f->Get("moddedbaselineAna/ClusterWidth");

  TCanvas * c1 = new TCanvas("c1","c1",800,800);
  h1[0]->SetLineColor(kRed);
  h2[0]->SetLineColor(kBlue);
  h1[0]->GetYaxis()->SetTitle("nEvents");
  h1[0]->GetXaxis()->SetTitle("nClusters");
  c1->SetLogx();
  //c1->SetLogy();
  h1[0]->SetStats(0);
  h1[0]->Draw();
  h2[0]->Draw("same"); 
  c1->SaveAs("img/ClusterMultComparison.png");  
  c1->SaveAs("img/ClusterMultComparison.pdf");  

  c1->Clear();
  h1[1]->SetLineColor(kRed);
  h2[1]->SetLineColor(kBlue);
  h2[1]->GetYaxis()->SetTitle("nClusters");
  h2[1]->GetXaxis()->SetTitle("Cluster Charge");
  h2[1]->GetXaxis()->SetRangeUser(0,10000);
  c1->SetLogy();
  h2[1]->SetStats(0);
  h2[1]->Draw();
  h1[1]->Draw("same");
  c1->SaveAs("img/ClusterChargeComparison.png");  
  c1->SaveAs("img/ClusterChargeComparison.pdf");  
  
  c1->Clear();
  h1[2]->SetLineColor(kRed);
  h2[2]->SetLineColor(kBlue);
  h2[2]->GetYaxis()->SetTitle("nClusters");
  h2[2]->GetXaxis()->SetTitle("Cluster Width");
  h2[2]->GetXaxis()->SetRangeUser(0,80);
  c1->SetLogx(0);
  h2[2]->SetStats(0);
  h2[2]->Draw();
  h1[2]->Draw("same");
  c1->SaveAs("img/ClusterWidthComparison.png");  
  c1->SaveAs("img/ClusterWidthComparison.pdf");  
  
 
}