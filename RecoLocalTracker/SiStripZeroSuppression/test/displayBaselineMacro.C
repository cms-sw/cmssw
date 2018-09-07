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

    
   

void  displayBaselineMacro(TString file, int limit){
  TFile *f;//, *fo;
  TString BaseDir;
  TString dir[4];
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
    f->cd();
//	fo->Write();
//	fo->Close();
    f->cd(dir[0]);
    
    TIter nextkey(gDirectory->GetListOfKeys());
    TKey *key;
	int objcounter=1;
	int histolimit =0;
    while ((key = (TKey*)nextkey())) {
      	if(histolimit< limit){
		histolimit++;
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

	h->SetLineWidth(2);
	h->SetXTitle("StripNumber");
	h->SetYTitle("Charge (ADC counts)");
	h->Draw("hist p l");
	f->cd();
	//f->cd(dir[1]);
	TH1F* hb = (TH1F*) f->Get(dir[1]+"/"+obj->GetName());
	
	if(hb!=0){
	  hb->SetLineWidth(2);
	  hb->SetLineStyle(2);
	  hb->SetLineColor(2);
	  //leg.AddEntry(hb,"clusters","lep");
	  hb->Draw("hist p l same");
	}
	
	f->cd();
	//f->cd(dir[1]);
	TH1F* hc = (TH1F*) f->Get(dir[2]+"/"+obj->GetName());
	
	if(hc!=0){
	  hc->SetLineWidth(2);
	  hc->SetLineStyle(2);
	  hc->SetLineColor(3);
	  //leg.AddEntry(hb,"clusters","lep");
	  hc->Draw("hist p l same");
	}
	TH1F* hd = (TH1F*) f->Get(dir[3]+"/"+obj->GetName());
	
	if(hd!=0){
	  hd->SetLineWidth(2);
	  hd->SetLineStyle(2);
	  hd->SetLineColor(6);
	  //leg.AddEntry(hb,"clusters","lep");
	  hd->Draw("hist p l same");
	}
	
	else
	  std::cout << "not found " << obj->GetName()<< std::endl;
	//leg.Draw();
    
	
	C->Update();
//	fo->cd();
//	C->Write();
	
	C->SaveAs(TString("img/")+obj->GetName()+TString(".png"));
	
	
      }
    }  
  }
}