#include <memory>
#include <TH1F.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TString.h>

using namespace std;

void PlotMacro(){

  int i, j, k;
  int nhistos;

  const int NDIRS     = 19;

  const int NLIHISTOS = 363;
  const int NMUHISTOS = 283;

  const int NDRAWN_HISTOS = 13;

  const float nref = 9050.;
  const float nval = 9050.;

  string sHistName;
  TString OutLabel, HistName;
  TString DirName;
  int DrawSwitch;

  TFile val_file("Idealttbar300p6.root");
  TFile ref_file("Idealttbar221.root");
  
  ifstream DirList("DirList.txt");

  TH1F* ref_histo[NDRAWN_HISTOS];
  TH1F* val_histo[NDRAWN_HISTOS];

  TCanvas *myc = new TCanvas("myc","",800,600);
  
  for (i = 0; i < NDIRS; i++){

    DirList>>DirName;
    cout<<DirName<<endl;
    std::auto_ptr<ifstream> HistoList;

    if (i < 10){
      HistoList = std::auto_ptr<ifstream>(new ifstream("LiHistoList.txt"));
      nhistos = NLIHISTOS;
    }
    else{
      HistoList = std::auto_ptr<ifstream>(new ifstream("MuHistoList.txt"));
      nhistos = NMUHISTOS;
    }
    
    k = 0;
    for (j = 0; j < nhistos; j++){
      (*HistoList)>>sHistName>>DrawSwitch;
      if (DrawSwitch == 0) continue;
      
      if (k >= NDRAWN_HISTOS) cout<<"Histogram index is out of range"<<endl;
      
      (*HistoList)>>OutLabel;
      
      HistName = sHistName;
      OutLabel += ("_"+DirName+".gif");
      
      ref_file.cd(DirName);
      ref_histo[k] = (TH1F*) gDirectory->Get(HistName);
      
      val_file.cd(DirName);
      val_histo[k] = (TH1F*) gDirectory->Get(HistName);
  
      ref_histo[k]->SetLineColor(kBlue);
      ref_histo[k]->SetStats(kFALSE);
      ref_histo[k]->SetLineWidth(2);

      val_histo[k]->SetLineColor(kRed);
      val_histo[k]->SetLineWidth(2);
      
//       if (j != 0){
// 	ref_histo[k]->Scale(1.0/nref);
// 	val_histo[k]->Scale(1.0/nval);
//       }

      //Create legend
      TLegend *leg = new TLegend(0.58, 0.91, 0.84, 0.99, "","brNDC");
      leg->SetBorderSize(2);
      leg->SetFillStyle(1001); //
      leg->AddEntry(ref_histo[k],"221","l");
      leg->AddEntry(val_histo[k],"300p6","l");
      
      ref_histo[k]->Draw("hist");
      val_histo[k]->Draw("hist same");
      
      leg->Draw();
      
      myc->SaveAs(OutLabel);
      
      k++;
    }

    HistoList->close(); //delete HistoList;
  }

}

