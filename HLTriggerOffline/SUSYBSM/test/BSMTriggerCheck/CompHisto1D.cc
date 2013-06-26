#include <TH1D.h>
#include <TCanvas.h>
#include <TLegend.h>
#include "CompHisto1D.hh"

CompHisto1D::CompHisto1D(TH1D* histo1_v, TH1D* histo2_v){

  histo1 = histo1_v;
  histo2 = histo2_v;
  label1 = "first set";
  label1 = "second set";
}

double CompHisto1D::Compare() {
  title = string(histo1->GetName());
  if(title != string(histo2->GetName()) ){ 
    cout << "CompHisto1D::Compare WARNING: histograms have different" << endl;
    cout << "                              labels. Using the label of first plot" << endl;
  }

  compCanvas = new TCanvas(title.c_str(),title.c_str(),300,300);

  double norm = 1./histo1->Integral();
  //  histo1->Scale(norm);
  for(int i=1; i<=histo1->GetNbinsX(); i++) {
    double content = histo1->GetBinContent(i);
    histo1->SetBinContent(i,content*norm);
    histo1->SetBinError(i,sqrt(content)*norm);
  }

  // space for the TLegend
  histo1->SetMaximum(1.5*histo1->GetMaximum());
  
  histo1->SetLineColor(2);
  //  histo1->SetMarkerStyle(20);
  histo1->SetStats(kFALSE);
  histo1->Draw("pe");

  norm = 1./histo2->Integral();
  //  histo2->Scale(norm);
  for(int i=1; i<=histo2->GetNbinsX(); i++) {
    double content = histo2->GetBinContent(i);
    histo2->SetBinContent(i,content*norm);
    histo2->SetBinError(i,sqrt(content)*norm);
  }

  histo2->SetLineColor(4);
  //  histo2->SetMarkerStyle(30);
  histo2->SetStats(kFALSE);
  histo2->Draw("pesame");
  
  TLegend *legend =new TLegend(0.4,0.72,0.95,0.87);
  legend->SetTextSize(0.03);
  legend->AddEntry(histo1,label1.c_str());
  legend->AddEntry(histo2,label2.c_str());
  legend->Draw();

  // calculate JetMET-like compatibility
  double chisq = myChisq();
  double kolmo = histo1->KolmogorovTest(histo2);

  return min(chisq,kolmo);

}

void CompHisto1D::SaveAsEps(){
  compCanvas->SaveAs((title+".eps").c_str());
}

double CompHisto1D::myChisq() {
  double chisq = 0.;
  int  nDOF = 0;
  for(int i=1; i<=histo1->GetXaxis()->GetNbins(); i++) {
    if(histo1->GetBinContent(i) != 0. ||
       histo2->GetBinContent(i) != 0.) {
      chisq += 
	std::pow(histo1->GetBinContent(i)-histo2->GetBinContent(i),2.)/
	(std::pow(histo1->GetBinError(i),2.) + std::pow(histo2->GetBinError(i),2.));
      nDOF += 1;
    }
  }

  return 
    1./(std::pow(2.,nDOF/2)*TMath::Gamma(nDOF/2))*
    std::pow(chisq, nDOF/2-1)*exp(-chisq/2.);
}
