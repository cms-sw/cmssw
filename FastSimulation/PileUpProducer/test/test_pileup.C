#include "DataFormats/FWLite/interface/Handle.h"

/*
Don't forget these commands, if they are not in your rootlogon.C:

   gSystem->Load("libFWCoreFWLite.so"); 
   AutoLibraryLoader::enable();
   gSystem->Load("libDataFormatsFWLite.so");

*/

void test_pileup()
{
  TFile file("TTbar_Tauola_7TeV_cfi_py_GEN_FASTSIM_HLT_PU.root");
  fwlite::Event ev(&file);

  Int_t nbins=50;
  bool verbose=true;
  TH1F* hpumc = new TH1F("PileupMixingContent","PileupMixingContent",nbins,0,(Double_t)nbins);
  TH1F* hpusi = new TH1F("PileupSummaryInfo","PileupSummaryInfo",nbins,0,(Double_t)nbins);

  for( ev.toBegin(); ! ev.atEnd(); ++ev) {

    // method 1
    if (verbose) std::cout << "##### PileupMixingContent " << std::endl;
    fwlite::Handle< PileupMixingContent > pmc;
    pmc.getByLabel(ev,"famosPileUp");
    hpumc->Fill(pmc.ptr()->getMix_Ninteractions().at(0));
    if (verbose) {
      std::cout <<" bunch crossing "<<pmc.ptr()->getMix_bunchCrossing().at(0)<<std::endl;
      std::cout <<" interaction number  "<<pmc.ptr()->getMix_Ninteractions().at(0)<<std::endl;
    }

    // method 2
    if (verbose) std::cout << "##### PileupSummaryInfo " << std::endl;
    fwlite::Handle< std::vector< PileupSummaryInfo > > psi;
    psi.getByLabel(ev,"addPileupInfo");
    hpusi->Fill(psi.ptr()->at(0).getPU_NumInteractions());
    if (verbose) {
      std::cout <<" bunch crossing "<< psi.ptr()->at(0).getBunchCrossing() <<std::endl;
      std::cout <<" interaction number  "<< psi.ptr()->at(0).getPU_NumInteractions()<<std::endl;
    }

  }

  TCanvas *canvas = new TCanvas("Pileup","Pileup",1000,1000);
  canvas->cd();
  hpumc->SetLineColor(2);
  hpusi->SetLineColor(1);
  hpumc->Draw();
  hpusi->Draw("same");
  TLegend* l = new TLegend(0.60,0.14,0.90,0.19);
  if (verbose) { // normally one trusts that the two histos are equal, so the legend can just be disabled
    l->SetTextSize(0.016);
    l->SetLineColor(1);
    l->SetLineWidth(1);
    l->SetLineStyle(1);
    l->SetFillColor(0);
    l->SetBorderSize(3);
    l->AddEntry(hpumc,"PileupMixingContent","LPF");
    l->AddEntry(hpusi,"PileupSummaryInfo","LPF");
    l->Draw();
  }
  TString namepdf = "test_pileup.pdf";
  canvas->Print(namepdf);   
  delete l;

  delete canvas;
}
