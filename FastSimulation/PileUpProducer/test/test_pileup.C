#include "DataFormats/FWLite/interface/Handle.h"

/*
Don't forget these commands, if they are not in your rootlogon.C:

   gSystem->Load("libFWCoreFWLite.so"); 
   AutoLibraryLoader::enable();
   gSystem->Load("libDataFormatsFWLite.so");

*/

/*
 nsls -l /castor/cern.ch/user/g/giamman/fastsim_pu
 mrw-r--r--   1 giamman  zh               1811480933 Jun 08 18:10 flat.root
 mrw-r--r--   1 giamman  zh               1831021005 Jun 09 11:05 poisson.root
*/

void test_pileup(TString input="rfio:/castor/cern.ch/user/g/giamman/fastsim_pu/flat.root")
{
  TRFIOFile file(input);
  bool verbose(0);

  TH1F* hprob = new TH1F("h","Pile-up",25,0,25); 

  
  fwlite::Event ev(&file);

  for( ev.toBegin(); ! ev.atEnd(); ++ev) {

    // method 1
    /*
    std::cout << "##### PileupMixingContent " << std::endl;
    fwlite::Handle< PileupMixingContent > pmc;
    pmc.getByLabel(ev,"famosPileUp");
    std::cout <<" bunch crossing "<<pmc.ptr()->getMix_bunchCrossing().at(0)<<std::endl;
    std::cout <<" interaction number  "<<pmc.ptr()->getMix_Ninteractions().at(0)<<std::endl;
    */

    // method 2
    if (verbose) std::cout << "##### PileupSummaryInfo " << std::endl;
    fwlite::Handle< std::vector< PileupSummaryInfo > > psi;
    psi.getByLabel(ev,"addPileupInfo");
    if (verbose) std::cout <<" bunch crossing "<< psi.ptr()->at(0).getBunchCrossing() <<std::endl;
    if (verbose) std::cout <<" interaction number  "<< psi.ptr()->at(0).getPU_NumInteractions()<<std::endl;
    hprob->Fill(psi.ptr()->at(0).getPU_NumInteractions());

  }
  
  TCanvas *c = new TCanvas("c", "Pile-up",0,0,700,500);
  gStyle->SetOptFit(1);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  c->SetHighLightColor(2);
  c->Range(-1.461538,-12.24238,1.615385,55.77082);
  c->SetFillColor(0);
  c->SetBorderMode(0);
  c->SetBorderSize(2);
  c->SetTickx(1);
  c->SetTicky(1);
  c->SetLeftMargin(0.15);
  c->SetTopMargin(0.07);
  c->SetBottomMargin(0.18);
  c->SetFrameFillStyle(0);
  c->SetFrameBorderMode(0);
  c->SetFrameFillStyle(0);
  c->SetFrameBorderMode(0);
  hprob->DrawCopy();
  
  TLegend *leg = new TLegend(0.58,0.75,0.88,0.90,NULL,"brNDC");
  leg->SetBorderSize(1);
  leg->SetLineColor(0);
  leg->SetLineStyle(0);
  leg->SetLineWidth(1);
  leg->SetFillColor(0);
  leg->SetFillStyle(0);
  leg->AddEntry(hprob,"generated PU","l");
  leg->SetTextSize(0.04);
  leg->SetFillColor(0);
  leg->Draw();

  c->Print("pileup.gif");

}
