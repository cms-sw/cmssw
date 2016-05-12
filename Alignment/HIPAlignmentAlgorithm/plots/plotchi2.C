#include "TTree.h"
#include "TText.h"
#include "TGraph.h"
#include "TFile.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TPaveStats.h"

TGraph* makegr(char* infileName, Double_t startPos = -300, Double_t step = 20, const Int_t n = 32, char* Det = "BPIX Half Barrels", char* ScanDirection = "delta z", char* plotVar = "Chi2", Double_t offset = 0, int lineColor = 1)
{

    gStyle->SetPadLeftMargin(0.15);
    gStyle->SetPadRightMargin(0.15);
    gStyle->SetOptFit();
    gStyle->SetOptStat(1111);
    gStyle->SetTitleFontSize(0.05);
    gStyle->SetTitleOffset(2, "Y");

    TFile* inFile = TFile::Open(infileName);

    char mytreeN[64];
    Double_t zPos[n - 1];
    Int_t zPosI[n-1];
    Double_t chi2 = 0;
    Int_t nhit = 0;
    Double_t chi2Vec[n - 1];
    Int_t nhitVec[n - 1];
    Double_t chi2ph[n - 1];

    for (int iter = 2; iter <= n; iter++) {
        snprintf(mytreeN, sizeof(mytreeN), "T9_%d", iter);
        TTree* mytree = (TTree*)inFile->Get(mytreeN);
        mytree->SetBranchAddress("AlignableChi2", &chi2);
        mytree->SetBranchAddress("Nhit", &nhit);
        mytree->GetEntry(0);
        //T9_1 has Chi2 at iter0, T9_2 has Chi2 at iter1
        zPos[iter - 2] = startPos + step * (iter - 2);
        zPosI[iter - 2] = startPos + step * (iter - 2);
        chi2Vec[iter - 2] = chi2 - offset;
        nhitVec[iter - 2] = nhit;
        if (nhit != 0)
            chi2ph[iter - 2] = chi2 / nhit;
        else
            chi2ph[iter - 2] = 0;
        cout << ScanDirection << " position=" << zPos[iter - 2] << ", chi2=" << chi2Vec[iter - 2] << ", hit=" << nhitVec[iter - 2] << ", chi2 per hit=" << chi2ph[iter - 2] << endl;
    }

    TGraph* gr;

    if (strcmp(plotVar, "Chi2") == 0)
        gr = new TGraph(n - 1, zPos, chi2Vec);
    else if (strcmp(plotVar, "Chi2PerHit") == 0)
        gr = new TGraph(n - 1, zPos, chi2ph);
    else if (strcmp(plotVar,"nhits")==0)
        gr = new TGraph (n-1,zPosI,nhitVec);

    gr->SetLineColor(lineColor);

    char title[50];
    sprintf(title, "%s scan of %s in %s", plotVar, ScanDirection, Det);

    gr->SetTitle(title);
    gr->GetXaxis()->SetTitle(ScanDirection);
    gr->GetYaxis()->SetTitle(plotVar);
    return gr;
}
void plotchi2()
{

    /************* START DEFINED BY USER *****************/

    char* infile = "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HIP/public/AlignmentCamp2016/CMSSW_8_0_3/src/Alignment/HIPAlignmentAlgorithm/hp1609_scanBPIXz/main/IOUserVariables.root";
    Double_t start = -300, end = 300, step = 20; //unit is um, the number of iterations must be bigger than (end-start)/step+2
    char* detName = "BPIX_Half_Barrels"; //scanned detector
    char* scanDir = "delta_z"; //scan direction
//    char* scanVar = "Chi2"; // choose from: Chi2, Chi2PerHit,nhits
    char* scanVar = "nhits"; // choose from: Chi2, Chi2PerHit,nhits

    /************** END DEFINED BY USER ******************/

    int n_iter = (end - start) / step + 2;
    TGraph* gr1 = makegr(infile, start, step, n_iter, detName, scanDir, scanVar);

    TCanvas* c = new TCanvas("c", "c", 1000, 10, 800, 800);

    c->cd();
    gr1->Draw("AC*");

    c->Update();
    TLine* l = new TLine(0, c->GetUymin(), 0, c->GetUymax());
    l->SetLineStyle(2);
    l->Draw();

    char plotName[50];
    sprintf(plotName, "%sscan_%s_%s.png", scanVar, scanDir, detName);

    c->SaveAs(plotName);

    /*
TGraph* gr1 = makegr(-300,20,32,"BPIX +x Half Barrel","z",Var,0,1);
TGraph* gr2 = makegr(-300,20,32,"BPIX -x Half Barrel","z",Var,0,2);

 TCanvas* c2 = new TCanvas("c2", "c2", 1000, 10, 800, 800);
  TMultiGraph *g = new TMultiGraph();
  g->Add(gr1);
  g->Add(gr2);
  TLegend* leg = new TLegend(.87,0.5,0.98,.9);
  leg->SetFillColor(0);
  leg->SetBorderSize(0);
  leg->AddEntry(gr1,"x>0","l");
  leg->AddEntry(gr2,"x<0","l");
  g->Draw("AC*");
  g->GetXaxis()->SetTitle("delta_z");
  g->GetYaxis()->SetTitle(Var);
  leg->Draw();
  c2->Update();
  char pn[50];
  sprintf(pn,"multigr_%s.eps",Var);
  c2->SaveAs(pn);
*/
}
