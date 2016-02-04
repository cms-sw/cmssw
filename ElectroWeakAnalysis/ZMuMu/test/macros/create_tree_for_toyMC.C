/*************************/
/*                       */ 
/* author: Pasquale Noli */
/* INFN Naples           */
/* Create TTree from     */
/* fit on Toy Montecarlo */
/*                       */
/*************************/
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "TObjArray.h"
#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include <iomanip>
#include <iostream>

using namespace std;

void create_tree_for_toyMC()()
{
  gROOT->Reset();

  Double_t Y;
  Double_t Y_true;
  Double_t dY;
  Double_t Tk;
  Double_t Tk_true;
  Double_t dTk;
  Double_t Sa;
  Double_t Sa_true;
  Double_t dSa;
  Double_t Iso;
  Double_t Iso_true;
  Double_t dIso;
  Double_t Hlt;
  Double_t Hlt_true;
  Double_t dHlt;
  Double_t chi2;

  TFile *f;
  TTree *tree;
  
  f = new TFile("fitResult.root","RECREATE");
  tree = new TTree("tree"," C data from ASCII file");

  tree->Branch("Y",&Y,"Y/D");
  tree->Branch("Y_true",&Y_true,"Y/D");
  tree->Branch("dY",&dY,"dY/D");
  tree->Branch("Tk",&Tk," Tk/D");
  tree->Branch("Tk_true",&Tk_true," Tk_true/D");
  tree->Branch("dTk",&dTk," dTk/D");
  tree->Branch("Sa",&Sa," Sa/D");
  tree->Branch("Sa_true",&Sa_true," Sa_true/D");
  tree->Branch("dSa",&dSa," dSa/D");
  tree->Branch("Iso",&Iso," Iso/D");
  tree->Branch("Iso_true",&Iso_true," Iso_true/D");
  tree->Branch("dIso",&dIso," dIso/D");
  tree->Branch("Hlt",&Hlt," Hlt/D");
  tree->Branch("Hlt_true",&Hlt_true," Hlt_true/D");
  tree->Branch("dHlt",&dHlt," dHlt/D");
  tree->Branch("chi2",&chi2," chi2/D");

  ifstream fin;
  fin.open("fitResult.txt");

  char line[1024];

  fin.getline(line, 1024);
  cout << line << endl;
  fin >> Y_true >> Tk_true >> Sa_true >> Iso_true >> Hlt_true;
  cout << "Yield = " << Y_true;
  cout << " eff_trk = " << Tk_true;
  cout << " eff_sa = " << Sa_true;
  cout << " eff_iso = " << Iso_true;
  cout << " eff_hlt = " << Hlt_true << endl;
  while(!(fin.eof())){
    Y = 0;
    fin >> Y >> dY >> Tk >>  dTk >>  Sa >>
      dSa >>  Iso >> dIso >>  Hlt >>  dHlt >>chi2;
    if(Y > 0)
      tree->Fill();
  }
  
  tree->Print();
  f->Write();
  f->Close();


}
