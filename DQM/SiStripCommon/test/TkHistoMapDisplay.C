/*
TkHistoMapDisplay macro to display TkHistoMap.
A canvas is created for each subdetector, containing the histograms of all the layers of the subdetector.
To each histogram can be overlapped the histgram that contains the detids, to easily discover problems.

usage

.L TkHistoMapDisplay.C

a) To create only display of the interesting histomap:

TkHistoMapDisplay("filaname","dirpat","mapName") like TkHistoMapDisplay("test.root","DQMData/Zmap","Zmap")

b) To overlap also the detids (very time expensive)

TkHistoMapDisplay("filaname","dirpat","mapName","filename of the reference file") like TkHistoMapDisplay("test.root","DQMData/Zmap","Zmap","test.root")

c) To overlap also the detids of a single subdetector

TkHistoMapDisplay("filaname","dirpat","mapName","filename of the reference file","subdetname") like TkHistoMapDisplay("test.root","DQMData/Zmap","Zmap","test.root","TIB")
*/

#include "TCanvas.h"
#include "TFile.h"
#include "TProfile2D.h"
#include "TString.h"
#include "iostream"

TFile *_file,*_reffile, *_afile;
TCanvas *CTIB, *CTOB, *CTEC, *CTID;
TProfile2D* h;

void plotTIB(TString fullPath, TString Option){
  std::cout<< fullPath+TString("_TIB") << std::endl;
  CTIB->cd(1);
  h=(TProfile2D*) _afile->Get(fullPath+TString("_TIB_L1"));
  h->Draw(Option);
  CTIB->cd(2);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TIB_L2"));
  h->Draw(Option);
  CTIB->cd(3);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TIB_L3"));
  h->Draw(Option);
  CTIB->cd(4);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TIB_L4"));
  h->Draw(Option);
}

void plotTOB(TString fullPath, TString Option){
  std::cout<< fullPath+TString("_TOB") << std::endl;
  CTOB->cd(1);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TOB_L1"));
  h->Draw(Option);
  CTOB->cd(2);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TOB_L2"));
  h->Draw(Option);
  CTOB->cd(3);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TOB_L3"));
  h->Draw(Option);
  CTOB->cd(4);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TOB_L4"));
  h->Draw(Option);
  CTOB->cd(5);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TOB_L5"));
  h->Draw(Option);
  CTOB->cd(6);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TOB_L6"));
  h->Draw(Option);
}

void plotTID(TString fullPath, TString Option){
  std::cout<< fullPath+TString("_TID") << std::endl;
  CTID->cd(1);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TID_D1"));
  h->Draw(Option);
  CTID->cd(2);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TID_D2"));
  h->Draw(Option);
  CTID->cd(3);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TID_D3"));
  h->Draw(Option);
}

void plotTEC(TString fullPath, TString Option){
  std::cout<< fullPath+TString("_TEC") << std::endl;
  CTEC->cd(1);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TEC_W1"));
  h->Draw(Option);
  CTEC->cd(2);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TEC_W2"));
  h->Draw(Option);
  CTEC->cd(3);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TEC_W3"));
  h->Draw(Option);
  CTEC->cd(4);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TEC_W4"));
  h->Draw(Option);
  CTEC->cd(5);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TEC_W5"));
  h->Draw(Option);
  CTEC->cd(6);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TEC_W6"));
  h->Draw(Option);
  CTEC->cd(7);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TEC_W7"));
  h->Draw(Option);
  CTEC->cd(8);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TEC_W8"));
  h->Draw(Option);
  CTEC->cd(9);
  h= (TProfile2D*) _afile->Get(fullPath+TString("_TEC_W9"));
  h->Draw(Option);
}

void TkHistoMapDisplay(TString File="", TString path="", TString TkmapName="", TString referenceFile="", TString refSubDet="", TString opt=""){

  std::cout << "File " << File.Data() << " reference " << referenceFile.Data()<< std::endl;


  _file=new TFile(File);
  
  CTIB=new TCanvas("TIB","TIB");
  CTOB=new TCanvas("TOB","TOB");
  CTID=new TCanvas("TID","TID");
  CTEC=new TCanvas("TEC","TEC");
  CTIB->Divide(2,2);
  CTOB->Divide(2,3);
  CTID->Divide(1,3);
  CTEC->Divide(3,3);
  
  _afile=_file;
  gStyle->SetOptStat(0);
  gStyle->SetPalette(1,0);
  if(opt=="")
    opt="COLZ";
  plotTIB(path+TString("/")+TkmapName, opt);
  plotTOB(path+TString("/")+TkmapName, opt);
  plotTID(path+TString("/")+TkmapName, opt);
  plotTEC(path+TString("/")+TkmapName, opt);
  
  
  if(referenceFile!=TString("")){
    
    path="DQMData/detId";
    TkmapName="detId";
    
    _reffile=new TFile(referenceFile);

    std::cout<< _reffile << std::endl;
    _afile=_reffile;
    std::cout<< _reffile << std::endl;
    gStyle->SetOptStat(0);

    if(refSubDet=="" || refSubDet=="TIB")
      plotTIB(path+TString("/")+TkmapName, "TEXTsames");
    if(refSubDet=="" || refSubDet=="TOB")
      plotTOB(path+TString("/")+TkmapName, "TEXTsames");
    if(refSubDet=="" || refSubDet=="TID")
      plotTID(path+TString("/")+TkmapName, "TEXTsames");
    if(refSubDet=="" || refSubDet=="TEC")
      plotTEC(path+TString("/")+TkmapName, "TEXTsames");

  }
 
}
