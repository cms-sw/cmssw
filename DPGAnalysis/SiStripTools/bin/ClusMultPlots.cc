#include "ClusMultPlots.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include "TPad.h"
#include "TFile.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TGraph.h"
#include "DPGAnalysis/SiStripTools/interface/CommonAnalyzer.h"
#include "TCanvas.h"
#include "TStyle.h"


void ClusMultPlots(const char* fullname, const char* pxmod, const char* strpmod, const char* corrmod,
		   const char* pxlabel, const char* strplabel, const char* corrlabel, const char* postfix, const char* shortname, const char* outtrunk) {


  char pxmodfull[300];
  sprintf(pxmodfull,"%s%s",pxmod,postfix);
  char pxlabfull[300];
  sprintf(pxlabfull,"%s%s",pxlabel,postfix);

  char strpmodfull[300];
  sprintf(strpmodfull,"%s%s",strpmod,postfix);
  char strplabfull[300];
  sprintf(strplabfull,"%s%s",strplabel,postfix);

  char corrmodfull[300];
  sprintf(corrmodfull,"%s%s",corrmod,postfix);
  char corrlabfull[300];
  sprintf(corrlabfull,"%s%s",corrlabel,postfix);


  //  char fullname[300];
  //  sprintf(fullname,"rootfiles/Tracking_PFG_%s.root",filename);


  TFile ff(fullname);

  gStyle->SetOptStat(111111);

  
  CommonAnalyzer capixel(&ff,"",pxmodfull,"EventProcs/Pixel");

  TH1F* pixel  = (TH1F*)capixel.getObject("nPixeldigi");
  if(pixel) {
    pixel->Draw();
    gPad->SetLogy(1);
    std::string plotfilename;
    plotfilename += outtrunk;
    plotfilename += shortname;
    plotfilename += "/pixel";
    plotfilename += pxlabfull;
    plotfilename += "_";
    plotfilename += shortname;
    plotfilename += ".gif";
    gPad->Print(plotfilename.c_str());
    delete pixel;
    gPad->SetLogy(0);
  }
 
  capixel.setPath("VtxCorr/Pixel");

  TH2F* pixelvtx  = (TH2F*)capixel.getObject("nPixeldigivsnvtx");
  if(pixelvtx) {
    pixelvtx->Draw("colz");
    //    TProfile* pixelvtxprof = pixelvtx->ProfileY("prof",1,-1,"");
    TProfile* pixelvtxprof = pixelvtx->ProfileX("prof",1,-1,"");
    pixelvtxprof->SetMarkerStyle(20);
    pixelvtxprof->SetMarkerSize(.4);
    /*
    TGraph tmp;
    for(unsigned bin=1;bin<pixelvtxprof->GetNbinsX()+1;++bin) {
      //      tmp.SetPoint(tmp.GetN(),pixelvtxprof->GetBinContent(bin),pixelvtxprof->GetBinCenter(bin));
      tmp.SetPoint(tmp.GetN(),pixelvtxprof->GetBinCenter(bin),pixelvtxprof->GetBinContent(bin));
    }
    tmp.SetMarkerStyle(20);
    tmp.Draw("p");
    */
    pixelvtxprof->Draw("esame");
    gPad->SetLogz(1);
    std::string plotfilename;
    plotfilename += outtrunk;
    plotfilename += shortname;
    plotfilename += "/pixelvtx";
    plotfilename += pxlabfull;
    plotfilename += "_";
    plotfilename += shortname;
    plotfilename += ".gif";
    gPad->Print(plotfilename.c_str());
    delete pixelvtx;
    gPad->SetLogz(0);
  }
 
  CommonAnalyzer castrip(&ff,"",strpmodfull,"EventProcs/TK");

  TH1F* tk  = (TH1F*)castrip.getObject("nTKdigi");
  if(tk) {
    tk->Draw();
    gPad->SetLogy(1);
    std::string plotfilename;
    plotfilename += outtrunk;
    plotfilename += shortname;
    plotfilename += "/tk";
    plotfilename += strplabfull;
    plotfilename += "_";
    plotfilename += shortname;
    plotfilename += ".gif";
    gPad->Print(plotfilename.c_str());
    delete tk;
    gPad->SetLogy(0);
  }
  
  castrip.setPath("VtxCorr/TK");

  TH2F* tkvtx  = (TH2F*)castrip.getObject("nTKdigivsnvtx");
  if(tkvtx) {
    tkvtx->Draw("colz");
    //    TProfile* tkvtxprof = tkvtx->ProfileY("prof2",1,-1,"");
    TProfile* tkvtxprof = tkvtx->ProfileX("prof2",1,-1,"");
    tkvtxprof->SetMarkerStyle(20);
    tkvtxprof->SetMarkerSize(.4);
    /*
    cout << tkvtxprof->GetNbinsX() << " " << tkvtxprof->GetSize() << " " << tkvtxprof->GetXaxis()->GetXbins()->GetSize() << endl;
    TGraph tmp;
    for(unsigned bin=1;bin<tkvtxprof->GetNbinsX()+1;++bin) {
      tmp.SetPoint(tmp.GetN(),tkvtxprof->GetBinContent(bin),tkvtxprof->GetBinCenter(bin));
    }
    tmp.SetMarkerStyle(20);
    tmp.Draw("p");
    */
    tkvtxprof->Draw("esame");
    gPad->SetLogz(1);
    std::string plotfilename;
    plotfilename += outtrunk;
    plotfilename += shortname;
    plotfilename += "/tkvtx";
    plotfilename += strplabfull;
    plotfilename += "_";
    plotfilename += shortname;
    plotfilename += ".gif";
    gPad->Print(plotfilename.c_str());
    delete tkvtx;
    gPad->SetLogz(0);
  }
 

  CommonAnalyzer cacorr(&ff,"",corrmodfull,"");


  TH1F* rat  = (TH1F*)cacorr.getObject("PixelOverTK");
  if(rat) {
    rat->Draw();
    gPad->SetLogy(1);
    std::string plotfilename;
    plotfilename += outtrunk;
    plotfilename += shortname;
    plotfilename += "/pixelovertk";
    plotfilename += corrlabfull;
    plotfilename += "_";
    plotfilename += shortname;
    plotfilename += ".gif";
    gPad->Print(plotfilename.c_str());
    delete rat;
    gPad->SetLogy(0);
  }


  TH2F* mult2d  = (TH2F*)cacorr.getObject("PixelVsTK");
  if(mult2d) {
    mult2d->Draw("colz");
    gPad->SetLogz(1);
    //    mult2d->GetXaxis()->SetRangeUser(0.,30000);
    //    mult2d->GetYaxis()->SetRangeUser(0.,15000);
    std::string plotfilename;
    plotfilename += outtrunk;
    plotfilename += shortname;
    plotfilename += "/pixelvstk";
    plotfilename += corrlabfull;
    plotfilename += "_";
    plotfilename += shortname;
    plotfilename += ".gif";
    gPad->Print(plotfilename.c_str());
    delete mult2d;
    gPad->SetLogz(0);
  }

  gStyle->SetOptStat(1111);

  ff.Close();
}

void ClusMultInvestPlots(const char* fullname, const char* mod, const char* label, const char* postfix, const char* subdet, const char* shortname, const char* outtrunk) {


  char modfull[300];
  sprintf(modfull,"%s%s",mod,postfix);
  char labfull[300];
  sprintf(labfull,"%s%s",label,postfix);

  //  char fullname[300];
  //  sprintf(fullname,"rootfiles/Tracking_PFG_%s.root",filename);


  TFile ff(fullname);

  gStyle->SetOptStat(111111);

  char subdirname[300];
  sprintf(subdirname,"EventProcs/%s",subdet);
  char histname[300];
  sprintf(histname,"n%sdigi",subdet);

  CommonAnalyzer ca(&ff,"",modfull,subdirname);

  TH1F* hist  = (TH1F*)ca.getObject(histname);
  if(hist) {
    hist->Draw();
    gPad->SetLogy(1);
    std::string plotfilename;
    plotfilename += outtrunk;
    plotfilename += shortname;
    plotfilename += "/";
    plotfilename += subdet;
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += shortname;
    plotfilename += ".gif";
    gPad->Print(plotfilename.c_str());
    delete hist;
    gPad->SetLogy(0);
  }
  gStyle->SetOptStat(1111);

  ff.Close();
}

void ClusMultCorrPlots(const char* fullname, const char* mod, const char* label, const char* postfix, const char* shortname, const char* outtrunk) {


  char modfull[300];
  sprintf(modfull,"%s%s",mod,postfix);
  char labfull[300];
  sprintf(labfull,"%s%s",label,postfix);


  //  char fullname[300];
  //  sprintf(fullname,"rootfiles/Tracking_PFG_%s.root",filename);


  TFile ff(fullname);

  gStyle->SetOptStat(111111);

  CommonAnalyzer ca(&ff,"",modfull,"");


  TH1F* rat  = (TH1F*)ca.getObject("PixelOverTK");
  if(rat) {
    rat->Draw();
    gPad->SetLogy(1);
    std::string plotfilename;
    plotfilename += outtrunk;
    plotfilename += shortname;
    plotfilename += "/pixelovertk";
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += shortname;
    plotfilename += ".gif";
    gPad->Print(plotfilename.c_str());
    delete rat;
    gPad->SetLogy(0);
  }


  TH2F* mult2d  = (TH2F*)ca.getObject("PixelVsTK");
  if(mult2d) {
    mult2d->Draw("colz");
    gPad->SetLogz(1);
    //    mult2d->GetXaxis()->SetRangeUser(0.,30000);
    //    mult2d->GetYaxis()->SetRangeUser(0.,15000);
    std::string plotfilename;
    plotfilename += outtrunk;
    plotfilename += shortname;
    plotfilename += "/pixelvstk";
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += shortname;
    plotfilename += ".gif";
    gPad->Print(plotfilename.c_str());
    delete mult2d;
    gPad->SetLogz(0);
  }

  gStyle->SetOptStat(1111);

  ff.Close();
}

void ClusMultVtxCorrPlots(const char* fullname, const char* mod, const char* label, const char* postfix, const char* subdet, const char* shortname, const char* outtrunk) {


  char modfull[300];
  sprintf(modfull,"%s%s",mod,postfix);
  char labfull[300];
  sprintf(labfull,"%s%s",label,postfix);

  //  char fullname[300];
  //  sprintf(fullname,"rootfiles/Tracking_PFG_%s.root",filename);


  TFile ff(fullname);

  gStyle->SetOptStat(111111);

  char subdirname[300];
  sprintf(subdirname,"VtxCorr/%s",subdet);
  char histname[300];
  sprintf(histname,"n%sdigivsnvtx",subdet);
  char profname[300];
  sprintf(profname,"n%sdigivsnvtxprof",subdet);
  
  CommonAnalyzer ca(&ff,"",modfull,subdirname);


  TH2F* histvtx  = (TH2F*)ca.getObject(histname);
  if(histvtx) {
    histvtx->Draw("colz");
    //    TProfile* histvtxprof = histvtx->ProfileY("prof",1,-1,"");
    TProfile* histvtxprof = 0;
    histvtxprof = (TProfile*)ca.getObject(profname);
    if(histvtxprof==0) {
      std::cout << "TProfile " << profname << " missing!" << std::endl;
      histvtxprof = histvtx->ProfileX("prof",1,-1,"");
    }
    histvtxprof->SetMarkerStyle(20);
    histvtxprof->SetMarkerSize(.4);
    /*
    TGraph tmp;
    for(unsigned bin=1;bin<histvtxprof->GetNbinsX()+1;++bin) {
      //      tmp.SetPoint(tmp.GetN(),histvtxprof->GetBinContent(bin),histvtxprof->GetBinCenter(bin));
      tmp.SetPoint(tmp.GetN(),histvtxprof->GetBinCenter(bin),histvtxprof->GetBinContent(bin));
    }
    tmp.SetMarkerStyle(20);
    tmp.Draw("p");
    */
    histvtxprof->Draw("esame");
    gPad->SetLogz(1);
    std::string plotfilename;
    plotfilename += outtrunk;
    plotfilename += shortname;
    plotfilename += "/";
    plotfilename += subdet;
    plotfilename += "vtx";
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += shortname;
    plotfilename += ".gif";
    gPad->Print(plotfilename.c_str());
    delete histvtx;
    gPad->SetLogz(0);
  }

  gStyle->SetOptStat(1111);

  ff.Close();
}

void ClusMultLumiCorrPlots(const char* fullname, const char* mod, const char* label,const char* postfix, const char* subdet, const char* shortname, const char* outtrunk) {


  char modfull[300];
  sprintf(modfull,"%s%s",mod,postfix);
  char labfull[300];
  sprintf(labfull,"%s%s",label,postfix);

  //  char fullname[300];
  //  sprintf(fullname,"rootfiles/Tracking_PFG_%s.root",filename);


  TFile ff(fullname);

  gStyle->SetOptStat(111111);

  char subdirname[300];
  sprintf(subdirname,"LumiCorr/%s",subdet);
  char histname[300];
  sprintf(histname,"n%sdigivslumi",subdet);
  char profname[300];
  sprintf(profname,"n%sdigivslumiprof",subdet);
  
  CommonAnalyzer ca(&ff,"",modfull,subdirname);


  TH2F* histlumi  = (TH2F*)ca.getObject(histname);
  if(histlumi) {
    histlumi->Draw("colz");
    //    TProfile* histlumiprof = histlumi->ProfileY("prof",1,-1,"");
    TProfile* histlumiprof = 0;
    histlumiprof = (TProfile*)ca.getObject(profname);
    if(histlumiprof==0) {
      std::cout << "TProfile " << profname << " missing!" << std::endl;
      histlumiprof = histlumi->ProfileX("prof",1,-1,"");
    }
    histlumiprof->SetMarkerStyle(20);
    histlumiprof->SetMarkerSize(.4);
    /*
    TGraph tmp;
    for(unsigned bin=1;bin<histlumiprof->GetNbinsX()+1;++bin) {
      //      tmp.SetPoint(tmp.GetN(),histlumiprof->GetBinContent(bin),histlumiprof->GetBinCenter(bin));
      tmp.SetPoint(tmp.GetN(),histlumiprof->GetBinCenter(bin),histlumiprof->GetBinContent(bin));
    }
    tmp.SetMarkerStyle(20);
    tmp.Draw("p");
    */
    histlumiprof->Draw("esame");
    gPad->SetLogz(1);
    std::string plotfilename;
    plotfilename += outtrunk;
    plotfilename += shortname;
    plotfilename += "/";
    plotfilename += subdet;
    plotfilename += "lumi";
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += shortname;
    plotfilename += ".gif";
    gPad->Print(plotfilename.c_str());
    delete histlumi;
    gPad->SetLogz(0);
  }

  gStyle->SetOptStat(1111);

  ff.Close();
}

