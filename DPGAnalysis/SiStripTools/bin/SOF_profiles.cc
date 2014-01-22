#include "SOF_profiles.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TProfile.h"
#include "TH1F.h"
#include "TDirectory.h"
#include "TLine.h"
#include "TGaxis.h"
#include "TLegend.h"
#include <iostream>

TCanvas* printSOF(TFile* file, const int run, const int firstLS,const int zoom) {

  TCanvas* canout = 0;

  char rname[400];
  sprintf(rname,"run_%d",run);

  if(file==0) return canout;

  TProfile* badmod = 0;
  TH1F* evtanydcs = 0;
  TH1F* evtdcson = 0;
  TH1F* evtnostrip = 0;

  char dname[400];

  sprintf(dname,"ssqhistory/%s",rname);
  if(file->cd(dname)) { badmod = (TProfile*)gDirectory->Get("badmodrun_HVDCS");  }

  sprintf(dname,"eventtimedistranydcs/%s",rname);
  if(file->cd(dname)) { evtanydcs = (TH1F*)gDirectory->Get("orbit");  }

  sprintf(dname,"eventtimedistribution/%s",rname);
  if(file->cd(dname)) { evtdcson = (TH1F*)gDirectory->Get("orbit");  }

  sprintf(dname,"eventtimedistrnostrip/%s",rname);
  if(file->cd(dname)) { evtnostrip = (TH1F*)gDirectory->Get("orbit");  }

  if(badmod && evtanydcs && evtdcson && evtnostrip) {

    badmod->SetStats(kFALSE);
    badmod->SetLineColor(kGreen);    badmod->SetMarkerColor(kGreen);
    evtanydcs->SetStats(kFALSE);
    evtanydcs->SetLineColor(kBlue);
    evtdcson->SetStats(kFALSE);
    evtdcson->SetLineColor(kRed);
    evtnostrip->SetStats(kFALSE);
    evtnostrip->SetLineColor(kRed);
    evtnostrip->SetFillColor(kRed);    evtnostrip->SetFillStyle(1000);

    canout = new TCanvas;

    badmod->Draw();
    badmod->GetYaxis()->SetRangeUser(1,100000);    badmod->GetYaxis()->SetTitle("");   badmod->SetTitle(rname);
    badmod->GetXaxis()->SetRangeUser((firstLS-zoom)*262144,(firstLS+zoom)*262144);
    evtanydcs->Draw("same");
    evtdcson->Draw("same");
    evtnostrip->Rebin(int(badmod->GetBinWidth(1)/evtnostrip->GetBinWidth(1)));
    std::cout << "rebin " << int(badmod->GetBinWidth(1)/evtnostrip->GetBinWidth(1)) << std::endl;
    evtnostrip->Draw("same");

    TGaxis* lsaxis = new TGaxis((firstLS-zoom)*262144,100000,(firstLS+zoom)*262144,100000,
				firstLS-zoom+1,firstLS+zoom+1,2*zoom,"-SM");
    //    TGaxis* lsaxis = new TGaxis(badmod->GetXaxis()->GetXmin(),100000,
    //				badmod->GetXaxis()->GetXmax(),100000,
    //				badmod->GetXaxis()->GetXmin()/262144+1,badmod->GetXaxis()->GetXmax()/262144+1,50,"-SM");
    lsaxis->Draw();

    TLine* line = new TLine((firstLS-1)*262144,1,(firstLS-1)*262144,100000);
    line->SetLineWidth(2);    line->SetLineStyle(2);
    line->Draw();

    TLegend* leg = new TLegend(.5,.65,.9,.85,"");
    leg->AddEntry(badmod,"Modules with HV off","l");
    leg->AddEntry(evtanydcs,"Events with any DCS bit","l");
    leg->AddEntry(evtdcson,"Events with DCS bit ON","l");
    leg->AddEntry(evtnostrip,"DCS bit ON No strip clus (masked FED)","f");
    leg->AddEntry(line,"first good LS (DCSonly JSON)","l");

    leg->SetFillStyle(0);
    leg->Draw();
    
    canout->SetLogy(1);
    

  }
  
  return canout;
}
