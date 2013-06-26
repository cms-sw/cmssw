{
#include <sstream>
#include <iostream>
#include <string.h> 
#include <vector.h> 
#include "triggerCscIdSector.h"

gROOT->Reset();
gROOT->SetStyle("Plain"); // to get rid of gray color of pad and have it white
gStyle->SetPalette(1,0); // 
std::ostringstream ss,ss1;

/// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// select run number
std::string run="62232";

//  input file with histograms
ss.str("");
ss<<"validationHists_"<<run<<".root";
TFile f_in(ss.str().c_str());
// output file with histograms
ss.str("");
ss<<"afeb_timing_"<<run<<".root";
TFile f_out(ss.str().c_str(),"RECREATE");

// folder in input file
std::string in_folder="AFEBTiming/";

/// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

/// common names and titles

// for CSC in given ME+- station/ring
std::vector<std::string> xTitle_ME_CSC;
xTitle_ME_CSC.push_back("ME-1/1 and ME+1/1 CSC #");
xTitle_ME_CSC.push_back("ME-1/2 and ME+1/2 CSC #");
xTitle_ME_CSC.push_back("ME-1/3 and ME+1/3 CSC #"); 
xTitle_ME_CSC.push_back("ME-2/1 and ME+2/1 CSC #");
xTitle_ME_CSC.push_back("ME-2/2 and ME+2/2 CSC #");
xTitle_ME_CSC.push_back("ME-3/1 and ME+3/1 CSC #");
xTitle_ME_CSC.push_back("ME-3/2 and ME+3/2 CSC #");
xTitle_ME_CSC.push_back("ME-4/1 and ME+4/1 CSC #");
xTitle_ME_CSC.push_back("ME-4/2 and ME+4/2 CSC #");

std::vector<std::string> xLabel_A_ME_CSC, xLabel_B_ME_CSC;
Int_t flag=2;
for(Int_t i=1;i<=36;i++) {
  
   ss.str("");
   if(flag==2) {
   Int_t cscnmb;
   if(i<19) cscnmb=i-19;
   if(i>18) cscnmb=i-18;
   if(cscnmb !=1) ss<<cscnmb;
   flag=0;
   if(cscnmb ==1) flag=1;
   }
   xLabel_A_ME_CSC.push_back(ss.str().c_str());
   flag++;
}

flag=4;
for(Int_t i=1;i<=72;i++) {
  
   ss.str("");
   if(flag==5) {
   Int_t cscnmb;
   if(i<37) cscnmb=i-37;
   if(i>36) cscnmb=i-36;
   if(cscnmb !=1) ss<<cscnmb;
   flag=0;
   if(cscnmb ==1) flag=1;
   }
   xLabel_B_ME_CSC.push_back(ss.str().c_str());
   flag++;
}

// for CSC in all ME
std::vector<std::string> yTitle_ME;
yTitle_ME.push_back("ME- 4/2"); yTitle_ME.push_back("ME- 4/1");
yTitle_ME.push_back("ME- 3/2"); yTitle_ME.push_back("ME- 3/1");
yTitle_ME.push_back("ME- 2/2"); yTitle_ME.push_back("ME- 2/1");
yTitle_ME.push_back("ME- 1/3"); yTitle_ME.push_back("ME- 1/2");
yTitle_ME.push_back("ME- 1/1");
yTitle_ME.push_back("ME+ 1/1");
yTitle_ME.push_back("ME+ 1/2"); yTitle_ME.push_back("ME+ 1/3");
yTitle_ME.push_back("ME+ 2/1"); yTitle_ME.push_back("ME+ 2/2");
yTitle_ME.push_back("ME+ 3/1"); yTitle_ME.push_back("ME+ 3/2");
yTitle_ME.push_back("ME+ 4/1"); yTitle_ME.push_back("ME+ 4/2");

// input hist names
std::vector<std::string> input_histName;
input_histName.push_back("afeb_time_bin_vs_afeb_occupancy_ME_");
input_histName.push_back("nmb_afeb_time_bins_vs_afeb_ME_");

// common titles
std::string title_afeb="AFEB";
std::string title_time_bin="Time Bin";
std::string title_nmb_time_bin="Nmb of Time Bins";
std::string title_entries="Entries";

// resulting hist names
std::string result_histName = "mean_afeb_time_bin_vs_afeb_csc_ME_";
std::string result_histTitle="AFEB Mean Time Bin";
std::string result_title_Y="AFEB";

std::string result_histNameEntries = "entries_afeb_time_bin_vs_afeb_csc_ME_";
std::string result_histTitleEntries="Entries AFEB Time Bin";

std::string result_histNameFr = "fraction_gt2_afeb_time_bin_vs_afeb_csc_ME_";
std::string result_histTitleFr="Fraction of (Nmb of AFEB time bins > 2), %";

std::string result_graphNameMean = "graph_mean_afeb_time_bin_vs_csc_ME_";
std::string result_graphTitleMean ="CSC Anode Mean Time Bin"; 

std::string result_histNameNmbAnodeTimeBins = "normal_nmb_afeb_time_bins_occupancy";
std::string result_histTitleNmbAnodeTimeBins ="Number of AFEB time bins normalized occupancy, %";

std::string result_graphNameMeanTrig = "graph_mean_afeb_time_bin_vs_trigger_csc_ME";
std::string result_graphTitleMeanTrig ="CSC Anode Mean Time Bin";


// folders in output hist file
f_out.cd();
f_out.mkdir("Summary");
f_out.mkdir("Results");
f_out.mkdir("Input_hists");
f_out.mkdir("Y_projections");
f_out.mkdir("Slices");


TH1F *hnnmboc_all;
TH2F *h2,*h2norm, *h_csc_me, *h2norm;
TH2F *hmean[9],*hnoc[9],*hnnmboc[9],*hentr[9],*hfrgt2[9];
TGraphErrors *gr_mean[9],*gr_mean_trig[2];
TCanvas *cgraph[9],*cgr_mean_trig[2];

Int_t gr_np[9];
Float_t gr_x[9][72];
Float_t gr_y[9][72];
Float_t gr_x_er[9][72]={0.0};
Float_t gr_y_er[9][72];

vector<Float_t> gr_y_trg_minus, gr_y_trg_plus,
                gr_y_trg_minus_er, gr_y_trg_plus_er,
                gr_x_trg_minus,gr_x_trg_plus,
                gr_x_trg_minus_er,gr_x_trg_plus_er;
std::vector<std::string> trg_minus_cscid, trg_plus_cscid;

// endcap, station, ring
Int_t esr[18]={111,112,113,121,122,131,132,141,142,
               211,212,213,221,222,231,232,241,242};
Int_t entries[18]={0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0};

// station, ring sequence, # of CSC and AFEB
Int_t sr[9]={11,12,13,21,22,31,32,41,42};
Int_t ncsc[9]={36,36,36,18,36,18,36,18,36};
Int_t nafeb[9]={18,24,12,42,24,36,24,36,24};
Int_t ncscmis=0;
vector<int> cscmis; cscmis.push_back(0);

/// book output histograms with all CSCs for given ME+- station,ring

for(Int_t isr=0;isr<9;isr++) {
  std::vector<std::string> xLabel_ME_CSC; xLabel_ME_CSC.clear();
     Int_t ny=nafeb[isr];
     Float_t ylow=1.0; Float_t yhigh=ny; yhigh=yhigh+1.0;
     if(ncsc[isr]==36) {
       Int_t nx=72; Float_t xlow=-35.0; Float_t xhigh=37.0;
       xLabel_ME_CSC=xLabel_B_ME_CSC;
     }
     if(ncsc[isr]==18) {
       Int_t nx=36; Float_t xlow=-17.0; Float_t xhigh=19.0;
       xLabel_ME_CSC=xLabel_A_ME_CSC;
     }
     gr_np[isr]=nx;
     for(Int_t n=0;n<nx;n++) {
        if(n==0) gr_x[isr][n]=xlow-1.0;
        if(n>0) {
          gr_x[isr][n]=gr_x[isr][n-1]+1.0;
          if(gr_x[isr][n]==0.0) gr_x[isr][n]=1.0;
	}  
     }
      
  // 2D hists for mean afeb time bin vs afeb and csc in given ME+-
     
     ss.str("");
     ss<<result_histName.c_str()<<sr[isr];
     ss1.str("");
     ss1<<result_histTitle<<" in run "<<run.c_str();
     hmean[isr]=new TH2F(ss.str().c_str(),ss1.str().c_str(),nx,xlow,xhigh,ny,ylow,yhigh);
     for(Int_t i=1;i<=nx;i++) hmean[isr]->GetXaxis()->SetBinLabel(i,xLabel_ME_CSC[i-1].c_str());
     hmean[isr]->SetStats(kFALSE);
     hmean[isr]->GetXaxis()->SetTitle(xTitle_ME_CSC[isr].c_str());
     hmean[isr]->GetYaxis()->SetTitle(result_title_Y.c_str());
     hmean[isr]->GetZaxis()->SetLabelSize(0.03);
     hmean[isr]->SetOption("COLZ");
     hmean[isr]->SetMinimum(4.0);
     hmean[isr]->SetMaximum(12.0);

  // 2D hists for entries afeb time bin vs afeb and csc in given ME+-
     ss.str("");
     ss<<result_histNameEntries.c_str()<<sr[isr];
     ss1.str("");
     ss1<<result_histTitleEntries<<" in run "<<run.c_str();
     hentr[isr]=new TH2F(ss.str().c_str(),ss1.str().c_str(),nx,xlow,xhigh,ny,ylow,yhigh);
     for(Int_t i=1;i<=nx;i++) hentr[isr]->GetXaxis()->SetBinLabel(i,xLabel_ME_CSC[i-1].c_str());
     hentr[isr]->SetStats(kFALSE);
     hentr[isr]->GetXaxis()->SetTitle(xTitle_ME_CSC[isr].c_str());
     hentr[isr]->GetYaxis()->SetTitle(result_title_Y.c_str());
     hentr[isr]->GetZaxis()->SetLabelSize(0.03);
     hentr[isr]->SetOption("COLZ");

  // 2D hists for fraction of nmb of afeb time bins > 2 vs afeb and csc in ME+-
     ss.str("");
     ss<<result_histNameFr.c_str()<<sr[isr];
     ss1.str("");
     ss1<<result_histTitleFr<<" in run "<<run.c_str();
     hfrgt2[isr]=new TH2F(ss.str().c_str(),ss1.str().c_str(),nx,xlow,xhigh,ny,ylow,yhigh);
     for(Int_t i=1;i<=nx;i++) hfrgt2[isr]->GetXaxis()->SetBinLabel(i,xLabel_ME_CSC[i-1].c_str());
     hfrgt2[isr]->SetStats(kFALSE);
     hfrgt2[isr]->GetXaxis()->SetTitle(xTitle_ME_CSC[isr].c_str());
     hfrgt2[isr]->GetYaxis()->SetTitle(result_title_Y.c_str());
     hfrgt2[isr]->GetZaxis()->SetLabelSize(0.03);
     hfrgt2[isr]->SetOption("COLZ");
     hfrgt2[isr]->SetMinimum(0.0);
     hfrgt2[isr]->SetMaximum(20.0);

     // 2D hists for normalized AFEB time bin occupancy, % 
     // vs afeb and csc in ME+-
     ss.str(""); ss1.str("");
     ss<<"normal_afeb_time_bin_vs_csc_ME_"<<sr[isr];
     ss1<<"Normalized AFEB time bin occupancy, %"<<" in run "<<run.c_str();
     hnoc[isr]=new TH2F(ss.str().c_str(),ss1.str().c_str(),nx,xlow,xhigh,16,0.0,16.0);
     for(Int_t i=1;i<=nx;i++) hnoc[isr]->GetXaxis()->SetBinLabel(i,xLabel_ME_CSC[i-1].c_str());
     hnoc[isr]->SetStats(kFALSE);
     hnoc[isr]->GetXaxis()->SetTitle(xTitle_ME_CSC[isr].c_str());
     hnoc[isr]->GetYaxis()->SetTitle(title_time_bin.c_str());
     hnoc[isr]->GetZaxis()->SetLabelSize(0.03);
     hnoc[isr]->SetOption("COLZ");
     hnoc[isr]->SetMinimum(0.0);
     hnoc[isr]->SetMaximum(100.0);
 
     // 2D hists for normalized number of AFEB time bin occupancy, % 
     // vs afeb and csc in ME+ 
     ss.str(""); ss1.str("");
     ss<<"normal_nmb_afeb_time_bins_vs_csc_ME_"<<sr[isr];
     ss1<<"Normalized Nmb of AFEB time bins occupancy, %"<<" in run "<<run.c_str();
     hnnmboc[isr]=new TH2F(ss.str().c_str(),ss1.str().c_str(),nx,xlow,xhigh,16,0.0,16.0);
     for(Int_t i=1;i<=nx;i++) hnnmboc[isr]->GetXaxis()->SetBinLabel(i,xLabel_ME_CSC[i-1].c_str());
     hnnmboc[isr]->SetStats(kFALSE);
     hnnmboc[isr]->GetXaxis()->SetTitle(xTitle_ME_CSC[isr].c_str());
     hnnmboc[isr]->GetYaxis()->SetTitle(title_nmb_time_bin.c_str());
     hnnmboc[isr]->GetZaxis()->SetLabelSize(0.03);
     hnnmboc[isr]->SetOption("COLZ");
     hnnmboc[isr]->SetMinimum(0.0);
     hnnmboc[isr]->SetMaximum(100.0);
} // end of for(Int_t isr=0

/// Histogram of number of anode time bins ON for all afebs
     ss.str(""); ss1.str("");
     ss<<result_histNameNmbAnodeTimeBins.c_str();
     ss1<<result_histTitleNmbAnodeTimeBins.c_str()<<" in run "<<run.c_str();
     hnnmboc_all=new TH1F(ss.str().c_str(),ss1.str().c_str(),16,0.0,16.0);
     hnnmboc_all->GetXaxis()->SetTitle(title_nmb_time_bin.c_str());
     hnnmboc_all->GetYaxis()->SetTitle("Normalized occupancy, %");
     
     hnnmboc_all->SetFillColor(4);
     
//***TCanvas *c1=new TCanvas("c1","canvas");

//***c1->cd();

/// get two types of input hists and analyze them

for(Int_t inp=1;inp<=2;inp++) {
  ss.str("");
  if(inp==1) ss<<"mean_afeb_time_bin_vs_csc_ME";
  if(inp==2) ss<<"rel_max_nmb_afeb_time_bins_vs_csc_ME";
  ss1.str("");
  if(inp==1) ss1<<"Mean AFEB time bin vs CSC and ME"<<" in run "<<run.c_str();
  if(inp==2) ss1<<"Max. Fraction of (Nmb of AFEB time bins > 2), %"<<" in run "<<run.c_str();
  gStyle->SetOptStat(0);

  // book two output 2D hists for mean time and number of entris vs CSC and ME
  h_csc_me=new TH2F(ss.str().c_str(),ss1.str().c_str(),36,1.0,37.0,18,1.0,19.0);
  h_csc_me->SetStats(kFALSE);
  h_csc_me->GetXaxis()->SetTitle("CSC #");
  for(Int_t i=1;i<=18;i++) h_csc_me->GetYaxis()->SetBinLabel(i,yTitle_ME[i-1].c_str());
  h_csc_me->GetZaxis()->SetLabelSize(0.03);
  h_csc_me->SetOption("COLZ");
  if(inp==1) {
    h_csc_me->SetMinimum(4.0);
    h_csc_me->SetMaximum(12.0);
  }
  if(inp==2) {
    h_csc_me->SetMinimum(0.0);
    h_csc_me->SetMaximum(20.0);
  }
  // cycle over 18 endcap/station/ring combinations

for(Int_t jesr=0;jesr<18;jesr++) {
  if(esr[jesr] != 142 && esr[jesr] != 242) {
  Int_t me;
  if(jesr<9) me=10+jesr;
  if(jesr>8) me=18-jesr;

  Int_t indisr;
  if(esr[jesr] < 200) indisr=esr[jesr]-100;
  if(esr[jesr] > 200) indisr=esr[jesr]-200;

  // define index for vector sr[9] (station,ring)
  for(Int_t i=0;i<8;i++) if(sr[i]==indisr) indisr=i; 

  // cycle over CSC 
  for(Int_t csc=1;csc<(ncsc[indisr]+1);csc++) {
     Int_t cscbin;
     // define bin cscbin in hists for ME+-
     if(esr[jesr]<200) cscbin=ncsc[indisr]+csc;
     if(esr[jesr]>200) cscbin=ncsc[indisr]-csc+1;

     Int_t idchamber=esr[jesr]*100+csc;
     ss.str("");
     ss<<in_folder.c_str()<<input_histName[inp-1].c_str()<<idchamber;
     f_in.cd();
     //gStyle->SetOptStat(0000011);
     TH2F *h2 = (TH2F*)f_in.Get(ss.str().c_str());
     if(h2 ==NULL && inp==1) {
       ncscmis++;
       cscmis.push_back(idchamber);
       //       std::cout<<"No chamber "<<idchamber<<std::endl;
     }
   if(h2 != NULL) {
     ss.str(""); ss1.str("");
     ss<<input_histName[inp-1].c_str()<<idchamber<<"_norm";
     ss1<<h2->GetTitle()<<" normalized, %"<<" in run "<<run.c_str();
     if(inp==1) {Int_t ny=16; Float_t ylow=0.0; Float_t yhigh=16.0;}
     if(inp==2) {Int_t ny=16; Float_t ylow=0.0; Float_t yhigh=16.0;}
     Int_t nx=nafeb[indisr]; Float_t xlow=1.0; Float_t xhigh=xlow+nx;
     h2norm=new TH2F(ss.str().c_str(),ss1.str().c_str(),nx,xlow,xhigh,ny,ylow,yhigh);
     h2norm->SetStats(kFALSE);
     
     f_out.cd("Input_hists");

     // saving original (modified to fraction in %), adding X,Y titles, 
     //color and "COLZ" option

     for(Int_t i=1;i<=h2->GetNbinsX();i++) {
       Float_t sum=0.0;
       for(Int_t j=1;j<=h2->GetNbinsY();j++) sum=sum+h2->GetBinContent(i,j);
       if(sum>0.0) {
         Float_t w;
         for(Int_t j=1;j<=h2->GetNbinsY();j++) {
	    w=100.0*h2->GetBinContent(i,j)/sum;
	    h2norm->SetBinContent(i,j,w);
	 }
       }
     }

     h2norm->GetXaxis()->SetTitle(title_afeb.c_str());
     if(inp==1) h2norm->GetYaxis()->SetTitle(title_time_bin.c_str());
     if(inp==2) h2norm->GetYaxis()->SetTitle(title_nmb_time_bin.c_str());
     h2norm->GetYaxis()->SetTitleOffset(1.2);
     h2norm->SetOption("COLZ");
     h2norm->Write();
 
     // saving Y projection of the whole 2D hist for given chamber

     ss.str("");
     ss<<input_histName[inp-1].c_str()<<idchamber<<"_Y_all";
     TH1D *h1d = h2->ProjectionY(ss.str().c_str(),1,h2->GetNbinsX(),"");
     if(inp==1) h1d->GetXaxis()->SetTitle(title_time_bin.c_str());
     if(inp==2) h1d->GetXaxis()->SetTitle(title_nmb_time_bin.c_str());
     h1d->GetYaxis()->SetTitle(title_entries.c_str());
     h1d->GetYaxis()->SetTitleOffset(1.2);
     gStyle->SetOptStat(1001111);
     f_out.cd("Y_projections");
     h1d->SetFillColor(4);
     h1d->Write();

     if(h1d->GetEntries() > 0) {
       Float_t entr=h1d->GetEntries();
       for(Int_t m=1; m<=h1d->GetNbinsX();m++) {
	 Float_t w=h1d->GetBinContent(m);
         if(inp==2) {
           Float_t fm=h1d->GetBinCenter(m);
           hnnmboc_all->Fill(fm,w);
         }
         w=100.0*w/entr;
         if(inp==1) hnoc[indisr]->SetBinContent(cscbin,m,w);
         if(inp==2) hnnmboc[indisr]->SetBinContent(cscbin,m,w);
       }
       
       Float_t mean=h1d->GetMean()-0.5;

       if(inp==1) {
         gr_y[indisr][cscbin-1]=mean;
         gr_y_er[indisr][cscbin-1]=h1d->GetMeanError(1);
         Float_t grmean=mean;

         if(mean<4.0) mean=4.0;
         if(mean>12.0) mean=12.0;
	 Int_t imean=mean; mean=imean;
         h_csc_me->SetBinContent(csc,me,mean);

         // for trigger sector graphs
         Int_t station=sr[indisr]/10;
         Int_t ring=sr[indisr]-station*10;
         Int_t sector= triggerSector(station,ring,csc);
         if(esr[jesr] > 200) sector=sector+6;
         Int_t trigcscid=triggerCscId(station,ring,csc);
	 /* Like in D. Wang presentation, though maybe different in ME1/1 */
         Float_t xtrigcsc = sector*100+station*20+trigcscid;
         ss.str("");
	 if(esr[jesr] < 200) { // ME+
           gr_y_trg_plus.push_back(grmean);
           gr_y_trg_plus_er.push_back(h1d->GetMeanError(1));
           gr_x_trg_plus.push_back(xtrigcsc);
           gr_x_trg_plus_er.push_back(0.0);
           ss<<"ME+"<<station<<"/"<<ring<<"/"<<csc;
           trg_plus_cscid.push_back(ss.str().c_str());
           ss.str("");  
	 }
	 if(esr[jesr] > 200) { // ME-
           gr_y_trg_minus.push_back(grmean);
           gr_y_trg_minus_er.push_back(h1d->GetMeanError(1));
           gr_x_trg_minus.push_back(xtrigcsc);
           gr_x_trg_minus_er.push_back(0.0);
           ss<<"ME-"<<station<<"/"<<ring<<"/"<<csc;
           trg_minus_cscid.push_back(ss.str().c_str());
           ss.str("");  
	 }
       }
     }
     delete h1d;   

     // saving slices, finding MEAN in each slice, fill 2D hist
     f_out.cd("Slices");
     Float_t maxsum=0.0;
     for(Int_t j=1;j<=h2->GetNbinsX();j++) {
        Int_t n=j;
        ss.str("");
        ss<<input_histName[inp-1].c_str()<<idchamber<<"_Y_"<<n;
        TH1D *h1d = h2->ProjectionY(ss.str().c_str(),j,j,"");
	//std::cout<<cscbin<<" "<<j<<" "<<h1d->GetEntries()<<std::endl;
        Float_t entr=h1d->GetEntries();
        if(entr > 0.0) {
          Float_t sum=entr-(h1d->GetBinContent(2)+h1d->GetBinContent(3));
          sum=100.0*sum/entr;
          if(sum>maxsum) maxsum=sum;
	  //  Float_t mean=h1d->GetMean();
          Float_t mean=h1d->GetMean()-0.5; // since June 03, 2008
          entries[jesr]=entries[jesr]+1;
          if(inp==1) {
            if(mean<4.0) mean=4.0;
            if(mean>12.0) mean=12.0;
	    Int_t imean=mean; mean=imean;
           hmean[indisr]->SetBinContent(cscbin,j,mean);
           hentr[indisr]->SetBinContent(cscbin,j,entr);
	  }
          if(inp==2) {
            if(sum>20.0) sum=20.0;
            hfrgt2[indisr]->SetBinContent(cscbin,j,sum);
	  }
          ss.str("");
          ss<<title_afeb<<" "<<n;
          h1d->GetXaxis()->SetTitle(ss.str().c_str());
          h1d->GetYaxis()->SetTitle(title_entries.c_str());
          h1d->GetYaxis()->SetTitleOffset(1.2);
          gStyle->SetOptStat(1001111);
          h1d->SetFillColor(4);
          h1d->Write();
	}
        delete h1d;
     }
     if(inp==2 && maxsum > 0.0) {
       if(maxsum > 20.0) maxsum=20.0;
       h_csc_me->SetBinContent(csc,me,maxsum);
     }
     delete h2norm;
   }
     delete h2;
  }
  } // end if not ME42
} // end of for(jesr=0
   f_out.cd("Summary");
   h_csc_me->Write();
   delete h_csc_me;    

} // end of for(inp=1

/// Write others resulting hists

f_out.cd("Results");
for(Int_t isr=0;isr<8;isr++) {
  if(hmean[isr] != NULL) {
       hmean[isr]->SetStats(kFALSE);
       hmean[isr]->Write();
  }
  if(hentr[isr] != NULL) {
       hentr[isr]->SetStats(kFALSE);
       hentr[isr]->Write();
  }
  if(hnoc[isr] != NULL) hnoc[isr]->Write();

  if(hnnmboc[isr] != NULL)     hnnmboc[isr]->Write();
  if(hfrgt2[isr] != NULL)     hfrgt2[isr]->Write();
}

Int_t movf=hnnmboc_all->GetNbinsX()+1; // overflow bin    
Double_t norm=100.0/(hnnmboc_all->Integral()+hnnmboc_all->GetBinContent(movf));
hnnmboc_all->Scale(norm);

c_hnnmboc_all_clone= new TCanvas(result_histNameNmbAnodeTimeBins.c_str());
c_hnnmboc_all_clone->cd();
c_hnnmboc_all_clone->SetLogy();
hnnmboc_all->SetStats(kFALSE);

hnnmboc_all->Draw();
TText txt;
txt.SetTextSize(0.05);

Float_t fr=hnnmboc_all->Integral()+hnnmboc_all->GetBinContent(movf)-
                                   hnnmboc_all->GetBinContent(1)-
                                   hnnmboc_all->GetBinContent(2)-
                                   hnnmboc_all->GetBinContent(3);
Int_t ifr=fr*1000.0;
fr=ifr; fr=fr/1000.0;
ss.str("");
ss<<"Nmb of Time Bins > 2 = "<<fr<<" %";
txt.DrawTextNDC(0.35,0.8,ss.str().c_str());

c_hnnmboc_all_clone->Update();
f_out.cd("Summary");
c_hnnmboc_all_clone->Write();


for(Int_t isr=0;isr<8;isr++) {
   delete hmean[isr]; delete hentr[isr]; delete hnoc[isr]; delete hnnmboc[isr];
   delete  hfrgt2[isr];
}

/// Report missing CSCs
std::cout<<"Missing csc "<<ncscmis<<std::endl;
if(ncscmis > 0) {
  for(Int_t jesr=0;jesr<18;jesr++) {
    Int_t n=0,npr=0;
    for(Int_t i=0;i<cscmis.size();i++) {
      Int_t idesr=cscmis[i]/100;
      if(esr[jesr]==idesr && n<10) {n++;npr++;std::cout<<cscmis[i]<<"   ";}
      if(n==10) {std::cout<<std::endl; n=0;} 
    }
    if(npr>0) std::cout<<std::endl;
  }
}

/// Plot and save graphs

// mean anode time bin vs CSC in given ME+-
for(Int_t isr=0;isr<8;isr++) {
     ss.str("");
     ss<<result_graphNameMean.c_str()<<sr[isr];
     std::cout<<isr+1<<" "<<ss.str().c_str()<<std::endl;
     cgraph[isr]= new TCanvas(ss.str().c_str());
     cgraph[isr]->cd();
     cgraph[isr]->SetGrid();
     Float_t x[72],y[72],ery[72],erx[72];
     if(gr_np[isr]>0) {
      for(Int_t n=0;n<gr_np[isr];n++) {
        x[n]=gr_x[isr][n];
        y[n]=gr_y[isr][n];
        erx[n]=0.0; 
        ery[n]=gr_y_er[isr][n];
      }
      gr_mean[isr]=new TGraphErrors(gr_np[isr],x,y,erx,ery);  
      ss1.str("");
      ss1<<result_graphTitleMean<<" in run "<<run.c_str();
      gr_mean[isr]->SetTitle(ss1.str().c_str());
      gr_mean[isr]->GetXaxis()->SetTitle(xTitle_ME_CSC[isr].c_str());
      gr_mean[isr]->GetYaxis()->SetTitle(title_time_bin.c_str());
      gr_mean[isr]->SetMinimum(4.0); 
      gr_mean[isr]->SetMaximum(12.0); 
      gr_mean[isr]->SetMarkerStyle(20);
      gr_mean[isr]->SetMarkerColor(4);
      gr_mean[isr]->SetMarkerSize(1.2);
      gr_mean[isr]->SetLineColor(4);    // Blue error bar
      gr_mean[isr]->Draw("APZ"); 
      cgraph[isr]->Update();
      f_out.cd("Results");
      cgraph[isr]->Write();
     }
}

// Text output for anode mean time vs CSC

std::cout<<std::endl;
for(Int_t i=0;i<trg_plus_cscid.size();i++) 
  //printf("%s %7.2f \n",trg_plus_cscid[i],gr_y_trg_plus[i]); 
  std::cout<<trg_plus_cscid[i]<<"   "<<gr_y_trg_plus[i]<<std::endl;
for(Int_t i=0;i<trg_minus_cscid.size();i++) 
  std::cout<<trg_minus_cscid[i]<<"   "<<gr_y_trg_minus[i]<<std::endl;

std::cout<<std::endl;
std::cout<<"Total CSCs "<<trg_plus_cscid.size()+trg_minus_cscid.size()<<std::endl;
std::cout<<std::endl;

// mean anode time bin vs csc in trigger sector for ME+ and ME-

Float_t x_tr[234],y_tr[234],ery_tr[234],erx_tr[234];
for(Int_t i=0;i<2;i++) {
     ss.str("");
     if(i==0) ss<<result_graphNameMeanTrig.c_str()<<"+";
     if(i==1) ss<<result_graphNameMeanTrig.c_str()<<"-";    
     if(i==0) Int_t x_size=gr_x_trg_plus.size();
     if(i==1) Int_t x_size=gr_x_trg_minus.size();
     std::cout<<i+1<<" "<<x_size<<std::endl;
     if(x_size > 0) {
      cgr_mean_trig[i] = new TCanvas(ss.str().c_str());
      cgr_mean_trig[i]->cd();
      cgr_mean_trig[i]->SetGrid();
      for(Int_t n=0;n<x_size;n++) {
        if(i==0) {
         y_tr[n]=gr_y_trg_plus[n];
         ery_tr[n]=gr_y_trg_plus_er[n];
        }
        if(i==1) {
         y_tr[n]=gr_y_trg_minus[n];
         ery_tr[n]=gr_y_trg_minus_er[n];
        }
        x_tr[n]=n+1;
        erx_tr[n]=0.0; 
      }
      gr_mean_trig[i]=new TGraphErrors(x_size,x_tr,y_tr,erx_tr,ery_tr);  
      ss1.str("");
      if(i==0) ss1<<"ME+ "<<result_graphTitleMeanTrig<<" in run "<<run.c_str();
      if(i==1) ss1<<"ME- "<<result_graphTitleMeanTrig<<" in run "<<run.c_str();
      gr_mean_trig[i]->SetTitle(ss1.str().c_str());
      gr_mean_trig[i]->GetXaxis()->SetTitle("ChamberId=SectorId*100+StationId*10+CSCID in increasing order");
      gr_mean_trig[i]->GetYaxis()->SetTitle(title_time_bin.c_str());
      gr_mean_trig[i]->SetMinimum(4.0);
      gr_mean_trig[i]->SetMaximum(12.0);
      gr_mean_trig[i]->SetMarkerStyle(20);
      gr_mean_trig[i]->SetMarkerColor(4);
      gr_mean_trig[i]->SetMarkerSize(1.0);
      gr_mean_trig[i]->SetLineColor(4);    // Blue error bar
      gr_mean_trig[i]->Draw("APZ"); 
      cgr_mean_trig[i]->Update();
      f_out.cd("Summary");
      cgr_mean_trig[i]->Write();
     }
}
f_out.Close();
}
