{
#include <sstream>
#include <iostream>
#include <string.h>
#include <vector.h>
#include <math.h>

gROOT->Reset();
gROOT->SetStyle("Plain"); // to get rid of gray color of pad and have it white
gStyle->SetPalette(1,0); // 
std::ostringstream ss,ss1;

/// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// select run number
std::string run="62232";
Int_t nminentries=50;

//  input file with histograms
ss.str("");
ss<<"validationHists_"<<run<<".root";
TFile f_in(ss.str().c_str());

// output file with histograms
ss.str("");
ss<<"gas_gain_fit_"<<run<<".root";
TFile f_out(ss.str().c_str(),"RECREATE");

// folder in input file
std::string in_folder="GasGain/";

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
  std::string input_histName = "gas_gain_rechit_adc_3_3_sum_location_ME_";  
  std::string input_title_X="Location=(layer-1)*nsegm+segm";
  std::string input_title_Y="3X3 ADC Sum";

  std::string slice_title_X="3X3 ADC Sum Location";

  Int_t ny=30;
  Float_t ylow=1.0, yhigh=31.0;
  std::string result_histName = "landau_peak_position_vs_location_csc_ME_";
  std::string result_histTitle="Landau peak position";
  std::string result_title_Y="Location=(layer-1)*nsegm+segm";

  std::string result_histNameError = "landau_peak_position_error_vs_location_csc_ME_";
  std::string result_histTitleError="Landau peak position error";

  std::string result_histNameEntries = "entries_gas_gain_vs_location_csc_ME_";
  std::string result_histTitleEntries="Entries 3X3 ADC Sum";

  std::string result_histNameLandauPeakPositionME = "landau_peak_position_ME";
  std::string result_histTitleLandauPeakPositionME="Landau peak position for ME";
  std::string result_title_X_LandauPeakPositionME="Landau peak position";
  std::string result_title_Y_LandauPeakPositionME="Entries";

  std::string result_histNameADCSum = "adcsum_ME_";
  std::string result_histTitleADCSum="3X3 ADC Sum for ME";
  std::string result_title_X_ADCSum="3X3 ADC Sum";
  std::string result_title_Y_ADCSum="Entries";

  std::string result_histNameDeltaHV = "delta_hv";
  std::string result_histTitleDeltaHV="Gas Gain equalizing Delta(HV)";
  std::string result_title_X_DeltaHV="Delta(HV), volts";
  std::string result_title_Y_DeltaHV="Entries";

std::string result_graphNameLandauCSCME = "graph_landau_peak_position_csc_ME_";
std::string result_graphTitleLandauCSCME = "Landau peak position vs CSC ME";

f_out.cd();
f_out.mkdir("Input_hists");
f_out.mkdir("Y_projections");
f_out.mkdir("Slices");
f_out.mkdir("Results");


TH2F *h2;
TH2F *h2mod;
TH2F *h;
TH2F *hlpp[9],*hlpperror[9],*hentr[9];
TH1F *hlppme[18],*hadcsum[9];
TH1F *hdeltahv, *widthlppratioME11,*widthlppratioME11, *statusflag;
TH2F *h_min_csc_me,*h_max_csc_me, *h2mod;

TGraphErrors *gr_lpp_csc_me[9];
TCanvas *cgraph[9];

vector<Float_t> gr_y[9],gr_y_er[9],gr_x[9];

Int_t esr[18]={111,112,113,121,122,131,132,141,142,
               211,212,213,221,222,231,232,241,242};
Int_t entries[18]={0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0};

// station, ring sequence, # of CSC and HV segments
Int_t sr[9]={11,12,13,21,22,31,32,41,42};
Int_t ncsc[9]={36,36,36,18,36,18,36,18,36};
Int_t nhvsegm[9]={6,18,18,18,30,18,30,18,30};   // number of HV segments per CSC              
Float_t dhv_coeff[9]={158.0,190.0,190.0,190.0,190.0,190.0,190.0,190.0,190.0};
Int_t nentr[9]={0};
Int_t ncscmis=0;
Int_t indsegm=0;
vector<Int_t> cscmis; cscmis.push_back(0);

vector<Int_t> selflag,locid, nentrloc;
selflag.clear(); locid.clear(); nentrloc.clear();

vector<Float_t> peak, peakpos, peakwidth, peakposer, dhvv, meanadclpp;
peak.clear(); peakpos.clear(); peakwidth.clear(); peakposer.clear(); 
dhvv.clear(); meanadclpp.clear();

Int_t locid_cur, nentrloc_cur, selflag_cur;
Float_t peak_cur, peakpos_cur, peakwidth_cur, peakposer_cur, dhv_cur;


/// book output histograms with all CSCs for given ME+- station,ring

for(Int_t isr=0;isr<8;isr++) {
  std::vector<std::string> xLabel_ME_CSC; xLabel_ME_CSC.clear();
     Int_t ny=nhvsegm[isr];
     Float_t ylow=1.0; Float_t yhigh=ny; yhigh=yhigh+1.0;
     if(ncsc[isr]==36) {
       Int_t nx=72; Float_t xlow=-35.0; Float_t xhigh=37.0;
       xLabel_ME_CSC=xLabel_B_ME_CSC;
     }
     if(ncsc[isr]==18) {
       Int_t nx=36; Float_t xlow=-17.0; Float_t xhigh=19.0;
       xLabel_ME_CSC=xLabel_A_ME_CSC;
     }

  // 2D hists for Landau peak in 3X3 ADC Sum and its error vs hv segment and 
  // csc in ME+-
     
     ss.str("");
     ss<<result_histName.c_str()<<sr[isr];
     ss1.str("");
     ss1<<result_histTitle<<" in run "<<run.c_str();
     hlpp[isr]=new TH2F(ss.str().c_str(),ss1.str().c_str(),nx,xlow,xhigh,ny,ylow,yhigh);
     for(Int_t i=1;i<=nx;i++) hlpp[isr]->GetXaxis()->SetBinLabel(i,xLabel_ME_CSC[i-1].c_str());
     hlpp[isr]->SetStats(kFALSE);
     hlpp[isr]->GetXaxis()->SetTitle(xTitle_ME_CSC[isr].c_str());
     hlpp[isr]->GetYaxis()->SetTitle(result_title_Y.c_str());
     hlpp[isr]->GetZaxis()->SetLabelSize(0.03);
     hlpp[isr]->SetOption("COLZ");
     hlpp[isr]->SetMinimum(0.0);
     hlpp[isr]->SetMaximum(2000.0);

     ss.str("");
     ss<<result_histNameError.c_str()<<sr[isr];
     ss1.str("");
     ss1<<result_histTitleError<<" in run "<<run.c_str();
     hlpperror[isr]=new TH2F(ss.str().c_str(),ss1.str().c_str(),nx,xlow,xhigh,ny,ylow,yhigh);
     for(Int_t i=1;i<=nx;i++) hlpperror[isr]->GetXaxis()->SetBinLabel(i,xLabel_ME_CSC[i-1].c_str());
     hlpperror[isr]->SetStats(kFALSE);
     hlpperror[isr]->GetXaxis()->SetTitle(xTitle_ME_CSC[isr].c_str());
     hlpperror[isr]->GetYaxis()->SetTitle(result_title_Y.c_str());
     hlpperror[isr]->GetZaxis()->SetLabelSize(0.03);
     hlpperror[isr]->SetOption("COLZ");
     hlpperror[isr]->SetMinimum(0.0);
     hlpperror[isr]->SetMaximum(100.0);


  // 2D hists for entries 3X3 ADC Sum vs hv segment and csc in given ME+-
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

     // 1D hists for 3X3 ADC Sum distribution in given ME 
     ss.str("");
     ss<< result_histNameADCSum <<sr[isr];
     ss1.str("");
     ss1<< result_histTitleADCSum<<sr[isr] <<" in run "<<run.c_str();
     hadcsum[isr]=new TH1F(ss.str().c_str(),ss1.str().c_str(),50,0.0,2000.0);
     
     hadcsum[isr]->GetXaxis()->SetTitle(result_title_X_ADCSum.c_str());
     hadcsum[isr]->GetYaxis()->SetTitle(result_title_Y_ADCSum.c_str());
     hadcsum[isr]->SetFillColor(4);

} // end of for(Int_t isr=0

  
  // 1D hists for gas gain as Landau peak position per csc in given ME 
for(Int_t jesr=0;jesr<18;jesr++) {
  if(esr[jesr] != 142 && esr[jesr] != 242) {
    if(esr[jesr] < 200) {Int_t isr=jesr;   std::string endcapsign="+";}
    if(esr[jesr] > 200) {Int_t isr=jesr-9; std::string endcapsign="-";}
     ss.str("");
     ss<< result_histNameLandauPeakPositionME <<endcapsign.c_str()<<sr[isr];
     ss1.str("");
     ss1<< result_histTitleLandauPeakPositionME<<endcapsign.c_str()<<sr[isr] <<" in run "<<run.c_str();
     hlppme[jesr]=new TH1F(ss.str().c_str(),ss1.str().c_str(),80,0.0,1600.0);
     
     hlppme[jesr]->GetXaxis()->SetTitle(result_title_X_LandauPeakPositionME.c_str());
     hlppme[jesr]->GetYaxis()->SetTitle(result_title_Y_LandauPeakPositionME.c_str());
     hlppme[jesr]->SetFillColor(4);
  }
}
// book two output 2D hists for min. and max. gas gain vs CSC and ME

  ss.str("");
  ss<<"min_landau_peak_position_csc_ME";
  ss1.str("");
  ss1<<"Min. Landau peak position vs CSC and ME"<<" in run "<<run.c_str();
  h_min_csc_me=new TH2F(ss.str().c_str(),ss1.str().c_str(),36,1.0,37.0,18,1.0,19.0);
  h_min_csc_me->SetStats(kFALSE);
  h_min_csc_me->GetXaxis()->SetTitle("CSC #");
  for(Int_t i=1;i<=18;i++) h_min_csc_me->GetYaxis()->SetBinLabel(i,yTitle_ME[i-1].c_str());
  h_min_csc_me->GetZaxis()->SetLabelSize(0.03);
  h_min_csc_me->SetOption("COLZ");
  h_min_csc_me->SetMinimum(0.0);
  h_min_csc_me->SetMaximum(2000.0);
 
  ss.str("");
  ss<<"max_landau_peak_position_csc_ME";
  ss1.str("");
  ss1<<"Max. Landau peak position vs CSC and ME"<<" in run "<<run.c_str();
  h_max_csc_me=new TH2F(ss.str().c_str(),ss1.str().c_str(),36,1.0,37.0,18,1.0,19.0);
  h_max_csc_me->SetStats(kFALSE);
  h_max_csc_me->GetXaxis()->SetTitle("CSC #");
  for(Int_t i=1;i<=18;i++) h_max_csc_me->GetYaxis()->SetBinLabel(i,yTitle_ME[i-1].c_str());
  h_max_csc_me->GetZaxis()->SetLabelSize(0.03);
  h_max_csc_me->SetOption("COLZ");
  h_max_csc_me->SetMinimum(0.0);
  h_max_csc_me->SetMaximum(2000.0);

  
  // 1D hist for delta hv  
  ss.str("");
  ss<< result_histNameDeltaHV;
  ss1.str("");
  ss1<< result_histTitleDeltaHV<<" in run "<<run.c_str();
  hdeltahv=new TH1F(ss.str().c_str(),ss1.str().c_str(),80,-200.0,200.0);
  hdeltahv->GetXaxis()->SetTitle(result_title_X_DeltaHV.c_str());
  hdeltahv->GetYaxis()->SetTitle(result_title_Y_DeltaHV.c_str());
  hdeltahv->SetFillColor(4);

  // 1D hist for ratio in Landau fit for ME11
  ss.str("");
  ss<< "widthlppratioME11";
  ss1.str("");
  ss1<< "Width to Landau Peak Ratio ME11 "<<" in run "<<run.c_str();
  widthlppratioME11=new TH1F(ss.str().c_str(),ss1.str().c_str(),100,0.0,1.0);
  widthlppratioME11->GetXaxis()->SetTitle("Width to Landau Peak Ratio ME11");
  widthlppratioME11->GetYaxis()->SetTitle("Entries");
  widthlppratioME11->SetFillColor(4);  

  // 1D hist for ratio in Landau fit for the rest of ME
  ss.str("");
  ss<< "widthlppratioME";
  ss1.str("");
  ss1<< "Width to Landau Peak Ratio ME "<<" in run "<<run.c_str();
  widthlppratioME=new TH1F(ss.str().c_str(),ss1.str().c_str(),100,0.0,1.0);
  widthlppratioME->GetXaxis()->SetTitle("Width to Landau Peak Ratio ME");
  widthlppratioME->GetYaxis()->SetTitle("Entries");
  widthlppratioME->SetFillColor(4);  

  // 1D hist for status flag
  ss.str("");
  ss<< "StatusFlag";
  ss1.str("");
  ss1<< "Status Flag "<<" in run "<<run.c_str();
  statusflag=new TH1F(ss.str().c_str(),ss1.str().c_str(),10,0.0,10.0);
  statusflag->GetXaxis()->SetTitle("Status Flag");
  statusflag->GetYaxis()->SetTitle("Entries");
  statusflag->SetFillColor(4); 

Int_t k=0;
TCanvas *c1=new TCanvas("c1","canvas");
c1->cd();

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
     ss<<in_folder.c_str()<<input_histName.c_str()<<idchamber;
     f_in.cd();
     TH2F *h2 = (TH2F*)f_in.Get(ss.str().c_str());
     if(h2 ==NULL) {
       ncscmis++;
       cscmis.push_back(idchamber);
     }
   if(h2 != NULL) {

     ss.str(""); ss1.str("");
     ss<<input_histName.c_str()<<idchamber;
     ss1<<h2->GetTitle()<<" in run "<<run.c_str();
     ny=h2->GetNbinsY(); Float_t ylow=h2->GetYaxis()->GetBinLowEdge(1);
                         Float_t yhigh=h2->GetYaxis()->GetBinLowEdge(ny)+
                                       h2->GetYaxis()->GetBinWidth(ny);
     Int_t nx=nhvsegm[indisr]; Float_t xlow=1.0; Float_t xhigh=xlow+nx;
     h2mod=new TH2F(ss.str().c_str(),ss1.str().c_str(),nx,xlow,xhigh,ny,ylow,yhigh);
     h2mod->SetStats(kFALSE);

     // saving original, adding X,Y titles, color and "BOX" option
     Float_t w;
     for(Int_t i=1;i<=h2->GetNbinsX();i++) { 
       if(i <= nhvsegm[indisr]) {
         for(Int_t j=1;j<=h2->GetNbinsY();j++) {
	    w=h2->GetBinContent(i,j);
	    h2mod->SetBinContent(i,j,w);
            Float_t x=h2->GetYaxis()->GetBinLowEdge(j);
            nentr[indisr]=nentr[indisr]+1;
            hadcsum[indisr]->Fill(x,w);
	 }
       }     
     }
     h2mod->GetXaxis()->SetTitle(input_title_X.c_str());
     h2mod->GetYaxis()->SetTitle(input_title_Y.c_str());
     h2mod->GetYaxis()->SetTitleOffset(1.2);
     h2mod->SetOption("COLZ");
     f_out.cd("Input_hists");     
     h2mod->Write();

     // saving Y projection of the whole 2D hist for given chamber
     ss.str("");
     ss<<input_histName.c_str()<<idchamber<<"_Y_all";
     TH1D *h1d = h2->ProjectionY(ss.str().c_str(),1,nhvsegm[indisr],"");
     h1d->GetXaxis()->SetTitle(input_title_Y.c_str());
     h1d->GetYaxis()->SetTitle("Entries");
     h1d->GetYaxis()->SetTitleOffset(1.2);
     h1d->SetFillColor(4);
     gStyle->SetOptStat(1001111);
     f_out.cd("Y_projections");

        // Fit by Landau peak the 3X3 ADC Sum distribution in the whole csc
     TF1 *fcsc=new TF1("fcsc","landau",0.0,2000.0);
     Float_t inpeak=5.0*h1d->GetMaximum(99999.0);
     Float_t inpeakpos=0.6*h1d->GetMean();
     Float_t inpeakwidth=0.3*inpeakpos;
     fcsc->SetParameters(inpeak,inpeakpos,inpeakwidth);
     //h1d->SetNdivisions(506);
     h1d->Fit("fcsc","BQ","",0.0,2000.0);
     Float_t lppcsc=fcsc->GetParameter(1);
     Float_t lppcscerr=fcsc->GetParError(1);

     h1d->Write(); 

     gr_y[indisr].push_back(lppcsc);
     gr_y_er[indisr].push_back(lppcscerr);
     Float_t xcsc=csc;
     if(esr[jesr] > 200) xcsc=-csc;
     gr_x[indisr].push_back(xcsc);

     delete h1d;   

     // saving slices, fitting Landau peak in each slice, fill 2D hist
     f_out.cd("Slices");
     Float_t mingasgain=10000.0;
     Float_t maxgasgain=0.0;
     for(Int_t j=1;j<=h2->GetNbinsX();j++) {
       if(j <= nhvsegm[indisr]) {
        Int_t n=j;
	locid_cur=idchamber*100+n;
        selflag_cur=0;
	peak_cur=0.0; peakpos_cur=0.0; peakwidth_cur=0.0; peakposer_cur=0.0;
        dhv_cur=0.0;

        ss.str("");
        ss<<input_histName.c_str()<<idchamber<<"_Y_"<<n;
        TH1D *h1d = h2->ProjectionY(ss.str().c_str(),j,j,"");
        nentrloc_cur=h1d->GetEntries();
        if( nentrloc_cur == 0) selflag_cur=1;
        if(nentrloc_cur > 0 && nentrloc_cur <= nminentries) selflag_cur=2;

	//	if(h1d->GetEntries() > 0 ) {
        if(nentrloc_cur > nminentries) {   // since Sep. 29, 2008
          /* Fit by Landau peak the 3X3 ADC Sum distribution in the slice*/
          TF1 *f1=new TF1("f1","landau",0.0,2000.0);
          Float_t inpeak=5.0*h1d->GetMaximum(99999.0);
          Float_t inpeakpos=0.6*h1d->GetMean();
          Float_t inpeakwidth=0.3*inpeakpos;
          f1->SetParameters(inpeak,inpeakpos,inpeakwidth);
          //h1d->SetNdivisions(506);
          h1d->Fit("f1","BQ","",0.0,2000.0);
          peak_cur=f1->GetParameter(0);
          peakpos_cur= f1->GetParameter(1);
          peakwidth_cur=f1->GetParameter(2);
          peakposer_cur=f1->GetParError(1);

          Float_t lpp=f1->GetParameter(1);
	  Float_t lpperr=f1->GetParError(1);

          Float_t ratio=0.0;
          if(peakpos_cur != 0.0) ratio=peakwidth_cur/peakpos_cur;
          if(sr[indisr] == 11) widthlppratioME11->Fill(ratio,1.0);
          if(sr[indisr] != 11) widthlppratioME->Fill(ratio,1.0);

          if(peakpos_cur < 0.0 || peakpos_cur > 2000.0 || 
	    ratio <= 0.0 || ratio>=1.0) selflag_cur=selflag_cur+4; // Fit failed

          if(lpperr>100.0) lpperr=100.0;
          Float_t entr=h1d->GetEntries();
          entries[jesr]=entries[jesr]+1;
          hlpp[indisr]->SetBinContent(cscbin,j,lpp);
          hlpperror[indisr]->SetBinContent(cscbin,j,lpperr);
          hentr[indisr]->SetBinContent(cscbin,j,entr);
          if(selflag_cur == 0 ) {
            hlppme[jesr]->Fill(lpp,1.0);
            if(lpp < mingasgain) mingasgain=lpp;
            if(lpp > maxgasgain) maxgasgain=lpp;
	  }
          ss.str("");
          ss<<slice_title_X<<" "<<n;
          h1d->GetXaxis()->SetTitle(ss.str().c_str());
          h1d->GetYaxis()->SetTitle("Entries");
          h1d->GetYaxis()->SetTitleOffset(1.2);
          h1d->SetFillColor(4);
          gStyle->SetOptStat(1001111);
          h1d->Write();

	} // end of if(h1d->GetEntries() > 0)
        delete h1d;
        Float_t dum=selflag_cur;
        statusflag->Fill(dum,1.0);

        locid.push_back(locid_cur);
        nentrloc.push_back(nentrloc_cur);
        selflag.push_back(selflag_cur);
        peak.push_back(peak_cur);
        peakpos.push_back(peakpos_cur);
        peakwidth.push_back(peakwidth_cur);
        peakposer.push_back(peakposer_cur);
        dhvv.push_back(dhv_cur);  // dhv will be modified later

       }  // end of if(j <= nhvsegm[indisr]) {
     }    // end of for(Int_t j=1;j<=h2->GetNbinsX();j++) {

   if(mingasgain < 10000.0) h_min_csc_me->SetBinContent(csc,me,mingasgain);
   if(maxgasgain > 0.0) h_max_csc_me->SetBinContent(csc,me,maxgasgain);
   delete h2;
   delete h2mod; 
   } // end of if(h2 != NULL)
  }  //  for(Int_t csc=1;csc<(ncsc[indisr]+1);csc++) {
  }  // end of   if(esr[jesr] != 142 && esr[jesr] != 242) {
}    // end of for(Int_t jesr=0;jesr<18;jesr++) { 


f_out.cd("Results");
for(Int_t isr=0;isr<8;isr++) {
  if(hlpp[isr] != NULL) {
       hlpp[isr]->SetStats(kFALSE);
       hlpp[isr]->Write();
  }
  if(hlpperror[isr] != NULL) {
       hlpperror[isr]->SetStats(kFALSE);
       hlpperror[isr]->Write();
  }
  if(hentr[isr] != NULL) {
       hentr[isr]->SetStats(kFALSE);
       hentr[isr]->Write();
  }
  if(hadcsum[isr] != NULL) {
       hadcsum[isr]->Write();
  }
}

for(Int_t jesr=0;jesr<18;jesr++) if(hlppme[jesr] != NULL) 
    hlppme[jesr]->Write();

h_min_csc_me->Write();
h_max_csc_me->Write();
widthlppratioME11->Write();
widthlppratioME->Write();
statusflag->Write();

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

// Print table

std::cout<<std::endl;
std::cout<<"Tables for CSC HV adjustment"<<std::endl;
std::cout<<std::endl;

Int_t station,ring,minentries=1000000,maxentries=0;
for(Int_t jesr=0;jesr<18;jesr++) {
  if(hlppme[jesr] != NULL) { 
  if(esr[jesr] < 200) {Int_t isr=jesr;   std::string endcapsign="+";}
  if(esr[jesr] > 200) {Int_t isr=jesr-9; std::string endcapsign="-";}
  station=sr[isr]/10; ring=sr[isr]-station*10;
  ss.str("");
  ss<<"ME"<<endcapsign.c_str()<<station<<"/"<<ring<< 
  "      MEAN(Landau peak position) "<<hlppme[jesr]->GetMean();
  std::cout<<ss.str().c_str()<<std::endl;
  }
}
std::cout<<std::endl;

std::cout<<"Total hv segments analyzed "<<locid.size()<<std::endl;
std::cout<<std::endl;

for(Int_t ind=0; ind<locid.size(); ind++) {
  Int_t jesr=locid[ind]/10000;
  for(Int_t i=0;i<18;i++) if(esr[i] == jesr) jesr=i;
  Float_t mean_adc_lpp=hlppme[jesr]->GetMean();
  meanadclpp.push_back(mean_adc_lpp);

  Int_t isr=locid[ind]/1000000;
  isr=locid[ind]-isr*1000000;
  isr=isr/10000;
  for(Int_t i=0;i<8;i++) if(sr[i] == isr) isr=i;

  if(selflag[ind] == 0) {   // statistics is acceptable and fit was OK 
    Float_t dhvc=dhv_coeff[isr];
    Float_t adc_lpp=peakpos[ind];
    Float_t dhv=-dhvc*log(adc_lpp/mean_adc_lpp);
    dhvv[ind]=dhv;
    hdeltahv->Fill(dhv,1.0);
  }
  printf("%7i%7.0f%3i%5i%6i%8.0f%8.0f%8.0f%8.0f%8.0f\n", locid[ind],meanadclpp[ind],selflag[ind],nminentries,nentrloc[ind],peak[ind],peakpos[ind],peakwidth[ind],peakposer[ind],dhvv[ind]);
}

f_out.cd("Results");
hdeltahv->Write();

/// Plot and save graphs of the mean Landau peak position vs CSC in ME
Float_t xgr[72],ygr[72],erygr[72],erxgr[72];
for(Int_t isr=0;isr<8;isr++) {
  if(gr_x[isr].size()>0) {
    Int_t np=gr_x[isr].size();
    for(Int_t i=0;i<np; i++) {
      xgr[i]=gr_x[isr][i];
      ygr[i]=gr_y[isr][i];
      erxgr[i]=0.0;
      erygr[i]=gr_y_er[isr][i];
    }
    ss.str("");
    ss<<result_graphNameLandauCSCME.c_str()<<sr[isr];

    cgraph[isr]= new TCanvas(ss.str().c_str());
    cgraph[isr]->cd();
    cgraph[isr]->SetGrid();
    gr_lpp_csc_me[isr]=new TGraphErrors(np,xgr,ygr,erxgr,erygr);
    ss1.str("");
    ss1<<result_graphTitleLandauCSCME<<sr[isr]<<" in run "<<run.c_str();
    gr_lpp_csc_me[isr]->SetTitle(ss1.str().c_str());
    gr_lpp_csc_me[isr]->GetXaxis()->SetTitle(xTitle_ME_CSC[isr].c_str());
    gr_lpp_csc_me[isr]->GetYaxis()->SetTitle("Landau peak position");
    gr_lpp_csc_me[isr]->SetMinimum(0.0);
    gr_lpp_csc_me[isr]->SetMaximum(500.0);
    if(isr==0) {
      gr_lpp_csc_me[isr]->SetMinimum(250.0);
      gr_lpp_csc_me[isr]->SetMaximum(1250.0);
    }
    gr_lpp_csc_me[isr]->SetMarkerStyle(20);
    gr_lpp_csc_me[isr]->SetMarkerColor(4);
    gr_lpp_csc_me[isr]->SetMarkerSize(1.2);
    gr_lpp_csc_me[isr]->SetLineColor(4);    // Blue error bar
    gr_lpp_csc_me[isr]->Draw("APZ"); 
    cgraph[isr]->Update();
    f_out.cd("Results");
    cgraph[isr]->Write();
    delete gr_lpp_csc_me[isr];
    delete cgraph[isr];
  }
}
//f_out.close();

FILE *File_out_recom;
ss.str("");
ss<<"dhv_recom_fit_"<<run<<".txt";
File_out_recom=fopen(ss.str().c_str(),"w");

for(Int_t ind=0; ind<locid.size(); ind++) {
  if(selflag[ind] == 0) {
    Int_t d=dhvv[ind];
    if(d > 100 ) d=100;
    if((d >= -15) && (d <= 15)) d=0; 
    if(d < -15 || d > 15) {
      Float_t df;
      if(d<0) {df=d-5; d=df/10; d=d*10;}
      if(d>0) {df=d+5; d=df/10; d=d*10;}
    }    
  fprintf(File_out_recom,"%7i%7.0f%3i%5i%6i%8.0f%8.0f%8.0f%8.0f%8.0f%8i\n", locid[ind],meanadclpp[ind],selflag[ind],nminentries,nentrloc[ind],peak[ind],peakpos[ind],peakwidth[ind],peakposer[ind],dhvv[ind],d);
  }
}
fclose(File_out_recom);

FILE *File_out_largedhvpos;
ss.str("");
ss<<"hv_segm_large_dhv_pos_"<<run<<".txt";
File_out_largedhvpos=fopen(ss.str().c_str(),"w");

for(Int_t ind=0; ind<locid.size(); ind++) {
  if(selflag[ind] == 0) {
    Int_t d=dhvv[ind];
    if(d > 100) {
      d=100;
      fprintf(File_out_largedhvpos,"%7i%7.0f%3i%5i%6i%8.0f%8.0f%8.0f%8.0f%8.0f%8i\n", locid[ind],meanadclpp[ind],selflag[ind],nminentries,nentrloc[ind],peak[ind],peakpos[ind],peakwidth[ind],peakposer[ind],dhvv[ind],d);
    }
  }
}
fclose(File_out_largedhvpos);

FILE *File_out_largedhvneg;
ss.str("");
ss<<"hv_segm_large_dhv_neg_"<<run<<".txt";
File_out_largedhvneg=fopen(ss.str().c_str(),"w");

for(Int_t ind=0; ind<locid.size(); ind++) {
  if(selflag[ind] == 0) {
    Int_t d=dhvv[ind];
    if(d <-100) {
      Float_t df;
      df=d-5; d=df/10; d=d*10;  
      fprintf(File_out_largedhvneg,"%7i%7.0f%3i%5i%6i%8.0f%8.0f%8.0f%8.0f%8.0f%8i\n", locid[ind],meanadclpp[ind],selflag[ind],nminentries,nentrloc[ind],peak[ind],peakpos[ind],peakwidth[ind],peakposer[ind],dhvv[ind],d);
    }
  }
}
fclose(File_out_largedhvneg);


FILE *File_out_nodata;
ss.str("");
ss<<"hv_segm_no_data_"<<run<<".txt";
File_out_nodata=fopen(ss.str().c_str(),"w");

for(Int_t ind=0; ind<locid.size(); ind++) {
  if(selflag[ind] == 1) 
  fprintf(File_out_nodata,"%7i%7.0f%3i%5i%6i%8.0f%8.0f%8.0f%8.0f%8i\n", locid[ind],meanadclpp[ind],selflag[ind],nminentries,nentrloc[ind],peak[ind],peakpos[ind],peakwidth[ind],peakposer[ind],dhvv[ind]);
  
}
fclose(File_out_nodata);

FILE *File_out_lowstat;
ss.str("");
ss<<"hv_segm_low_stat_"<<run<<".txt";
File_out_lowstat=fopen(ss.str().c_str(),"w");

for(Int_t ind=0; ind<locid.size(); ind++) {
  if((selflag[ind] & 2) > 0) 
  fprintf(File_out_lowstat,"%7i%7.0f%3i%5i%6i%8.0f%8.0f%8.0f%8.0f%8i\n", locid[ind],meanadclpp[ind],selflag[ind],nminentries,nentrloc[ind],peak[ind],peakpos[ind],peakwidth[ind],peakposer[ind],dhvv[ind]);
  
}
fclose(File_out_lowstat);

FILE *File_out_failedfit;
ss.str("");
ss<<"hv_segm_failed_fit_"<<run<<".txt";
File_out_failedfit=fopen(ss.str().c_str(),"w");

for(Int_t ind=0; ind<locid.size(); ind++) {
  if((selflag[ind] & 4) > 0) 
  fprintf(File_out_failedfit,"%7i%7.0f%3i%5i%6i%8.0f%8.0f%8.0f%8.0f%8i\n", locid[ind],meanadclpp[ind],selflag[ind],nminentries,nentrloc[ind],peak[ind],peakpos[ind],peakwidth[ind],peakposer[ind],dhvv[ind]); 
}
fclose(File_out_failedfit);
}
