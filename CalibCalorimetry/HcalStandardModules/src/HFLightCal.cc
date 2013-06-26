// Analysis of HF LED/Laser run: 
// SPE calibration for low light intensity or raw SPE calibration for high light intensity
// and HF performance based on this analysis
//
// Igor Vodopiyanov. Oct-2007 .... update Sept-2008
// Thanks G.Safronov, M.Mohammadi, F.Ratnikov
//
#include <memory>
#include <string>
#include <iostream>

#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "math.h"
#include "TMath.h"
#include "TF1.h"

#include "CalibCalorimetry/HcalStandardModules/interface/HFLightCal.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace std;
Int_t Nev, runN=0, eventN=0;
Int_t itsmax[26][36][2];
Int_t itspinmax[8][3];

namespace {
  //bool verbose = true;
  bool verbose = false;
}

HFLightCal::HFLightCal (const edm::ParameterSet& fConfiguration) :
  hfDigiCollectionTag_(fConfiguration.getParameter<edm::InputTag>("hfDigiCollectionTag")),
  hcalCalibDigiCollectionTag_(fConfiguration.getParameter<edm::InputTag>("hcalCalibDigiCollectionTag")) {

  //std::string histfile = fConfiguration.getUntrackedParameter<string>("rootFile");
  histfile = fConfiguration.getUntrackedParameter<string>("rootFile");
  textfile = fConfiguration.getUntrackedParameter<string>("textFile");
  prefile = fConfiguration.getUntrackedParameter<string>("preFile");
}

HFLightCal::~HFLightCal () {
  //delete mFile;
}

void HFLightCal::beginJob() {

  char htit[64];
  Int_t neta,nphi,ndepth,nmax,nquad,npin;

  std::cout<<std::endl<<"HFLightCal beginJob: --> "<<std::endl;

  // Read info about signal timing in TS from PreAnalysis
  mFile = new TFile (histfile.c_str(),"RECREATE");
  if ((tFile = fopen(textfile.c_str(),"w"))==NULL) {
    printf("\nNo output textfile open\n\n");
    std::cout<<"Problem with output textFILE => exit"<<std::endl;
    exit(1);
  }
  //if ((preFile = fopen("hf_preanal.txt","r"))==NULL){
  if ((preFile = fopen(prefile.c_str(),"r"))==NULL){
    printf("\nNo input pre-file open\n\n");
    std::cout<<"Problem with input textFILE => exit"<<std::endl;
    exit(1);
  }
  rewind(preFile);
  for (int i=0; i<1728; i++) {
    fscanf(preFile,"%d%d%d%d\r",&neta,&nphi,&ndepth,&nmax);
    //std::cout<<neta<<"  "<<nphi<<"  "<<ndepth<<"  "<<nmax<<std::endl;
    if (neta>=29 && neta<=41 && nphi<72 && nphi>0 && ndepth>0 && ndepth<=2) 
      itsmax[neta-29][(nphi-1)/2][ndepth-1] = nmax;
    else if (neta<=-29 && neta>=-41 && nphi<72 && nphi>0 && ndepth>0 && ndepth<=2) 
      itsmax[13-neta-29][(nphi-1)/2][ndepth-1] = nmax;
    else {
      std::cout<<"Input pre-file: wrong channel record:"<<std::endl;
      std::cout<<"eta="<<neta<<"  phi="<<nphi<<"  depth="<<ndepth<<"  max="<<nmax<<std::endl;
    }
  }
  for (int i=0; i<24; i++) {
    fscanf(preFile,"%d%d%d\r",&nquad,&npin,&nmax);
    //std::cout<<nquad<<"  "<<npin<<"  "<<nmax<<std::endl;
    if (nquad>0 && nquad<=4 && npin<=3 && npin>0) 
      itspinmax[nquad-1][npin-1] = nmax;
    else if (nquad<0 && nquad>=-4 && npin<=3 && npin>0) 
      itspinmax[4-nquad-1][npin-1] = nmax;
    else {
      std::cout<<"Input pre-file: wrong PIN record:"<<std::endl;
      std::cout<<"quad="<<nquad<<"  pin="<<npin<<"  max="<<nmax<<std::endl;
    }
  }

  // General Histos
  htmax = new TH1F("htmax","Max TS",10,-0.5,9.5);
  htmean = new TH1F("htmean","Mean signal TS",100,0,10);
  hsignalmean = new TH1F("hsignalmean","Mean ADC 4maxTS",1201,-25,30000);
  hsignalrms = new TH1F("hsignalrms","RMS ADC 4maxTS",500,0,500);
  hpedmean = new TH1F("hpedmean","Mean ADC 4lowTS",200,-10,90);
  hpedrms = new TH1F("hpedrms","RMS ADC 4lowTS",200,0,100);
  hspes = new TH1F("hspes","SPE if measured",200,0,40);
  hnpevar = new TH1F("hnpevar","~N PE input",500,0,500);
  hsignalmapP = new TH2F("hsignalmapP","Mean(Response) - Mean(Pedestal) HFP;#eta;#phi",26,28.5,41.5,36,0,72);
  hsignalRMSmapP = new TH2F("hsignalRMSmapP","RMS Response HFP;#eta;#phi",26,28.5,41.5,36,0,72);
  hnpemapP = new TH2F("hnpemapP","~N PE input HFP;#eta;#phi",26,28.5,41.5,36,0,72);
  hnpemapP->SetOption("COLZ");hsignalmapP->SetOption("COLZ");hsignalRMSmapP->SetOption("COLZ");
  hsignalmapM = new TH2F("hsignalmapM","Mean(Response) - Mean(Pedestal) HFM;#eta;#phi",26,-41.5,-28.5,36,0,72);
  hsignalRMSmapM = new TH2F("hsignalRMSmapM","RMS Response HFM;#eta;#phi",26,-41.5,-28.5,36,0,72);
  hnpemapM = new TH2F("hnpemapM","~N PE input HFM;#eta;#phi",26,-41.5,-28.5,36,0,72);
  hnpemapM->SetOption("COLZ");hsignalmapM->SetOption("COLZ");hsignalRMSmapM->SetOption("COLZ");
  // Channel-by-channel histos
  for (int i=0;i<13;i++) for (int j=0;j<36;j++) for (int k=0;k<2;k++) {
    if (i>10 && j%2==0) continue;
    sprintf(htit,"ts_+%d_%d_%d",i+29,j*2+1,k+1);
    hts[i][j][k] = new TH1F(htit,htit,10,-0.5,9.5); // TimeSlices (pulse shape)
    sprintf(htit,"tsmean_+%d_%d_%d",i+29,j*2+1,k+1);
    htsm[i][j][k] = new TH1F(htit,htit,100,0,10);   // Mean signal time estimated from TS 
    sprintf(htit,"sp_+%d_%d_%d",i+29,j*2+1,k+1);
    hsp[i][j][k] = new TH1F(htit,htit,1201,-25,30000); // Big-scale spectrum (linear ADC)
    sprintf(htit,"spe_+%d_%d_%d",i+29,j*2+1,k+1);
    hspe[i][j][k] = new TH1F(htit,htit,200,-9.5,190.5); // Small-scale spectrum (linear ADC)
    sprintf(htit,"ped_+%d_%d_%d",i+29,j*2+1,k+1);
    hped[i][j][k] = new TH1F(htit,htit,200,-9.5,190.5); // Pedestal spectrum
    sprintf(htit,"ts_-%d_%d_%d",i+29,j*2+1,k+1);
    hts[i+13][j][k] = new TH1F(htit,htit,10,-0.5,9.5);
    sprintf(htit,"tsmean_-%d_%d_%d",i+29,j*2+1,k+1);
    htsm[i+13][j][k] = new TH1F(htit,htit,100,0,10);  
    sprintf(htit,"sp_-%d_%d_%d",i+29,j*2+1,k+1);
    hsp[i+13][j][k] = new TH1F(htit,htit,1201,-25,30000);
    sprintf(htit,"spe_-%d_%d_%d",i+29,j*2+1,k+1);
    hspe[i+13][j][k] = new TH1F(htit,htit,200,-9.5,190.5); 
    sprintf(htit,"ped_-%d_%d_%d",i+29,j*2+1,k+1);
    hped[i+13][j][k] = new TH1F(htit,htit,200,-9.5,190.5); 
  } 
  // PIN-diodes histos
  for (int i=0;i<4;i++) for (int j=0;j<3;j++) {
    sprintf(htit,"ts_PIN%d_+Q%d",j+1,i+1);
    htspin[i][j] = new TH1F(htit,htit,10,-0.5,9.5);
    sprintf(htit,"sp_PIN%d_+Q%d",j+1,i+1);
    hsppin[i][j] = new TH1F(htit,htit,1601,-25,40000);
    sprintf(htit,"spe_PIN%d_+Q%d",j+1,i+1);
    hspepin[i][j] = new TH1F(htit,htit,200,-9.5,190.5);
    sprintf(htit,"ped_PIN%d_+Q%d",j+1,i+1);
    hpedpin[i][j] = new TH1F(htit,htit,200,-9.5,190.5);
    sprintf(htit,"tsmean_PIN%d_+Q%d",j+1,i+1);
    htsmpin[i][j] = new TH1F(htit,htit,100,0,10);  
    sprintf(htit,"ts_PIN%d_-Q%d",j+1,i+1);
    htspin[i+4][j] = new TH1F(htit,htit,10,-0.5,9.5);
    sprintf(htit,"sp_PIN%d_-Q%d",j+1,i+1);
    hsppin[i+4][j] = new TH1F(htit,htit,1601,-25,40000);
    sprintf(htit,"spe_PIN%d_-Q%d",j+1,i+1);
    hspepin[i+4][j] = new TH1F(htit,htit,200,-9.5,190.5);
    sprintf(htit,"ped_PIN%d_-Q%d",j+1,i+1);
    hpedpin[i+4][j] = new TH1F(htit,htit,200,-9.5,190.5);
    sprintf(htit,"tsmean_PIN%d_-Q%d",j+1,i+1);
    htsmpin[i+4][j] = new TH1F(htit,htit,100,0,10);  
  }
  std::cout<<std::endl<<"histfile="<<histfile.c_str()<<"  textfile="<<textfile.c_str()<<std::endl;
  return;
}

void HistSpecs(TH1F* hist, Double_t &mean, Double_t &rms, Double_t range=4) {
  Double_t xmin,xmax;
  mean=hist->GetMean();
  rms=hist->GetRMS();
  xmin=hist->GetXaxis()->GetXmin();
  xmax=hist->GetXaxis()->GetXmax();
  hist->SetAxisRange(mean-range*rms-100,mean+range*rms+100);
  mean=hist->GetMean();
  rms=hist->GetRMS();
  hist->SetAxisRange(mean-range*rms-100,mean+range*rms+100);
  mean=hist->GetMean();
  rms=hist->GetRMS();
  hist->SetAxisRange(xmin,xmax);
  return;
}

Double_t FitFun(Double_t *x, Double_t *par) { 
// Spectra fit function: Pedestal Gaussian + asymmetric 1PE + 2PE +3PE peaks

  Double_t sum,xx,A0,C0,r0,sigma0,mean1,sigma1,A1,C1,r1,mean2,sigma2,A2,C2,r2,mean3,sigma3,A3,C3,r3;

  const Double_t k0=2.0, k1=1.6, k2=2.0;

  xx=x[0];
  sigma0 = par[2];
  A0 = 2*Nev/(2+2*par[0]+par[0]*par[0]+pow(par[0],3)/3+pow(par[0],4)/12+
	      pow(par[0],5)/60+pow(par[0],6)/360);
  r0 = ((xx-par[1])/sigma0);
  C0 = 1/(sigma0* TMath::Exp(-k0*k0/2)/k0 +
	  sigma0*sqrt(2*3.14159)*0.5*(1+TMath::Erf(k0/1.41421)));
  //sum = 1/(sqrt(2*3.14159)*par[2])*A0*TMath::Exp(-0.5*r0*r0);
  if(r0 < k0) sum = C0*A0*TMath::Exp(-0.5*r0*r0);
  else sum = C0*A0*TMath::Exp(0.5*k0*k0-k0*r0);

  mean1 = par[1]+par[3];
  sigma1 = par[4];
  //sigma1 = 1.547+0.125*par[3]+0.004042*par[3]*par[3];
  //sigma1 = (sigma1+(9.1347e-3+3.845e-2*par[3])*par[4]*2.0)*par[2];
  A1 = A0*par[0];
  C1 = 1/(sigma1* TMath::Exp(-k1*k1/2)/k1 +
	  sigma1*sqrt(2*3.14159)*0.5*(1+TMath::Erf(k1/1.41421)));
  r1 = ((xx-mean1)/sigma1);
  if(r1 < k1) sum += C1*A1*TMath::Exp(-0.5*r1*r1);
  else sum += C1*A1*TMath::Exp(0.5*k1*k1-k1*r1);

  mean2 = 2*par[3]+par[1];
  sigma2 = sqrt(2*sigma1*sigma1 - pow(par[2],2));
  //A2 = A0*par[5]*par[0]*par[0]/2;
  A2 = A0*par[0]*par[0]/2;
  C2 = 1/(sigma2* TMath::Exp(-k2*k2/2)/k2 +
	  sigma2*sqrt(2*3.14159)*0.5*(1+TMath::Erf(k2/1.41421)));
  r2 = ((xx-mean2)/sigma2);
  if(r2 < k2) sum += C2*A2*TMath::Exp(-0.5*r2*r2);
  else sum += C2*A2*TMath::Exp(0.5*k2*k2-k2*r2);

  mean3 = 3*par[3]+par[1];
  sigma3 = sqrt(3*sigma1*sigma1 - 2*pow(par[2],2));
  A3 = A0*par[0]*par[0]*par[0]/6;
  C3 = 1/(sigma3*sqrt(2*3.14159));
  r3 = ((xx-mean3)/sigma3);
  sum += C3*A3*TMath::Exp(-0.5*r3*r3);

  return sum;
}

void HFLightCal::endJob(void)
{
  Double_t mean,rms,meanped,rmsped,npevar;
  Double_t par[5],dspe=0,dnpe;
  Int_t tsmax,intspe;
  std::cout<<std::endl<<"HFLightCal endJob --> ";
  fprintf(tFile,"#RunN %d   Events processed %d",runN,eventN);

  for (int i=0;i<26;i++) for (int j=0;j<36;j++) for (int k=0;k<2;k++) {
    if (i>10 && i<13 && j%2==0) continue;
    if (i>23 && j%2==0) continue;
    meanped=rmsped=mean=rms=0;
    if (hsp[i][j][k]->Integral()>0) {
      HistSpecs(hped[i][j][k],meanped,rmsped);
      HistSpecs(hsp[i][j][k],mean,rms);
      if (hspe[i][j][k]->Integral()>hsp[i][j][k]->Integral()*0.9 || mean<100) {
	HistSpecs(hspe[i][j][k],mean,rms);
      }
      hsignalmean->Fill(mean);
      hsignalrms->Fill(rms);
      hpedmean->Fill(meanped);
      hpedrms->Fill(rmsped);
    }
  }

  meanped=hpedmean->GetMean();
  rmsped=hpedrms->GetMean();
  mean=hsignalmean->GetMean();
  rms=hsignalrms->GetMean();
  fprintf(tFile,"   MeanInput=<%.2f> [linADCcount]   RMS=<%.2f>\n",mean,rms);
  fprintf(tFile,"#eta/phi/depth  sum4maxTS     RMS      ~N_PE  sum4lowTS     RMS  maxTS  SPE +/- Err   Comment\n");
  TF1* fPed = new TF1("fPed","gaus",0,120);
  fPed->SetNpx(200);
  TF1 *fTot = new TF1("fTot",FitFun ,0,200,5);
  fTot->SetNpx(800);
  for (int i=0;i<26;i++) for (int j=0;j<36;j++) for (int k=0;k<2;k++) {
    if (i>10 && i<13 && j%2==0) continue;
    if (i>23 && j%2==0) continue;
    HistSpecs(hped[i][j][k],meanped,rmsped);
    HistSpecs(hsp[i][j][k],mean,rms);
    par[3]=0;
    if (hspe[i][j][k]->Integral()>hsp[i][j][k]->Integral()*0.9 || mean<100) {
      HistSpecs(hspe[i][j][k],mean,rms);
      if (hspe[i][j][k]->Integral(1,(int) (meanped+3*rmsped+12))/Nev>0.1) {
	//if (hspe[i][j][k]->Integral()>100 && mean-meanped<100) {
	if (mean+rms*3-meanped-rmsped*3>2 && rmsped>0) { // SPE fit if low intensity>0
	  par[1] = meanped;
	  par[2] = rmsped;
	  par[0] = hped[i][j][k]->GetMaximum();
	  fPed->SetParameters(par);
	  hped[i][j][k]->Fit(fPed,"BQ0");
	  fPed->GetParameters(&par[0]);
	  hped[i][j][k]->Fit(fPed,"B0Q","",par[1]-par[2]*3,par[1]+par[2]*3);
	  fPed->GetParameters(par);
	  hped[i][j][k]->Fit(fPed,"BLIQ","",par[1]-par[2]*3,par[1]+par[2]*3);
	  fPed->GetParameters(&par[0]);
	  Nev = (int) hspe[i][j][k]->Integral();
	  par[0]=0.1;
	  par[3]=10;
	  par[4]=6;
	  fTot->SetParameters(par);
	  fTot->SetParLimits(0,0,2);
	  //fTot->FixParameter(1,par[1]);
	  fTot->SetParLimits(1,par[1]-1,par[1]+1);
	  fTot->FixParameter(2,par[2]);
	  fTot->SetParLimits(3,1.2,100);
	  //fTot->SetParLimits(4,-1.64,1.64);
	  //fTot->SetParLimits(5,0.5,3);
	  hspe[i][j][k]->Fit(fTot,"BLEQ","");
	  fTot->GetParameters(par);
	  hspe[i][j][k]->Fit(fTot,"BLEQ","",-10,par[1]+par[3]*5);
	  fTot->GetParameters(par);
	  dspe=fTot->GetParError(3);
	  dnpe=fTot->GetParError(0);
	  if (par[3]<1.21 || dnpe>par[0]) par[3]=-1;
	  else if (par[0]>1.96 || par[3]>49) par[3]=0;
	  else {
	    hspes->Fill(par[3]);
	  }
	} 
      }
    }

    // NPE
    npevar=0;
    if (par[3]>0) npevar=par[0];                          // NPE from SPE fit
    else {                                                // NPE from high intensity signal
      if (hspe[i][j][k]->Integral()>hsp[i][j][k]->Integral()*0.98) {
	HistSpecs(hspe[i][j][k],mean,rms,3);
      }
      else {
	HistSpecs(hsp[i][j][k],mean,rms,3);
      }
      if (rmsped>0) {
	if (rms*rms-rmsped*rmsped>1 && mean>meanped) {
	  npevar=(mean-meanped)*(mean-meanped)/(rms*rms-rmsped*rmsped);
	}
	else if (mean<100) {
	  intspe=int(hspe[i][j][k]->Integral());
	  hspe[i][j][k]->SetAxisRange(meanped+rmsped*4,300);
	  npevar=hspe[i][j][k]->Integral()/intspe;
	  if (npevar>0.01) npevar=-1;
	  else npevar=0;
	  hspe[i][j][k]->SetAxisRange(-20,300);
	}
      }
    }
    if (npevar>5.0e-5) hnpevar->Fill(npevar);

    if (i<13) {
      hsignalmapP->Fill(i+28.6+k/2.0,j*2+1,mean-meanped); 
      hsignalRMSmapP->Fill(i+28.6+k/2.0,j*2+1,rms);
      if (npevar>0) hnpemapP->Fill(i+28.6+k/2.0,j*2+1,npevar);
      fprintf(tFile,"%3d%4d%5d  %11.2f%8.2f",i+29,j*2+1,k+1,mean,rms);
    }
    else {
      fprintf(tFile,"%3d%4d%5d  %11.2f%8.2f",13-i-29,j*2+1,k+1,mean,rms);
      hsignalmapM->Fill(13-i-28.6-k/2.0,j*2+1,mean-meanped);
      hsignalRMSmapM->Fill(13-i-28.6-k/2.0,j*2+1,rms);
      if (npevar>0) hnpemapM->Fill(13-i-28.6-k/2.0,j*2+1,npevar);
    }
    if (npevar>0) fprintf(tFile,"  %9.4f",npevar);
    else  fprintf(tFile,"      0    ");
    fprintf(tFile,"   %8.2f%8.2f",meanped,rmsped);
    tsmax=hts[i][j][k]->GetMaximumBin()-1;
    fprintf(tFile," %4d",tsmax);
    if (par[3]>0 && par[3]<99)  fprintf(tFile,"%8.2f%7.2f",par[3],dspe);
    else if (npevar>0) fprintf(tFile,"%8.2f    0  ",(mean-meanped)/npevar);
    else fprintf(tFile,"     0      0  ");

    // Diagnostics 
    fprintf(tFile,"    ");
    if (hsp[i][j][k]->GetEntries()<=0) fprintf(tFile,"NoSignal\n");
    else if (hsp[i][j][k]->GetEntries()<=10) fprintf(tFile,"Nev<10\n");
    else {
      if (hsp[i][j][k]->Integral()<=10 || mean>12000)  fprintf(tFile,"SignalOffRange\n");
      else {
	if (hsp[i][j][k]->Integral()<100)  fprintf(tFile,"Nev<100/");
	if (npevar>0 && par[3]>0 && (npevar*Nev<10 || npevar<0.001)) 
	  fprintf(tFile,"LowSignal/");
	else if (npevar==0 && fabs(mean-meanped)<3) fprintf(tFile,"LowSignal/");
	if (par[3]<0)  fprintf(tFile,"BadFit/");
	else if (par[3]==0)  fprintf(tFile,"NoSPEFit/");
	else if (par[3]>0 && npevar>1)   fprintf(tFile,"NPE>1/");
	if (npevar<0)   fprintf(tFile,"Problem/");
	if (mean<2) fprintf(tFile,"LowMean/");
	if (rms<0.5) fprintf(tFile,"LowRMS/"); 
	if (meanped<-1) fprintf(tFile,"LowPed/");
	else if (meanped>25) fprintf(tFile,"HighPed/"); 
	if (rmsped<0.5 && rmsped>0) fprintf(tFile,"NarrowPed/"); 
	else if (rmsped>10) fprintf(tFile,"WidePed/");
	if (hped[i][j][k]->GetBinContent(201)>10) fprintf(tFile,"PedOffRange"); 
	fprintf(tFile,"-\n");
      }
    }
  }

  for (int i=0;i<8;i++) for (int j=0;j<3;j++) {
    HistSpecs(hpedpin[i][j],meanped,rmsped);
    HistSpecs(hsppin[i][j],mean,rms);
    if (hspepin[i][j]->Integral()>hsppin[i][j]->Integral()*0.9 || mean<100) {
      HistSpecs(hspepin[i][j],mean,rms);
    }
    if (i<4) fprintf(tFile," PIN%d  +Q%d  %12.2f  %6.2f",j+1,i+1,mean,rms);
    else fprintf(tFile," PIN%d  -Q%d  %12.2f  %6.2f",j+1,i-3,mean,rms);
    fprintf(tFile,"  %15.2f  %6.2f",meanped,rmsped);
    tsmax=htspin[i][j]->GetMaximumBin()-1;
    fprintf(tFile,"  %4d\n",tsmax);
  } 

  mFile->Write();
  mFile->Close();
  fclose(tFile);
  std::cout<<std::endl<<" --endJob-- done"<<std::endl;
  return;
}

void HFLightCal::analyze(const edm::Event& fEvent, const edm::EventSetup& fSetup) {

  // event ID
  edm::EventID eventId = fEvent.id();
  int runNer = eventId.run ();
  int eventNumber = eventId.event ();
  if (runN==0) runN=runNer;
  eventN++;
  if (verbose) std::cout << "========================================="<<std::endl
			 << "run/event: "<<runNer<<'/'<<eventNumber<<std::endl;

  Double_t buf[20];
  Double_t maxADC,signal,ped=0,meant;
  Int_t maxisample=0,i1=3,i2=6;

  // HF PIN-diodes
  edm::Handle<HcalCalibDigiCollection> calib;  
  fEvent.getByLabel(hcalCalibDigiCollectionTag_, calib);
  if (verbose) std::cout<<"Analysis-> total CAL digis= "<<calib->size()<<std::endl;

  /* COMMENTED OUT by J. Mans (7-28-2008) as major changes needed with new Calib DetId 
   re-commented out by R.Ofierzynski (11.Nov.2008) - to be able to provide a consistent code for CMSSW_3_0_0_pre3:
   major changes are needed for the new Calib DetId which does not have the old methods any more

  for (unsigned j = 0; j < calib->size (); ++j) {
    const HcalCalibDataFrame digi = (*calib)[j];
    HcalElectronicsId elecId = digi.elecId();
    HcalCalibDetId calibId = digi.id();
    if (verbose) std::cout<<calibId.sectorString().c_str()<<" "<<calibId.rbx()<<" "<<elecId.fiberChanId()<<std::endl;
    int isector = calibId.rbx()-1;
    int ipin = elecId.fiberChanId();
    int iside = -1;
    if (calibId.sectorString() == "HFP") iside = 0; 
    else if (calibId.sectorString() == "HFM") iside = 4;
    maxisample = itspinmax[isector+iside][ipin]-1;

    if (iside != -1) {
      for (int isample = 0; isample < digi.size(); ++isample) {
	int adc = digi[isample].adc();
	int capid = digi[isample].capid ();
	double linear_ADC = digi[isample].nominal_fC();
	if (verbose) std::cout<<"PIN linear_ADC = "<<linear_ADC<<"  atMAXTS="<<maxisample<<std::endl;
	htspin[isector+iside][ipin]->Fill(isample,linear_ADC);
	buf[isample]=linear_ADC;
      }
      i1=maxisample-1;
      i2=maxisample+2;
      if (i1<0) {i1=0;i2=3;}
      else if (i2>9) {i1=6;i2=9;} 
      if      (i1==0) ped=buf[8]+buf[9]+buf[6]+buf[7];
      else if (i1==1) ped=buf[8]+buf[9]+buf[6]+buf[7];
      else if (i1==2) ped=buf[0]+buf[1]+buf[6]+buf[7];
      else if (i1==3) ped=buf[0]+buf[1]+buf[2]+buf[7];
      else if (i1>=4) ped=buf[0]+buf[1]+buf[2]+buf[3];
      signal=0;
      for (ii=0;ii<4;ii++) signal+=TMath::Max(buf[ii+i1],ped/4.0);
      hsppin[isector+iside][ipin]->Fill(signal);
      hspepin[isector+iside][ipin]->Fill(signal);
      hpedpin[isector+iside][ipin]->Fill(ped);

      // Mean signal time estimation
      ped=ped/4;
      meant=0;
      for (ii=0;ii<4;ii++) meant+=(TMath::Max(buf[ii+i1],ped)-ped)*(ii+i1);
      if (signal-ped*4>0) meant/=(signal-ped*4); 
      else meant=i1+1;
      htsmpin[isector+iside][ipin]->Fill(meant);
    }
  }
  */  

  // HF
  edm::Handle<HFDigiCollection> hf_digi;
  fEvent.getByLabel(hfDigiCollectionTag_, hf_digi);
  if (verbose) std::cout<<"Analysis-> total HF digis= "<<hf_digi->size()<<std::endl;

  for (unsigned ihit = 0; ihit < hf_digi->size (); ++ihit) {
    const HFDataFrame& frame = (*hf_digi)[ihit];
    HcalDetId detId = frame.id();
    int ieta = detId.ieta();
    int iphi = detId.iphi();
    int depth = detId.depth();
    if (verbose) std::cout <<"HF digi # " <<ihit<<": eta/phi/depth: "
			   <<ieta<<'/'<<iphi<<'/'<< depth << std::endl;

    if (ieta>0) ieta = ieta-29;
    else ieta = 13-ieta-29;

    for (int ii=0; ii<10; ii++) buf[ii]=0;
    maxADC=-99;
    for (int isample = 0; isample < frame.size(); ++isample) {
      int adc = frame[isample].adc();
      int capid = frame[isample].capid ();
      double linear_ADC = frame[isample].nominal_fC();
      double nominal_fC = detId.subdet () == HcalForward ? 2.6 *  linear_ADC : linear_ADC;

      if (verbose) std::cout << "Analysis->     HF sample # " << isample 
			     << ", capid=" << capid 
			     << ": ADC=" << adc 
			     << ", linearized ADC=" << linear_ADC
			     << ", nominal fC=" << nominal_fC <<std::endl;

      hts[ieta][(iphi-1)/2][depth-1]->Fill(isample,linear_ADC);
      buf[isample]=linear_ADC;
      /*
      if (maxADC<linear_ADC) {
	maxADC=linear_ADC;
	maxisample=isample;
      }
      */
    }

    maxADC=-99;
    for (int ii=0; ii<frame.size(); ii++) {
      signal=buf[ii];
      if      (ii<2) signal -= (buf[ii+4]+buf[ii+8])/2.0;
      else if (ii<4) signal -= buf[ii+4];
      else if (ii<6) signal -= (buf[ii+4]+buf[ii-4])/2.0;
      else if (ii<8) signal -= buf[ii-4];
      else           signal -= (buf[ii-4]+buf[ii-8])/2.0;
      if (signal>maxADC) {
	maxADC=signal;
	maxisample=ii;
      }
    }
    //maxisample=itsmax[ieta][(iphi-1)/2][depth-1]-1;
    if (abs(maxisample-itsmax[ieta][(iphi-1)/2][depth-1]+1)>1)  maxisample=itsmax[ieta][(iphi-1)/2][depth-1]-1;
    if (verbose) std::cout<<eventNumber<<"/"<<ihit<<": maxTS="<<maxisample<<endl;

    // Signal = four capIDs found by PreAnal, Pedestal = four capIDs off the signal
    htmax->Fill(maxisample);
    i1=maxisample-1;
    i2=maxisample+2;
    if (i1<0) {i1=0;i2=3;}
    else if (i2>9) {i1=6;i2=9;} 
    else if (i2<9 && maxisample<=itsmax[ieta][(iphi-1)/2][depth-1]-1) {
      if (buf[i1]<buf[i2+1]) {i1=i1+1;i2=i2+1;}
    }
    signal=buf[i1]+buf[i1+1]+buf[i1+2]+buf[i1+3];
    hsp[ieta][(iphi-1)/2][depth-1]->Fill(signal);
    hspe[ieta][(iphi-1)/2][depth-1]->Fill(signal);
    /*
    if      (i1==0) ped=(buf[4]+buf[8])/2.0+(buf[5]+buf[9])/2.0+buf[6]+buf[7];
    else if (i1==1) ped=(buf[0]+buf[8])/2.0+(buf[5]+buf[9])/2.0+buf[6]+buf[7];
    else if (i1==2) ped=(buf[0]+buf[8])/2.0+(buf[1]+buf[9])/2.0+buf[6]+buf[7];
    else if (i1==3) ped=(buf[0]+buf[8])/2.0+(buf[1]+buf[9])/2.0+buf[2]+buf[7];
    else if (i1==4) ped=(buf[0]+buf[8])/2.0+(buf[1]+buf[9])/2.0+buf[2]+buf[3];
    else if (i1==5) ped=(buf[0]+buf[4])/2.0+(buf[1]+buf[9])/2.0+buf[2]+buf[3];
    else if (i1==6) ped=(buf[0]+buf[4])/2.0+(buf[1]+buf[5])/2.0+buf[2]+buf[3];
    */
    
    if      (i1<2) ped=buf[8]+buf[9]+buf[6]+buf[7];
    else if (i1==2) ped=buf[6]+buf[9]+buf[7]+buf[0];
    else if (i1==3) ped=buf[0]+buf[1]+buf[2]+buf[7];
    else if (i1>=4) ped=buf[0]+buf[1]+buf[2]+buf[3];
    
    hped[ieta][(iphi-1)/2][depth-1]->Fill(ped);

    // Mean signal time estimation
    ped=ped/4;
    meant=(buf[i1]-ped)*i1+(buf[i1+1]-ped)*(i1+1)+(buf[i1+2]-ped)*(i1+2)+(buf[i1+3]-ped)*(i1+3);
    meant /= (buf[i1]-ped)+(buf[i1+1]-ped)+(buf[i1+2]-ped)+(buf[i1+3]-ped);
    htmean->Fill(meant);
    htsm[ieta][(iphi-1)/2][depth-1]->Fill(meant);
  }
}

