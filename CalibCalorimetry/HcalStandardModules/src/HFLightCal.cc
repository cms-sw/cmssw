// Analysis of HF LED/Laser run: 
// SPE calibration for low light intensity or raw SPE calibration for high light intensity
// and HF performance based on this analysis
//
// Igor Vodopiyanov. Oct-2007
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

HFLightCal::HFLightCal (const edm::ParameterSet& fConfiguration) {
  //std::string histfile = fConfiguration.getUntrackedParameter<string>("rootFile");
  histfile = fConfiguration.getUntrackedParameter<string>("rootFile");
  textfile = fConfiguration.getUntrackedParameter<string>("textFile");
  prefile = fConfiguration.getUntrackedParameter<string>("preFile");
}

HFLightCal::~HFLightCal () {
  //delete mFile;
}

void HFLightCal::beginJob(const edm::EventSetup& fSetup) {

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
  hsignalmean = new TH1F("hsignalmean","Mean ADC 4maxTS",1000,-0.5,9999.5);
  hsignalrms = new TH1F("hsignalrms","RMS ADC 4maxTS",500,0,500);
  hpedmean = new TH1F("hpedmean","Mean ADC 4lowTS",500,0,200);
  hpedrms = new TH1F("hpedrms","RMS ADC 4lowTS",500,0,100);
  hspes = new TH1F("hspes","SPE if measured",200,0,40);
  hnpevar = new TH1F("hnpevar","~N PE input",500,0,500);
  hsignalmapP = new TH2F("hsignalmapP","Mean(Response) - Mean(Pedestal) <+>;#eta;#phi",26,28.5,41.5,36,0,72);
  hsignalRMSmapP = new TH2F("hsignalRMSmapP","RMS Response <+>;#eta;#phi",26,28.5,41.5,36,0,72);
  hnpemapP = new TH2F("hnpemapP","~N PE input <+>;#eta;#phi",26,28.5,41.5,36,0,72);
  hnpemapP->SetOption("COLZ");hsignalmapP->SetOption("COLZ");hsignalRMSmapP->SetOption("COLZ");
  hsignalmapM = new TH2F("hsignalmapM","Mean(Response) - Mean(Pedestal) <->;#eta;#phi",26,-41.5,-28.5,36,0,72);
  hsignalRMSmapM = new TH2F("hsignalRMSmapM","RMS Response <->;#eta;#phi",26,-41.5,-28.5,36,0,72);
  hnpemapM = new TH2F("hnpemapM","~N PE input <->;#eta;#phi",26,-41.5,-28.5,36,0,72);
  hnpemapM->SetOption("COLZ");hsignalmapM->SetOption("COLZ");hsignalRMSmapM->SetOption("COLZ");
  // Channel-by-channel histos
  for (int i=0;i<13;i++) for (int j=0;j<36;j++) for (int k=0;k<2;k++) {
    if (i>10 && j%2==0) continue;
    sprintf(htit,"ts_+%d_%d_%d",i+29,j*2+1,k+1);
    hts[i][j][k] = new TH1F(htit,htit,10,-0.5,9.5); // TimeSlices (pulse shape)
    sprintf(htit,"tsmean_+%d_%d_%d",i+29,j*2+1,k+1);
    htsm[i][j][k] = new TH1F(htit,htit,100,0,10);   // Mean signal time estimated from TS 
    sprintf(htit,"sp_+%d_%d_%d",i+29,j*2+1,k+1);
    hsp[i][j][k] = new TH1F(htit,htit,1000,-0.5,9999.5); // Big-scale spectrum (linear ADC)
    sprintf(htit,"spe_+%d_%d_%d",i+29,j*2+1,k+1);
    hspe[i][j][k] = new TH1F(htit,htit,200,-2.5,197.5); // Small-scale spectrum (linear ADC)
    sprintf(htit,"ped_+%d_%d_%d",i+29,j*2+1,k+1);
    hped[i][j][k] = new TH1F(htit,htit,100,-2.5,97.5); // Pedestal spectrum
    sprintf(htit,"ts_-%d_%d_%d",i+29,j*2+1,k+1);
    hts[i+13][j][k] = new TH1F(htit,htit,10,-0.5,9.5);
    sprintf(htit,"tsmean_-%d_%d_%d",i+29,j*2+1,k+1);
    htsm[i+13][j][k] = new TH1F(htit,htit,100,0,10);  
    sprintf(htit,"sp_-%d_%d_%d",i+29,j*2+1,k+1);
    hsp[i+13][j][k] = new TH1F(htit,htit,1000,-0.5,9999.5);
    sprintf(htit,"spe_-%d_%d_%d",i+29,j*2+1,k+1);
    hspe[i+13][j][k] = new TH1F(htit,htit,200,-2.5,197.5); 
    sprintf(htit,"ped_-%d_%d_%d",i+29,j*2+1,k+1);
    hped[i+13][j][k] = new TH1F(htit,htit,100,-2.5,97.5); 
  } 
  // PIN-diodes histos
  for (int i=0;i<4;i++) for (int j=0;j<3;j++) {
    sprintf(htit,"ts_PIN%d_+Q%d",j+1,i+1);
    htspin[i][j] = new TH1F(htit,htit,10,-0.5,9.5);
    sprintf(htit,"sp_PIN%d_+Q%d",j+1,i+1);
    hsppin[i][j] = new TH1F(htit,htit,1000,-0.5,9999.5);
    sprintf(htit,"spe_PIN%d_+Q%d",j+1,i+1);
    hspepin[i][j] = new TH1F(htit,htit,200,-2.5,197.2);
    sprintf(htit,"ped_PIN%d_+Q%d",j+1,i+1);
    hpedpin[i][j] = new TH1F(htit,htit,200,-2.5,197.2);
    sprintf(htit,"ts_PIN%d_-Q%d",j+1,i+1);
    htspin[i+4][j] = new TH1F(htit,htit,10,-0.5,9.5);
    sprintf(htit,"sp_PIN%d_-Q%d",j+1,i+1);
    hsppin[i+4][j] = new TH1F(htit,htit,1000,-0.5,9999.5);
    sprintf(htit,"spe_PIN%d_-Q%d",j+1,i+1);
    hspepin[i+4][j] = new TH1F(htit,htit,200,-2.5,197.2);
    sprintf(htit,"ped_PIN%d_-Q%d",j+1,i+1);
    hpedpin[i+4][j] = new TH1F(htit,htit,200,-2.5,197.2);
  }
  std::cout<<std::endl<<"histfile="<<histfile.c_str()<<"  textfile="<<textfile.c_str()<<std::endl;
  return;
}

Double_t FitFun(Double_t *x, Double_t *par) { 
// Spectra fit function: Pedestal Gaussian + asymmetric 1PE and 2PE peaks

  Double_t sum,xx,A0,C0,r0,sigma0,mean1,sigma1,A1,C1,r1,mean2,sigma2,A2,C2,r2;
  const Double_t k0=2.0,k1=1.0, k2=1.2;

  xx=x[0];
  sigma0 = par[2];
  A0 = 2*Nev/(2+2*par[0]+par[0]*par[0]);
  r0 = ((xx-par[1])/sigma0);
  C0 = 1/(sigma0* TMath::Exp(-k0*k0/2)/k0 +
	  sigma0*sqrt(2*3.14159)*0.5*(1+TMath::Erf(k0/1.41421)));
  //sum = 1/(sqrt(2*3.14159)*par[2])*A0*TMath::Exp(-0.5*r0*r0);
  if(r0 < k0) sum = C0*A0*TMath::Exp(-0.5*r0*r0);
  else sum = C0*A0*TMath::Exp(0.5*k0*k0-k0*r0);

  mean1 = par[1]+par[3];
  sigma1 = par[4];
  A1 = A0*par[0];
  C1 = 1/(sigma1* TMath::Exp(-k1*k1/2)/k1 +
	  sigma1*sqrt(2*3.14159)*0.5*(1+TMath::Erf(k1/1.41421)));
  r1 = ((xx-mean1)/sigma1);
  if(r1 < k1) sum += C1*A1*TMath::Exp(-0.5*r1*r1);
  else sum += C1*A1*TMath::Exp(0.5*k1*k1-k1*r1);

  mean2 = 2*par[3]+par[1];
  sigma2 = sqrt(2*par[4]*par[4] - par[2]*par[2]);
  A2 = A0*par[0]*par[0]/2;
  C2 = 1/(sigma2* TMath::Exp(-k2*k2/2)/k2 +
	  sigma2*sqrt(2*3.14159)*0.5*(1+TMath::Erf(k2/1.41421)));
  r2 = ((xx-mean2)/sigma2);
  if(r2 < k2) sum += C2*A2*TMath::Exp(-0.5*r2*r2);
  else sum += C2*A2*TMath::Exp(0.5*k2*k2-k2*r2);

  return sum;
}

void HFLightCal::endJob(void)
{
  Double_t mean,rms,meanped,rmsped,maxc,npevar,npevarm;
  Double_t par[5],dspe=0,dnpe;
  Int_t tsmax;
  std::cout<<std::endl<<"HFLightCal endJob --> ";
  fprintf(tFile,"#RunN %d   Events processed %d",runN,eventN);

  for (int i=0;i<26;i++) for (int j=0;j<36;j++) for (int k=0;k<2;k++) {
    if (i>10 && i<13 && j%2==0) continue;
    if (i>23 && j%2==0) continue;
    meanped=rmsped=mean=rms=0;
    if (hsp[i][j][k]->Integral()>0) {
      meanped=hped[i][j][k]->GetMean();
      rmsped=hped[i][j][k]->GetRMS();
      if (hspe[i][j][k]->Integral()>hsp[i][j][k]->Integral()*0.9) {
	mean=hspe[i][j][k]->GetMean();
	rms=hspe[i][j][k]->GetRMS();
      }
      else {
	mean=hsp[i][j][k]->GetMean();
	rms=hsp[i][j][k]->GetRMS();
      }
      hsignalmean->Fill(mean);
      hsignalrms->Fill(rms);
      hpedmean->Fill(meanped);
      hpedrms->Fill(rmsped);
      if (rms*rms-rmsped*rmsped>1 && mean>meanped && meanped>0) {
	hnpevar->Fill((mean-meanped)*(mean-meanped)/(rms*rms-rmsped*rmsped));
      }
    }
  }

  meanped=hpedmean->GetMean();
  rmsped=hpedrms->GetMean();
  mean=hsignalmean->GetMean();
  rms=hsignalrms->GetMean();
  npevarm=hnpevar->GetMean();
  hnpevar->Reset();
  fprintf(tFile,"   Photoelectrons input <%.2f>\n",npevarm);
  fprintf(tFile,"#eta/phi/depth  sum4maxTS     RMS     ~N_PE   sum4lowTS     RMS  maxTS  SPE +/- Err   Comment\n");
  TF1* fPed = new TF1("fPed","gaus",0,120);
  fPed->SetNpx(200);
  TF1 *fTot = new TF1("fTot",FitFun ,0,160,5);
  fTot->SetNpx(200);
  for (int i=0;i<26;i++) for (int j=0;j<36;j++) for (int k=0;k<2;k++) {
    if (i>10 && i<13 && j%2==0) continue;
    if (i>23 && j%2==0) continue;
    meanped=hped[i][j][k]->GetMean();
    rmsped=hped[i][j][k]->GetRMS();
    par[3]=0;
    mean=hsp[i][j][k]->GetMean();
    rms=hsp[i][j][k]->GetRMS();
    if (hspe[i][j][k]->Integral()>hsp[i][j][k]->Integral()*0.9 || mean<100) {
      mean=hspe[i][j][k]->GetMean();
      rms=hspe[i][j][k]->GetRMS();
      if (hspe[i][j][k]->Integral()>100) {
	if (mean+rms*3-meanped-rmsped*3>2 && rmsped>0 && meanped>0) { // SPE fit if low intensity>0
	  par[1] = meanped;
	  par[2] = rmsped;
	  par[0] = hped[i][j][k]->GetMaximum();
	  fPed->SetParameters(par);
	  hped[i][j][k]->Fit(fPed,"BQ0");
	  fPed->GetParameters(&par[0]);
	  hped[i][j][k]->Fit(fPed,"B0Q","",par[1]-par[2]*3,par[1]+par[2]*3);
	  fPed->GetParameters(par);
	  hped[i][j][k]->Fit(fPed,"BLQ","",par[1]-par[2]*2.5,par[1]+par[2]*2.5);
	  fPed->GetParameters(&par[0]);
	  Nev = (int) hspe[i][j][k]->Integral();
	  par[0]=0.1;
	  par[3]=10;
	  par[4]=6;
	  fTot->SetParameters(par);
	  fTot->SetParLimits(0,0,2);
	  fTot->FixParameter(1,par[1]);
	  fTot->FixParameter(2,par[2]);
	  fTot->SetParLimits(3,2.2,100);
	  fTot->SetParLimits(4,par[2]+0.5,100);
	  hspe[i][j][k]->Fit(fTot,"BL0Q","");
	  fTot->GetParameters(par);
	  maxc=par[3]*2+par[1]+2.5*sqrt(2*par[4]*par[4]-par[2]*par[2]);
	  hspe[i][j][k]->Fit(fTot,"BLEQ","",0,maxc);
	  fTot->GetParameters(par);
	  dspe=fTot->GetParError(3);
	  dnpe=fTot->GetParError(0);
	  if (par[3]<2.21  || dnpe>par[0] || par[4]<par[2]+0.6 || par[4]>99) par[3]=-1;
	  else if (par[0]>1.96 || par[3]>95) par[3]=0;
	  else hspes->Fill(par[3]);
	} 
      }
    }

    // NPE
    npevar=0;
    if (par[3]>0 && mean<(par[3]+meanped)) npevar=par[0]; // NPE from SPE fit
    else {                                                // NPE from high intensity signal
      if (hspe[i][j][k]->Integral()>hsp[i][j][k]->Integral()*0.98) {
	hspe[i][j][k]->SetAxisRange(mean-3*rms,mean+3*rms);
	mean=hspe[i][j][k]->GetMean();
	rms=hspe[i][j][k]->GetRMS();
	hspe[i][j][k]->SetAxisRange(-2.5,197.2);
      }
      else {
	hsp[i][j][k]->SetAxisRange(mean-3*rms,mean+3*rms);
	mean=hsp[i][j][k]->GetMean();
	rms=hsp[i][j][k]->GetRMS();
	hsp[i][j][k]->SetAxisRange(-0.5,9997.5);
      }
      if (rms*rms-rmsped*rmsped>1 && mean>meanped && meanped>0) 
	npevar=(mean-meanped)*(mean-meanped)/(rms*rms-rmsped*rmsped);
    }
    if (npevar>5.0e-5) hnpevar->Fill(npevar);

    if (i<13) {
      hsignalmapP->Fill(i+28.6+k/2.0,j*2+1,mean-meanped); 
      hsignalRMSmapP->Fill(i+28.6+k/2.0,j*2+1,rms);
      hnpemapP->Fill(i+28.6+k/2.0,j*2+1,npevar);
      fprintf(tFile,"%3d%4d%5d  %11.2f%8.2f",i+29,j*2+1,k+1,mean,rms);
    }
    else {
      fprintf(tFile,"%3d%4d%5d  %11.2f%8.2f",13-i-29,j*2+1,k+1,mean,rms);
      hsignalmapM->Fill(13-i-28.6-k/2.0,j*2+1,mean-meanped);
      hsignalRMSmapM->Fill(13-i-28.6-k/2.0,j*2+1,rms);
      hnpemapM->Fill(13-i-28.6-k/2.0,j*2+1,npevar);
    }
    fprintf(tFile,"  %9.4f",npevar);
    if (hped[i][j][k]->Integral()<=0 && hped[i][j][k]->GetBinContent(201)>1) {
      meanped=-1;
      rmsped=-1;
    }
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
      if (hsp[i][j][k]->Integral()<=10)  fprintf(tFile,"SignalOffRange\n");
      else {
	if (hsp[i][j][k]->Integral()<100)  fprintf(tFile,"Nev<100/");
	if (npevar>0) {
	  if (npevar<npevarm/5) fprintf(tFile,"LowNPE/");
	  else if (npevar>npevarm*5) fprintf(tFile,"HighNPE/");
	  if (par[3]>0 && (npevar*Nev<10 || npevar<0.001)) fprintf(tFile,"LowSignal/");
	}
	else if (fabs(mean-meanped)<5) fprintf(tFile,"LowSignal/");
	if (par[3]<0)  fprintf(tFile,"BadFit/");
	else if (par[3]==0)  fprintf(tFile,"NoSPEFit/");
	else if (par[3]>0 && npevar>1)   fprintf(tFile,"NPE>1/");
	if (mean<2) fprintf(tFile,"LowMean/");
	if (rms<0.5) fprintf(tFile,"LowRMS/"); 
	if (meanped==-1) fprintf(tFile,"Ped>200/");
	else if (meanped<2) fprintf(tFile,"LowPed/");
	else if (meanped>25) fprintf(tFile,"HighPed/"); 
	if (rmsped<0.5 && rmsped>0) fprintf(tFile,"NarrowPed/"); 
	else if (rmsped>10) fprintf(tFile,"WidePed/");
	fprintf(tFile,"-\n");
      }
    }
  }

  for (int i=0;i<8;i++) for (int j=0;j<3;j++) {
    meanped=hpedpin[i][j]->GetMean();
    rmsped=hpedpin[i][j]->GetRMS();
    mean=hsppin[i][j]->GetMean();
    rms=hsppin[i][j]->GetRMS();
    if (hspepin[i][j]->Integral()+100>hsppin[i][j]->Integral() || mean<100) {
      mean=hspepin[i][j]->GetMean();
      rms=hspepin[i][j]->GetRMS();
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
  Double_t maxADC,signal,ped,meant;
  Int_t ii,maxisample=0,i1=3,i2=6;

  // HF PIN-diodes
  edm::Handle<HcalCalibDigiCollection> calib;  
  fEvent.getByType(calib);
  if (verbose) std::cout<<"Analysis-> total CAL digis= "<<calib->size()<<std::endl;

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
    }
  }
  
  // HF
  edm::Handle<HFDigiCollection> hf_digi;
  fEvent.getByType(hf_digi);
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
    maxisample = itsmax[ieta][(iphi-1)/2][depth-1]-1;
    if (verbose) std::cout <<"Max-i-sample = " <<maxisample<<std::endl;

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
    }

    // Signal is four capIDs found by PreAnal, Pedestal is four capID off the signal
    htmax->Fill(maxisample);
    i1=maxisample-1;
    i2=maxisample+2;
    if (i1<0) {i1=0;i2=3;}
    else if (i2>9) {i1=6;i2=9;} 
    signal=buf[i1]+buf[i1+1]+buf[i1+2]+buf[i1+3];
    hsp[ieta][(iphi-1)/2][depth-1]->Fill(signal);
    hspe[ieta][(iphi-1)/2][depth-1]->Fill(signal);
    if      (i1==0) ped=buf[8]+buf[9]+buf[6]+buf[7];
    else if (i1==1) ped=buf[8]+buf[9]+buf[6]+buf[7];
    else if (i1==2) ped=buf[0]+buf[1]+buf[6]+buf[7];
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

