// Analysis of HF LED/Laser run: 
// Case when signal has random time position in TS
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

#include "CalibCalorimetry/HcalStandardModules/interface/HFLightCalRand.h"

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
Int_t NEvents, runNumb=0, EventN=0;

namespace {
  //bool verbose = true;
  bool verbose = false;
}

HFLightCalRand::HFLightCalRand (const edm::ParameterSet& fConfiguration) {
  //std::string histfile = fConfiguration.getUntrackedParameter<string>("rootFile");
  histfile = fConfiguration.getUntrackedParameter<string>("rootFile");
  textfile = fConfiguration.getUntrackedParameter<string>("textFile");
}

HFLightCalRand::~HFLightCalRand () {
  //delete mFile;
}

void HFLightCalRand::beginJob(const edm::EventSetup& fSetup) {

  char htit[64];
  mFile = new TFile (histfile.c_str(),"RECREATE");
  if ((tFile = fopen(textfile.c_str(),"w"))==NULL)printf("\nNo textfile open\n\n");
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
  std::cout<<std::endl<<"beginJob: histfile="<<histfile.c_str()<<"  textfile="<<textfile.c_str()<<std::endl;
  return;
}

Double_t Fit3Peak(Double_t *x, Double_t *par) { 
// Spectra fit function: Pedestal Gaussian + asymmetric 1PE and 2PE peaks

  Double_t sum,xx,A0,C0,r0,sigma0,mean1,sigma1,A1,C1,r1,mean2,sigma2,A2,C2,r2;
  const Double_t k0=2.0,k1=1.0, k2=1.2;

  xx=x[0];
  sigma0 = par[2];
  A0 = 2*NEvents/(2+2*par[0]+par[0]*par[0]);
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

void HFLightCalRand::endJob(void)
{
  Double_t mean,rms,meanped,rmsped,maxc,npevar;
  Double_t par[5],parm[5],dspe=0,dnpe;
  Int_t tsmax;
  fprintf(tFile,"#RunN %d   Events processed %d",runNumb,EventN);
  std::cout<<"endJob: histos processing..."<<std::endl;
  std::cout<<"RunN= "<<runNumb<<"  Events processed= "<<EventN<<std::endl;

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
  TF1 *fTot = new TF1("fTot",Fit3Peak ,0,200,5);
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
	  fPed->GetParameters(&parm[0]);
	  NEvents = (int) hspe[i][j][k]->Integral();
	  par[0]=0.1;
	  par[3]=10;
	  par[4]=6;
	  fTot->SetParameters(par);
	  fTot->SetParLimits(0,0,2);
	  //fTot->FixParameter(1,par[1]);
	  //fTot->FixParameter(2,par[2]);
	  fTot->SetParLimits(1,par[1]*0.8,par[1]*1.4);
	  fTot->SetParLimits(2,par[2]*0.9,par[2]);
	  fTot->SetParLimits(3,2.2,100);
	  fTot->SetParLimits(4,par[2]+0.5,100);
	  hspe[i][j][k]->Fit(fTot,"BL0Q","");
	  fTot->GetParameters(par);
	  maxc=par[3]*2+par[1]+2.5*sqrt(2*par[4]*par[4]-par[2]*par[2]);
	  maxc=TMath::Max(maxc,par[1]+20);
	  hspe[i][j][k]->Fit(fTot,"BLEQ","",0,maxc);
	  //hspe[i][j][k]->Fit(fTot,"BLEQ","");
	  fTot->GetParameters(par);
	  dspe=fTot->GetParError(3);
	  dnpe=fTot->GetParError(0);
	  if (par[3]<2.21 || dnpe>par[0] || par[4]<par[2]+0.6 || par[4]>99) par[3]=-1;
	  else if (par[0]>1.96 || par[3]>95) par[3]=0;
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
	hspe[i][j][k]->SetAxisRange(mean-3*rms,mean+3*rms);
	mean=hspe[i][j][k]->GetMean();
	rms=hspe[i][j][k]->GetRMS();
	hspe[i][j][k]->SetAxisRange(mean-3*rms,mean+3*rms);
	mean=hspe[i][j][k]->GetMean();
	rms=hspe[i][j][k]->GetRMS();
	hspe[i][j][k]->SetAxisRange(-2.5,197.2);
      }
      else {
	hsp[i][j][k]->SetAxisRange(mean-3*rms,mean+3*rms);
	mean=hsp[i][j][k]->GetMean();
	rms=hsp[i][j][k]->GetRMS();
	hsp[i][j][k]->SetAxisRange(mean-3*rms,mean+3*rms);
	mean=hsp[i][j][k]->GetMean();
	rms=hsp[i][j][k]->GetRMS();
	hsp[i][j][k]->SetAxisRange(-0.5,9999.5);
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
	if (npevar>0 && par[3]>0 && (npevar*NEvents<10 || npevar<0.001)) 
	  fprintf(tFile,"LowSignal/");
	else if (npevar==0 && fabs(mean-meanped)<5) fprintf(tFile,"LowSignal/");
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

void HFLightCalRand::analyze(const edm::Event& fEvent, const edm::EventSetup& fSetup) {

  // event ID
  edm::EventID eventId = fEvent.id();
  int runNumber = eventId.run ();
  int eventNumber = eventId.event ();
  if (runNumb==0) runNumb=runNumber;
  EventN++;
  if (verbose) std::cout << "========================================="<<std::endl
			 << "run/event: "<<runNumber<<'/'<<eventNumber<<std::endl;

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

    if (iside != -1) {
      maxADC=-99;
      for (int isample = 0; isample < digi.size(); ++isample) {
	int adc = digi[isample].adc();
	int capid = digi[isample].capid ();
	double linear_ADC = digi[isample].nominal_fC();
	if (verbose) std::cout<<"PIN linear_ADC = "<<linear_ADC<<std::endl;
	htspin[isector+iside][ipin]->Fill(isample,linear_ADC);
	buf[isample]=linear_ADC;
	if (maxADC<linear_ADC) {
	  maxADC=linear_ADC;
	  maxisample=isample;
	}
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
			     << ", nominal fC=" << nominal_fC << std::endl;


      hts[ieta][(iphi-1)/2][depth-1]->Fill(isample,linear_ADC);
      buf[isample]=linear_ADC;
      if (maxADC<linear_ADC) {
	maxADC=linear_ADC;
	maxisample=isample;
      }
    }
     
    // Signal is four capIDs around maxTS, Pedestal is four capID off the signal
    maxADC=-99;
    for (int ibf=0; ibf<7; ibf++) {
      Double_t sumbuf=0;
      for (int jbf=0; jbf<4; jbf++) {
	sumbuf += buf[ibf+jbf];
	if (ibf+jbf<2) sumbuf -= (buf[ibf+jbf+4]+buf[ibf+jbf+8])/2.0;
	else if (ibf+jbf<4) sumbuf -= buf[ibf+jbf+4];
	else if (ibf+jbf<6) sumbuf -= (buf[ibf+jbf-4]+buf[ibf+jbf+4])/2.0;
	else if (ibf+jbf<8) sumbuf -= buf[ibf+jbf-4];
	else if (ibf+jbf<10) sumbuf -= (buf[ibf+jbf-4]+buf[ibf+jbf-8])/2.0;
      }
      if (sumbuf>maxADC) {
	maxADC=sumbuf;
	maxisample=ibf+1;
      }
      htmax->Fill(maxisample);
    }
    i1=maxisample-1;
    i2=maxisample+2;
    if (i1<0) {i1=0;i2=3;}
    else if (i2>9) {i1=6;i2=9;} 
    signal=buf[i1]+buf[i1+1]+buf[i1+2]+buf[i1+3];
    hsp[ieta][(iphi-1)/2][depth-1]->Fill(signal);
    hspe[ieta][(iphi-1)/2][depth-1]->Fill(signal);
    if      (i1==0) ped=(buf[4]+buf[8])/2.0+(buf[5]+buf[9])/2.0+buf[6]+buf[7];
    else if (i1==1) ped=(buf[0]+buf[8])/2.0+(buf[5]+buf[9])/2.0+buf[6]+buf[7];
    else if (i1==2) ped=(buf[0]+buf[8])/2.0+(buf[1]+buf[9])/2.0+buf[6]+buf[7];
    else if (i1==3) ped=(buf[0]+buf[8])/2.0+(buf[1]+buf[9])/2.0+buf[2]+buf[7];
    else if (i1==4) ped=(buf[0]+buf[8])/2.0+(buf[1]+buf[9])/2.0+buf[2]+buf[3];
    else if (i1==5) ped=(buf[0]+buf[4])/2.0+(buf[1]+buf[9])/2.0+buf[2]+buf[3];
    else if (i1==6) ped=(buf[0]+buf[4])/2.0+(buf[1]+buf[5])/2.0+buf[2]+buf[3];
    hped[ieta][(iphi-1)/2][depth-1]->Fill(ped);

    // Mean signal time estimation
    ped=ped/4;
    meant=(buf[i1]-ped)*i1+(buf[i1+1]-ped)*(i1+1)+(buf[i1+2]-ped)*(i1+2)+(buf[i1+3]-ped)*(i1+3);
    meant /= (buf[i1]-ped)+(buf[i1+1]-ped)+(buf[i1+2]-ped)+(buf[i1+3]-ped);
    htmean->Fill(meant);
    htsm[ieta][(iphi-1)/2][depth-1]->Fill(meant);
  }
}

