// Pre-Analysis for HFLightCal:
// finding time position of signal in TS
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

#include "CalibCalorimetry/HcalStandardModules/interface/HFPreLightCal.h"

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
Int_t run_NN=0, event_NN=0;

namespace {
  //bool verbose = true;
  bool verbose = false;
}

HFPreLightCal::HFPreLightCal (const edm::ParameterSet& fConfiguration) :
  hfDigiCollectionTag_(fConfiguration.getParameter<edm::InputTag>("hfDigiCollectionTag")),
  hcalCalibDigiCollectionTag_(fConfiguration.getParameter<edm::InputTag>("hcalCalibDigiCollectionTag")) {

  //std::string histfile = fConfiguration.getUntrackedParameter<string>("rootFile");
  histfile = fConfiguration.getUntrackedParameter<string>("rootPreFile");
  textfile = fConfiguration.getUntrackedParameter<string>("textPreFile");
}

HFPreLightCal::~HFPreLightCal () {
  //delete mFile;
}

void HFPreLightCal::beginJob() {

  char htit[64];
  std::cout<<std::endl<<"HFPreLightCal beginJob: --> ";

  mFile = new TFile (histfile.c_str(),"RECREATE");
  if ((tFile = fopen(textfile.c_str(),"w"))==NULL) {
    printf("\nNo textfile open\n\n");
    std::cout<<"Problem with output Pre-textFILE => exit"<<std::endl;
    exit(1);
  }
  // Histos
  htsmax = new TH1F("htsmax","Max TS",100,0,10);
  htspinmax = new TH1F("htspinmax","Max TS PIN",100,0,10);
  // Channel-by-channel histos
  for (int i=0;i<13;i++) for (int j=0;j<36;j++) for (int k=0;k<2;k++) {
    if (i>10 && j%2==0) continue;
    sprintf(htit,"tspre_+%d_%d_%d",i+29,j*2+1,k+1);
    hts[i][j][k] = new TH1F(htit,htit,10,-0.5,9.5); // TimeSlices (pulse shape)
    sprintf(htit,"tspre_-%d_%d_%d",i+29,j*2+1,k+1);
    hts[i+13][j][k] = new TH1F(htit,htit,10,-0.5,9.5); // TimeSlices (pulse shape)
  } 
  // PIN-diodes histos
  for (int i=0;i<4;i++) for (int j=0;j<3;j++) {
    sprintf(htit,"tspre_PIN%d_+Q%d",j+1,i+1);
    htspin[i][j] = new TH1F(htit,htit,10,-0.5,9.5);
    sprintf(htit,"tspre_PIN%d_-Q%d",j+1,i+1);
    htspin[i+4][j] = new TH1F(htit,htit,10,-0.5,9.5);
  }
  std::cout<<"histfile="<<histfile.c_str()<<"  textfile="<<textfile.c_str()<<std::endl;
  return;
}

void HFPreLightCal::endJob(void)
{
  Double_t sum,cont;
  Int_t tsmax;

  std::cout<<std::endl<<"HFPreLightCal endJob --> ";

  for (int i=0;i<26;i++) for (int j=0;j<36;j++) for (int k=0;k<2;k++) {
    if (i>10 && i<13 && j%2==0) continue;
    if (i>23 && j%2==0) continue;
    sum=tsmax=0;
    for (int ii=1; ii<=10; ii++) {
      cont = hts[i][j][k]->GetBinContent(ii);
      if (ii<3) cont=cont-(hts[i][j][k]->GetBinContent(ii+4)+hts[i][j][k]->GetBinContent(ii+8))/2;
      else if (ii<5) cont=cont-hts[i][j][k]->GetBinContent(ii+4);
      else if (ii<7) cont=cont-(hts[i][j][k]->GetBinContent(ii-4)+hts[i][j][k]->GetBinContent(ii+4))/2;
      else if (ii<9) cont=cont-hts[i][j][k]->GetBinContent(ii-4);
      else cont=cont-(hts[i][j][k]->GetBinContent(ii-4)+hts[i][j][k]->GetBinContent(ii-8))/2;
      if (cont>sum) {
	sum = cont;
	tsmax=ii;
      }
    }
    htsmax->Fill(tsmax);
    if (i<13) fprintf(tFile," %d  %d  %d  %d\n",i+29,j*2+1,k+1,tsmax);
    else      fprintf(tFile," %d  %d  %d  %d\n",13-i-29,j*2+1,k+1,tsmax);
  }

  for (int i=0;i<8;i++) for (int j=0;j<3;j++) {
    sum=tsmax=0;
    tsmax = htspin[i][j]->GetMaximumBin();
    htspinmax->Fill(tsmax);
    if (i<4) fprintf(tFile,"%d  %d  %d\n",i+1,j+1,tsmax);
    else     fprintf(tFile,"%d  %d  %d\n",-i+3,j+1,tsmax);
  } 

  mFile->Write();
  mFile->Close();
  fclose(tFile);
  std::cout<<" Nevents = "<<event_NN<<std::endl;
  return;
}

void HFPreLightCal::analyze(const edm::Event& fEvent, const edm::EventSetup& fSetup) {

  // event ID
  edm::EventID eventId = fEvent.id();
  int runNumber = eventId.run ();
  int eventNumber = eventId.event ();
  if (run_NN==0) run_NN=runNumber;
  event_NN++;
  if (verbose) std::cout << "========================================="<<std::endl
			 << "run/event: "<<runNumber<<'/'<<eventNumber<<std::endl;

  // HF PIN-diodes
  edm::Handle<HcalCalibDigiCollection> calib;  
  fEvent.getByLabel(hcalCalibDigiCollectionTag_, calib);
  if (verbose) std::cout<<"Analysis-> total CAL digis= "<<calib->size()<<std::endl;
  /* COMMENTED OUT by J. Mans (7-28-2008) as major changes needed with new Calib DetId 

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
      for (int isample = 0; isample < digi.size(); ++isample) {
	int adc = digi[isample].adc();
	int capid = digi[isample].capid ();
	double linear_ADC = digi[isample].nominal_fC();
	if (verbose) std::cout<<"PIN linear_ADC = "<<linear_ADC<<std::endl;
	htspin[isector+iside][ipin]->Fill(isample,linear_ADC);
      }
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
    }
  }
  return;
}

