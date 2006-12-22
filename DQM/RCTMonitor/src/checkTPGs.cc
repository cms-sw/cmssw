#include "DQM/RCTMonitor/src/checkTPGs.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

checkTPGs::checkTPGs(const edm::ParameterSet& iConfig)
{
}

checkTPGs::~checkTPGs()
{
}


void checkTPGs::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::ESHandle<CaloTPGTranscoder> transcoder;
  iSetup.get<CaloTPGRecord>().get(transcoder);
  edm::Handle<EcalTrigPrimDigiCollection> ecal;
  edm::Handle<HcalTrigPrimDigiCollection> hcal;
  iEvent.getByType(ecal);
  iEvent.getByLabel("hcalTriggerPrimitiveDigis",hcal);
  EcalTrigPrimDigiCollection ecalCollection = *ecal;
  HcalTrigPrimDigiCollection hcalCollection = *hcal;
  int nEcalDigi = ecalCollection.size();
  if (nEcalDigi>4032) {nEcalDigi=4032;}
  float ecalSum = 0;
  float ecalMax = 0;
  for (int i = 0; i < nEcalDigi; i++){
    unsigned short energy = ecalCollection[i].compressedEt();
    float et = float(energy)/2.;  // Temporarily ET is hardcoded to be in 0.5 GeV steps in linear scale
    if(et > 1) ecalSum += et;
    if(et > ecalMax) ecalMax = et;
  }
  float hbSum = 0;
  float heSum = 0;
  float hfSum = 0;
  float hbMax = 0;
  float heMax = 0;
  float hfMax = 0;
  int nHcalDigi = hcalCollection.size();
  if (nHcalDigi != 4176){ cout << "There are " << nHcalDigi << " instead of 4176!" << endl;}
  // incl HF 4032 + 144 = 4176
  for (int i = 0; i < nHcalDigi; i++){
    short ieta = (short) hcalCollection[i].id().ieta(); 
    unsigned short absIeta = (unsigned short) abs(ieta);
    unsigned short energy = hcalCollection[i].SOI_compressedEt();     // access only sample of interest
    float et = transcoder->hcaletValue(absIeta, energy);
    if (absIeta <= 28){
      if(absIeta <= 24) {
	hbSum += et;
	if(et > hbMax) hbMax = et;
      }
      if(absIeta > 24) {
	heSum += et;
	if(et > heMax) heMax = et;
      }
    }
    else if ((absIeta >= 29) && (absIeta <= 32)){
      hfSum += et;
      if(et > hfMax) hfMax = et;
    }
  }
  cout << endl;
  cout << "ecalSum = " << ecalSum << "\thbSum = " << hbSum << "\theSum = " << heSum << "\thfSum = " << hfSum << endl;
  cout << "ecalMax = " << ecalMax << "\thbMax = " << hbMax << "\theMax = " << heMax << "\thfMax = " << hfMax << endl;
  cout << endl;
}
