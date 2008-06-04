// -*- C++ -*-
//
// Class:      EcalTPGAnalyzer
//
//
// Original Author:  Pascal Paganini
//
//


// system include files
#include <memory>
#include <utility>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "EcalTPGAnalyzer.h"

#include <TMath.h>
#include <sstream>

using namespace edm;
class CaloSubdetectorGeometry;

EcalTPGAnalyzer::EcalTPGAnalyzer(const edm::ParameterSet&  iConfig)
{
  label_= iConfig.getParameter<std::string>("Label");
  producer_= iConfig.getParameter<std::string>("Producer");
  digi_label_= iConfig.getParameter<std::string>("DigiLabel");
  digi_producerEB_=  iConfig.getParameter<std::string>("DigiProducerEB");
  digi_producerEE_=  iConfig.getParameter<std::string>("DigiProducerEE");
  emul_label_= iConfig.getParameter<std::string>("EmulLabel");
  emul_producer_=  iConfig.getParameter<std::string>("EmulProducer");
  allowTP_ = iConfig.getParameter<bool>("ReadTriggerPrimitives");
  useEE_ = iConfig.getParameter<bool>("UseEndCap");
  adcCut_ =  iConfig.getParameter<int>("ADCCut");
  shapeCut_ =  iConfig.getParameter<int>("shapeCut");
  occupancyCut_ =  iConfig.getParameter<int>("occupancyCut");
  tpgRef_ =  iConfig.getParameter<int>("TPGEmulatorIndexRef") ;

  histfile_ = new TFile("histosTPG.root","RECREATE");

  //histo
  for (int iSM  = 0 ; iSM<36 ; iSM++) {
    std::stringstream name ;
    name<<"shape_"<<iSM+1 ;
    shape_[iSM] = new TH2F(name.str().c_str(),"Tower Pulses",11, -1, 10, 120, -1.1, 1.1) ;
    shape_[iSM]->GetXaxis()->SetTitle("sample nb") ;
  }
  shapeMax_ = new TH2F("shapeMax","Crystal Max Shapes",11, -1, 10, 120, -1.1, 1.1) ; 
  shapeMax_->GetXaxis()->SetTitle("sample nb") ;

  occupancyTP_ = new TH2F("occupancyTP", "Occupancy TP data", 72, 1, 73, 38, -19, 19) ;
  occupancyTP_->GetYaxis()->SetTitle("eta index") ;
  occupancyTP_->GetXaxis()->SetTitle("phi index") ;
  occupancyTPEmul_ = new TH2F("occupancyTPEmul", "Occupancy TP emulator", 72, 1, 73, 38, -19, 19) ;
  occupancyTPEmul_->GetYaxis()->SetTitle("eta index") ;
  occupancyTPEmul_->GetXaxis()->SetTitle("phi index") ;
  occupancyTPEmulMax_ = new TH2F("occupancyTPEmulMax", "Occupancy TP emulator max", 72, 1, 73, 38, -19, 19) ;
  occupancyTPEmulMax_->GetYaxis()->SetTitle("eta index") ;
  occupancyTPEmulMax_->GetXaxis()->SetTitle("phi index") ;

  crystalVsTP_ = new TH2F("crystalVsTP", "tower Ener vs TP", 256, 0., 256., 301, -1., 300.) ;
  crystalVsTP_->GetXaxis()->SetTitle("TP (ADC)") ;
  crystalVsTP_->GetYaxis()->SetTitle("Tower E(ADC)") ;
  crystalVsEmulTP_ = new TH2F("crystalVsEmulTP", "tower Ener vs Emulator TP", 256, 0., 256., 301, -1., 300.) ;
  crystalVsEmulTP_->GetXaxis()->SetTitle("TP (ADC)") ;
  crystalVsEmulTP_->GetYaxis()->SetTitle("Tower E(ADC)") ;
  crystalVsEmulTPMax_ = new TH2F("crystalVsEmulTPMax", "tower Ener vs Max Emulator TP", 256, 0., 256., 301, -1., 300.) ;
  crystalVsEmulTPMax_->GetXaxis()->SetTitle("TP (ADC)") ;
  crystalVsEmulTPMax_->GetYaxis()->SetTitle("Tower E(ADC)") ;
  TPVsEmulTP_ = new TH2F("TPVsEmulTP", "TP vs Emulator TP", 256, 0., 256., 256, 0., 256.) ;
  TPVsEmulTP_->GetXaxis()->SetTitle("TP Emul (ADC)") ;
  TPVsEmulTP_->GetYaxis()->SetTitle("TP Data (ADC)") ;
  
  TP_ = new TH1F("TP", "TP", 256, 0., 256.) ;
  TP_->GetXaxis()->SetTitle("TP (ADC)") ;
  TPEmul_ = new TH1F("TPEmul", "TP Emulator", 256, 0., 256.) ;
  TPEmul_->GetXaxis()->SetTitle("TP (ADC)") ;
  TPEmulMax_ = new TH1F("TPEmulMax", "TP Emulator max", 256, 0., 256.) ;
  TPEmulMax_->GetXaxis()->SetTitle("TP (ADC)") ;

  TPMatchEmul_ = new TH1F("TPMatchEmul", "TP data matching Emulator",6 , -1., 5.) ;
  TPEmulMaxIndex_ = new TH1F("TPEmulMaxIndex", "Index of the max TP from Emulator", 6, -1., 5.) ;
}


EcalTPGAnalyzer::~EcalTPGAnalyzer()
{
  for (int iSM=0 ; iSM<36 ; iSM++) shape_[iSM]->Write() ;
  shapeMax_->Write() ;
  occupancyTP_->Write() ;
  occupancyTPEmul_->Write() ;
  occupancyTPEmulMax_->Write() ;
  crystalVsTP_->Write() ;
  crystalVsEmulTP_->Write() ;
  crystalVsEmulTPMax_->Write() ;
  TPVsEmulTP_->Write() ;
  TP_->Write() ;
  TPEmul_->Write() ;
  TPEmulMax_->Write() ;
  TPMatchEmul_->Write() ; 
  TPEmulMaxIndex_->Write() ;

  histfile_->Write();
  histfile_->Close();
}

void EcalTPGAnalyzer::beginJob(const edm::EventSetup& evtSetup)
{
  // geometry
  ESHandle<CaloGeometry> theGeometry;
  ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle, theBarrelGeometry_handle;
  evtSetup.get<IdealGeometryRecord>().get( theGeometry );
  evtSetup.get<IdealGeometryRecord>().get("EcalEndcap",theEndcapGeometry_handle);
  evtSetup.get<IdealGeometryRecord>().get("EcalBarrel",theBarrelGeometry_handle);
  evtSetup.get<IdealGeometryRecord>().get(eTTmap_);
  theEndcapGeometry_ = &(*theEndcapGeometry_handle);
  theBarrelGeometry_ = &(*theBarrelGeometry_handle);
}


void EcalTPGAnalyzer::analyze(const edm::Event& iEvent, const  edm::EventSetup & iSetup)
{
  using namespace edm;
  using namespace std;
  
  map<EcalTrigTowerDetId, towerEner> mapTower ;
  map<EcalTrigTowerDetId, towerEner>::iterator itTT ;
  
  // Get EB xtal digi inputs
  edm::Handle<EBDigiCollection> digiEB;
  iEvent.getByLabel(digi_label_, digi_producerEB_, digiEB);
  
  float E_xtal_Max = -999. ;
  EBDataFrame dfMax ;

  for (unsigned int i=0;i<digiEB.product()->size();i++) {
    const EBDataFrame & df = (*(digiEB.product()))[i];    
    int gain, adc ;
    float E_xtal = 0. ; 
    int theSamp = 0 ;
    float mean = 0., max = -999 ; 
    for (int samp = 0 ; samp<10 ; samp++) {
      adc = df[samp].adc() ;
      if (samp<2) mean += adc ;
      if (adc>max) {
	max = adc ;
	theSamp = samp ;
      }
    }
    mean /= 2. ;
    if (mean>0 && max >= mean + adcCut_) {
      gain = df[theSamp].gainId() ;
      adc = df[theSamp].adc() ;
      if (gain == 1) E_xtal = adc-mean ;
      if (gain == 2) E_xtal = 2.*adc-mean ;
      if (gain == 3) E_xtal = 12.*adc-mean ;
      if (gain == 0) E_xtal = 12.*adc-mean ;
    }
    if (E_xtal > E_xtal_Max) {
      E_xtal_Max = E_xtal ;
      dfMax = df ;
    }
    
    const EBDetId & id=df.id();
    const EcalTrigTowerDetId towid= id.tower();
    itTT = mapTower.find(towid) ;
    if (itTT != mapTower.end()) {
      (itTT->second).eRec_ += E_xtal ;
      for (int samp = 0 ; samp<10 ; samp++) {
	gain = df[samp].gainId() ;
	adc = df[samp].adc() ;
	if (gain == 1) (itTT->second).data_[samp] += adc-mean ;
	if (gain == 2) (itTT->second).data_[samp] += 2.*adc-mean ;
	if (gain == 3) (itTT->second).data_[samp] += 12.*adc-mean ;
	if (gain == 0) (itTT->second).data_[samp] += 12.*adc-mean ;
      }
      (itTT->second).iphi_ = towid.iphi() ;
      (itTT->second).ieta_ = towid.ieta() ;
      (itTT->second).iSM_ = towid.iDCC() ;
    }
    else {
      towerEner tE ;
      tE.eRec_ = E_xtal ;
      for (int samp = 0 ; samp<10 ; samp++) {
	gain = df[samp].gainId() ;
	adc = df[samp].adc() ;
	if (gain == 1) tE.data_[samp] = adc-mean ;
	if (gain == 2) tE.data_[samp] = 2.*adc-mean ;
	if (gain == 3) tE.data_[samp] = 12.*adc-mean ;
	if (gain == 0) tE.data_[samp] = 12.*adc-mean ;
      }
      tE.iphi_ = towid.iphi() ;
      tE.ieta_ = towid.ieta() ;
      tE.iSM_ = towid.iDCC() ;
      mapTower[towid] = tE ;
    }
  }


  if (useEE_) {
    // Get EE xtal digi inputs
    edm::Handle<EEDigiCollection> digiEE;
    iEvent.getByLabel(digi_label_, digi_producerEE_, digiEE);
    
    for (unsigned int i=0;i<digiEE.product()->size();i++) {
      const EEDataFrame & df = (*(digiEE.product()))[i];    
      int gain, adc ;
      float E_xtal = 0. ; 
      int theSamp = 0 ;
      float mean = 0., max = -999 ; 
      for (int samp = 0 ; samp<10 ; samp++) {
	adc = df[samp].adc() ;
	if (samp<2) mean += adc ;
	if (adc>max) {
	  max = adc ;
	  theSamp = samp ;
	}
      }
      mean /= 2 ;
      if (mean>0 && max >= mean + adcCut_) {
	gain = df[theSamp].gainId() ;
	adc = df[theSamp].adc() ;
	if (gain == 1) E_xtal = (adc-mean) ;
	if (gain == 2) E_xtal = 2.*(adc-mean) ;
	if (gain == 3) E_xtal = 12.*(adc-mean) ;
	if (gain == 0) E_xtal = 12.*(adc-mean) ;
      }
      const EEDetId & id=df.id();
      const EcalTrigTowerDetId towid = (*eTTmap_).towerOf(id);
      itTT = mapTower.find(towid) ;
      if (itTT != mapTower.end()) {
	(itTT->second).eRec_ += E_xtal ;
	for (int samp = 0 ; samp<10 ; samp++) (itTT->second).data_[samp] += df[samp].adc()-mean ;
	(itTT->second).iphi_ = towid.iphi() ;
	(itTT->second).ieta_ = towid.ieta() ;
      }
      else {
	towerEner tE ;
	tE.eRec_ = E_xtal ;
	for (int samp = 0 ; samp<10 ; samp++) tE.data_[samp] = df[samp].adc()-mean ;
	tE.iphi_ = towid.iphi() ;
	tE.ieta_ = towid.ieta() ;
	mapTower[towid] = tE ;
      }
    }
  }
  
 
  // Get TP data  
  if (allowTP_) {
    edm::Handle<EcalTrigPrimDigiCollection> tp;
    iEvent.getByLabel(label_,producer_,tp);
  
    for (unsigned int i=0;i<tp.product()->size();i++) {
      EcalTriggerPrimitiveDigi d = (*(tp.product()))[i];
      const EcalTrigTowerDetId TPtowid= d.id();
      
      itTT = mapTower.find(TPtowid) ;
      if (itTT != mapTower.end()) {
	(itTT->second).tpgADC_ = d.compressedEt() ;
	(itTT->second).ttf_ = d.ttFlag() ;
	(itTT->second).fg_ = d.fineGrain() ;      
      }
      else {
	towerEner tE ;
	tE.iphi_ = TPtowid.iphi() ;
	tE.ieta_ = TPtowid.ieta() ;
	tE.tpgADC_ = d.compressedEt() ;
	tE.ttf_ = d.ttFlag() ;
	tE.fg_ = d.fineGrain() ;    
	mapTower[TPtowid] = tE ;
      }

      //if (d.compressedEt()>0) std::cout<<"Data (phi,eta, Et, i) ="<<TPtowid.iphi()<<" "<<TPtowid.ieta()<<" "<<d.compressedEt()<<std::endl ;
    }
  }


  // Get Emulators TP
  edm::Handle<EcalTrigPrimDigiCollection> tpEmul ;
  iEvent.getByLabel(emul_label_, emul_producer_, tpEmul);
  for (unsigned int i=0;i<tpEmul.product()->size();i++) {
    EcalTriggerPrimitiveDigi d = (*(tpEmul.product()))[i];
    const EcalTrigTowerDetId TPtowid= d.id();
    itTT = mapTower.find(TPtowid) ;
    if (itTT != mapTower.end())
      for (int j=0 ; j<5 ; j++) (itTT->second).tpgEmul_[j] = (d[j].raw() & 0x1ff) ;
    else {
      towerEner tE ;
      tE.iphi_ = TPtowid.iphi() ;
      tE.ieta_ = TPtowid.ieta() ;
      for (int j=0 ; j<5 ; j++) tE.tpgEmul_[j] = (d[j].raw() & 0x1ff) ;
      mapTower[TPtowid] = tE ;
    }

//     for (int j=0 ; j<5 ; j++) {
//       if ((d[j].raw() & 0xff)>0) std::cout<<"Emulateur (phi,eta, Et, i) ="<<TPtowid.iphi()<<" "<<TPtowid.ieta()<<" "<<(d[j].raw() & 0xff)<<" "<<j<<std::endl ;
//     }
  }


  
  // fill histograms
  fillShape(dfMax) ;
  for (itTT = mapTower.begin() ; itTT != mapTower.end() ; ++itTT ) {
    fillShape(itTT->second) ;
    fillOccupancyPlots(itTT->second) ;
    fillEnergyPlots(itTT->second) ;
    fillTPMatchPlots(itTT->second) ;
  }
  
}

void EcalTPGAnalyzer::fillShape(EBDataFrame & df)
{
  float max = -999 ;
  int gain, adc ;
  float data = -999;
  if (df[0].gainId() != 1 || df[1].gainId() != 1) return ; // first 2 samples must be in gain x12
  float mean = 0.5*(df[0].adc()+df[1].adc()) ;
  for (int i = 0 ; i<10 ; i++) {
    gain = df[i].gainId() ;
    adc = df[i].adc() ;
    if (gain == 1) data = adc-mean ;
    if (gain == 2) data = 2.*adc-mean ;
    if (gain == 3) data = 12.*adc-mean ;
    if (gain == 0) data = 12.*adc-mean ;
    if (data>max) max = data ;
  }
  if (max>0. && max>=shapeCut_)
    for (int i=0 ; i<10 ; i++) {
      gain = df[i].gainId() ;
      adc = df[i].adc() ;
      if (gain == 1) data = adc-mean ;
      if (gain == 2) data = 2.*adc-mean ;
      if (gain == 3) data = 12.*adc-mean ;
      if (gain == 0) data = 12.*adc-mean ;
      shapeMax_->Fill(i, data/max) ;
    }
}
  
void EcalTPGAnalyzer::fillShape(towerEner & t)
{
  float max = 0. ;
  for (int i=0 ; i<10 ; i++) if (t.data_[i]>max) max = t.data_[i] ;
  if (max>0 && max>=shapeCut_) {
    for (int i=0 ; i<10 ; i++) shape_[t.iSM_-1]->Fill(i, t.data_[i]/max, max) ;
  }
}

void  EcalTPGAnalyzer::fillOccupancyPlots(towerEner & t)
{
  int max = 0 ;
  for (int i=0 ; i<5 ; i++) if ((t.tpgEmul_[i]&0xff)>max) max = (t.tpgEmul_[i]&0xff) ; 
  if (max >= occupancyCut_) occupancyTPEmulMax_->Fill(t.iphi_, t.ieta_) ;
  if ((t.tpgEmul_[tpgRef_]&0xff) >= occupancyCut_) occupancyTPEmul_->Fill(t.iphi_, t.ieta_) ;
  if (t.tpgADC_ >= occupancyCut_) occupancyTP_->Fill(t.iphi_, t.ieta_) ;
}

void  EcalTPGAnalyzer::fillEnergyPlots(towerEner & t)
{
  int max = 0 ;
  for (int i=0 ; i<5 ; i++) if ((t.tpgEmul_[i]&0xff)>max) max = (t.tpgEmul_[i]&0xff) ; 
  crystalVsTP_->Fill(t.tpgADC_, t.eRec_) ;
  crystalVsEmulTP_->Fill((t.tpgEmul_[tpgRef_]&0xff), t.eRec_) ;
  crystalVsEmulTPMax_->Fill(max, t.eRec_) ;
  TPVsEmulTP_->Fill((t.tpgEmul_[tpgRef_]&0xff), t.tpgADC_) ;
  TP_->Fill(t.tpgADC_) ;
  TPEmul_->Fill((t.tpgEmul_[tpgRef_]&0xff)) ;
  TPEmulMax_->Fill(max) ;
}

void EcalTPGAnalyzer::fillTPMatchPlots(towerEner & t)
{
  bool match(false) ;
  if (t.tpgADC_>0) {
    for (int i=0 ; i<5 ; i++)
      if ((t.tpgEmul_[i]&0xff) == t.tpgADC_) {
	TPMatchEmul_->Fill(i) ;
	match = true ;
      }
    if (!match) TPMatchEmul_->Fill(-1) ;
  }

  int max = 0 ;
  int index = -1 ;
  for (int i=0 ; i<5 ; i++) 
    if ((t.tpgEmul_[i]&0xff)>max) {
      max = (t.tpgEmul_[i]&0xff) ; 
      index = i ;
    }
  if (max>0) TPEmulMaxIndex_->Fill(index) ;

}
