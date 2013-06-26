// -*- C++ -*-
//
// Package:    HcalQLPlotAnal
// Class:      HcalQLPlotAnal
// 
/**\class HcalQLPlotAnal HcalQLPlotAnal.cc MyEDProducts/HcalPlotter/src/HcalQLPlotAnal.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Phillip R. Dudero
//         Created:  Tue Jan 16 21:11:37 CST 2007
// $Id: HcalQLPlotAnalAlgos.cc,v 1.4 2008/01/05 22:27:39 elmer Exp $
//
//


// system include files
#include <memory>
#include <math.h>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTBCalo/HcalPlotter/src/HcalQLPlotAnalAlgos.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalRecHit/interface/HcalCalibRecHit.h"
#include "TH1.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HcalQLPlotAnalAlgos::HcalQLPlotAnalAlgos(const char *outputFilename,
					 edm::ParameterSet histoParams)
{
  triggerID_=HcalQLPlotHistoMgr::UNKNOWN;

  mf_     = new TFile(outputFilename,"RECREATE");
  histos_ = new HcalQLPlotHistoMgr(mf_,histoParams);
}


//
// member functions
//

void HcalQLPlotAnalAlgos::end(void)
{
  mf_->Write();
}

void HcalQLPlotAnalAlgos::SetEventType(const HcalTBTriggerData& trigd)
{
  if( trigd.wasInSpillPedestalTrigger()  ||
      trigd.wasOutSpillPedestalTrigger() ||
      trigd.wasSpillIgnorantPedestalTrigger() )
                                triggerID_=HcalQLPlotHistoMgr::PEDESTAL;
  if( trigd.wasLEDTrigger() )   triggerID_=HcalQLPlotHistoMgr::LED;
  if( trigd.wasLaserTrigger() ) triggerID_=HcalQLPlotHistoMgr::LASER;
  if( trigd.wasBeamTrigger() )  triggerID_=HcalQLPlotHistoMgr::BEAM;

  if( triggerID_ == HcalQLPlotHistoMgr::UNKNOWN) {
    edm::LogError("HcalQLPlotAnalAlgos::begin") <<
      "Trigger Type unrecognized, aborting";
    std::exception e;
    throw e;
  }
}

void HcalQLPlotAnalAlgos::processRH(const HBHERecHitCollection& hbherhc,
				    const HBHEDigiCollection& hbhedgc)
{
  HBHERecHitCollection::const_iterator it;

  for (it  = hbherhc.begin(); 
       it != hbherhc.end();
       it++) {
    HcalDetId id (it->id());
    HcalElectronicsId eid;
    HBHEDigiCollection::const_iterator dit = hbhedgc.find(id);
    if (dit != hbhedgc.end())
      eid = dit->elecId();
    else {
      edm::LogWarning("HcalQLPlotAnalAlgos::processRH") <<
	"No digi found for id" << id;
      continue;
    }

    TH1* ehist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::ENERGY,triggerID_);
    if (ehist){
      ehist->Fill(it->energy());
    }

    TH1* thist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::TIME,triggerID_);
    if (thist){
      thist->Fill(it->time());
    }
  }
}

void HcalQLPlotAnalAlgos::processRH(const HORecHitCollection& horhc,
				    const HODigiCollection& hodgc)
{
  HORecHitCollection::const_iterator it;

  for (it  = horhc.begin(); 
       it != horhc.end();
       it++) {
    HcalDetId id (it->id());
    HcalElectronicsId eid;
    HODigiCollection::const_iterator dit = hodgc.find(id);
    if (dit != hodgc.end())
      eid = dit->elecId();
    else {
      edm::LogWarning("HcalQLPlotAnalAlgos::processRH") <<
	"No digi found for id" << id;
      continue;
    }

    TH1* ehist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::ENERGY,triggerID_);
    if (ehist){
      ehist->Fill(it->energy());
    }

    TH1* thist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::TIME,triggerID_);
    if (thist){
      thist->Fill(it->time());
    }
  }
}

void HcalQLPlotAnalAlgos::processRH(const HFRecHitCollection& hfrhc,
				    const HFDigiCollection& hfdgc)
{
  HFRecHitCollection::const_iterator it;

  for (it  = hfrhc.begin(); 
       it != hfrhc.end();
       it++) {
    HcalDetId id (it->id());
    HcalElectronicsId eid;
    HFDigiCollection::const_iterator dit = hfdgc.find(id);
    if (dit != hfdgc.end())
      eid = dit->elecId();
    else {
      edm::LogWarning("HcalQLPlotAnalAlgos::processRH") <<
	"No digi found for id" << id;
      continue;
    }

    TH1* ehist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::ENERGY,triggerID_);
    if (ehist){
      ehist->Fill(it->energy());
    }

    TH1* thist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::TIME,triggerID_);
    if (thist){
      thist->Fill(it->time());
    }
  }
}

void HcalQLPlotAnalAlgos::processDigi(const HBHEDigiCollection& hbhedigic)
{
  HBHEDigiCollection::const_iterator it;

  for (it  = hbhedigic.begin(); 
       it != hbhedigic.end();
       it++) {
    HcalDetId id (it->id());
    HcalElectronicsId eid (it->elecId());

    TH1* phist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::PULSE,triggerID_);
    if (phist){
      for (int bin=0; bin<it->size(); bin++)
	phist->Fill(bin*1.0,(*it)[bin].nominal_fC());
    }

    if (triggerID_ == HcalQLPlotHistoMgr::PEDESTAL) {
      phist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::ADC,triggerID_);
      if (phist){
	for (int bin=0; bin<it->size(); bin++)
	  phist->Fill((*it)[bin].adc());
      }
    }
  }
}

void HcalQLPlotAnalAlgos::processDigi(const HODigiCollection& hodigic)
{
  HODigiCollection::const_iterator it;

  for (it  = hodigic.begin(); 
       it != hodigic.end();
       it++) {
    HcalDetId id (it->id());
    HcalElectronicsId eid (it->elecId());

    TH1* phist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::PULSE,triggerID_);
    if (phist){
      for (int bin=0; bin<it->size(); bin++)
	phist->Fill(bin*1.0,(*it)[bin].nominal_fC());
    }

    if (triggerID_ == HcalQLPlotHistoMgr::PEDESTAL) {
      phist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::ADC,triggerID_);
      if (phist){
	for (int bin=0; bin<it->size(); bin++)
	  phist->Fill((*it)[bin].adc());
      }
    }
  }
}

void HcalQLPlotAnalAlgos::processDigi(const HFDigiCollection& hfdigic)
{
  HFDigiCollection::const_iterator it;

  for (it  = hfdigic.begin(); 
       it != hfdigic.end();
       it++) {
    HcalDetId id (it->id());
    HcalElectronicsId eid (it->elecId());

    TH1* phist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::PULSE,triggerID_);
    if (phist){
      for (int bin=0; bin<it->size(); bin++)
	phist->Fill(bin*1.0,(*it)[bin].nominal_fC());
    }

    if (triggerID_ == HcalQLPlotHistoMgr::PEDESTAL) {
      phist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::ADC,triggerID_);
      if (phist){
	for (int bin=0; bin<it->size(); bin++)
	  phist->Fill((*it)[bin].adc());
      }
    }
  }
}

HcalCalibRecHit HcalQLPlotAnalAlgos::recoCalib(const HcalCalibDataFrame& cdigi,
					       double calibFC2GeV)
{
  double nominal_ped = (cdigi[0].nominal_fC() + cdigi[1].nominal_fC())/2.0;

  double totamp = 0.0;
  double maxA = -1e99;
  int    maxI = -1;
  for (int i=0; i<cdigi.size(); i++) {
    double ampl = (cdigi[i].nominal_fC()-nominal_ped)*calibFC2GeV;
    totamp += ampl;

    if (ampl > maxA) {
      maxA = ampl;
      maxI = i;
    }
  }

  maxA=fabs(maxA);
  float t0 = (maxI > 0) ? (fabs((cdigi[maxI-1].nominal_fC()-nominal_ped))*calibFC2GeV):0.0;
  float t2 = fabs((cdigi[maxI+1].nominal_fC()-nominal_ped)*calibFC2GeV);    
  float wpksamp = (maxA + 2.0*t2) / (t0 + maxA + t2);
  float time = (maxI - cdigi.presamples() + wpksamp)*25.0;

  return HcalCalibRecHit(cdigi.id(),totamp,time);    
}

void HcalQLPlotAnalAlgos::processDigi(const HcalCalibDigiCollection& calibdigic,
				      double calibFC2GeV)
{
  HcalCalibDigiCollection::const_iterator it;

  for (it  = calibdigic.begin(); 
       it != calibdigic.end();
       it++) {
    HcalCalibDetId     id (it->id());
    HcalElectronicsId eid (it->elecId());

    TH1* phist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::PULSE,triggerID_);
    if (phist){
      for (int bin=0; bin<it->size(); bin++)
	phist->Fill(bin*1.0,(*it)[bin].nominal_fC());
    }

    // HACK-reco the calib digi into a rechit:
    //
    HcalCalibRecHit rh = recoCalib(*it, calibFC2GeV);

    TH1* ehist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::ENERGY,triggerID_);
    if (ehist){
      ehist->Fill(rh.amplitude());
    }

    TH1* thist=histos_->GetAHistogram(id,eid,HcalQLPlotHistoMgr::TIME,triggerID_);
    if (thist){
      thist->Fill(rh.time());
    }
  }
}

