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
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTBCalo/HcalPlotter/src/HcalQLPlotAnalAlgos.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
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
    edm::LogError("HcalQLPlotAnalAlgos::begin") << "Trigger Type unrecognized, aborting";
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
    HBHEDigiCollection::const_iterator eit = hbhedgc.find(id);
    if (eit != hbhedgc.end())
      eid = eit->elecId();
    else {
      edm::LogWarning("HcalQLPlotAnalAlgos::processRH") <<
	"No electronics ID found for id" << id;
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
    HODigiCollection::const_iterator eit = hodgc.find(id);
    if (eit != hodgc.end())
      eid = eit->elecId();
    else {
      edm::LogWarning("HcalQLPlotAnalAlgos::processRH") <<
	"No electronics ID found for id" << id;
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
    HFDigiCollection::const_iterator eit = hfdgc.find(id);
    if (eit != hfdgc.end())
      eid = eit->elecId();
    else {
      edm::LogWarning("HcalQLPlotAnalAlgos::processRH") <<
	"No electronics ID found for id" << id;
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
  }
}
