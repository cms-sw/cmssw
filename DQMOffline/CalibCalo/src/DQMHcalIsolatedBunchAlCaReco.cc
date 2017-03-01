/*
 * \file DQMHcalIsolatedBunchAlCaReco.cc
 *
 * \author Olga Kodolova
 * 
 *
 *
 * Description: Monitoring of Phi Symmetry Calibration Stream  
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// DQM include files

#include "DQMServices/Core/interface/MonitorElement.h"

// work on collections

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMOffline/CalibCalo/src/DQMHcalIsolatedBunchAlCaReco.h"

// ******************************************
// constructors
// *****************************************

DQMHcalIsolatedBunchAlCaReco::DQMHcalIsolatedBunchAlCaReco(const edm::ParameterSet& ps) {
  //
  // Input from configurator file 
  //
  folderName_   = ps.getUntrackedParameter<std::string>("FolderName","ALCAStreamHcalIsolatedBunch");
  trigName_     = ps.getParameter<std::string>("TriggerName");
  plotAll_      = ps.getUntrackedParameter<bool>("PlotAll",true);
  
  hbhereco_     = consumes<HBHERecHitCollection>(ps.getParameter<edm::InputTag>("hbheInput"));
  horeco_       = consumes<HORecHitCollection>(ps.getParameter<edm::InputTag>("hoInput"));
  hfreco_       = consumes<HFRecHitCollection>(ps.getParameter<edm::InputTag>("hfInput"));
  trigResult_   = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResult"));
    
}

DQMHcalIsolatedBunchAlCaReco::~DQMHcalIsolatedBunchAlCaReco() { }

//--------------------------------------------------------
void DQMHcalIsolatedBunchAlCaReco::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & irun, edm::EventSetup const & isetup) {
 
  // create and cd into new folder
  ibooker.setCurrentFolder(folderName_);
  h_Event_   = ibooker.book1D("hEvent","Selection summary",10,0,10);
  h_hbhehit_ = ibooker.book1D("hHBHEHit","Size of HBHE Collection",200,0,2000);
  h_hohit_   = ibooker.book1D("hHOHit",  "Size of HO Collection",  200,0,2000);
  h_hfhit_   = ibooker.book1D("hHFHit",  "Size of HF Collection",  200,0,2000);

}

//-------------------------------------------------------------

void DQMHcalIsolatedBunchAlCaReco::analyze(const edm::Event& iEvent, 
					   const edm::EventSetup& iSetup ){  
 
  bool accept(false);
  /////////////////////////////TriggerResults
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(trigResult_, triggerResults);
  if (triggerResults.isValid()) {
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);
    const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
    for (unsigned int iHLT=0; iHLT<triggerResults->size(); iHLT++) {
      int hlt    = triggerResults->accept(iHLT);
      if (hlt > 0) {
	if (triggerNames_[iHLT].find(trigName_.c_str())!=std::string::npos) {
	  accept = true;
	  break;
	}
      }
    }
  }
  h_Event_->Fill(0.);
  if (accept) h_Event_->Fill(1.);

  if (accept || plotAll_) {
    edm::Handle<HBHERecHitCollection> hbhe;
    iEvent.getByToken(hbhereco_, hbhe);
    if (!hbhe.isValid()) {
      edm::LogInfo("HcalCalib") << "Cannot get hbhe product!" << std::endl;
    } else {
      h_hbhehit_->Fill(hbhe->size());
    }
  
    edm::Handle<HFRecHitCollection> hf;
    iEvent.getByToken(hfreco_, hf);
    if (!hf.isValid()) {
      edm::LogInfo("HcalCalib") << "Cannot get hf product!" << std::endl;
    } else {
      h_hfhit_->Fill(hf->size());
    }
  
    edm::Handle<HORecHitCollection> ho;
    iEvent.getByToken(horeco_, ho);
    if (!ho.isValid()) {
      edm::LogInfo("HcalCalib") << "Cannot get ho product!" << std::endl;
    } else {
      h_hohit_->Fill(ho->size());
    }
  }
	
} //analyze
