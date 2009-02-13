// -*- C++ -*-
//
// Package:    HcalCalibTypeFilter
// Class:      HcalCalibTypeFilter
// 
/**\class HcalCalibTypeFilter HcalCalibTypeFilter.cc filter/HcalCalibTypeFilter/src/HcalCalibTypeFilter.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Giovanni FRANZONI
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HcalCalibTypeFilter.cc,v 1.7 2008/09/02 08:25:39 gruen Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <iostream>

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

//
// class declaration
//

class HcalCalibTypeFilter : public edm::EDFilter {
public:
  explicit HcalCalibTypeFilter(const edm::ParameterSet&);
  virtual ~HcalCalibTypeFilter();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
  
  std::string DataLabel_ ;
  bool        Summary_ ;
  std::vector<int> eventsByType ; 

};


//
// constructors and destructor
//
HcalCalibTypeFilter::HcalCalibTypeFilter(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed

  DataLabel_  = iConfig.getParameter<std::string>("InputLabel") ;
  Summary_    = iConfig.getParameter<bool>("FilterSummary") ;   
}


HcalCalibTypeFilter::~HcalCalibTypeFilter()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HcalCalibTypeFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  if (iEvent.isRealData()) {

    edm::Handle<FEDRawDataCollection> rawdata;  
    iEvent.getByLabel(DataLabel_,rawdata);
  
    // checking FEDs for calibration information
    int calibType = -1 ; 
    for (int i=FEDNumbering::getHcalFEDIds().first; 
	 i<=FEDNumbering::getHcalFEDIds().second; i++) {
      const FEDRawData& fedData = rawdata->FEDData(i) ; 
      if ( fedData.size() < 24 ) continue ; 
      int value = ((const HcalDCCHeader*)(fedData.data()))->getCalibType() ; 
      if ( calibType < 0 ) {
	calibType = value ; 
      } else { 
	if ( calibType != value ) 
	  edm::LogWarning("HcalCalibTypeFilter") << "Conflicting calibration types found: " 
						 << calibType << " vs. " << value
						 << ".  Staying with " << calibType ; 
      }
    }

    LogDebug("HcalCalibTypeFilter") << "Calibration type is: " << calibType ; 

    eventsByType.at(calibType)++ ; 
    return ( calibType != 0 ) ; 

} else {
  return true;
}

}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalCalibTypeFilter::beginJob(const edm::EventSetup&)
{
  eventsByType.clear() ; 
  eventsByType.resize(8,0) ; 
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalCalibTypeFilter::endJob() {
  if ( Summary_ )
    edm::LogWarning("HcalCalibTypeFilter") << "Summary of filter decisions: " 
					   << eventsByType.at(hc_Null) << "(No Calib), " 
					   << eventsByType.at(hc_Pedestal) << "(Pedestal), " 
					   << eventsByType.at(hc_RADDAM) << "(RADDAM), " 
					   << eventsByType.at(hc_HBHEHPD) << "(HBHE/HPD), " 
					   << eventsByType.at(hc_HOHPD) << "(HO/HPD), " 
					   << eventsByType.at(hc_HFPMT) << "(HF/PMT)" ;  
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalCalibTypeFilter);
