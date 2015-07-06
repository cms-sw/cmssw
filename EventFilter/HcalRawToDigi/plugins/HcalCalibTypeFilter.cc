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
  virtual void beginJob() override ;
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;
  
  // ----------member data ---------------------------
 
  edm::EDGetTokenT<FEDRawDataCollection> tok_data_; 
  bool        Summary_ ;
  std::vector<int> CalibTypes_ ;   
  std::vector<int> eventsByType ; 

};


//
// constructors and destructor
//
HcalCalibTypeFilter::HcalCalibTypeFilter(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed

  tok_data_  = consumes<FEDRawDataCollection>(edm::InputTag(iConfig.getParameter<std::string>("InputLabel") ));
  Summary_    = iConfig.getUntrackedParameter<bool>("FilterSummary",false) ;
  CalibTypes_ = iConfig.getParameter< std::vector<int> >("CalibTypes") ; 
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
  
  edm::Handle<FEDRawDataCollection> rawdata;  
  iEvent.getByToken(tok_data_,rawdata);
  
  if(!rawdata.isValid()){
     return false;
  }

  // checking FEDs for calibration information
  int calibType = -1 ; int numEmptyFEDs = 0 ; 
  std::vector<int> calibTypeCounter(8,0) ; 
  for (int i=FEDNumbering::MINHCALFEDID;
       i<=FEDNumbering::MAXHCALFEDID; i++) {
      const FEDRawData& fedData = rawdata->FEDData(i) ; 
      if ( fedData.size() < 24 ) numEmptyFEDs++ ; 
      if ( fedData.size() < 24 ) continue ; 
      int value = ((const HcalDCCHeader*)(fedData.data()))->getCalibType() ; 
      calibTypeCounter.at(value)++ ; // increment the counter for this calib type
  }
  
  int maxCount = 0 ;
  int numberOfFEDIds = FEDNumbering::MAXHCALFEDID  - FEDNumbering::MINHCALFEDID + 1 ; 
  for (unsigned int i=0; i<calibTypeCounter.size(); i++) {
      if ( calibTypeCounter.at(i) > maxCount ) { calibType = i ; maxCount = calibTypeCounter.at(i) ; } 
      if ( maxCount == numberOfFEDIds ) break ;
  }
  if ( maxCount != (numberOfFEDIds-numEmptyFEDs) )
      edm::LogWarning("HcalCalibTypeFilter") << "Conflicting calibration types found.  Assigning type " 
                                             << calibType ; 
  LogDebug("HcalCalibTypeFilter") << "Calibration type is: " << calibType ; 
  eventsByType.at(calibType)++ ;
  for (unsigned int i=0; i<CalibTypes_.size(); i++) 
      if ( calibType == CalibTypes_.at(i) ) return true ;
  return false ; 
}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalCalibTypeFilter::beginJob()
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
