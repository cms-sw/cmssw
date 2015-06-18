// -*- C++ -*-
//
// Package:    HLTHcalCalibTypeFilter
// Class:      HLTHcalCalibTypeFilter
// 
/**\class HLTHcalCalibTypeFilter HLTHcalCalibTypeFilter.cc filter/HLTHcalCalibTypeFilter/src/HLTHcalCalibTypeFilter.cc

Description: Filter to select HCAL abort gap events

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Bryan DAHMES
//         Created:  Tue Jan 22 13:55:00 CET 2008
//
//


// system include files
#include <string>
#include <iostream>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include "HLTrigger/special/interface/HLTHcalCalibTypeFilter.h"

//
// constructors and destructor
//
HLTHcalCalibTypeFilter::HLTHcalCalibTypeFilter(const edm::ParameterSet& config) :
  DataInputToken_( consumes<FEDRawDataCollection>( config.getParameter<edm::InputTag>("InputTag") ) ),
  CalibTypes_( config.getParameter< std::vector<int> >("CalibTypes") ),
  Summary_(  config.getUntrackedParameter<bool>("FilterSummary", false) ),
  eventsByType_()
{
  for (auto & i : eventsByType_) i = 0;
}


HLTHcalCalibTypeFilter::~HLTHcalCalibTypeFilter()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}

void
HLTHcalCalibTypeFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputTag",edm::InputTag("source"));
  std::vector<int> temp; for (int i=1; i<=5; i++) temp.push_back(i);
  desc.add<std::vector<int> >("CalibTypes", temp);
  desc.addUntracked<bool>("FilterSummary",false);
  descriptions.add("hltHcalCalibTypeFilter",desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTHcalCalibTypeFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  using namespace edm;
  
  edm::Handle<FEDRawDataCollection> rawdata;  
  iEvent.getByToken(DataInputToken_,rawdata);
  
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
  int numberOfFEDIds = FEDNumbering::MAXHCALFEDID - FEDNumbering::MINHCALFEDID + 1 ; 
  for (unsigned int i=0; i<calibTypeCounter.size(); i++) {
      if ( calibTypeCounter.at(i) > maxCount ) { calibType = i ; maxCount = calibTypeCounter.at(i) ; } 
      if ( maxCount == numberOfFEDIds ) break ;
  }
  if ( calibType < 0 ) return false ; // No HCAL FEDs, thus no calibration type
  if ( maxCount != (numberOfFEDIds-numEmptyFEDs) )
      edm::LogWarning("HLTHcalCalibTypeFilter") << "Conflicting calibration types found.  Assigning type " 
                                             << calibType ; 
  LogDebug("HLTHcalCalibTypeFilter") << "Calibration type is: " << calibType ;
  if (Summary_)
    ++eventsByType_.at(calibType);

  for (unsigned int i=0; i<CalibTypes_.size(); i++) 
      if ( calibType == CalibTypes_.at(i) ) return true ;
  return false ; 
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTHcalCalibTypeFilter::endJob() {
  if ( Summary_ )
    edm::LogWarning("HLTHcalCalibTypeFilter") << "Summary of filter decisions: " 
                                              << eventsByType_.at(hc_Null)      << "(No Calib), " 
                                              << eventsByType_.at(hc_Pedestal)  << "(Pedestal), " 
                                              << eventsByType_.at(hc_RADDAM)    << "(RADDAM), " 
                                              << eventsByType_.at(hc_HBHEHPD)   << "(HBHE/HPD), " 
                                              << eventsByType_.at(hc_HOHPD)     << "(HO/HPD), " 
                                              << eventsByType_.at(hc_HFPMT)     << "(HF/PMT)" ;  
}
