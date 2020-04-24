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
#include "EventFilter/HcalRawToDigi/interface/HcalUHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/AMC13Header.h"
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

  //    some inits
  int numZeroes(0), numPositives(0);

  //    loop over all HCAL FEDs
  for (int fed=FEDNumbering::MINHCALFEDID;
       fed<=FEDNumbering::MAXHCALuTCAFEDID; fed++) 
  {
      //    skip FEDs in between VME and uTCA    
      if (fed>FEDNumbering::MAXHCALFEDID && fed<FEDNumbering::MINHCALuTCAFEDID)
            continue;

      //    get raw data and check if there are empty feds
      const FEDRawData& fedData = rawdata->FEDData(fed) ; 
      if ( fedData.size() < 24 ) continue ;

      if (fed<=FEDNumbering::MAXHCALFEDID)
      {
          //    VME get event type
          int eventtype = ((const HcalDCCHeader*)(fedData.data()))->getCalibType(); 
          if (eventtype==0) numZeroes++; else numPositives++;
      }
      else 
      {
          //    UTCA
          hcal::AMC13Header const *hamc13 = (hcal::AMC13Header const*) fedData.data();
          for (int iamc=0; iamc<hamc13->NAMC(); iamc++)
          {
              HcalUHTRData uhtr(hamc13->AMCPayload(iamc), hamc13->AMCSize(iamc));
              int eventtype = uhtr.getEventType();
              if (eventtype==0) numZeroes++; else numPositives++;
          }
      }
  }

  //
  //    if there are FEDs with Non-Collission event type, check what the majority is
  //    if calibs - true
  //    if 0s - false
  //
  if (numPositives>0)
  {
    if (numPositives>numZeroes) return true;
    else
        edm::LogWarning("HLTHcalCalibTypeFilter") 
            << "Conflicting Calibration Types found";
  }

  //    return false if there are no positives
  //    and if the majority has 0 calib type
  return false;
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
