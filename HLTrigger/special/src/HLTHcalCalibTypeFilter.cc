/** \class HLTHcalCalibTypeFilter
 *
 * See header file for documentation
 *
 *  $Date: 2008/01/09 14:16:15 $
 *  $Revision: 1.3 $
 *
 *  \author Bryan DAHMES
 *
 */

// include files
#include "HLTrigger/special/interface/HLTHcalCalibTypeFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <iostream>
#include <memory>

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

//
// constructors and destructor
//
HLTHcalCalibTypeFilter::HLTHcalCalibTypeFilter(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed

  DataLabel_  = iConfig.getParameter<std::string>("InputLabel") ;
  Summary_    = iConfig.getUntrackedParameter<bool>("FilterSummary",false) ;
  CalibTypes_ = iConfig.getParameter< std::vector<int> >("CalibTypes") ; 
}


HLTHcalCalibTypeFilter::~HLTHcalCalibTypeFilter()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTHcalCalibTypeFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  edm::Handle<FEDRawDataCollection> rawdata;  
  iEvent.getByLabel(DataLabel_,rawdata);
  
  // checking FEDs for calibration information
  int calibType = -1 ;
  std::vector<int> calibTypeCounter(8,0) ; 
  for (int i=FEDNumbering::getHcalFEDIds().first; 
       i<=FEDNumbering::getHcalFEDIds().second; i++) {
      const FEDRawData& fedData = rawdata->FEDData(i) ; 
      if ( fedData.size() < 24 ) continue ; 
      int value = ((const HcalDCCHeader*)(fedData.data()))->getCalibType() ; 
      calibTypeCounter.at(value)++ ; // increment the counter for this calib type
  }
  int maxCount = 0 ;
  int numberOfFEDIds = FEDNumbering::getHcalFEDIds().second - FEDNumbering::getHcalFEDIds().first + 1 ; 
  for (unsigned int i=0; i<calibTypeCounter.size(); i++) {
      if ( calibTypeCounter.at(i) > maxCount ) { calibType = i ; maxCount = calibTypeCounter.at(i) ; } 
      if ( maxCount == numberOfFEDIds ) break ;
  }
  if ( maxCount != numberOfFEDIds ) 
      edm::LogWarning("HLTHcalCalibTypeFilter") << "Conflicting calibration types found.  Assigning type " 
                                             << calibType ; 
  LogDebug("HLTHcalCalibTypeFilter") << "Calibration type is: " << calibType ; 
  eventsByType.at(calibType)++ ;
  for (unsigned int i=0; i<CalibTypes_.size(); i++) 
      if ( calibType == CalibTypes_.at(i) ) return true ;
  return false ; 
}

// ------------ method called once each job just before starting event loop  ------------
void 
HLTHcalCalibTypeFilter::beginJob(const edm::EventSetup&)
{
  eventsByType.clear() ; 
  eventsByType.resize(8,0) ; 
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTHcalCalibTypeFilter::endJob() {
  if ( Summary_ )
    edm::LogWarning("HLTHcalCalibTypeFilter") << "Summary of filter decisions: " 
					   << eventsByType.at(hc_Null) << "(No Calib), " 
					   << eventsByType.at(hc_Pedestal) << "(Pedestal), " 
					   << eventsByType.at(hc_RADDAM) << "(RADDAM), " 
					   << eventsByType.at(hc_HBHEHPD) << "(HBHE/HPD), " 
					   << eventsByType.at(hc_HOHPD) << "(HO/HPD), " 
					   << eventsByType.at(hc_HFPMT) << "(HF/PMT)" ;  
}
