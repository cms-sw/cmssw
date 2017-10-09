
/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
 */


#include "DQM/DTMonitorClient/src/DTDCSSummary.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


using namespace std;
using namespace edm;



DTDCSSummary::DTDCSSummary(const ParameterSet& pset) {

  bookingdone = 0;

}

DTDCSSummary::~DTDCSSummary() {}

void DTDCSSummary::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,
                         edm::LuminosityBlock const & lumiSeg, edm::EventSetup const & context) {

  if (bookingdone) return;

  ibooker.setCurrentFolder("DT/EventInfo/DCSContents");
  // global fraction
  totalDCSFraction = ibooker.bookFloat("DTDCSSummary");  
  totalDCSFraction->Fill(-1);
  // Wheel "fractions" -> will be 0 or 1
  for(int wheel = -2; wheel != 3; ++wheel) {
    stringstream streams;
    streams << "DT_Wheel" << wheel;
    dcsFractions[wheel] = ibooker.bookFloat(streams.str());
    dcsFractions[wheel]->Fill(-1);
  }
  
  bookingdone = 1; 
}

void DTDCSSummary::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {}
