
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/05 10:15:46 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
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



DTDCSSummary::DTDCSSummary(const ParameterSet& pset) {}




DTDCSSummary::~DTDCSSummary() {}



void DTDCSSummary::beginJob(){
  // get the DQMStore
  theDbe = Service<DQMStore>().operator->();
  
  // book the ME
  theDbe->setCurrentFolder("DT/EventInfo/DCSContents");
  // global fraction
  totalDCSFraction = theDbe->bookFloat("DTDCSSummary");  
  totalDCSFraction->Fill(-1);
  // Wheel "fractions" -> will be 0 or 1
  for(int wheel = -2; wheel != 3; ++wheel) {
    stringstream streams;
    streams << "DT_Wheel" << wheel;
    dcsFractions[wheel] = theDbe->bookFloat(streams.str());
    dcsFractions[wheel]->Fill(-1);
  }

}



void DTDCSSummary::beginLuminosityBlock(const LuminosityBlock& lumi, const  EventSetup& setup) {
}




void DTDCSSummary::endLuminosityBlock(const LuminosityBlock&  lumi, const  EventSetup& setup){}



void DTDCSSummary::endJob() {}



void DTDCSSummary::analyze(const Event& event, const EventSetup& setup){}



