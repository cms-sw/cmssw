
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/12/12 18:04:17 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */


#include "DQM/DTMonitorClient/src/DTDAQInfo.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"





using namespace std;
using namespace edm;



DTDAQInfo::DTDAQInfo(const ParameterSet& pset) {}




DTDAQInfo::~DTDAQInfo() {}



void DTDAQInfo::beginJob(const EventSetup& setup){
  // get the DQMStore
  theDbe = Service<DQMStore>().operator->();
  
  // book the ME
  theDbe->setCurrentFolder("DT/EventInfo/DAQContents");
  // global fraction
  totalDAQFraction = theDbe->bookFloat("DTDaqFraction");  
  totalDAQFraction->Fill(-1);
  // Wheel "fractions" -> will be 0 or 1
  for(int wheel = -2; wheel != 3; ++wheel) {
    stringstream streams;
    streams << "DT_Wheel" << wheel;
    daqFractions[wheel] = theDbe->bookFloat(streams.str());
    daqFractions[wheel]->Fill(-1);
  }

  //

}



void DTDAQInfo::beginLuminosityBlock(const LuminosityBlock& lumi, const  EventSetup& setup) {
  
  // create a record key for RunInfoRcd
  eventsetup::EventSetupRecordKey recordKey(eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));

  if(setup.find(recordKey) != 0) { 
    //get fed summary information
    ESHandle<RunInfo> sumFED;
    setup.get<RunInfoRcd>().get(sumFED);    
    vector<int> fedInIDs = sumFED->m_fed_in;   

    int fedCount=0;

    // the range of DT feds
    static int FEDIDmin = FEDNumbering::MINDTFEDID;
    static int FEDIDMax = FEDNumbering::MAXDTFEDID;
    static int nFeds = FEDIDMax-FEDIDmin-1;

    // loop on all active feds
    for(vector<int>::const_iterator fed = fedInIDs.begin();
	fed != fedInIDs.end();
	++fed) {
      // check if the fed is in the DT range
      int wheelNumber = *fed - 772;; 
      if(wheelNumber >= -2 && wheelNumber < 3) { // this is a DT FED
	daqFractions[wheelNumber]->Fill(1);
	fedCount++;
      }
    }   

    //Fill total fraction of active feds
    if(nFeds > 0) totalDAQFraction->Fill(fedCount/nFeds);
    else {
      totalDAQFraction->Fill(-1);
      for(int wheel = -2; wheel != 3; ++wheel) {
	daqFractions[wheel]->Fill(-1);
      }
    }
  } else {      
    LogWarning("DQM|DTMonitorClient|DTDAQInfo") << "*** Warning: record key not found for RunInfoRcd" << endl;
    totalDAQFraction->Fill(-1);               
    for(int wheel = -2; wheel != 3; ++wheel) {
      daqFractions[wheel]->Fill(-1);
    }
    return; 
  }
}




void DTDAQInfo::endLuminosityBlock(const LuminosityBlock&  lumi, const  EventSetup& setup){}



void DTDAQInfo::endJob() {}



void DTDAQInfo::analyze(const Event& event, const EventSetup& setup){}



