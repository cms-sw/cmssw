
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/05 10:15:45 $
 *  $Revision: 1.6 $
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


#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"



using namespace std;
using namespace edm;



DTDAQInfo::DTDAQInfo(const ParameterSet& pset) {}




DTDAQInfo::~DTDAQInfo() {}



void DTDAQInfo::beginJob(){
  // get the DQMStore
  theDbe = Service<DQMStore>().operator->();
  
  // book the ME
  // global fraction
  theDbe->setCurrentFolder("DT/EventInfo");
  totalDAQFraction = theDbe->bookFloat("DAQSummary");  
  totalDAQFraction->Fill(-1);

  // map
  daqMap = theDbe->book2D("DAQSummaryMap","DT Certification Summary Map",12,1,13,5,-2,3);
  daqMap->setAxisTitle("sector",1);
  daqMap->setAxisTitle("wheel",2);


  // Wheel "fractions" -> will be 0 or 1
  theDbe->setCurrentFolder("DT/EventInfo/DAQContents");
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
    // reset to 0
    totalDAQFraction->Fill(0.);
    daqFractions[-2]->Fill(0.);
    daqFractions[-1]->Fill(0.);
    daqFractions[-0]->Fill(0.);
    daqFractions[1]->Fill(0.);
    daqFractions[2]->Fill(0.);

    daqMap->Reset();

    //get fed summary information
    ESHandle<RunInfo> sumFED;
    setup.get<RunInfoRcd>().get(sumFED);    
    vector<int> fedInIDs = sumFED->m_fed_in;   


    // the range of DT feds
    static int FEDIDmin = FEDNumbering::MINDTFEDID;
    static int FEDIDMax = FEDNumbering::MAXDTFEDID;

    // loop on all active feds
    for(vector<int>::const_iterator fed = fedInIDs.begin();
	fed != fedInIDs.end();
	++fed) {
      // check if the fed is in the DT range
      if(!(*fed >= FEDIDmin && *fed <= FEDIDMax)) continue;

      // check if the 12 channels are connected to any sector and fill the wheel percentage accordignly
      int wheel = -99;
      int sector = -99;
      int dummy = -99;
      for(int ros = 1; ros != 13; ++ros) {
	if(!mapping->readOutToGeometry(*fed,ros,2,2,2,wheel,dummy,sector,dummy,dummy,dummy)) {
	  LogTrace("DQM|DTMonitorClient|DTDAQInfo")
	    << "FED: " << *fed << " Ch: " << ros << " wheel: " << wheel << " Sect: " << sector << endl;
	  daqFractions[wheel]->Fill(daqFractions[wheel]->getFloatValue() + 1./12.);
	  totalDAQFraction->Fill(totalDAQFraction->getFloatValue() + 1./60.);
	  daqMap->Fill(sector,wheel);
	}
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



void DTDAQInfo::beginRun(const Run& run, const EventSetup& setup) {
  // retrieve the mapping
  setup.get<DTReadOutMappingRcd>().get(mapping);
}
