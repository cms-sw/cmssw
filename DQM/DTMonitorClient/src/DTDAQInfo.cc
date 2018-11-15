/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
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



DTDAQInfo::DTDAQInfo(const ParameterSet& pset) {

  bookingdone = false;
  checkUros = pset.getUntrackedParameter<bool>("checkUros",true);
}

DTDAQInfo::~DTDAQInfo() {}

  void DTDAQInfo::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, edm::LuminosityBlock const & lumiSeg, 
                                   edm::EventSetup const & setup) {

  if (!bookingdone) {
  // retrieve the mapping
  setup.get<DTReadOutMappingRcd>().get(mapping);

  // book the ME
  // global fraction

  ibooker.setCurrentFolder("DT/EventInfo");

  totalDAQFraction = ibooker.bookFloat("DAQSummary");  
  totalDAQFraction->Fill(-1);

  // map
  daqMap = ibooker.book2D("DAQSummaryMap","DT Certification Summary Map",12,1,13,5,-2,3);
  daqMap->setAxisTitle("sector",1);
  daqMap->setAxisTitle("wheel",2);

  // Wheel "fractions" -> will be 0 or 1

  ibooker.setCurrentFolder("DT/EventInfo/DAQContents");
  for(int wheel = -2; wheel != 3; ++wheel) {
    stringstream streams;
    streams << "DT_Wheel" << wheel;

    daqFractions[wheel] = ibooker.bookFloat(streams.str());
    daqFractions[wheel]->Fill(-1);
  }
  bookingdone = true; 
  }  //booking done

  if(auto runInfoRec = setup.tryToGet<RunInfoRcd>()) {
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
    runInfoRec->get(sumFED);
    vector<int> fedInIDs = sumFED->m_fed_in;   

    // the range of DT feds
    static const int FEDIDmin = FEDNumbering::MINDTFEDID;
    static const int FEDIDMax = FEDNumbering::MAXDTFEDID; 
    
    //FIXME for uROS FEDIDs once mapping has been defined
    if (checkUros) { LogTrace("DQM|DTMonitorClient|DTDAQInfo")
            << "Checking uROS FEDs as Legacy FEDs"<< endl;}

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

void DTDAQInfo::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {}
