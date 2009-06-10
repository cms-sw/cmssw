
/*
 * \file DTBlockedROChannelsTest.cc
 * 
 * $Date: 2009/06/10 10:53:26 $
 * $Revision: 1.3 $
 * \author G. Cerminara - University and INFN Torino
 *
 */

#include <DQM/DTMonitorClient/src/DTBlockedROChannelsTest.h>

//Framework
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <string>


using namespace std;
using namespace edm;


DTBlockedROChannelsTest::DTBlockedROChannelsTest(const ParameterSet& ps) : nevents(0) {
  LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
    << "[DTBlockedROChannelsTest]: Constructor";

  // prescale on the # of LS to update the test
  prescaleFactor = ps.getUntrackedParameter<int>("diagnosticPrescale", 1);
}



DTBlockedROChannelsTest::~DTBlockedROChannelsTest() {
  LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
    << "DataIntegrityTest: analyzed " << nupdates << " updates";
}



// book histos
void DTBlockedROChannelsTest::beginJob(const EventSetup& context) {
  LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
    << "[DTBlockedROChannelsTest]: BeginJob";

  //nSTAEvents = 0;
  nupdates = 0;
  run=0;

  dbe = Service<DQMStore>().operator->();
  
  // book the summary histogram
  dbe->setCurrentFolder("DT/00-ROChannels");
  summaryHisto = dbe->book2D("ROChannelSummary","Summary Blocked RO Channels",12,1,13,5,-2,3);
  summaryHisto->setAxisTitle("Sector",1);
  summaryHisto->setAxisTitle("Wheel",2);

  for(int wheel = -2; wheel != 3; ++wheel) {
    stringstream namestream;  namestream << "ROChannelSummary_W" << wheel;
    stringstream titlestream; titlestream << "Blocked RO Channels (Wh " << wheel << ")";
    wheelHitos[wheel] = dbe->book2D(namestream.str().c_str(),titlestream.str().c_str(),12,1,13,4,1,5);
    wheelHitos[wheel]->setAxisTitle("Sector",1);
    wheelHitos[wheel]->setBinLabel(1,"MB1",2);
    wheelHitos[wheel]->setBinLabel(2,"MB2",2);
    wheelHitos[wheel]->setBinLabel(3,"MB3",2);
    wheelHitos[wheel]->setBinLabel(4,"MB4",2);
  }


}




void DTBlockedROChannelsTest::beginRun(const Run& run, const EventSetup& context) {
  // get the RO mapping
  context.get<DTReadOutMappingRcd>().get(mapping);
  
  // fill the map of the robs per chamber
  for(int dduId = FEDNumbering::MINDTFEDID; dduId<=FEDNumbering::MAXDTFEDID; ++dduId) { //loop over DDUs
    for(int ros = 1; ros != 13; ++ros) { // loop over ROSs
      for(int rob = 1; rob != 26; ++rob) { // loop over ROBs	
	int wheel = 0;
	int station = 0;
	int sector = 0;
	int dummy = 0;
	if(!mapping->readOutToGeometry(dduId,ros,rob-1,2,2,wheel,station,sector,dummy,dummy,dummy)) {
	  DTChamberId chId(wheel, station, sector);
	  if(chamberMap.find(chId) == chamberMap.end()) {
	    chamberMap[chId] = DTRobBinsMap(dduId, ros, dbe);
	    chamberMap[chId].addRobBin(rob);
	  } 
	  chamberMap[chId].addRobBin(rob);
	} else {
	   LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
	     << "[DTRobBinsMap] FED: " << dduId << " ROS " << ros << " ROB: " << rob-1
	     << " not in the mapping!" << endl;
	}
      }
    }
  }
}



void DTBlockedROChannelsTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg,
						   EventSetup const& context) {
  LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
    <<"[DTBlockedROChannelsTest]: Begin of LS transition";

  // Get the run number
  run = lumiSeg.run();
  nevents = 0;
}



void DTBlockedROChannelsTest::analyze(const Event& e, const EventSetup& context){
  // count the analyzed events
  nevents++;
  if(nevents%1000 == 0)
    LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
      << "[DTBlockedROChannelsTest]: "<<nevents<<" events";
}



void DTBlockedROChannelsTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();
  
  // prescale factor
  if (nLumiSegs%prescaleFactor != 0) return;
  
  LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
    <<"[DTBlockedROChannelsTest]: End of LS " << nLumiSegs << ", performing client operations";


  // counts number of updats 
  nupdates++;

  // skip empty LSs
  if(nevents == 0) return;

  // reset the histos
  summaryHisto->Reset();
  for(int wheel = -2; wheel != 3; ++wheel) {
    wheelHitos[wheel]->Reset();
  }
  


  // loop over all chambers and fill the wheel plots
  for(map<DTChamberId, DTRobBinsMap>::iterator chAndRobs = chamberMap.begin();
      chAndRobs != chamberMap.end(); ++chAndRobs) {
    DTChamberId chId = (*chAndRobs).first;
    double scale = 1.;
    int sectorForPlot = chId.sector();
    if(sectorForPlot == 13 || (sectorForPlot == 4 && chId.station() ==4)) {
      sectorForPlot = 4;
      scale = 0.5;
    } else if(sectorForPlot == 14 || (sectorForPlot == 10 && chId.station() ==4)) {
      sectorForPlot = 10;
      scale = 0.5;
    }
    // NOTE: can be called only ONCE per event per each chamber
    double chPercent = (*chAndRobs).second.getChamberPercentage(); 
    wheelHitos[chId.wheel()]->Fill(sectorForPlot, chId.station(),
				   scale*chPercent);
    // Fill the summary
    summaryHisto->Fill(sectorForPlot, chId.wheel(), 0.25*scale*chPercent);
  }


}



void DTBlockedROChannelsTest::endJob(){
  LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
    <<"[DTBlockedROChannelsTest] endjob called!";
}



int DTBlockedROChannelsTest::readOutToGeometry(int dduId, int ros, int& wheel, int& sector){

  int dummy;
  return mapping->readOutToGeometry(dduId,ros,2,2,2,wheel,dummy,sector,dummy,dummy,dummy);

}





DTBlockedROChannelsTest::DTRobBinsMap::DTRobBinsMap(const int fed, const int ros, const DQMStore* dbe) {
  // get the pointer to the corresondig histo
  stringstream mename; mename << "DT/00-DataIntegrity/FED" << fed << "/ROS" << ros
			      << "/FED" << fed << "_ROS" << ros << "_ROSError";
  meROS = dbe->get(mename.str());
}




DTBlockedROChannelsTest::DTRobBinsMap::DTRobBinsMap() {
  meROS = 0;
}



DTBlockedROChannelsTest::DTRobBinsMap::~DTRobBinsMap() {}



// add a rob to the set of robs
void DTBlockedROChannelsTest::DTRobBinsMap::addRobBin(int robBin) {
  robsAndValues[robBin] = getValueRobBin(robBin);
}



    
int DTBlockedROChannelsTest::DTRobBinsMap::getValueRobBin(int robBin) const {
  int value = 0;
  if(meROS) {
    value += (int)meROS->getBinContent(9,robBin);
    value += (int)meROS->getBinContent(11,robBin);
  }
  return value;
}




bool DTBlockedROChannelsTest::DTRobBinsMap::robChanged(int robBin) {
  // check that this is a valid ROB for this map (= it has been added!)
  if(robsAndValues.find(robBin) == robsAndValues.end()) {
     LogWarning("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
       << "[DTRobBinsMap]***Error: ROB: " << robBin << " is not valid" << endl;
    return false;
  }

  int newValue = getValueRobBin(robBin);
  if(newValue > robsAndValues[robBin]) {
    robsAndValues[robBin] = newValue;
    return true;
  }
  return false;
}




double DTBlockedROChannelsTest::DTRobBinsMap::getChamberPercentage() {
  int nChangedROBs = 0;
  for(map<int, int>::const_iterator robAndValue = robsAndValues.begin();
      robAndValue != robsAndValues.end(); ++robAndValue) {
    if(robChanged((*robAndValue).first)) nChangedROBs++;
  }
  return 1.-((double)nChangedROBs/(double)robsAndValues.size());
}


// FIXME: move to SealModule
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_ANOTHER_FWK_MODULE(DTBlockedROChannelsTest);
