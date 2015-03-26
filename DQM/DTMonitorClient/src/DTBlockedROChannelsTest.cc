/*
 * \file DTBlockedROChannelsTest.cc
 * 
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
#include "DQM/DTMonitorModule/interface/DTTimeEvolutionHisto.h"
#include <iostream>
#include <string>


using namespace std;
using namespace edm;


DTBlockedROChannelsTest::DTBlockedROChannelsTest(const ParameterSet& ps) : nevents(0),
  neventsPrev(0),
  prevNLumiSegs(0),
  prevTotalPerc(0),
  hSystFractionVsLS(0) 
{
  LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
    << "[DTBlockedROChannelsTest]: Constructor";

  // prescale on the # of LS to update the test
  prescaleFactor = ps.getUntrackedParameter<int>("diagnosticPrescale", 1);

  offlineMode = ps.getUntrackedParameter<bool>("offlineMode", true);
}



DTBlockedROChannelsTest::~DTBlockedROChannelsTest() {
  LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
    << "DataIntegrityTest: analyzed " << nupdates << " updates";
}


void DTBlockedROChannelsTest::beginRun(const Run& run, const EventSetup& setup)
{

  nupdates = 0;
  return;
}


void DTBlockedROChannelsTest::fillChamberMap( DQMStore::IGetter & igetter, const EventSetup& context) {
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
        if(!mapping->readOutToGeometry(dduId,ros,rob-1,0,2,wheel,station,sector,dummy,dummy,dummy) ||
            !mapping->readOutToGeometry(dduId,ros,rob-1,0,16,wheel,station,sector,dummy,dummy,dummy)) {
          DTChamberId chId(wheel, station, sector);
          if(chamberMap.find(chId) == chamberMap.end()) {
            chamberMap[chId] = DTRobBinsMap(igetter, dduId, ros);
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
  // loop over all chambers and remove the init flag
  for(map<DTChamberId, DTRobBinsMap>::iterator chAndRobs = chamberMap.begin();
      chAndRobs != chamberMap.end(); ++chAndRobs) {
    chAndRobs->second.init(false);
  }
}


void DTBlockedROChannelsTest::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,
                                                    edm::LuminosityBlock const & lumiSeg, edm::EventSetup const& context){

  //FR moved the following from beginJob!

  // book the summary histogram

  if (wheelHitos.size()==0) { // this is an attempt to make these bookings only once!

    ibooker.setCurrentFolder("DT/00-ROChannels");
    summaryHisto = ibooker.book2D("ROChannelSummary","Summary Blocked RO Channels",12,1,13,5,-2,3);
    summaryHisto->setAxisTitle("Sector",1);
    summaryHisto->setAxisTitle("Wheel",2);

    for(int wheel = -2; wheel != 3; ++wheel) {
      stringstream namestream;  namestream << "ROChannelSummary_W" << wheel;
      stringstream titlestream; titlestream << "Blocked RO Channels (Wh " << wheel << ")";
      wheelHitos[wheel] = ibooker.book2D(namestream.str().c_str(),titlestream.str().c_str(),12,1,13,4,1,5);
      wheelHitos[wheel]->setAxisTitle("Sector",1);
      wheelHitos[wheel]->setBinLabel(1,"MB1",2);
      wheelHitos[wheel]->setBinLabel(2,"MB2",2);
      wheelHitos[wheel]->setBinLabel(3,"MB3",2);
      wheelHitos[wheel]->setBinLabel(4,"MB4",2);
    }

    if(!offlineMode) {
      hSystFractionVsLS = new DTTimeEvolutionHisto(ibooker, "EnabledROChannelsVsLS", "% RO channels",
        500, 5, true, 3);
    }
  }  // end attempt to make these bookings only once!


  //FR moved here from beginRun

  if (chamberMap.size() == 0) fillChamberMap(igetter, context);

  //FR moved here from beginLuminosityBlock
  run = lumiSeg.run();
 
  //FR moved here from endLuminosityBlock
  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if (nLumiSegs%prescaleFactor != 0 || offlineMode) return;

  LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
    <<"[DTBlockedROChannelsTest]: End of LS " << nLumiSegs << ". Client called in online mode, performing client operations";

  performClientDiagnostic(igetter);

  // counts number of updats 
  nupdates++;
 
}

void DTBlockedROChannelsTest::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter){

  if (offlineMode) {
    LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
      <<"[DTBlockedROChannelsTest] endRun called. Client called in offline mode, performing operations.";
    performClientDiagnostic(igetter);
  }
}

void DTBlockedROChannelsTest::performClientDiagnostic(DQMStore::IGetter & igetter) {

  //FR: I've commented the if below. Either in online mode or in offline mode, when the diagnostic is called
  // compute first the number of events. It will be: event/lumisection in the online case, it will be: total number
  // of events (neventsPrev=0) in the offline case, when the diagnostic is called only once from the dqmEndJob

  //if(nevents == 0) { // hack to work also in offline DQM
   MonitorElement *procEvt =  igetter.get("DT/EventInfo/processedEvents");
   if(procEvt != 0) {
     int procEvents = procEvt->getIntValue();
     nevents = procEvents - neventsPrev;
     neventsPrev = procEvents;
   }
   //}

  double totalPerc = prevTotalPerc;
  // check again!
  if(nevents != 0) { // skip the computation if no events in the last LS

    // reset the histos
    summaryHisto->Reset();
    for(int wheel = -2; wheel != 3; ++wheel) {
      wheelHitos[wheel]->Reset();
    }

    totalPerc = 0.;

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
      double chPercent = (*chAndRobs).second.getChamberPercentage(igetter); 
      wheelHitos[chId.wheel()]->Fill(sectorForPlot, chId.station(),
          scale*chPercent);
      totalPerc += chPercent*scale*1./240.; // CB has to be 240 as double stations are taken into account by scale factor

      // Fill the summary
      summaryHisto->Fill(sectorForPlot, chId.wheel(), 0.25*scale*chPercent);
    }
  }

  if(!offlineMode) { // fill trend histo only in online
    hSystFractionVsLS->accumulateValueTimeSlot(totalPerc);
    hSystFractionVsLS->updateTimeSlot(nLumiSegs, nevents);
    prevTotalPerc = totalPerc;
  }

}

int DTBlockedROChannelsTest::readOutToGeometry(int dduId, int ros, int& wheel, int& sector){

  int dummy;
  return mapping->readOutToGeometry(dduId,ros,2,2,2,wheel,dummy,sector,dummy,dummy,dummy);

}

DTBlockedROChannelsTest::DTRobBinsMap::DTRobBinsMap(DQMStore::IGetter & igetter, const int fed, const int ros) : 
  rosBin(ros),
  init_(true),
  rosValue(0)
{

  // get the pointer to the corresondig histo
  stringstream mename; mename << "DT/00-DataIntegrity/FED" << fed << "/ROS" << ros
    << "/FED" << fed << "_ROS" << ros << "_ROSError";
  rosHName = mename.str();

  stringstream whname; whname << "DT/00-DataIntegrity/FED" << fed
    << "/FED" << fed << "_ROSStatus";
  dduHName = whname.str();

  meROS = igetter.get(rosHName);
  meDDU = igetter.get(dduHName);
}

DTBlockedROChannelsTest::DTRobBinsMap::DTRobBinsMap() : init_(true),
  meROS(0),
  meDDU(0){}

  DTBlockedROChannelsTest::DTRobBinsMap::~DTRobBinsMap() {}



  // add a rob to the set of robs
  void DTBlockedROChannelsTest::DTRobBinsMap::addRobBin(int robBin) {
    robsAndValues[robBin] = getValueRobBin(robBin);
  }




  int DTBlockedROChannelsTest::DTRobBinsMap::getValueRobBin(int robBin) const {
    if (init_)
      return 0;
    int value = 0;
    if(meROS) {
      value += (int)meROS->getBinContent(9,robBin);
      value += (int)meROS->getBinContent(11,robBin);
    }
    return value;
  }




int DTBlockedROChannelsTest::DTRobBinsMap::getValueRos() const {
  int value = 0;
  if(meDDU) {
    value += (int)meDDU->getBinContent(2,rosBin);
    value += (int)meDDU->getBinContent(10,rosBin);
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




double DTBlockedROChannelsTest::DTRobBinsMap::getChamberPercentage(DQMStore::IGetter & igetter) {
  meROS = igetter.get(rosHName);
  meDDU = igetter.get(dduHName);
  int nChangedROBs = 0;

  // check if ros status has changed
  int newValue = getValueRos();
  if(newValue > rosValue) {
    rosValue= newValue;
    return 0.;
  }

  for(map<int, int>::const_iterator robAndValue = robsAndValues.begin();
      robAndValue != robsAndValues.end(); ++robAndValue) {
    if(robChanged((*robAndValue).first)) nChangedROBs++;
  }
  return 1.-((double)nChangedROBs/(double)robsAndValues.size());
}


void DTBlockedROChannelsTest::DTRobBinsMap::readNewValues(DQMStore::IGetter & igetter) {
  meROS = igetter.get(rosHName);
  meDDU = igetter.get(dduHName);

  rosValue = getValueRos();
  for(map<int, int>::const_iterator robAndValue = robsAndValues.begin();
      robAndValue != robsAndValues.end(); ++robAndValue) {
    robChanged((*robAndValue).first);
  }
}




// FIXME: move to SealModule
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DTBlockedROChannelsTest);
