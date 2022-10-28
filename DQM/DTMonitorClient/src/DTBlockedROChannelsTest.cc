/*
 * \file DTBlockedROChannelsTest.cc
 * 
 * \author G. Cerminara - University and INFN Torino
 *
 */

#include "DQM/DTMonitorClient/src/DTBlockedROChannelsTest.h"

//Framework
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/DTMonitorModule/interface/DTTimeEvolutionHisto.h"
#include <iostream>
#include <string>

using namespace std;
using namespace edm;

DTBlockedROChannelsTest::DTBlockedROChannelsTest(const ParameterSet& ps)
    : nevents(0),
      neventsPrev(0),
      prevNLumiSegs(0),
      prevTotalPerc(0),
      mappingToken_(esConsumes<edm::Transition::BeginRun>()),
      hSystFractionVsLS(nullptr) {
  LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest") << "[DTBlockedROChannelsTest]: Constructor";

  // prescale on the # of LS to update the test
  prescaleFactor = ps.getUntrackedParameter<int>("diagnosticPrescale", 1);

  offlineMode = ps.getUntrackedParameter<bool>("offlineMode", true);

  checkUros = ps.getUntrackedParameter<bool>("checkUros", true);
}

DTBlockedROChannelsTest::~DTBlockedROChannelsTest() {
  LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
      << "DataIntegrityTest: analyzed " << nupdates << " updates";
}

void DTBlockedROChannelsTest::beginRun(const Run& run, const EventSetup& setup) {
  // get the RO mapping
  mapping = &setup.getData(mappingToken_);
  nupdates = 0;
  return;
}

void DTBlockedROChannelsTest::fillChamberMap(DQMStore::IGetter& igetter, const EventSetup& context) {
  int dummy = 0;
  bool tenDDU = !mapping->readOutToGeometry(779, 7, 1, 1, 1, dummy, dummy, dummy, dummy, dummy, dummy);

  if (checkUros) {
    for (int crate = FEDNumbering::MINDTUROSFEDID; crate <= FEDNumbering::MAXDTUROSFEDID; ++crate) {  //loop over FEDs
      for (int mapSlot = 1; mapSlot != 13; ++mapSlot) {  //loop over mapSlot
        for (int link = 0; link != 72; ++link) {         //loop over links
                                                         //skip non existing links
          if (mapSlot == 6)
            continue;
          if (crate == 1370 && mapSlot > 5)
            continue;
          if ((mapSlot == 5 || mapSlot == 11) && link > 11)
            continue;

          int wheel = 0;
          int station = 0;
          int sector = 0;

          int dduId = theDDU(crate, mapSlot, link, tenDDU);
          int ros = theROS(mapSlot, link);
          int rob = theROB(mapSlot, link);

          //          mapping->readOutToGeometry(dduId,ros,rob,2,2,wheel,station,sector,dummy,dummy,dummy);
          readOutToGeometry(dduId, ros, rob, wheel, station, sector);
          if (station > 0) {
            //std::cout<<" FED "<<crate<<" mapSlot "<< mapSlot<<" Link "<<link<<" Wh "<<wheel<<" station "<<station<<" sector "<<sector <<endl;
            DTChamberId chId(wheel, station, sector);
            if (chamberMapUros.find(chId) == chamberMapUros.end()) {
              chamberMapUros[chId] = DTLinkBinsMap(igetter, dduId, ros);
              chamberMapUros[chId].addLinkBin(link % 24);
            }
            chamberMapUros[chId].addLinkBin(link % 24);
          } else {
            LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
                << "[DTLinkBinsMap] FED: " << crate << "mapSlot: " << mapSlot << " Link: " << link
                << " not in the mapping!" << endl;
          }
        }  //loop on links
      }    //loop on mapSlots
    }      //loop on crates
  }        //checkUros
  else {
    // fill the map of the robs per chamber
    // //FIXME: monitoring only real used FEDs
    for (int dduId = FEDNumbering::MINDTFEDID; dduId <= FEDNumbering::MAXDTFEDID; ++dduId) {  //loop over DDUs
      for (int ros = 1; ros != 13; ++ros) {                                                   // loop over ROSs
        for (int rob = 1; rob != 26; ++rob) {                                                 // loop over ROBs
          int wheel = 0;
          int station = 0;
          int sector = 0;
          if (!mapping->readOutToGeometry(dduId, ros, rob - 1, 0, 2, wheel, station, sector, dummy, dummy, dummy) ||
              !mapping->readOutToGeometry(dduId, ros, rob - 1, 0, 16, wheel, station, sector, dummy, dummy, dummy)) {
            DTChamberId chId(wheel, station, sector);
            if (chamberMap.find(chId) == chamberMap.end()) {
              chamberMap[chId] = DTRobBinsMap(igetter, dduId, ros);
              chamberMap[chId].addRobBin(rob);
            }
            chamberMap[chId].addRobBin(rob);
          } else {
            LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
                << "[DTRobBinsMap] FED: " << dduId << " ROS " << ros << " ROB: " << rob - 1 << " not in the mapping!"
                << endl;
          }
        }
      }
    }
    // loop over all chambers and remove the init flag
    for (map<DTChamberId, DTRobBinsMap>::iterator chAndRobs = chamberMap.begin(); chAndRobs != chamberMap.end();
         ++chAndRobs) {
      chAndRobs->second.init(false);
    }
  }  //Legacy
}

void DTBlockedROChannelsTest::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                                    DQMStore::IGetter& igetter,
                                                    edm::LuminosityBlock const& lumiSeg,
                                                    edm::EventSetup const& context) {
  //FR moved the following from beginJob!

  // book the summary histogram

  if (wheelHistos.empty()) {  // this is an attempt to make these bookings only once!

    ibooker.setCurrentFolder("DT/00-ROChannels");
    summaryHisto = ibooker.book2D("ROChannelSummary", "Summary Blocked RO Channels", 12, 1, 13, 5, -2, 3);
    summaryHisto->setAxisTitle("Sector", 1);
    summaryHisto->setAxisTitle("Wheel", 2);

    for (int wheel = -2; wheel != 3; ++wheel) {
      stringstream namestream;
      namestream << "ROChannelSummary_W" << wheel;
      stringstream titlestream;
      titlestream << "Blocked RO Channels (Wh " << wheel << ")";
      wheelHistos[wheel] = ibooker.book2D(namestream.str().c_str(), titlestream.str().c_str(), 12, 1, 13, 4, 1, 5);
      wheelHistos[wheel]->setAxisTitle("Sector", 1);
      wheelHistos[wheel]->setBinLabel(1, "MB1", 2);
      wheelHistos[wheel]->setBinLabel(2, "MB2", 2);
      wheelHistos[wheel]->setBinLabel(3, "MB3", 2);
      wheelHistos[wheel]->setBinLabel(4, "MB4", 2);
    }

    if (!offlineMode) {
      hSystFractionVsLS = new DTTimeEvolutionHisto(ibooker, "EnabledROChannelsVsLS", "% RO channels", 500, 5, true, 3);
    }
  }  // end attempt to make these bookings only once!

  //FR moved here from beginRun

  if (checkUros) {
    if (chamberMapUros.empty())
      fillChamberMap(igetter, context);
  } else {
    if (chamberMap.empty())
      fillChamberMap(igetter, context);
  }

  //FR moved here from beginLuminosityBlock
  run = lumiSeg.run();

  //FR moved here from endLuminosityBlock
  // counts number of lumiSegs
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if (nLumiSegs % prescaleFactor != 0 || offlineMode)
    return;

  LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
      << "[DTBlockedROChannelsTest]: End of LS " << nLumiSegs
      << ". Client called in online mode, performing client operations";

  performClientDiagnostic(igetter);

  // counts number of updats
  nupdates++;
}

void DTBlockedROChannelsTest::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (offlineMode) {
    LogTrace("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
        << "[DTBlockedROChannelsTest] endRun called. Client called in offline mode, performing operations.";
    performClientDiagnostic(igetter);
  }
}

void DTBlockedROChannelsTest::performClientDiagnostic(DQMStore::IGetter& igetter) {
  //FR: I've commented the if below. Either in online mode or in offline mode, when the diagnostic is called
  // compute first the number of events. It will be: event/lumisection in the online case, it will be: total number
  // of events (neventsPrev=0) in the offline case, when the diagnostic is called only once from the dqmEndJob

  //if(nevents == 0) { // hack to work also in offline DQM
  MonitorElement* procEvt = igetter.get("DT/EventInfo/processedEvents");
  if (procEvt != nullptr) {
    int procEvents = procEvt->getIntValue();
    nevents = procEvents - neventsPrev;
    neventsPrev = procEvents;
  }
  //}

  double totalPerc = prevTotalPerc;
  // check again!
  if (nevents != 0) {  // skip the computation if no events in the last LS

    // reset the histos
    summaryHisto->Reset();
    for (int wheel = -2; wheel != 3; ++wheel) {
      wheelHistos[wheel]->Reset();
    }

    totalPerc = 0.;

    if (checkUros) {
      // loop over all chambers and fill the wheel plots
      for (map<DTChamberId, DTLinkBinsMap>::iterator chAndLinks = chamberMapUros.begin();
           chAndLinks != chamberMapUros.end();
           ++chAndLinks) {
        DTChamberId chId = (*chAndLinks).first;
        double scale = 1.;
        int sectorForPlot = chId.sector();
        if (sectorForPlot == 13 || (sectorForPlot == 4 && chId.station() == 4)) {
          sectorForPlot = 4;
          scale = 0.5;
        } else if (sectorForPlot == 14 || (sectorForPlot == 10 && chId.station() == 4)) {
          sectorForPlot = 10;
          scale = 0.5;
        }

        // NOTE: can be called only ONCE per event per each chamber
        double chPercent = (*chAndLinks).second.getChamberPercentage(igetter);
        wheelHistos[chId.wheel()]->Fill(sectorForPlot, chId.station(), scale * chPercent);
        totalPerc += chPercent * scale * 1. /
                     240.;  // CB has to be 240 as double stations are taken into account by scale factor

        // Fill the summary
        summaryHisto->Fill(sectorForPlot, chId.wheel(), 0.25 * scale * chPercent);
      }
    }       //Uros case
    else {  //Legacy case
      // loop over all chambers and fill the wheel plots
      for (map<DTChamberId, DTRobBinsMap>::iterator chAndRobs = chamberMap.begin(); chAndRobs != chamberMap.end();
           ++chAndRobs) {
        DTChamberId chId = (*chAndRobs).first;
        double scale = 1.;
        int sectorForPlot = chId.sector();
        if (sectorForPlot == 13 || (sectorForPlot == 4 && chId.station() == 4)) {
          sectorForPlot = 4;
          scale = 0.5;
        } else if (sectorForPlot == 14 || (sectorForPlot == 10 && chId.station() == 4)) {
          sectorForPlot = 10;
          scale = 0.5;
        }

        // NOTE: can be called only ONCE per event per each chamber
        double chPercent = (*chAndRobs).second.getChamberPercentage(igetter);
        wheelHistos[chId.wheel()]->Fill(sectorForPlot, chId.station(), scale * chPercent);
        totalPerc += chPercent * scale * 1. /
                     240.;  // CB has to be 240 as double stations are taken into account by scale factor

        // Fill the summary
        summaryHisto->Fill(sectorForPlot, chId.wheel(), 0.25 * scale * chPercent);
      }
    }  //Legacy case
  }    //nevents != 0

  if (!offlineMode) {  // fill trend histo only in online
    hSystFractionVsLS->accumulateValueTimeSlot(totalPerc);
    hSystFractionVsLS->updateTimeSlot(nLumiSegs, nevents);
    prevTotalPerc = totalPerc;
  }
}

int DTBlockedROChannelsTest::readOutToGeometry(int dduId, int ros, int rob, int& wheel, int& station, int& sector) {
  int dummy = 0;
  return mapping->readOutToGeometry(dduId, ros, rob, 2, 2, wheel, station, sector, dummy, dummy, dummy);
}

DTBlockedROChannelsTest::DTRobBinsMap::DTRobBinsMap(DQMStore::IGetter& igetter, const int fed, const int ros)
    : rosBin(ros), init_(true), rosValue(0) {
  // get the pointer to the corresondig histo
  // Legacy
  stringstream mename;
  mename << "DT/00-DataIntegrity/FED" << fed << "/ROS" << ros << "/FED" << fed << "_ROS" << ros << "_ROSError";
  rosHName = mename.str();

  stringstream whname;
  whname << "DT/00-DataIntegrity/FED" << fed << "/FED" << fed << "_ROSStatus";
  dduHName = whname.str();

  meROS = igetter.get(rosHName);
  meDDU = igetter.get(dduHName);
}

DTBlockedROChannelsTest::DTRobBinsMap::DTRobBinsMap() : init_(true), meROS(nullptr), meDDU(nullptr) {}

DTBlockedROChannelsTest::DTRobBinsMap::~DTRobBinsMap() {}

// add a rob to the set of robs
void DTBlockedROChannelsTest::DTRobBinsMap::addRobBin(int robBin) { robsAndValues[robBin] = getValueRobBin(robBin); }

int DTBlockedROChannelsTest::DTRobBinsMap::getValueRobBin(int robBin) const {
  if (init_)
    return 0;
  int value = 0;
  if (meROS) {
    value += (int)meROS->getBinContent(9, robBin);
    value += (int)meROS->getBinContent(11, robBin);
  }
  return value;
}

int DTBlockedROChannelsTest::DTRobBinsMap::getValueRos() const {
  int value = 0;
  if (meDDU) {
    value += (int)meDDU->getBinContent(2, rosBin);
    value += (int)meDDU->getBinContent(10, rosBin);
  }
  return value;
}

bool DTBlockedROChannelsTest::DTRobBinsMap::robChanged(int robBin) {
  // check that this is a valid ROB for this map (= it has been added!)
  if (robsAndValues.find(robBin) == robsAndValues.end()) {
    LogWarning("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
        << "[DTRobBinsMap]***Error: ROB: " << robBin << " is not valid" << endl;
    return false;
  }

  int newValue = getValueRobBin(robBin);
  if (newValue > robsAndValues[robBin]) {
    robsAndValues[robBin] = newValue;
    return true;
  }
  return false;
}

double DTBlockedROChannelsTest::DTRobBinsMap::getChamberPercentage(DQMStore::IGetter& igetter) {
  meROS = igetter.get(rosHName);
  meDDU = igetter.get(dduHName);
  int nChangedROBs = 0;

  // check if ros status has changed
  int newValue = getValueRos();
  if (newValue > rosValue) {
    rosValue = newValue;
    return 0.;
  }

  for (map<int, int>::const_iterator robAndValue = robsAndValues.begin(); robAndValue != robsAndValues.end();
       ++robAndValue) {
    if (robChanged((*robAndValue).first))
      nChangedROBs++;
  }
  return 1. - ((double)nChangedROBs / (double)robsAndValues.size());
}

void DTBlockedROChannelsTest::DTRobBinsMap::readNewValues(DQMStore::IGetter& igetter) {
  meROS = igetter.get(rosHName);
  meDDU = igetter.get(dduHName);

  rosValue = getValueRos();
  for (map<int, int>::const_iterator robAndValue = robsAndValues.begin(); robAndValue != robsAndValues.end();
       ++robAndValue) {
    robChanged((*robAndValue).first);
  }
}

// uROS starting on 2018
DTBlockedROChannelsTest::DTLinkBinsMap::DTLinkBinsMap(DQMStore::IGetter& igetter, const int ddu, const int ros)
    : init_(true) {
  int wheel = (ddu - 770) % 5 - 2;

  // get the pointer to the corresondig histo
  urosHName = "DT/00-DataIntegrity/Wheel" + to_string(wheel) + "/Sector" + to_string(ros) + "/W" + to_string(wheel) +
              "_Sector" + to_string(ros) + "_ROSError";
  meuROS = igetter.get(urosHName);
}

DTBlockedROChannelsTest::DTLinkBinsMap::DTLinkBinsMap() : init_(false), meuROS(nullptr) {}

DTBlockedROChannelsTest::DTLinkBinsMap::~DTLinkBinsMap() {}

void DTBlockedROChannelsTest::DTLinkBinsMap::addLinkBin(int linkBin) {
  linksAndValues[linkBin] = getValueLinkBin(linkBin);
}

int DTBlockedROChannelsTest::DTLinkBinsMap::getValueLinkBin(int linkBin) const {
  if (!init_)
    return 0;
  int value = 0;
  if (meuROS) {
    value += (int)meuROS->getBinContent(5, linkBin);  //ONLY NotOKFlag
  }
  return value;
}

bool DTBlockedROChannelsTest::DTLinkBinsMap::linkChanged(int linkBin) {
  // check that this is a valid Link for this map (= it has been added!)
  if (linksAndValues.find(linkBin) == linksAndValues.end()) {
    LogWarning("DTDQM|DTRawToDigi|DTMonitorClient|DTBlockedROChannelsTest")
        << "[DTLinkBinsMap]***Error: Link: " << linkBin << " is not valid" << endl;
    return false;
  }

  int newValue = getValueLinkBin(linkBin);
  if (newValue > linksAndValues[linkBin]) {
    linksAndValues[linkBin] = newValue;
    return true;
  }
  return false;
}

double DTBlockedROChannelsTest::DTLinkBinsMap::getChamberPercentage(DQMStore::IGetter& igetter) {
  meuROS = igetter.get(urosHName);
  int nChangedLinks = 0;

  for (map<int, int>::const_iterator linkAndValue = linksAndValues.begin(); linkAndValue != linksAndValues.end();
       ++linkAndValue) {
    if (linkChanged((*linkAndValue).first))
      nChangedLinks++;
  }
  return 1. - ((double)nChangedLinks / (double)linksAndValues.size());
}

void DTBlockedROChannelsTest::DTLinkBinsMap::readNewValues(DQMStore::IGetter& igetter) {
  meuROS = igetter.get(urosHName);

  for (map<int, int>::const_iterator linkAndValue = linksAndValues.begin(); linkAndValue != linksAndValues.end();
       ++linkAndValue) {
    linkChanged((*linkAndValue).first);
  }
}

// Conversions
int DTBlockedROChannelsTest::theDDU(int crate, int slot, int link, bool tenDDU) {
  int ros = theROS(slot, link);

  int ddu = 772;
  //if (crate == 1368) { ddu = 775; }
  //Needed just in case this FED should be used due to fibers lenght

  if (crate == FEDNumbering::MINDTUROSFEDID) {
    if (slot < 7)
      ddu = 770;
    else
      ddu = 771;
  }

  if (crate == (FEDNumbering::MINDTUROSFEDID + 1)) {
    ddu = 772;
  }

  if (crate == FEDNumbering::MAXDTUROSFEDID) {
    if (slot < 7)
      ddu = 773;
    else
      ddu = 774;
  }

  if (ros > 6 && tenDDU && ddu < 775)
    ddu += 5;

  return ddu;
}

int DTBlockedROChannelsTest::theROS(int slot, int link) {
  if (slot % 6 == 5)
    return link + 1;

  int ros = (link / 24) + 3 * (slot % 6) - 2;
  return ros;
}

int DTBlockedROChannelsTest::theROB(int slot, int link) {
  if (slot % 6 == 5)
    return 23;

  int rob = link % 24;
  if (rob < 15)
    return rob;
  if (rob == 15)
    return 24;
  return rob - 1;
}

// FIXME: move to SealModule
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DTBlockedROChannelsTest);
