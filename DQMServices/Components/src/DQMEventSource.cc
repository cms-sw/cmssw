/** \file 
 *
 *  $Date: 2007/03/26 14:37:04 $
 *  $Revision: 1.2 $
 *  \author S. Bolognesi - M. Zanetti
 */

#include "DQMServices/Components/interface/DQMEventSource.h"
#include <DataFormats/Common/interface/EventID.h>
#include <DataFormats/Common/interface/Timestamp.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <DQMServices/UI/interface/MonitorUIRoot.h>
#include "DQMServices/ClientConfig/interface/SubscriptionHandle.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include <DQMServices/Core/interface/MonitorElementBaseT.h>

#include <iostream>
#include <string>
#include <sys/time.h>


using namespace edm;
using namespace std;


DQMEventSource::DQMEventSource(const ParameterSet& pset, 
			       const InputSourceDescription& desc) 
  : RawInputSource(pset,desc), updatesCounter(0){

  // hold off to the monitor interface 
  mui = new MonitorUIRoot(pset.getUntrackedParameter<string>("server", "localhost"), 
			  pset.getUntrackedParameter<int>("port", 9090), 
			  pset.getUntrackedParameter<string>("name", "DTDQMClient"), 
			  pset.getUntrackedParameter<int>("reconnect_delay_secs", 5), 
			  pset.getUntrackedParameter<bool>("actAsServer", true));

  subscriber=new SubscriptionHandle;
  qtHandler=new QTestHandle;


  getQualityTestsFromFile = pset.getUntrackedParameter<bool>("getQualityTestsFromFile", true);
  skipUpdates = pset.getUntrackedParameter<int>("numberOfUpdatesToBeSkipped", 1);

  // subscribe to MEs and configure the quality tests
  subscriber->getMEList(pset.getUntrackedParameter<string>("meSubscriptionList", "MESubscriptionList.xml")); 
  if (getQualityTestsFromFile)
    qtHandler->configureTests(pset.getUntrackedParameter<string>("qtList", "QualityTests.xml"),mui);

  iRunMEName = pset.getUntrackedParameter<string>("iRunMEName", "Collector/FU0/iRun");
  iEventMEName = pset.getUntrackedParameter<string>("iEventMEName", "Collector/FU0/iEvent");
  timeStampMEName = pset.getUntrackedParameter<string>("timeStampMEName", "Collector/FU0/timeStamp");

  subscriber->makeSubscriptions(mui);
  
  // getting the run coordinates 
  RunNumber_t iRun = 0;
  MonitorElementInt * iRun_p = dynamic_cast<MonitorElementInt*>(mui->get(iRunMEName));
  if (iRun_p) iRun = iRun_p->getValue(); 

  setRunNumber(iRun);

}


std::auto_ptr<Event> DQMEventSource::readOneEvent() {

  // the "onUpdate" call. 
  mui->doMonitoring();


  if (getQualityTestsFromFile) qtHandler->attachTests(mui);

  EventNumber_t iEvent = 0;
  MonitorElementInt * iEvent_p = dynamic_cast<MonitorElementInt*>(mui->get(iEventMEName));
  if (iEvent_p) iEvent = iEvent_p->getValue(); 
  else iEvent=updatesCounter;

  TimeValue_t tStamp = 0;
  MonitorElementInt * tStamp_p = dynamic_cast<MonitorElementInt*>(mui->get(timeStampMEName));
  if (tStamp_p) tStamp = tStamp_p->getValue(); 
  else tStamp = 1;
  
  RunNumber_t iRun = 0; // fake 
  EventID eventId(iRun,iEvent);
  Timestamp timeStamp (tStamp);

  // make a fake event containing no data but the evId and runId from DQM sources
  std::auto_ptr<Event> e = makeEvent(eventId,timeStamp);
  
  // run the quality tests: skip the first when the ME created by the Clients are not yet there
  if (updatesCounter > skipUpdates && getQualityTestsFromFile) {
    cout<<"[DQMEventSource]: Running the quality tests"<<endl;
    mui->runQTests();
  }

  // counting the updates
  updatesCounter++;

  return e;
}
