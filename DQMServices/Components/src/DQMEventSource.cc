/** \file 
 *
 *  $Date: 2007/03/31 11:07:53 $
 *  $Revision: 1.5 $
 *  \author S. Bolognesi - M. Zanetti
 */

#include "DQMServices/Components/interface/DQMEventSource.h"
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

  getMESubscriptionListFromFile = pset.getUntrackedParameter<bool>("getMESubscriptionListFromFile", true);
  getQualityTestsFromFile = pset.getUntrackedParameter<bool>("getQualityTestsFromFile", true);
  skipUpdates = pset.getUntrackedParameter<int>("numberOfUpdatesToBeSkipped", 1);

  // subscribe to MEs and configure the quality tests
  if (getMESubscriptionListFromFile)
    subscriber->getMEList(pset.getUntrackedParameter<string>("meSubscriptionList", "MESubscriptionList.xml")); 
  if (getQualityTestsFromFile)
    qtHandler->configureTests(pset.getUntrackedParameter<string>("qtList", "QualityTests.xml"),mui);

  iRunMEName = pset.getUntrackedParameter<string>("iRunMEName", "Collector/FU0/EventInfo/iRun");
  iEventMEName = pset.getUntrackedParameter<string>("iEventMEName", "Collector/FU0/EventInfo/iEvent");
  timeStampMEName = pset.getUntrackedParameter<string>("timeStampMEName", "Collector/FU0/EventInfo/timeStamp");


}


std::auto_ptr<Event> DQMEventSource::readOneEvent() {

  // the "onUpdate" call. 
  mui->doMonitoring();

  if (getMESubscriptionListFromFile) subscriber->makeSubscriptions(mui);

  if (getQualityTestsFromFile) qtHandler->attachTests(mui);
  
  // getting the run coordinates 
  RunNumber_t iRun = 0;
  MonitorElementInt * iRun_p = dynamic_cast<MonitorElementInt*>(mui->get(iRunMEName));
  if (iRun_p) iRun = iRun_p->getValue(); 
  setRunNumber(iRun); // <<=== here is where the run is set

  EventNumber_t iEvent = 0;
  MonitorElementInt * iEvent_p = dynamic_cast<MonitorElementInt*>(mui->get(iEventMEName));
  if (iEvent_p) iEvent = iEvent_p->getValue(); 
  else iEvent=updatesCounter; // if the event is not received set number of updates as eventId 

  TimeValue_t tStamp = 0;
  MonitorElementInt * tStamp_p = dynamic_cast<MonitorElementInt*>(mui->get(timeStampMEName));
  if (tStamp_p) tStamp = tStamp_p->getValue(); 
  else tStamp = 1;
  
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
