#include "EventFilter/TriggerAdapter/interface/Constants.h"
#include "EventFilter/AutoBU/interface/AutoTrigger.h"
#include "EventFilter/Utilities/interface/DebugUtils.h"

#include <toolbox/task/WorkLoopFactory.h>

#include "xgi/Utils.h"

#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"

#include <iostream>

void evf::AutoTrigger::sendNTriggers(const unsigned int n)
throw (rubuilder::ta::exception::Exception)
{
  toolbox::mem::Reference *bufRef = 0;
  unsigned int            i       = 0;
  
  uint32_t eventType = 1;
  uint32_t orbit = orbitCurrent_.value_;
  evf::l1cond l1(-1,eventNumber_.value_);
  for(i=0; i<n; i++)
    {
      bufRef = triggerGenerator_.generate
        (
	 poolFactory_,            // poolFactory
	 triggerPool_,            // pool
	 tid_,                    // initiatorAddress
	 evmTid_,                 // targetAddress
	 triggerSourceId_.value_, // triggerSourceId
	 eventNumber_.value_,     // eventNumber
	 eventType,
	 &l1,
	 orbit
	 );

      // Update parameters showing message payloads and counts
      {
	I2O_MESSAGE_FRAME *stdMsg =
	  (I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
	
	I2O_EVM_TRIGGER_Payload_.value_ += (stdMsg->MessageSize << 2) -
	  sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
	I2O_EVM_TRIGGER_LogicalCount_.value_++;
	I2O_EVM_TRIGGER_I2oCount_.value_++;
      }
      
      try
        {
	  appContext_->postFrame
            (
	     bufRef,
                appDescriptor_,
	     evmDescriptor_,
	     i2oExceptionHandler_,
	     evmDescriptor_
	     );
        }
      catch(xcept::Exception &e)
        {
	  std::stringstream oss;
	  
	  oss << "Failed to send dummy trigger";
	  oss << " (eventNumber=" << eventNumber_ << ")";
	  
	  XCEPT_RETHROW(rubuilder::ta::exception::Exception, oss.str(), e);
        }
      catch(...)
        {
	  std::stringstream oss;
	  
	  oss << "Failed to send dummy trigger";
	  oss << " (eventNumber=" << eventNumber_ << ")";
	  oss << " : Unknown exception";
	  
	  XCEPT_RAISE(rubuilder::ta::exception::Exception, oss.str());
        }
      
      // Increment the event number taking into account the 24-bit dynamic rance (wrap if necessary)
      eventNumber_ = (eventNumber_ + 1) % 0x1000000;
      sem_wait(&triggerSem_);
    }
}
void evf::AutoTrigger::enableAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  startLsCycle();
  startMonitorCycle();
  Base::enableAction(e);
}
void evf::AutoTrigger::haltAction(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  Base::haltAction(e);
  stopLsCycle();
  stopMonitorCycle();
}

void evf::AutoTrigger::defaultWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
    std::string path;
    cgicc::CgiEnvironment cgie(in);
    path = cgie.getPathInfo() + "?" + cgie.getQueryString();
    try 
      {
	cgicc::Cgicc cgi(in);
	if ( xgi::Utils::hasFormElement(cgi,"nextn") )
	  {

	    targetRate_.value_ = 
	      xgi::Utils::getFormElement(cgi, "nextn")->getIntegerValue();
	    delta_ = int((2.-double(targetRate_.value_)/100000.)/double(targetRate_.value_)*1.e6+3.);
	    std::cout << "web setting target rate and delta " << delta_ << std::endl;
	  }
      }
    catch (const std::exception & e) 
      {
	// don't care if it did not work
      }
    
    out->getHTTPResponseHeader().addHeader("Content-Type", "text/html");

    *out << "<html>"                                              << std::endl;

    *out << "<head>"                                              << std::endl;
    *out << "<link type=\"text/css\" rel=\"stylesheet\"";
    *out << " href=\"/" << urn_ << "/styles.css\"/>"              << std::endl;
    *out << "<title>"                                             << std::endl;
    *out << xmlClass_ << instance_ << " MAIN"                     << std::endl;
    *out << "</title>"                                            << std::endl;
    *out << "</head>"                                             << std::endl;

    *out << "<body>"                                              << std::endl;

    printWebPageTitleBar
    (
        out,
        rubuilder::ta::APP_ICON,
        "Main",
        rubuilder::ta::APP_ICON,
        rubuilder::ta::DBG_ICON,
        rubuilder::ta::FSM_ICON
    );

    *out << "<table>"                                             << std::endl;
    *out << "<tr valign=\"top\">"                                 << std::endl;
    *out << "  <td>"                                              << std::endl;
    try
    {
        printParamsTable(in, out, "Standard configuration", stdConfigParams_);
    }
    catch(xcept::Exception &e)
    {
        XCEPT_RETHROW(xgi::exception::Exception,
            "Failed to print standard configuration table", e);
    }
    *out << "  </td>"                                             << std::endl;
    *out << "  <td width=\"64\">"                                 << std::endl;
    *out << "  </td>"                                             << std::endl;
    *out << "  <td>"                                              << std::endl;
    try
    {
        printParamsTable(in, out, "Standard monitoring", stdMonitorParams_);
    }
    catch(xcept::Exception &e)
    {
        XCEPT_RETHROW(xgi::exception::Exception,
            "Failed to print standard monitoring table", e);
    }
    *out << "  </td>"                                             << std::endl;
    *out << "  <td>"                                              << std::endl;
    *out << "<table frame=\"void\" rules=\"rows\" class=\"params\">"
	 << std::endl;
    *out << "  <tr>"                                              << std::endl;
    *out << "    <th colspan=2>"                                  << std::endl;
    *out << "      Trigger Params"                                << std::endl;
    *out << "    </th>"                                           << std::endl;
    *out << "  </tr>"                                             << std::endl;
    *out << "  <tr>"                                              << std::endl;
    // Name
    *out << "    <td>"                                           << std::endl;
    *out << "      Orbit"                                        << std::endl;
    *out << "    </td>"                                          << std::endl;
    // Value
    *out << "    <td>"                                           << std::endl;
    std::string str;

    try
      {
	str = orbitCurrent_.toString();
      }
    catch(xcept::Exception &e)
      {
	str = e.what();
      }
    *out << "      " << str << std::endl;
    *out << "    </td>"                                           << std::endl;
    *out << "  </tr>"                                             << std::endl;
    *out << "  <tr>"                                              << std::endl;
    // Name
    *out << "    <td>"                                            << std::endl;
    *out << "      LumiSection"                                   << std::endl;
    *out << "    </td>"                                           << std::endl;
    // Value
    *out << "    <td>"                                            << std::endl;
    *out << "      " << lsCurrent_.value_			  << std::endl;
    *out << "    </td>"                                           << std::endl;
    *out << "  </tr>"                                             << std::endl;
    *out << "  <tr>"                                              << std::endl;
    // Name
    *out << "    <td>"                                            << std::endl;
    *out << "      Duration"                                      << std::endl;
    *out << "    </td>"                                           << std::endl;
    // Value
    *out << "    <td>"                                            << std::endl;
    *out << "      " << lsStartTime_.tv_sec - lsEndTime_.tv_sec
	 << std::endl;
    *out << "    </td>"                                           << std::endl;
    *out << "  </tr>"                                             << std::endl;

    // Name
    *out << "    <td>"                                            << std::endl;
    *out << "      Measured Rate"                                 << std::endl;
    *out << "    </td>"                                           << std::endl;
    // Value
    *out << "    <td>"                                            << std::endl;
    *out << "      " << measuredRate_.value_
	 << std::endl;
    *out << "    </td>"                                           << std::endl;
    *out << "  </tr>"                                             << std::endl;

    // Name
    *out << "    <td>"                                            << std::endl;
    *out << "      Target Rate"                                 << std::endl;
    *out << "    </td>"                                           << std::endl;
    // Value
    *out << "    <td>"                                            << std::endl;
    *out << "      " << targetRate_.value_
	 << std::endl;
    *out << "    </td>"                                           << std::endl;
    *out << "  </tr>"                                             << std::endl;


    // Name
    *out << "    <td>"                                            << std::endl;
    *out << "      Deadtime Fraction"                             << std::endl;
    *out << "    </td>"                                           << std::endl;
    // Value
    *out << "    <td>"                                            << std::endl;
    *out << "      " << fractionDead_.value_
	 << std::endl;
    *out << "    </td>"                                           << std::endl;
    *out << "  </tr>"                                             << std::endl;


    *out << "</table>"                                            << std::endl;

    *out << "  </td>"                                             << std::endl;
    *out << "</tr>"                                               << std::endl;
    *out << "<tr>"                                                << std::endl;
    *out << "  <td>"                                              << std::endl;
    *out << cgicc::form().set("method","GET").set("action", path ) 
	 << std::endl;
    *out << "Set rate " << std::endl;
    *out << cgicc::input().set("type","text").set("name","nextn").set("value",targetRate_.toString()) << std::endl;
    *out << cgicc::input().set("type","submit").set("value","DoIt")    << std::endl;
    *out << cgicc::form()					       << std::endl;  

    *out << "  </td>"                                             << std::endl;
    *out << "</tr>"                                               << std::endl;

    *out << "</table>"                                            << std::endl;
    *out << "</body>"                                             << std::endl;
    *out << "</html>"                                             << std::endl;
}

void evf::AutoTrigger::startLsCycle() throw (evf::Exception)
{
  lsExit_ = false;
  gettimeofday(&lsEndTime_,0);
  lsStartTime_ = lsEndTime_;
  try {
    wlLumiSectionCycle_ =
      toolbox::task::getWorkLoopFactory()->getWorkLoop("LumiSection",
						       "waiting");
    if (!wlLumiSectionCycle_->isActive()) wlLumiSectionCycle_->activate();
    asLumiSectionCycle_ = 
      toolbox::task::bind(this,&AutoTrigger::lsCycle,
				      "LumiSection");
    wlLumiSectionCycle_->submit(asLumiSectionCycle_);
  }
  catch (xcept::Exception& e) {
    std::string msg = "Failed to start workloop 'LumiSection'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }

}

void evf::AutoTrigger::stopLsCycle() throw (evf::Exception)
{
  lsExit_=true;
}

void evf::AutoTrigger::startMonitorCycle() throw (evf::Exception)
{
  monitorExit_ = false;

  try {
    wlMonitorCycle_ =
      toolbox::task::getWorkLoopFactory()->getWorkLoop("RateMonitor",
						       "waiting");
    if (!wlMonitorCycle_->isActive()) wlMonitorCycle_->activate();
    asMonitorCycle_ = 
      toolbox::task::bind(this,&AutoTrigger::monitorCycle,
				      "RateMonitor");
    wlMonitorCycle_->submit(asMonitorCycle_);
  }
  catch (xcept::Exception& e) {
    std::string msg = "Failed to start workloop 'Monitor'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
  try {
    wlMachineCycle_ =
      toolbox::task::getWorkLoopFactory()->getWorkLoop("TriggerProcessor",
						       "waiting");
    if (!wlMachineCycle_->isActive()) wlMachineCycle_->activate();
    asMachineCycle_ = 
      toolbox::task::bind(this,&AutoTrigger::machineCycle,
				      "RateMonitor");
    wlMachineCycle_->submit(asMachineCycle_);
  }
  catch (xcept::Exception& e) {
    std::string msg = "Failed to start workloop 'Machine'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }

}

void evf::AutoTrigger::stopMonitorCycle() throw (evf::Exception)
{
  monitorExit_=true;
}


bool evf::AutoTrigger::lsCycle(toolbox::task::WorkLoop* wl)
{
  if(lsExit_) return false;
  uint32_t lsn = 0;
  for(int i = 0; i < 100; i++) {
    for(int j = 0; j < 1800; j++)
      int a = i+j;
    //	  cout <<"*****"<< delta << endl;
    
    //	  ::nanosleep(&ts,0);
  }
  orbitCurrent_.value_++;
  lsn = orbitCurrent_.value_/0x00100000 + 1;
  if(lsn!=lsCurrent_.value_)
    {lsEndTime_=lsStartTime_;gettimeofday(&lsStartTime_,0);lsCurrent_.value_ = lsn;}
  
  return true;
}

bool evf::AutoTrigger::monitorCycle(toolbox::task::WorkLoop* wl)
{
  deadBuckets_.value_ = 0;
  if(monitorExit_) return false;
  gettimeofday(&monitorStartTime_,0);
  unsigned int startEvno = eventNumber_.value_;
  ::sleep(10);
  gettimeofday(&monitorEndTime_,0);
  unsigned int stopEvno = eventNumber_.value_;
  double deltaTimeSecs = monitorEndTime_.tv_sec - monitorStartTime_.tv_sec +
    (monitorEndTime_.tv_usec - monitorStartTime_.tv_usec)/1.e6;
  measuredRate_.value_ = double(stopEvno - startEvno)/ deltaTimeSecs;
  int dead =0;
  int live =0;
  sem_getvalue(&triggerSem_,&live);
  sem_getvalue(&deadtimeSem_,&dead);
  std::cout << "currently " << live << "triggers in the live and " 
	    << dead << " triggers in the dead queue " << std::endl;
  while(sem_trywait(&deadtimeSem_)==0)
    deadBuckets_.value_++;
  sem_getvalue(&triggerSem_,&live);
  sem_getvalue(&deadtimeSem_,&dead);
  std::cout << "currently " << live << "triggers in the live and " 
	    << dead << " triggers in the dead queue " << std::endl;

  fractionDead_.value_ = double(deadBuckets_.value_)/double(deltaTimeSecs*targetRate_.value_);
  return true;
}
bool evf::AutoTrigger::machineCycle(toolbox::task::WorkLoop* wl)
{
  if(lsExit_) return false;
  int semval = 0;
  sem_getvalue(&triggerSem_, &semval);
  int tdelta = (*rndPoiss_)()*delta_;
  int a = 0;
  for(int i = 0; i < 100; i++) {
    for(int j = 0; j < tdelta; j++)
      a = i+j;
  }
  if(a>0 && semval<100)sem_post(&triggerSem_); else sem_post(&deadtimeSem_);
  bx_ = tdelta;

  return true;
}
/**
 * Provides the factory method for the instantiation of autotrigger applications.
 */
XDAQ_INSTANTIATOR_IMPL(evf::AutoTrigger)

