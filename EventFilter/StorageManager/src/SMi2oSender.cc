/*
   Author: Harry Cheung, FNAL

   Description:
     XDAQ application that is meant to receive locally I2O frames
     from the FU HLT application, and sends them via the network
     to the Storage Manager.
     See CMS EventFilter wiki page for further notes.

   Modification:
     version 1.1 2005/11/23
       Initial implementation, only creates pool and destination.
       It does not relay I2O frames. This is done directly in the
       I2O Output module in this preproduction version.
       Uses a global pointer. Needs changes for production version.
     version 1.2 2005/12/15
       Changed to using a committed heap memory pool allocator and
       a way to set its size.
       Expanded global variable to enable statistics to be collected.
       Added default home page to show statistics.
     version 1.3 2006/01/24
        Changed to hardwire to class testStorageManager as receiver
*/

#include "EventFilter/StorageManager/interface/SMi2oSender.h"
#include "EventFilter/StorageManager/interface/i2oStorageManagerMsg.h"

#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationGroup.h"

#include "toolbox/mem/CommittedHeapAllocator.h"
#include "toolbox/mem/Reference.h"
#include "toolbox/mem/Pool.h"

// for performance measurements
#include "xdata/UnsignedLong.h"
#include "xdata/Double.h"
#include "EventFilter/StorageManager/interface/SMPerformanceMeter.h"

#include "xcept/tools.h"
#include "xgi/Method.h"

#include "i2o/Method.h"
#include "i2o/utils/include/i2o/utils/AddressMap.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

////////////////////////// global variable for interaction with I2OConsumer ///////////////
struct SMFU_data
{
  SMFU_data();

  xdaq::Application* app;
  toolbox::mem::Pool *pool;
  xdaq::ApplicationDescriptor* destination;

  // for performance measurements
  xdata::UnsignedLong samples_; //number of samples (frames) per measurement
  sto::SMPerformanceMeter *pmeter_;
  // measurements for last set of samples
  xdata::Double databw_;      // bandwidth in MB/s
  xdata::Double datarate_;    // number of frames/s
  xdata::Double datalatency_; // micro-seconds/frame
  xdata::UnsignedLong totalsamples_; //number of samples (frames) per measurement
  xdata::Double duration_;        // time for run in seconds
  xdata::Double meandatabw_;      // bandwidth in MB/s
  xdata::Double meandatarate_;    // number of frames/s
  xdata::Double meandatalatency_; // micro-seconds/frame
  xdata::Double maxdatabw_;       // maximum bandwidth in MB/s
  xdata::Double mindatabw_;       // minimum bandwidth in MB/s

};

SMFU_data::SMFU_data()
{
  app = 0;
  pool = 0;
  destination = 0;

  // for performance measurements
  samples_ = 100; // measurements every 25MB (about)
  databw_ = 0.;
  datarate_ = 0.;
  datalatency_ = 0.;
  totalsamples_ = 0;
  duration_ = 0.;
  meandatabw_ = 0.;
  meandatarate_ = 0.;
  meandatalatency_ = 0.;
  pmeter_ = new sto::SMPerformanceMeter();
  pmeter_->init(samples_);
  maxdatabw_ = 0.;
  mindatabw_ = 999999.;
}

// define a global variable and a global function to return pointer
// until we change the EventFilter/Processor/FUEventProcessor and 
// EventFilter/Processor/EventProcessor to provide these via the
// service set.

static SMFU_data SMfudata;

xdaq::Application* getMyXDAQPtr() { return SMfudata.app; }
toolbox::mem::Pool *getMyXDAQPool() { return SMfudata.pool; }
xdaq::ApplicationDescriptor* getMyXDAQDest() { return SMfudata.destination; }

void addMyXDAQMeasurement(unsigned long size)
{
  // for bandwidth performance measurements
  if ( SMfudata.pmeter_->addSample(size) )
  {
    //LOG4CPLUS_INFO(SMfudata.app->getApplicationLogger(),
    //  toolbox::toString("measured latency: %f for size %d",SMfudata.pmeter_->latency(), size));
    //LOG4CPLUS_INFO(SMfudata.app->getApplicationLogger(),
    //  toolbox::toString("latency:  %f, rate: %f,bandwidth %f, size: %d\n",
    //  SMfudata.pmeter_->latency(),SMfudata.pmeter_->rate(),SMfudata.pmeter_->bandwidth(),size));
    // new measurement; so update
    SMfudata.databw_ = SMfudata.pmeter_->bandwidth();
    SMfudata.datarate_ = SMfudata.pmeter_->rate();
    SMfudata.datalatency_ = SMfudata.pmeter_->latency();
    SMfudata.totalsamples_ = SMfudata.pmeter_->totalsamples();
    SMfudata.duration_ = SMfudata.pmeter_->duration();
    SMfudata.meandatabw_ = SMfudata.pmeter_->meanbandwidth();
    SMfudata.meandatarate_ = SMfudata.pmeter_->meanrate();
    SMfudata.meandatalatency_ = SMfudata.pmeter_->meanlatency();
    if(SMfudata.databw_ > SMfudata.maxdatabw_) SMfudata.maxdatabw_ = SMfudata.databw_;
    if(SMfudata.databw_ < SMfudata.mindatabw_) SMfudata.mindatabw_ = SMfudata.databw_;
  }
}

xdata::UnsignedLong getMyXDAQsamples() { return SMfudata.samples_; }
xdata::Double getMyXDAQdatabw() { return SMfudata.databw_; }
xdata::Double getMyXDAQdatarate() { return SMfudata.datarate_; }
xdata::Double getMyXDAQdatalatency() { return SMfudata.datalatency_; }
xdata::UnsignedLong getMyXDAQtotalsamples() { return SMfudata.totalsamples_; }
xdata::Double getMyXDAQduration() { return SMfudata.duration_; }
xdata::Double getMyXDAQmeandatabw() { return SMfudata.meandatabw_; }
xdata::Double getMyXDAQmeandatarate() { return SMfudata.meandatarate_; }
xdata::Double getMyXDAQmeandatalatency() { return SMfudata.meandatalatency_; }
xdata::Double getMyXDAQmaxdatabw() { return SMfudata.maxdatabw_; }
xdata::Double getMyXDAQmindatabw() { return SMfudata.mindatabw_; }

////////////////////////// end global variable for interaction with I2OConsumer ///////////////

//--------------------------------------------------------------
SMi2oSender::SMi2oSender(xdaq::ApplicationStub * s)
  throw (xdaq::exception::Exception): xdaq::Application(s)
{
  // create memory pool and find destinations
  // to be used for I2O frame relaying
  // currently also used by I2OConsumer output module to avoid
  // changes to EventFilter/Processor/FUEventProcessor, EventProcessor

  LOG4CPLUS_INFO(this->getApplicationLogger(),"Making SMi2oSender");
  // Create a memory pool

  std::string poolName = "SMi2oPool";
  // Set committed memeory pool size - currently hardwired!
  committedpoolsize_ = 0x8000000; // 128MB
  //committedpoolsize_ = 0x1000000; // 16MB
  try
  {
    toolbox::mem::CommittedHeapAllocator *allocator =
                  new toolbox::mem::CommittedHeapAllocator(committedpoolsize_);
    toolbox::net::URN urn("toolbox-mem-pool", poolName);
    toolbox::mem::MemoryPoolFactory *poolFactory =
                  toolbox::mem::getMemoryPoolFactory();
    pool_ = poolFactory->createPool(urn, allocator);

    LOG4CPLUS_INFO(getApplicationLogger(),
                   "Created memory pool: " << poolName);
    LOG4CPLUS_INFO(getApplicationLogger(), "Setting high and low memory watermark");
    pool_->setHighThreshold( (unsigned long)(committedpoolsize_ * 0.9));
    pool_->setLowThreshold(  (unsigned long)(committedpoolsize_ * 0.7));
    // check the settings at the beginning
    std::cout << " max committed size " 
              << pool_->getMemoryUsage().getCommitted() << std::endl;
    std::cout << " mem size used (bytes) " 
              << pool_->getMemoryUsage().getUsed() << std::endl;
    std::cout << " max possible mem size " 
              << pool_->getMemoryUsage().getMax() << std::endl;
  }
  catch (toolbox::mem::exception::Exception& e)
  {
    string s = "Failed to create pool: " + poolName;

    LOG4CPLUS_FATAL(getApplicationLogger(), s);
    XCEPT_RETHROW(xcept::Exception, s, e);
  }

  // Get XDAQ application destinations - currently hardwired!
  try{
    destinations_=
    getApplicationContext()->getApplicationGroup()->
//HEREHERE
    //getApplicationDescriptors("testI2OReceiver"); // hardwire here for now
    getApplicationDescriptors("testStorageManager"); // hardwire here for now
  }
  catch(xdaq::exception::ApplicationDescriptorNotFound e)
  {
    LOG4CPLUS_ERROR(this->getApplicationLogger(),
                    "No testI2OReceiver available in configuration");
  }
  catch(...)
  {
      LOG4CPLUS_FATAL(this->getApplicationLogger(),
                      "Unknown error in looking up connectable testI2OReceiver");
  }
  // set global variable to share with the i2o output module
  SMfudata.app = this;
  SMfudata.pool = pool_;
  if(destinations_.size()>0)
  {
    firstDestination_ = destinations_[0];
    SMfudata.destination = destinations_[0];
  }
  else
  {
    LOG4CPLUS_ERROR(this->getApplicationLogger(),
                    "SMi2oSender::No receiver in configuration");
  }

  xgi::bind(this,&SMi2oSender::defaultWebPage, "Default");
  xgi::bind(this,&SMi2oSender::css, "styles.css");
}

#include <iomanip>

void SMi2oSender::defaultWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  //std::cout << "default web page called" << std::endl;
  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"/" <<  getApplicationDescriptor()->getURN()
       << "/styles.css\"/>"                   << endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() << " instance "
       << getApplicationDescriptor()->getInstance()
       << "</title>"     << endl;
    *out << "<table border=\"0\" width=\"100%\">"                      << endl;
    *out << "<tr>"                                                     << endl;
    *out << "  <td align=\"left\">"                                    << endl;
    *out << "    <img"                                                 << endl;
    *out << "     align=\"middle\""                                    << endl;
    *out << "     src=\"/daq/evb/examples/fu/images/fu64x64.gif\""     << endl;
    *out << "     alt=\"main\""                                        << endl;
    *out << "     width=\"64\""                                        << endl;
    *out << "     height=\"64\""                                       << endl;
    *out << "     border=\"\"/>"                                       << endl;
    *out << "    <b>"                                                  << endl;
    *out << getApplicationDescriptor()->getClassName() << " instance "
         << getApplicationDescriptor()->getInstance()                  << endl;
    *out << "    </b>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/daq/xdaq/hyperdaq/images/HyperDAQ.jpg\""    << endl;
    *out << "       alt=\"HyperDAQ\""                                  << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                      << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/" << getApplicationDescriptor()->getURN()
         << "/debug\">"                   << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/daq/evb/bu/images/debug32x32.gif\""         << endl;
    *out << "       alt=\"debug\""                                     << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                     << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "</tr>"                                                    << endl;
    *out << "</table>"                                                 << endl;

  *out << "<hr/>"                                                    << endl;
  *out << "<table>"                                                  << endl;
  *out << "<tr valign=\"top\">"                                      << endl;
  *out << "  <td>"                                                   << endl;

  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << std::endl;
  *out << "<colgroup> <colgroup align=\"rigth\">"                    << std::endl;
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Memory Pool Usage"                            << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;

        *out << "<tr>" << std::endl;
        *out << "<th >" << std::endl;
        *out << "Parameter" << std::endl;
        *out << "</th>" << std::endl;
        *out << "<th>" << std::endl;
        *out << "Value" << std::endl;
        *out << "</th>" << std::endl;
        *out << "</tr>" << std::endl;
        *out << "<tr>" << std::endl;
          *out << "<td >" << std::endl;
          *out << "Memory Comitted (Bytes)" << std::endl;
          *out << "</td>" << std::endl;
          *out << "<td align=right>" << std::endl;
          *out << pool_->getMemoryUsage().getCommitted() << std::endl;
          *out << "</td>" << std::endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << std::endl;
          *out << "<td >" << std::endl;
          *out << "Memory Used (Bytes)" << std::endl;
          *out << "</td>" << std::endl;
          *out << "<td align=right>" << std::endl;
          *out << pool_->getMemoryUsage().getUsed() << std::endl;
          *out << "</td>" << std::endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << std::endl;
          *out << "<td >" << std::endl;
          *out << "Percent Memory Used (%)" << std::endl;
          *out << "</td>" << std::endl;
          *out << "<td align=right>" << std::endl;
          double memused = (double)pool_->getMemoryUsage().getUsed();
          double memmax = (double)pool_->getMemoryUsage().getCommitted();
          double percentused = 0.0;
          if(pool_->getMemoryUsage().getCommitted() != 0)
            percentused = 100.0*(memused/memmax);
          *out << std::fixed << std::showpoint
               << std::setw(6) << std::setprecision(2) << percentused << std::endl;
          *out << "</td>" << std::endl;
        *out << "  </tr>" << endl;
// performance statistics
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Performance for last " << getMyXDAQsamples() << " frames"<< endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;
        *out << "<tr>" << std::endl;
          *out << "<td >" << std::endl;
          *out << "Bandwidth (MB/s)" << std::endl;
          *out << "</td>" << std::endl;
          *out << "<td align=right>" << std::endl;
          *out << getMyXDAQdatabw() << std::endl;
          *out << "</td>" << std::endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << std::endl;
          *out << "<td >" << std::endl;
          *out << "Rate (Frame-chains/s)" << std::endl;
          *out << "</td>" << std::endl;
          *out << "<td align=right>" << std::endl;
          *out << getMyXDAQdatarate() << std::endl;
          *out << "</td>" << std::endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << std::endl;
          *out << "<td >" << std::endl;
          *out << "Latency (us/frame-chain)" << std::endl;
          *out << "</td>" << std::endl;
          *out << "<td align=right>" << std::endl;
          *out << getMyXDAQdatalatency() << std::endl;
          *out << "</td>" << std::endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << std::endl;
          *out << "<td >" << std::endl;
          *out << "Maximum Bandwidth (MB/s)" << std::endl;
          *out << "</td>" << std::endl;
          *out << "<td align=right>" << std::endl;
          *out << getMyXDAQmaxdatabw() << std::endl;
          *out << "</td>" << std::endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << std::endl;
          *out << "<td >" << std::endl;
          *out << "Minimum Bandwidth (MB/s)" << std::endl;
          *out << "</td>" << std::endl;
          *out << "<td align=right>" << std::endl;
          *out << getMyXDAQmindatabw() << std::endl;
          *out << "</td>" << std::endl;
        *out << "  </tr>" << endl;
// mean performance statistics for whole run
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Mean Performance for " << getMyXDAQtotalsamples() << " frames, duration "
         << getMyXDAQduration() << " seconds" << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;
        *out << "<tr>" << std::endl;
          *out << "<td >" << std::endl;
          *out << "Bandwidth (MB/s)" << std::endl;
          *out << "</td>" << std::endl;
          *out << "<td align=right>" << std::endl;
          *out << getMyXDAQmeandatabw() << std::endl;
          *out << "</td>" << std::endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << std::endl;
          *out << "<td >" << std::endl;
          *out << "Rate (Frame-chains/s)" << std::endl;
          *out << "</td>" << std::endl;
          *out << "<td align=right>" << std::endl;
          *out << getMyXDAQmeandatarate() << std::endl;
          *out << "</td>" << std::endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << std::endl;
          *out << "<td >" << std::endl;
          *out << "Latency (us/frame-chain)" << std::endl;
          *out << "</td>" << std::endl;
          *out << "<td align=right>" << std::endl;
          *out << getMyXDAQmeandatalatency() << std::endl;
          *out << "</td>" << std::endl;
        *out << "  </tr>" << endl;

  *out << "</table>" << std::endl;

  *out << "  </td>"                                                  << endl;
  *out << "</table>"                                                 << endl;

  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;
}

/**
 * Provides factory method for the instantiation of FU applications
 */

extern "C" xdaq::Application * instantiate_SMi2oSender(xdaq::ApplicationStub * stub )
{
        std::cout << "Going to construct a SMi2oSender instance " << endl;
        return new SMi2oSender(stub);
}
