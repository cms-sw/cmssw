/*
   Description:
     XDAQ application that is meant to act as the interface to the
     output output running in the FU HLT application. It provides
     the pointers to the XDAQ quantities needed by the framework
     I2O output module, and provides utilities for creating the
     chain of fragments and ensuring sufficient space in the memory pool.
     See the CMS EvF Storage Manager wiki page for further notes.

   $Id$
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "EventFilter/StorageManager/interface/SMi2oSender.h"
#include "EventFilter/StorageManager/interface/i2oStorageManagerMsg.h"

#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationGroup.h"

#include "toolbox/mem/CommittedHeapAllocator.h"
#include "toolbox/mem/Reference.h"
#include "toolbox/mem/Pool.h"

// for performance measurements
#include "xdata/UnsignedInteger32.h"
#include "xdata/Double.h"
#include "EventFilter/StorageManager/interface/SMPerformanceMeter.h"

#include "xcept/tools.h"
#include "xgi/Method.h"

#include "i2o/Method.h"
#include "i2o/utils/include/i2o/utils/AddressMap.h"

#include <exception>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>


using namespace std;

////////////////////////// global variable for interaction with I2OConsumer ///////////////
struct SMFU_data
{
  SMFU_data();

  xdaq::Application* app;
  toolbox::mem::Pool *pool;
  vector<xdaq::ApplicationDescriptor*> destination;
  xdaq::ApplicationDescriptor* primarydest;
  // for performance measurements
  xdata::UnsignedInteger32 samples_; //number of samples (frames) per measurement
  stor::SMPerformanceMeter *pmeter_;
  // measurements for last set of samples
  xdata::Double databw_;      // bandwidth in MB/s
  xdata::Double datarate_;    // number of frames/s
  xdata::Double datalatency_; // micro-seconds/frame
  xdata::UnsignedInteger32 totalsamples_; //number of samples (frames) per measurement
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
  pmeter_ = new stor::SMPerformanceMeter();
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
xdaq::ApplicationDescriptor* getMyXDAQDest(unsigned int instance) 
{ 
  SMi2oSender *sd = (SMi2oSender *)SMfudata.app;
  sd->setDestinations();
  if(instance < SMfudata.destination.size())
    return SMfudata.destination[instance]; 
  else
    return 0;
}
xdaq::ApplicationDescriptor* getMyXDAQDest() 
{ 
  SMi2oSender *sd = (SMi2oSender *)SMfudata.app;
  sd->setDestinations();
  return SMfudata.primarydest; 
}

void addMyXDAQMeasurement(unsigned int size)
{
  // for bandwidth performance measurements
  if ( SMfudata.pmeter_->addSample(size) )
  {
    //std::cout <<
    //  toolbox::toString("measured latency: %f for size %d",SMfudata.pmeter_->latency(), size);
    //std::cout <<
    //  toolbox::toString("latency:  %f, rate: %f,bandwidth %f, size: %d\n",
    //  SMfudata.pmeter_->latency(),SMfudata.pmeter_->rate(),SMfudata.pmeter_->bandwidth(),size);
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

xdata::UnsignedInteger32 getMyXDAQsamples() { return SMfudata.samples_; }
xdata::Double getMyXDAQdatabw() { return SMfudata.databw_; }
xdata::Double getMyXDAQdatarate() { return SMfudata.datarate_; }
xdata::Double getMyXDAQdatalatency() { return SMfudata.datalatency_; }
xdata::UnsignedInteger32 getMyXDAQtotalsamples() { return SMfudata.totalsamples_; }
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
  //  committedpoolsize_ = 0x8000000; // 128MB
  committedpoolsize_ = 0x4000000; // 64MB
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
    LOG4CPLUS_INFO(getApplicationLogger(), " max committed size " 
              << pool_->getMemoryUsage().getCommitted());
    LOG4CPLUS_INFO(getApplicationLogger(), " mem size used (bytes) " 
              << pool_->getMemoryUsage().getUsed());
    LOG4CPLUS_INFO(getApplicationLogger(), " max possible mem size " 
              << pool_->getMemoryUsage().getMax());
  }
  catch (toolbox::mem::exception::Exception& e)
  {
    string s = "Failed to create pool: " + poolName;

    LOG4CPLUS_FATAL(getApplicationLogger(), s);
    XCEPT_RETHROW(xcept::Exception, s, e);
  }


  xdata::InfoSpace *ispace = getApplicationInfoSpace();
  primarysm_ = 0;
  ispace->fireItemAvailable("primarySMInstance", &primarysm_);
  LOG4CPLUS_INFO(this->getApplicationLogger(),
		 "Primary StorageManager instance" << primarysm_);

  // Get XDAQ application destinations - currently hardwired!
  try{
    destinations_=
    getApplicationContext()->getDefaultZone()->
    getApplicationDescriptors("testStorageManager"); // hardwire here for now
  }
  catch(xdaq::exception::ApplicationDescriptorNotFound e)
  {
    LOG4CPLUS_ERROR(this->getApplicationLogger(),
                    "No StorageManager available in configuration");
  }
  catch(...)
  {
      LOG4CPLUS_FATAL(this->getApplicationLogger(),
                      "Unknown error in looking up connectable StorageManagers");
  }
  // set global variable to share with the i2o output module
  SMfudata.app = this;
  SMfudata.pool = pool_;

  xgi::bind(this,&SMi2oSender::defaultWebPage, "Default");
  xgi::bind(this,&SMi2oSender::css, "styles.css");
}

void SMi2oSender::setDestinations()
{

  set<xdaq::ApplicationDescriptor*>::iterator idest;
  if(destinations_.size()>0)
  {
    SMfudata.destination.resize(destinations_.size());

    for(idest = destinations_.begin(); idest != destinations_.end(); idest++)
      {
	SMfudata.destination[(*idest)->getInstance()] = (*idest);
	if((*idest)->getInstance() == primarysm_)
	      firstDestination_ = (*idest);
      }
    SMfudata.primarydest = firstDestination_;
  }
  else
  {
    LOG4CPLUS_ERROR(this->getApplicationLogger(),
                    "SMi2oSender::No receiver in configuration");
  }
}

void SMi2oSender::defaultWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  std::cout << "SMi2oSender default web page called" << std::endl;
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
    *out << "      " << "Performance for last " << getMyXDAQsamples() << " frame chains posted"<< endl;
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
    *out << "      " << "Mean Performance for " << getMyXDAQtotalsamples() << " frame-chains, duration "
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

/**
 * Splits an arbitrarily large data buffer into a chain of I2O
 * fragments of fixed maximum size.  This method treats each fragment
 * as a I2O_SM_MULTIPART_MESSAGE_FRAME structure
 * (see EventFilter/StorageManager/interface/i2oStorageManagerMsg.h) but
 * leaves room for additional header data using the trueHeaderSize argument.
 *
 * This method can throw the following exceptions:
 * - cms::Exception with category="InvalidInputValue" if one of
 *   the consistency checks on the input arguments fails
 * - other exceptions related to creating pool entries
 */
toolbox::mem::Reference
    *SMi2oSender::createI2OFragmentChain(char *rawData,
                                         unsigned int rawDataSize,
                                         toolbox::mem::Pool *fragmentPool,
                                         unsigned int maxFragmentDataSize,
                                         unsigned int trueHeaderSize,
                                         unsigned short functionCode,
                                         xdaq::Application *sourceApp,
                                         xdaq::ApplicationDescriptor *destAppDesc,
                                         unsigned int& numBytesSent)
{
  // Assumes that sizes are already valid and multiple of 64 bits as
  // is the case for StreamerI2OWriter (test performed during ctor)
  // rawDataSize         = total bytes to fragment starting at rawData
  // maxFragmentDataSize = size for the data
  // trueHeaderSize      = size of all I2O headers
  // trueHeaderSize + maxFragmentDataSize = i2o max size and multiple of 64 bits

  // determine the number of fragments that are needed
  unsigned int fragmentCount = (rawDataSize / maxFragmentDataSize);
  unsigned int remainder = (rawDataSize % maxFragmentDataSize);
  if(remainder > 0) ++fragmentCount;
  //std::cout << "createI2OFragmentChain: number of frames needed = " << fragmentCount
  //            << " remainder = " << remainder << std::endl;

  // verify that the pool has room for the fragments or wait
  SMi2oSender::waitForPoolSpaceIfNeeded(fragmentPool);

  // create the fragments as elements in a chain of XDAQ Reference objects
  unsigned int currentDataIndex = 0;
  int remainingDataSize = rawDataSize;
  numBytesSent = 0;
  toolbox::mem::Reference *head = NULL;
  toolbox::mem::Reference *tail = NULL;

  // catch exceptions so that we can free allocated elements in the
  // chain before re-throwing the exception
  try
  {
    for(int fragIdx = 0; fragIdx < (int)fragmentCount; ++fragIdx)
    {
      // determine the size of data to be stored in this fragment
      unsigned int dataFragmentSize = maxFragmentDataSize;
      unsigned int fragmentSize = dataFragmentSize + trueHeaderSize;
      if(remainingDataSize < (int)maxFragmentDataSize)
      {
        dataFragmentSize = (unsigned int)remainingDataSize;
        // Only in this case do we need to ensure a multiple of 64 bits is sent
        // as dataFragmentSize + trueHeaderSize is always a multiple of 64 bits
        fragmentSize = dataFragmentSize + trueHeaderSize;
        if((fragmentSize & 0x7) != 0)
        {
          // round it up to ensure sufficient space
          fragmentSize = ((fragmentSize >> 3) + 1) << 3;
        }
      }

      // allocate the fragment buffer from the pool
      toolbox::mem::Reference *bufRef =
        toolbox::mem::getMemoryPoolFactory()->getFrame(fragmentPool, fragmentSize);

      // set up pointers to the allocated buffer
      I2O_MESSAGE_FRAME *stdMsg =
          (I2O_MESSAGE_FRAME*) bufRef->getDataLocation();
      I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
          (I2O_PRIVATE_MESSAGE_FRAME*) stdMsg;
      I2O_SM_MULTIPART_MESSAGE_FRAME *msg =
          (I2O_SM_MULTIPART_MESSAGE_FRAME*) stdMsg;

      // zero out the memory buffer - is this a waste of cycles?
      memset(msg, 0x00, fragmentSize);

      // fill in the relevant fields in the I2O_MESSAGE_FRAME
      // (see $XDAQ_ROOT/daq/extern/i2o/include/i2o/shared/i2omsg.h)
      stdMsg->VersionOffset = 0;
      stdMsg->MsgFlags      = 0;  // normal message (not multicast)
      stdMsg->MessageSize   = fragmentSize >> 2;
      stdMsg->Function      = I2O_PRIVATE_MESSAGE;
      try
      {
        stdMsg->InitiatorAddress = i2o::utils::getAddressMap()->
          getTid(sourceApp->getApplicationDescriptor());
      }
      catch (xdaq::exception::ApplicationDescriptorNotFound excpt)
      {
        edm::LogError("createI2OFragmentChain") 
                      << "SMi2oSender::createI2OFragmentChain: exception in "
                      << "getting source tid "
                      <<  xcept::stdformat_exception_history(excpt);
      }
      try
      {
        stdMsg->TargetAddress =
          i2o::utils::getAddressMap()->getTid(destAppDesc);
      }
      catch (xdaq::exception::ApplicationDescriptorNotFound excpt)
      {
        edm::LogError("createI2OFragmentChain") 
                  << "SMi2oSender::createI2OFragmentChain: "
                  << "exception in getting destination tid "
                  <<  xcept::stdformat_exception_history(excpt);
      }

      // fill in the relevant fields in the I2O_PRIVATE_MESSAGE_FRAME
      // (see $XDAQ_ROOT/daq/extern/i2o/include/i2o/shared/i2omsg.h)
      pvtMsg->XFunctionCode  = functionCode;
      pvtMsg->OrganizationID = XDAQ_ORGANIZATION_ID;

      // fill in the necessary fields in the I2O_SM_MULTIPART_MESSAGE_FRAME
      // (see $CMSSW_RELEASE_BASE/src/EventFilter/StorageManager/interface/
      // i2oStorageManagerMsg.h)
      msg->dataSize = dataFragmentSize;
      msg->hltLocalId = sourceApp->getApplicationDescriptor()->getLocalId();
      msg->hltInstance = sourceApp->getApplicationDescriptor()->getInstance();
      try
      {
        msg->hltTid = i2o::utils::getAddressMap()->
          getTid(sourceApp->getApplicationDescriptor());
      }
      catch (xdaq::exception::ApplicationDescriptorNotFound excpt)
      {
        edm::LogError("createI2OFragmentChain")
                  << "SMi2oSender::createI2OFragmentChain: exception in "
                  << "getting source tid "
                  <<  xcept::stdformat_exception_history(excpt);
      }

      msg->numFrames = fragmentCount;
      msg->frameCount = fragIdx;
      msg->originalSize = rawDataSize;

      // Fill in the long form of the source (HLT) identifier
      int copySize;
      std::string url = sourceApp->getApplicationDescriptor()->
          getContextDescriptor()->getURL();
      if(url.size() > MAX_I2O_SM_URLCHARS)
      {
        edm::LogInfo("createI2OFragmentChain")
                     << "SMi2oSender: Error! Source URL truncated";
        copySize = MAX_I2O_SM_URLCHARS;
      }
      else
      {
        copySize = url.size();
      }
      for(int idx = 0; idx < copySize; idx++)
      {
        msg->hltURL[idx] = url[idx];
      }
      std::string classname = sourceApp->getApplicationDescriptor()->
          getClassName();
      if(classname.size() > MAX_I2O_SM_URLCHARS)
      {
        edm::LogInfo("createI2OFragmentChain")
                     << "SMi2oSender: Error! Source ClassName truncated";
        copySize = MAX_I2O_SM_URLCHARS;
      }
      else
      {
        copySize = classname.size();
      }
      for(int idx = 0; idx < copySize; idx++)
      {
        msg->hltClassName[idx] = classname[idx];
      }

      // update the chain pointers as needed
      if(fragIdx == 0)
      {
        head = bufRef;
        tail = bufRef;
      }
      else
      {
        tail->setNextReference(bufRef);
        tail = bufRef;
      }

      // fill in the data for this fragment
      if(dataFragmentSize != 0)
      {
        char *dataTarget = (char *) msg + trueHeaderSize;
        std::copy(rawData + currentDataIndex, 
                  rawData + currentDataIndex + dataFragmentSize, dataTarget);
      }

      // need to set the actual buffer size that is being sent
      bufRef->setDataSize(fragmentSize);

      // update indices and sizes as needed
      remainingDataSize -= dataFragmentSize;
      currentDataIndex += dataFragmentSize;
      numBytesSent += fragmentSize;
      // should check that remainingDataSize doesn't go negative (shouldn't never do!)
    }
  }
  // catch all exceptions
  catch (...)
  {

    // free up any chain elements that we have already allocated
    if(head != NULL)
    {
      head->release();
      head = NULL;
    }

    // re-throw the exception
    throw;
  }

  // return the first element in the chain
  return head;
}

/**
 * Prints debug information for the specified fragment chain to stdout.
 */
void SMi2oSender::debugFragmentChain(toolbox::mem::Reference *head)
{
  cout << "SMi2oSender::debugFragmentChain for "
       << hex << head << dec << endl;
  toolbox::mem::Reference *currentElement = head;
  int elementCount = 0;
  while(currentElement != NULL)
  {
    cout << "*** Fragment " << elementCount << endl;
    cout << " - buffer address = " << hex << currentElement->getBuffer()
         << dec << endl;
    cout << " - data location = " << hex << currentElement->getDataLocation()
         << dec << endl;
    cout << " - data offset = 0x" << hex << currentElement->getDataOffset()
         << dec << endl;
    cout << " - data size = 0x" << hex << currentElement->getDataSize()
         << dec << endl;

    char *bufPtr = (char *) currentElement->getDataLocation();
    bufPtr += currentElement->getDataOffset();
    for(unsigned int idx = 0; idx < currentElement->getDataSize(); idx++)
    {
      if((idx % 24) == 0) {cout << endl;}
      else if((idx % 4) == 0) {cout << " ";}
      int val = 0xff & (int) bufPtr[idx];
      if(val >= 0 && val <= 15) {cout << "0";}
      cout << hex << val << dec;
    }
    cout << endl;

    elementCount++;
    currentElement = currentElement->getNextReference();
  }
  cout << " -> number of fragments in the chain is " << elementCount << endl;
}

/**
 * Tests if there is sufficient space in the specified memory pool
 * and waits for space to become available if not.
 */
void SMi2oSender::waitForPoolSpaceIfNeeded(toolbox::mem::Pool *targetPool)
{
  // return immediately if there is sufficient space available
  if(!targetPool->isHighThresholdExceeded()) {return;}

  // otherwise, wait for space to become available
  edm::LogInfo("waitForPoolSpaceIfNeeded")
                 << " High threshold exceeded in memory pool "
                 << hex << targetPool << dec
                 << ", max committed size "
                 << targetPool->getMemoryUsage().getCommitted()
                 << ", mem size used (bytes) "
                 << targetPool->getMemoryUsage().getUsed();
  unsigned int yc = 0;
  while(targetPool->isLowThresholdExceeded())
  {
    usleep(50000);
    yc++;
  }
  edm::LogInfo("waitForPoolSpaceIfNeeded") <<
               "Yielded " << yc << " times before low threshold reached";
}
