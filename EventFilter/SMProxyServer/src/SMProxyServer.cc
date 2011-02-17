// $Id: SMProxyServer.cc,v 1.44 2010/09/20 15:16:49 mommsen Exp $

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <sys/stat.h>

#include "EventFilter/SMProxyServer/interface/SMProxyServer.h"
#include "EventFilter/StorageManager/interface/ConsumerPipe.h"

#include "EventFilter/Utilities/interface/ParameterSetRetriever.h"

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"

#include "IOPool/Streamer/interface/MsgHeader.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/OtherMessage.h"
#include "IOPool/Streamer/interface/ConsRegMessage.h"
#include "IOPool/Streamer/interface/HLTInfo.h"
#include "IOPool/Streamer/interface/Utilities.h"
#include "IOPool/Streamer/interface/TestFileReader.h"

#include "xcept/tools.h"

#include "xgi/Method.h"

#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xoap/domutils.h"

#include "xdata/InfoSpaceFactory.h"

#include "boost/algorithm/string/case_conv.hpp"
#include "cgicc/Cgicc.h"
#include "zlib.h"

using namespace edm;
using namespace std;
using namespace stor;

SMProxyServer::SMProxyServer(xdaq::ApplicationStub * s)
  throw (xdaq::exception::Exception) :
  xdaq::Application(s),
  fsm_(this), 
  reasonForFailedState_(),
  ah_(0), 
  collateDQM_(false),
  archiveDQM_(false),
  archiveIntervalDQM_(0),
  filePrefixDQM_("/tmp/DQM"),
  purgeTimeDQM_(DEFAULT_PURGE_TIME),
  readyTimeDQM_(DEFAULT_READY_TIME),
  useCompressionDQM_(true),
  compressionLevelDQM_(1),
  receivedEvents_(0),
  receivedDQMEvents_(0),
  mybuffer_(7000000),
  fairShareES_(false),
  connectedSMs_(0), 
  storedDQMEvents_(0), 
  sentEvents_(0),
  sentDQMEvents_(0), 
  storedVolume_(0.),
  progressMarker_("Idle")
{  
  LOG4CPLUS_INFO(this->getApplicationLogger(),"Making SMProxyServer");


  ah_   = new edm::AssertHandler();
  fsm_.initialize<SMProxyServer>(this);

  // Careful with next line: state machine fsm_ has to be setup first
  setupFlashList();

  xdata::InfoSpace *ispace = getApplicationInfoSpace();

  ispace->fireItemAvailable("stateName",     fsm_.stateName());
  ispace->fireItemAvailable("connectedSMs",  &connectedSMs_);
  ispace->fireItemAvailable("storedDQMEvents",  &storedDQMEvents_);
  ispace->fireItemAvailable("sentEvents",    &sentEvents_);
  ispace->fireItemAvailable("sentDQMEvents",    &sentDQMEvents_);
  ispace->fireItemAvailable("SMRegistrationList",&smRegList_);
  //ispace->fireItemAvailable("closedFiles",&closedFiles_);
  //ispace->fireItemAvailable("fileList",&fileList_);
  //ispace->fireItemAvailable("eventsInFile",&eventsInFile_);
  //ispace->fireItemAvailable("fileSize",&fileSize_);

  // Bind web interface
  xgi::bind(this,&SMProxyServer::defaultWebPage,       "Default");
  xgi::bind(this,&SMProxyServer::css,                  "styles.css");
  xgi::bind(this,&SMProxyServer::smsenderWebPage,      "smsenderlist");
  xgi::bind(this,&SMProxyServer::DQMOutputWebPage,     "DQMoutputStatus");
  xgi::bind(this,&SMProxyServer::eventdataWebPage,     "geteventdata");
  xgi::bind(this,&SMProxyServer::headerdataWebPage,    "getregdata");
  xgi::bind(this,&SMProxyServer::consumerWebPage,      "registerConsumer");
  xgi::bind(this,&SMProxyServer::consumerListWebPage,  "consumerList");
  xgi::bind(this,&SMProxyServer::DQMeventdataWebPage,  "getDQMeventdata");
  xgi::bind(this,&SMProxyServer::DQMconsumerWebPage,   "registerDQMConsumer");
  xgi::bind(this,&SMProxyServer::eventServerWebPage,   "EventServerStats");

  xgi::bind(this,&SMProxyServer::receiveEventWebPage,     "pushEventData");
  xgi::bind(this,&SMProxyServer::receiveDQMEventWebPage,  "pushDQMEventData");

  ispace->fireItemAvailable("collateDQM",     &collateDQM_);
  ispace->fireItemAvailable("archiveDQM",     &archiveDQM_);
  ispace->fireItemAvailable("archiveIntervalDQM",  &archiveIntervalDQM_);
  ispace->fireItemAvailable("purgeTimeDQM",   &purgeTimeDQM_);
  ispace->fireItemAvailable("readyTimeDQM",   &readyTimeDQM_);
  ispace->fireItemAvailable("filePrefixDQM",  &filePrefixDQM_);
  ispace->fireItemAvailable("useCompressionDQM",  &useCompressionDQM_);
  ispace->fireItemAvailable("compressionLevelDQM",  &compressionLevelDQM_);
  //nLogicalDisk_   = 0;

  ispace->fireItemAvailable("rcmsStateListener", fsm_.rcmsStateListener());
  ispace->fireItemAvailable("foundRcmsStateListener", fsm_.foundRcmsStateListener());
  // 21-Nov-2008, KAB: the findRcmsStateListener call needs to go after the
  // calls to add the RCMS vars to the application infospace.
  fsm_.findRcmsStateListener();

  // added for Event Server
  maxESEventRate_ = 100.0;  // hertz
  ispace->fireItemAvailable("maxESEventRate",&maxESEventRate_);
  maxESDataRate_ = 1024.0;  // MB/sec
  ispace->fireItemAvailable("maxESDataRate",&maxESDataRate_);
  maxEventRequestRate_ = 25.0;  // hertz
  ispace->fireItemAvailable("maxEventRequestRate",&maxEventRequestRate_);
  activeConsumerTimeout_ = 60;  // seconds
  ispace->fireItemAvailable("activeConsumerTimeout",&activeConsumerTimeout_);
  idleConsumerTimeout_ = 60;  // seconds
  ispace->fireItemAvailable("idleConsumerTimeout",&idleConsumerTimeout_);
  consumerQueueSize_ = 10;
  ispace->fireItemAvailable("consumerQueueSize",&consumerQueueSize_);
  //ispace->fireItemAvailable("fairShareES",&fairShareES_);
  DQMmaxESEventRate_ = 1.0;  // hertz
  ispace->fireItemAvailable("DQMmaxESEventRate",&DQMmaxESEventRate_);
  maxDQMEventRequestRate_ = 1.0;  // hertz
  ispace->fireItemAvailable("maxDQMEventRequestRate",&maxDQMEventRequestRate_);
  DQMactiveConsumerTimeout_ = 60;  // seconds
  ispace->fireItemAvailable("DQMactiveConsumerTimeout",&DQMactiveConsumerTimeout_);
  DQMidleConsumerTimeout_ = 120;  // seconds
  ispace->fireItemAvailable("DQMidleConsumerTimeout",&DQMidleConsumerTimeout_);
  DQMconsumerQueueSize_ = 10;
  ispace->fireItemAvailable("DQMconsumerQueueSize",&DQMconsumerQueueSize_);
  esSelectedHLTOutputModule_ = "hltOutputDQM";
  ispace->fireItemAvailable("esSelectedHLTOutputModule",&esSelectedHLTOutputModule_);
  esSelectedEventSelection_.clear();
  ispace->fireItemAvailable("esSelectedEventSelection",&esSelectedEventSelection_);
  TriggerSelector_ = xdata::String();
  ispace->fireItemAvailable("TriggerSelector",&TriggerSelector_);
  selectionFromClient_=false;
  ispace->fireItemAvailable("selectionFromClient",&selectionFromClient_);
  allowMissingSM_ = false;
  ispace->fireItemAvailable("allowMissingSM",&allowMissingSM_);
  dropOldLumisectionEvents_ = false;
  ispace->fireItemAvailable("dropOldLumisectionEvents",&dropOldLumisectionEvents_);
  enableDQMSM_ = true;
  ispace->fireItemAvailable("enableDQMSM",&enableDQMSM_);
  maxConnectionRetries_ = 5;
  ispace->fireItemAvailable("maxConnectionRetries",&maxConnectionRetries_);

  //those are relevant only when consumer defines a SM connection

  queueTimeout_=0;
  ispace->fireItemAvailable("queueTimeout",&queueTimeout_);
  alwaysRestartQueue_=false;
  ispace->fireItemAvailable("alwaysRestartQueue",&alwaysRestartQueue_);

  timeoutCounter_=queueTimeout_;
  queueCreated_=false;

  //prepare workloop
  timeoutWorkLoop_ = toolbox::task::getWorkLoopFactory()->getWorkLoop("queueTimeout", "waiting");
  asTimeout_ = toolbox::task::bind(this,&SMProxyServer::queueTimeout,
		  "queueTimeout");
  timeoutWorkLoop_->submit(asTimeout_);

  // for performance measurements
  ispace->fireItemAvailable("receivedSamples4Stats",&samples_);
  ispace->fireItemAvailable("receivedPeriod4Stats",&period4samples_);
  samples_          = 100; // measurements every 100 samples
  period4samples_   = 5; // measurements every 5 seconds
  instantBandwidth_ = 0.;
  instantRate_      = 0.;
  instantLatency_   = 0.;
  totalSamples_     = 0;
  duration_         = 0.;
  meanBandwidth_    = 0.;
  meanRate_         = 0.;
  meanLatency_      = 0.;
  maxBandwidth_     = 0.;
  minBandwidth_     = 999999.;
  receivedVolume_   = 0.;
  outreceivedVolume_ = 0.;
  outinstantBandwidth_ = 0.;
  outinstantRate_      = 0.;
  outinstantLatency_   = 0.;
  outtotalSamples_     = 0;
  outduration_         = 0.;
  outmeanBandwidth_    = 0.;
  outmeanRate_         = 0.;
  outmeanLatency_      = 0.;
  outmaxBandwidth_     = 0.;
  outminBandwidth_     = 999999.;
  instantBandwidth2_ = 0.;
  instantRate2_      = 0.;
  instantLatency2_   = 0.;
  totalSamples2_     = 0;
  duration2_         = 0.;
  meanBandwidth2_    = 0.;
  meanRate2_         = 0.;
  meanLatency2_      = 0.;
  maxBandwidth2_     = 0.;
  minBandwidth2_     = 999999.;
  outinstantBandwidth2_ = 0.;
  outinstantRate2_      = 0.;
  outinstantLatency2_   = 0.;
  outtotalSamples2_     = 0;
  outduration2_         = 0.;
  outmeanBandwidth2_    = 0.;
  outmeanRate2_         = 0.;
  outmeanLatency2_      = 0.;
  outmaxBandwidth2_     = 0.;
  outminBandwidth2_     = 999999.;

  outpmeter_ = new stor::SMPerformanceMeter();
  outpmeter_->init(samples_, period4samples_);

  //string        xmlClass = getApplicationDescriptor()->getClassName();
  //unsigned long instance = getApplicationDescriptor()->getInstance();
  //ostringstream sourcename;
  // sourcename << xmlClass << "_" << instance;
  //sourcename << instance;
  //sourceId_ = sourcename.str();
  //smParameter_ -> setSmInstance(sourceId_);  // sourceId_ can be removed ...

  // Need this to deserialize the streamer data
  edm::RootAutoLibraryLoader::enable();
}

SMProxyServer::~SMProxyServer()
{
  delete ah_;
  delete outpmeter_;
}

xoap::MessageReference
SMProxyServer::ParameterGet(xoap::MessageReference message)
  throw (xoap::exception::Exception)
{
  connectedSMs_.value_ = smsenders_.size();
  return Application::ParameterGet(message);
}




//////////// ***  Performance //////////////////////////////////////////////////////////
void SMProxyServer::addMeasurement(unsigned long size)
{
  // for input bandwidth performance measurements
  if (dpm_.get() != NULL)
  {
    dpm_->addMeasurement(size);
  }
}

void SMProxyServer::addOutMeasurement(unsigned long size)
{
// for bandwidth performance measurements, first sample based
  if ( outpmeter_->addSample(size) )
  {
    // Copy measurements for our record
    stor::SMPerfStats stats = outpmeter_->getStats();
    outinstantBandwidth_= stats.shortTermCounter_->getValueRate();
    outinstantRate_     = stats.shortTermCounter_->getSampleRate();
    outinstantLatency_  = 1000000.0 / outinstantRate_;
    double now = ForeverCounter::getCurrentTime();
    outtotalSamples_    = stats.longTermCounter_->getSampleCount();
    outduration_        = stats.longTermCounter_->getDuration(now);
    outmeanBandwidth_   = stats.longTermCounter_->getValueRate(now);
    outmeanRate_        = stats.longTermCounter_->getSampleRate(now);
    outmeanLatency_     = 1000000.0 / outmeanRate_;
    outmaxBandwidth_    = stats.maxBandwidth_;
    outminBandwidth_    = stats.minBandwidth_;
  }

  // for time period bandwidth performance measurements
  if ( outpmeter_->getStats().shortPeriodCounter_->hasValidResult() )
  {
    // Copy measurements for our record
    stor::SMPerfStats stats = outpmeter_->getStats();
    outinstantBandwidth2_= stats.shortPeriodCounter_->getValueRate();
    outinstantRate2_     = stats.shortPeriodCounter_->getSampleRate();
    outinstantLatency2_  = 1000000.0 / outinstantRate2_;
    double now = ForeverCounter::getCurrentTime();
    outtotalSamples2_    = stats.longTermCounter_->getSampleCount();
    outduration2_        = stats.longTermCounter_->getDuration(now);
    outmeanBandwidth2_   = stats.longTermCounter_->getValueRate(now);
    outmeanRate2_        = stats.longTermCounter_->getSampleRate(now);
    outmeanLatency2_     = 1000000.0 / outmeanRate2_;
    outmaxBandwidth2_    = stats.maxBandwidth2_;
    outminBandwidth2_    = stats.minBandwidth2_;
  }
  outreceivedVolume_ = outpmeter_->totalvolumemb();

}

//////////// *** Default web page //////////////////////////////////////////////////////////
void SMProxyServer::defaultWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
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
    *out << "     src=\"/rubuilder/fu/images/fu64x64.gif\""     << endl;
    *out << "     alt=\"main\""                                        << endl;
    *out << "     width=\"64\""                                        << endl;
    *out << "     height=\"64\""                                       << endl;
    *out << "     border=\"\"/>"                                       << endl;
    *out << "    <b>"                                                  << endl;
    *out << getApplicationDescriptor()->getClassName() << " instance "
         << getApplicationDescriptor()->getInstance()                  << endl;
    *out << "      " << fsm_.stateName()->toString()                   << endl;
    *out << "    </b>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/hyperdaq/images/HyperDAQ.jpg\""    << endl;
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
    *out << "       src=\"/rubuilder/fu/images/debug32x32.gif\""         << endl;
    *out << "       alt=\"debug\""                                     << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                     << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "</tr>"                                                    << endl;
    if(fsm_.stateName()->value_ == "Failed")
    {
      *out << "<tr>"                                         << endl;
      *out << " <td>"                                        << endl;
      *out << "<textarea rows=" << 5 << " cols=60 scroll=yes";
      *out << " readonly title=\"Reason For Failed\">"               << endl;
      *out << reasonForFailedState_                                  << endl;
      *out << "</textarea>"                                          << endl;
      *out << " </td>"                                       << endl;
      *out << "</tr>"                                        << endl;
    }
    *out << "</table>"                                                 << endl;

  *out << "<hr/>"                                                    << endl;
  *out << "<table>"                                                  << endl;
  *out << "<tr valign=\"top\">"                                      << endl;
  *out << "  <td>"                                                   << endl;

  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << endl;
  *out << "<colgroup> <colgroup align=\"right\">"                    << endl;
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Input and Output Statistics"                  << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;

    if (dpm_.get() != NULL)
    {
      receivedEvents_ = dpm_->receivedevents();
      receivedDQMEvents_ = dpm_->receivedDQMevents();
      stor::SMPerfStats stats = dpm_->getStats();

      instantBandwidth_= stats.shortTermCounter_->getValueRate();
      instantRate_     = stats.shortTermCounter_->getSampleRate();
      instantLatency_  = 1000000.0 / instantRate_;
      double now = ForeverCounter::getCurrentTime();
      totalSamples_    = stats.longTermCounter_->getSampleCount();
      duration_        = stats.longTermCounter_->getDuration(now);
      meanBandwidth_   = stats.longTermCounter_->getValueRate(now);
      meanRate_        = stats.longTermCounter_->getSampleRate(now);
      meanLatency_     = 1000000.0 / meanRate_;
      maxBandwidth_    = stats.maxBandwidth_;
      minBandwidth_    = stats.minBandwidth_;
      // for time period bandwidth performance measurements
      if ( stats.shortPeriodCounter_->hasValidResult() )
      {
        instantBandwidth2_= stats.shortPeriodCounter_->getValueRate();
        instantRate2_     = stats.shortPeriodCounter_->getSampleRate();
        instantLatency2_  = 1000000.0 / instantRate2_;
        double now = ForeverCounter::getCurrentTime();
        totalSamples2_    = stats.longTermCounter_->getSampleCount();
        duration2_        = stats.longTermCounter_->getDuration(now);
        meanBandwidth2_   = stats.longTermCounter_->getValueRate(now);
        meanRate2_        = stats.longTermCounter_->getSampleRate(now);
        meanLatency2_     = 1000000.0 / meanRate2_;
        maxBandwidth2_    = stats.maxBandwidth2_;
        minBandwidth2_    = stats.minBandwidth2_;
      }
      receivedVolume_ = dpm_->totalvolumemb();
    }

        *out << "<tr>" << endl;
        *out << "<th >" << endl;
        *out << "Parameter" << endl;
        *out << "</th>" << endl;
        *out << "<th>" << endl;
        *out << "Value" << endl;
        *out << "</th>" << endl;
        *out << "</tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Events Received" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << receivedEvents_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "DQMEvents Received" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << receivedDQMEvents_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "DQMEvents Stored" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << storedDQMEvents_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Events sent to consumers" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << sentEvents_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "DQMEvents sent to consumers" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << sentDQMEvents_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
// performance statistics
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Input Performance for last " << samples_ << " HTTP posts" << " (and last " << period4samples_ << " sec)" << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << instantBandwidth_ << " (" << instantBandwidth2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Rate (Posts/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << instantRate_ << " (" << instantRate2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Latency (us/post)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << instantLatency_ << " (" << instantLatency2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Maximum Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << maxBandwidth_ << " (" << maxBandwidth2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Minimum Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << minBandwidth_ << " (" << minBandwidth2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
// mean performance statistics for whole run
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Mean Performance for " << totalSamples_ << " (" << totalSamples2_ << ")" << " posts, duration "
         << duration_ << " (" << duration2_ << ")" << " seconds" << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << meanBandwidth_ << " (" << meanBandwidth2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Rate (Posts/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << meanRate_ << " (" << meanRate2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Latency (us/post)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << meanLatency_ << " (" << meanLatency2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Data Volume (MB)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << receivedVolume_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
// performance statistics
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Output Performance for last " << samples_ << " HTTP posts"<< " (and last " << period4samples_ << " sec)" << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outinstantBandwidth_ << " (" << outinstantBandwidth2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Rate (Posts/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outinstantRate_ << " (" << outinstantRate2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Latency (us/post)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outinstantLatency_ << " (" << outinstantLatency2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Maximum Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outmaxBandwidth_ << " (" << outmaxBandwidth2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Minimum Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outminBandwidth_ << " (" << outminBandwidth2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
// mean performance statistics for whole run
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Mean Performance for " << outtotalSamples_ << " (" << outtotalSamples2_ << ")" << " posts, duration "
         << outduration_ << " seconds" << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outmeanBandwidth_ << " (" << outmeanBandwidth2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Rate (Posts/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outmeanRate_ << " (" << outmeanRate2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Latency (us/post)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outmeanLatency_ << " (" << outmeanLatency2_ << ")" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Data Volume (MB)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outreceivedVolume_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;


  *out << "</table>" << endl;

  *out << "  </td>"                                                  << endl;
  *out << "</table>"                                                 << endl;
// now for SM sender list statistics
  *out << "<hr/>"                                                    << endl;
  *out << "<table>"                                                  << endl;
  *out << "<tr valign=\"top\">"                                      << endl;
  *out << "  <td>"                                                   << endl;

  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << endl;
  *out << "<colgroup> <colgroup align=\"rigth\">"                    << endl;
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "SM Sender Information"                            << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;

    *out << "<tr>" << endl;
    *out << "<th >" << endl;
    *out << "Parameter" << endl;
    *out << "</th>" << endl;
    *out << "<th>" << endl;
    *out << "Value" << endl;
    *out << "</th>" << endl;
    *out << "</tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Number of SM Senders" << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          *out << smsenders_.size() << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;

  *out << "</table>" << endl;

  *out << "  </td>"                                                  << endl;
  *out << "</table>"                                                 << endl;
  //---- separate pages for FU senders and Streamer Output
  *out << "<hr/>"                                                 << endl;
  std::string url = getApplicationDescriptor()->getContextDescriptor()->getURL();
  std::string urn = getApplicationDescriptor()->getURN();
  *out << "<a href=\"" << url << "/" << urn << "/smsenderlist" << "\">" 
       << "SM Sender list web page" << "</a>" << endl;
  *out << "<hr/>"                                                 << endl;
  *out << "<a href=\"" << url << "/" << urn << "/DQMoutputStatus" << "\">" 
       << "DQM Output Status web page" << "</a>" << endl;
  *out << "<hr/>"                                                 << endl;
  *out << "<a href=\"" << url << "/" << urn << "/EventServerStats?update=off"
       << "\">Event Server Statistics" << "</a>" << endl;

  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;
}


//////////// *** smsender web page //////////////////////////////////////////////////////////
void SMProxyServer::smsenderWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  bool senderReady = false;
  if (dpm_.get() != NULL ){
    if (dpm_->isFullyRegistered()){
      std::vector<std::string> smList = dpm_->getSmList();
      for (unsigned int i=0;i<smList.size();i++){
        smsenders_[smList[i]]=true;  
      }
      senderReady = true;
    }
  }
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
    *out << "     src=\"/rubuilder/fu/images/fu64x64.gif\""     << endl;
    *out << "     alt=\"main\""                                        << endl;
    *out << "     width=\"64\""                                        << endl;
    *out << "     height=\"64\""                                       << endl;
    *out << "     border=\"\"/>"                                       << endl;
    *out << "    <b>"                                                  << endl;
    *out << getApplicationDescriptor()->getClassName() << " instance "
         << getApplicationDescriptor()->getInstance()                  << endl;
    *out << "      " << fsm_.stateName()->toString()                   << endl;
    *out << "    </b>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/hyperdaq/images/HyperDAQ.jpg\""    << endl;
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
    *out << "       src=\"/rubuilder/fu/images/debug32x32.gif\""         << endl;
    *out << "       alt=\"debug\""                                     << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                     << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "</tr>"                                                    << endl;
    if(fsm_.stateName()->value_ == "Failed")
    {
      *out << "<tr>"                                         << endl;
      *out << " <td>"                                        << endl;
      *out << "<textarea rows=" << 5 << " cols=60 scroll=yes";
      *out << " readonly title=\"Reason For Failed\">"               << endl;
      *out << reasonForFailedState_                                  << endl;
      *out << "</textarea>"                                          << endl;
      *out << " </td>"                                       << endl;
      *out << "</tr>"                                        << endl;
    }
    *out << "</table>"                                                 << endl;

  *out << "<hr/>"                                                    << endl;

// now for SM sender list statistics
  *out << "<table>"                                                  << endl;
  *out << "<tr valign=\"top\">"                                      << endl;
  *out << "  <td>"                                                   << endl;

  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << endl;
  *out << "<colgroup> <colgroup align=\"rigth\">"                    << endl;
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "FU Sender List"                            << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;

    *out << "<tr>" << endl;
    *out << "<th >" << endl;
    *out << "Parameter" << endl;
    *out << "</th>" << endl;
    *out << "<th>" << endl;
    *out << "Value" << endl;
    *out << "</th>" << endl;
    *out << "</tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Number of SM Senders" << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          *out << smsenders_.size() << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
    if(smsenders_.size() > 0) {
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "SM Sender URL" << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          *out << "Registered?" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
    }
    std::map< std::string, bool >::iterator si(smsenders_.begin()), se(smsenders_.end());
    for( ; si != se; ++si) {
        *out << "<tr>" << endl;
          *out << "<td>" << endl;
          *out << si->first << endl;
          *out << "</td>" << endl;
          *out << "<td>" << endl;
          if(si->second && senderReady) 
            *out << "Yes" << endl;
          else if (senderReady)
            *out << "No" << endl;
          else
            *out << "Waiting for all SMs to be ready" << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
    }

  *out << "</table>" << endl;

  *out << "  </td>"                                                  << endl;
  *out << "</table>"                                                 << endl;

  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;
}


//////////// *** streamer file output web page //////////////////////////////////////////////////////////
void SMProxyServer::DQMOutputWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
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
    *out << "     src=\"/rubuilder/fu/images/fu64x64.gif\""     << endl;
    *out << "     alt=\"main\""                                        << endl;
    *out << "     width=\"64\""                                        << endl;
    *out << "     height=\"64\""                                       << endl;
    *out << "     border=\"\"/>"                                       << endl;
    *out << "    <b>"                                                  << endl;
    *out << getApplicationDescriptor()->getClassName() << " instance "
         << getApplicationDescriptor()->getInstance()                  << endl;
    *out << "      " << fsm_.stateName()->toString()                   << endl;
    *out << "    </b>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/hyperdaq/images/HyperDAQ.jpg\""    << endl;
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
    *out << "       src=\"/rubuilder/fu/images/debug32x32.gif\""         << endl;
    *out << "       alt=\"debug\""                                     << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                     << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "</tr>"                                                    << endl;
    if(fsm_.stateName()->value_ == "Failed")
    {
      *out << "<tr>"                                         << endl;
      *out << " <td>"                                        << endl;
      *out << "<textarea rows=" << 5 << " cols=60 scroll=yes";
      *out << " readonly title=\"Reason For Failed\">"               << endl;
      *out << reasonForFailedState_                                  << endl;
      *out << "</textarea>"                                          << endl;
      *out << " </td>"                                       << endl;
      *out << "</tr>"                                        << endl;
    }
    *out << "</table>"                                                 << endl;

    *out << "<hr/>"                                                    << endl;

  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;
}


//////////// *** get event data web page //////////////////////////////////////////////////////////
void SMProxyServer::eventdataWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  boost::mutex::scoped_lock ql(queue_lock_);
  // default the message length to zero
  int len=0;

  // determine the consumer ID from the event request
  // message, if it is available.
  unsigned int consumerId = 0;
  std::string lengthString = in->getenv("CONTENT_LENGTH");
  unsigned long contentLength = std::atol(lengthString.c_str());
  if (contentLength > 0) 
    {
      auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
      in->read(&(*bufPtr)[0], contentLength);
      OtherMessageView requestMessage(&(*bufPtr)[0]);
      if (requestMessage.code() == Header::EVENT_REQUEST)
	{
	  uint8 *bodyPtr = requestMessage.msgBody();
	  consumerId = convert32(bodyPtr);
	}
    }
  
  // first test if SMProxyServer is in Enabled state and registry is filled
  // this must be the case for valid data to be present
  bool haveHeaderAlready = false;
  if(dpm_.get() != NULL) haveHeaderAlready = dpm_->haveHeader();
  if(fsm_.stateName()->toString() == "Enabled" && haveHeaderAlready)
  {
    boost::shared_ptr<EventServer> eventServer;
    if (dpm_.get() != NULL)
    {
      eventServer = dpm_->getEventServer();
    }
    if (eventServer.get() != NULL)
    {
      // if we've stored a "registry warning" in the consumer pipe, send
      // that instead of an event so that the consumer can react to
      // the warning
      boost::shared_ptr<ConsumerPipe> consPtr =
        eventServer->getConsumer(consumerId);
      if (consPtr.get() != NULL && consPtr->hasRegistryWarning())
      {
        std::vector<char> registryWarning = consPtr->getRegistryWarning();
        const char* from = &registryWarning[0];
        unsigned int msize = registryWarning.size();
        if(mybuffer_.capacity() < msize) mybuffer_.resize(msize);
        unsigned char* pos = (unsigned char*) &mybuffer_[0];

        copy(from,from+msize,pos);
        len = msize;
        consPtr->clearRegistryWarning();
      }
      else
      {
        boost::shared_ptr< std::vector<char> > bufPtr =
          eventServer->getEvent(consumerId);
        if (bufPtr.get() != NULL)
        {
          EventMsgView msgView(&(*bufPtr)[0]);

          unsigned char* from = msgView.startAddress();
          unsigned int dsize = msgView.size();
          if(dsize > mybuffer_.capacity() ) mybuffer_.resize(dsize);
          unsigned char* pos = (unsigned char*) &mybuffer_[0];

          copy(from,from+dsize,pos);
          len = dsize;
          FDEBUG(10) << "sending event " << msgView.event() << std::endl;
          ++sentEvents_;
          addOutMeasurement(len);
        }
      }
    }

    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);
  } // else send DONE message as response
  else
  {
    OtherMessageBuilder othermsg(&mybuffer_[0],Header::DONE);
    len = othermsg.size();

    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);
  }

  //reset timeout counter if consumer keeps asking for events
    timeoutCounter_=queueTimeout_;
    queueInactive_=false;

}


//////////// *** callback for header request (registry) web page ////////////////////////////////////////
void SMProxyServer::headerdataWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  boost::mutex::scoped_lock ql(queue_lock_);
  unsigned int len = 0;

  // determine the consumer ID from the header request
  // message, if it is available.
  auto_ptr< vector<char> > httpPostData;
  unsigned int consumerId = 0;
  std::string lengthString = in->getenv("CONTENT_LENGTH");
  unsigned long contentLength = std::atol(lengthString.c_str());
  if (contentLength > 0) {
    auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
    in->read(&(*bufPtr)[0], contentLength);
    OtherMessageView requestMessage(&(*bufPtr)[0]);
    if (requestMessage.code() == Header::HEADER_REQUEST)
    {
      uint8 *bodyPtr = requestMessage.msgBody();
      consumerId = convert32(bodyPtr);
    }

    // save the post data for use outside the "if" block scope in case it is
    // useful later (it will still get deleted at the end of the method)
    httpPostData = bufPtr;
  }

  // check we are in the right state
  // first test if SMProxyServer is in Enabled state and registry is filled
  // this must be the case for valid data to be present
  if(fsm_.stateName()->toString() == "Enabled" && dpm_.get() != NULL &&
     dpm_->getInitMsgCollection().get() != NULL &&
     dpm_->getInitMsgCollection()->size() > 0)
    {
      std::string errorString;
      InitMsgSharedPtr serializedProds;
      boost::shared_ptr<EventServer> eventServer = dpm_->getEventServer();
      if (eventServer.get() != NULL)
      {
        boost::shared_ptr<ConsumerPipe> consPtr =
          eventServer->getConsumer(consumerId);
        if (consPtr.get() != NULL)
        {
          boost::shared_ptr<InitMsgCollection> initMsgCollection =
            dpm_->getInitMsgCollection();
          try
          {
            std::string hltOMLabel = consPtr->getHLTOutputSelection();
            serializedProds =
              initMsgCollection->getElementForOutputModule(hltOMLabel);
            if (serializedProds.get() != NULL)
            {
              Strings triggerNameList;
              InitMsgView initView(&(*serializedProds)[0]);
              initView.hltTriggerNames(triggerNameList);

              uint32 outputModuleId;
              if (initView.protocolVersion() >= 6) {
                outputModuleId = initView.outputModuleId();
              }
              else {
                std::string moduleLabel = initView.outputModuleLabel();
                uLong crc = crc32(0L, Z_NULL, 0);
                Bytef* crcbuf = (Bytef*) moduleLabel.data();
                crc = crc32(crc, crcbuf, moduleLabel.length());
                outputModuleId = static_cast<uint32>(crc);
              }
              consPtr->initializeSelection(triggerNameList,
                                           outputModuleId);
            }
          }
          catch (const edm::Exception& excpt)
          {
            errorString = excpt.what();
          }
          catch (const cms::Exception& excpt)
          {
            //errorString.append(excpt.what());
            errorString.append("ERROR: The configuration for this ");
            errorString.append("consumer does not specify an HLT output ");
            errorString.append("module.\nPlease specify one of the HLT ");
            errorString.append("output modules listed below as the ");
            errorString.append("SelectHLTOutput parameter ");
            errorString.append("in the InputSource configuration.\n");
            errorString.append(initMsgCollection->getSelectionHelpString());
            errorString.append("\n");
          }
        }
      }
      if (errorString.length() > 0) {
        len = errorString.length();
      }
      else if (serializedProds.get() != NULL) {
        len = serializedProds->size();
      }
      else {
        len = 0;
      }
      if (mybuffer_.capacity() < len) mybuffer_.resize(len);
      if (errorString.length() > 0) {
        const char *errorBytes = errorString.c_str();
        for (unsigned int i=0; i<len; ++i) mybuffer_[i]=errorBytes[i];
      }
      else if (serializedProds.get() != NULL) {
        for (unsigned int i=0; i<len; ++i) mybuffer_[i]=(*serializedProds)[i];
      }
    }

  out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
  out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
  out->write((char*) &mybuffer_[0],len);

  //reset timeout counter if consumer keeps asking for header
  timeoutCounter_=queueTimeout_;
  queueInactive_=false;

}


////////////////////////////// consumer registration web page ////////////////////////////
void SMProxyServer::consumerWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  // 04-Mar-2008, KAB - added the requirement that the DataProcessManager
  // has registered with the SM event server.  This is to ensure that we
  // have a copy of the stream-to-trigger-selection map before we accept
  // any consumer registrations and try to translate a (proxy) consumer 
  // stream-based selection request.  At some point, we should fix up the
  // tests on whether the dpm_ shared pointer is valid (can we even get here
  // without it being valid?)
  boost::mutex::scoped_lock ql(queue_lock_);

  if (!selectionFromClient_ || !(fsm_.stateName()->toString() == "Enabled")) {

    if (! (dpm_.get() != NULL 
	   && dpm_->haveRegWithEventServer() 
           && fsm_.stateName()->toString() == "Enabled"))
    {
      //write back response and finish
      out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
      out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
      out->write((char*) &mybuffer_[0],0);
      return;
    }
  }

  std::string consumerName = "None provided";
  std::string consumerRequest = "<>";
  std::string consumerHost = in->getenv("REMOTE_HOST");

  // read the consumer registration message from the http input stream
  std::string lengthString = in->getenv("CONTENT_LENGTH");
  unsigned long contentLength = std::atol(lengthString.c_str());
  if (contentLength > 0)
  {
    auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
    in->read(&(*bufPtr)[0], contentLength);
    ConsRegRequestView requestMessage(&(*bufPtr)[0]);
    consumerName = requestMessage.getConsumerName();
    std::string reqString = requestMessage.getRequestParameterSet();
    if (reqString.size() >= 2) consumerRequest = reqString;
  }

  // create the buffer to hold the registration reply message
  const int BUFFER_SIZE = 100;
  char msgBuff[BUFFER_SIZE];

  // fetch the event server
  // (it and/or the job controller may not have been created yet)

  // fetch the event selection request from the consumer request
  edm::ParameterSet requestParamSet(consumerRequest);

  // 26-Jan-2009, KAB: an ugly hack to get ParameterSet to serialize
  // the parameters that we need.  A better solution is in the works.
  try {
    double rate =
      requestParamSet.getUntrackedParameter<double>("maxEventRequestRate",
	  -999.0);
    if (rate == -999.0) {
      rate = requestParamSet.getParameter<double>("TrackedMaxRate");
      requestParamSet.addUntrackedParameter<double>("maxEventRequestRate",
	  rate);
    }
  }
  catch (...) {}
  try {
    std::string hltOMLabel =
      requestParamSet.getUntrackedParameter<std::string>("SelectHLTOutput",
	  "NoneFound");
    if (hltOMLabel == "NoneFound") {
      hltOMLabel =
	requestParamSet.getParameter<std::string>("TrackedHLTOutMod");
      requestParamSet.addUntrackedParameter<std::string>("SelectHLTOutput",
	  hltOMLabel);
    }
  }
  catch (...) {}
  try {
    edm::ParameterSet tmpPSet1 =
      requestParamSet.getUntrackedParameter<edm::ParameterSet>("SelectEvents",
	  edm::ParameterSet());
    if (tmpPSet1.empty()) {
      Strings path_specs = 
	requestParamSet.getParameter<Strings>("TrackedEventSelection");
      if (! path_specs.empty()) {
	edm::ParameterSet tmpPSet2;
	tmpPSet2.addParameter<Strings>("SelectEvents", path_specs);
	requestParamSet.addUntrackedParameter<edm::ParameterSet>("SelectEvents",
	    tmpPSet2);
      }
    }
  }
  catch (...) {}

  //read variables
  Strings selectionRequest =
    EventSelector::getEventSelectionVString(requestParamSet);


  // pull the rate request out of the consumer parameter set, too
  double maxEventRequestRate =
    requestParamSet.getUntrackedParameter<double>("maxEventRequestRate", 1.0);

  // pull the HLT output module selection out of the PSet
  // (default is empty string)
  std::string hltOMLabel =
    requestParamSet.getUntrackedParameter<std::string>("SelectHLTOutput", std::string());

  //get the optional TriggerSelector, which is tracked parameter
  std::string tTS_ = std::string();
  try { 
    tTS_ = requestParamSet.getParameter<std::string>("TriggerSelector");
  } 
  catch(...) {}

  if (selectionFromClient_) {

    if (queueCreated_ && !queueInactive_ && !alwaysRestartQueue_) {
      LOG4CPLUS_WARN(getApplicationLogger(),"Queue already exists, new client will be connected to it\n");
    }
    else {
      if (queueCreated_) destroyQueue();
      std::cout << "----Client parameters:----\n";
      maxEventRequestRate_ = maxEventRequestRate; 
      std::cout << "maxEventRequestRate: "<< maxEventRequestRate_ <<"\n";
      esSelectedHLTOutputModule_=hltOMLabel;
      std::cout << "hltOutputModule: " << esSelectedHLTOutputModule_.toString() <<"\n";

      TriggerSelector_ = tTS_;
      if (!TriggerSelector_.toString().empty()) 
	std::cout << "TriggerSelection: " << TriggerSelector_.toString() << std::endl;
      else {
	esSelectedEventSelection_.clear();
	for (size_t i=0;i<selectionRequest.size();i++) esSelectedEventSelection_.push_back(selectionRequest.at(i));
	std::cout << "SelectEvents:" <<"\n";
	for(unsigned int i = 0; i < esSelectedEventSelection_.elements(); ++i)
	  cout << " " << esSelectedEventSelection_[i].toString();
	cout << "." << std::endl;
      }
      std::cout << "----End Client Parameters----\n";

      createQueue();
      //start the Proxy
      dpm_->start();
    }
  }

  boost::shared_ptr<EventServer> eventServer = dpm_->getEventServer();

  // if no event server, tell the consumer that we're not ready
  if (eventServer.get() == NULL) {
    ConsRegResponseBuilder respMsg(msgBuff, BUFFER_SIZE,
	ConsRegResponseBuilder::ES_NOT_READY, 0);
    // debug message so that compiler thinks respMsg is used
    FDEBUG(20) << "Registration response size =  " << respMsg.size() << std::endl;
  }
  else {
    // create the local consumer interface and add it to the event server
    boost::shared_ptr<ConsumerPipe>
      consPtr(new ConsumerPipe(consumerName, "normal",
	    activeConsumerTimeout_.value_,
	    idleConsumerTimeout_.value_,
	    tTS_, selectionRequest,
	    maxEventRequestRate,
	    hltOMLabel,
	    consumerHost, consumerQueueSize_));
    eventServer->addConsumer(consPtr);

    // build the registration response into the message buffer
    ConsRegResponseBuilder respMsg(msgBuff, BUFFER_SIZE,
	0, consPtr->getConsumerId());
    // debug message so that compiler thinks respMsg is used
    FDEBUG(20) << "Registration response size =  " <<
      respMsg.size() << std::endl;
  }

  // send the response
  ConsRegResponseView responseMessage(msgBuff);
  unsigned int len = responseMessage.size();
  if(len > mybuffer_.capacity() ) mybuffer_.resize(len);
  for (int i=0; i<(int)len; i++) mybuffer_[i]=msgBuff[i];

  out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
  out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
  out->write((char*) &mybuffer_[0],len);
}

void SMProxyServer::consumerListWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  char buffer[65536];

  out->getHTTPResponseHeader().addHeader("Content-Type", "application/xml");
  sprintf(buffer,
	  "<?xml version=\"1.0\" encoding=\"iso-8859-1\"?>\n<Monitor>\n");
  out->write(buffer,strlen(buffer));

  if(fsm_.stateName()->toString() == "Enabled")
  {
    sprintf(buffer, "<ConsumerList>\n");
    out->write(buffer,strlen(buffer));

    boost::shared_ptr<EventServer> eventServer;
    if (dpm_.get() != NULL)
    {
      eventServer = dpm_->getEventServer();
    }
    if (eventServer.get() != NULL)
    {
      std::map< uint32, boost::shared_ptr<ConsumerPipe> > consumerTable = 
	eventServer->getConsumerTable();
      std::map< uint32, boost::shared_ptr<ConsumerPipe> >::const_iterator 
	consumerIter;
      for (consumerIter = consumerTable.begin();
	   consumerIter != consumerTable.end();
	   consumerIter++)
      {
	boost::shared_ptr<ConsumerPipe> consumerPipe = consumerIter->second;
	sprintf(buffer, "<Consumer>\n");
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Name>%s</Name>\n",
		consumerPipe->getConsumerName().c_str());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<ID>%d</ID>\n", consumerPipe->getConsumerId());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Time>%d</Time>\n", 
		(int)consumerPipe->getLastEventRequestTime());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Host>%s</Host>\n", 
		consumerPipe->getHostName().c_str());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Events>%d</Events>\n", consumerPipe->getEvents());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Failed>%d</Failed>\n", 
		consumerPipe->getPushEventFailures());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Idle>%d</Idle>\n", consumerPipe->isIdle());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Disconnected>%d</Disconnected>\n", 
		consumerPipe->isDisconnected());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Ready>%d</Ready>\n", consumerPipe->isReadyForEvent());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "</Consumer>\n");
	out->write(buffer,strlen(buffer));
      }
    }
    boost::shared_ptr<DQMEventServer> dqmServer;
    if (dpm_.get() != NULL)
    {
      dqmServer = dpm_->getDQMEventServer();
    }
    if (dqmServer.get() != NULL)
    {
      std::map< uint32, boost::shared_ptr<DQMConsumerPipe> > dqmTable = 
	dqmServer->getConsumerTable();
      std::map< uint32, boost::shared_ptr<DQMConsumerPipe> >::const_iterator 
	dqmIter;
      for (dqmIter = dqmTable.begin();
	   dqmIter != dqmTable.end();
	   dqmIter++)
      {
	boost::shared_ptr<DQMConsumerPipe> dqmPipe = dqmIter->second;
	sprintf(buffer, "<DQMConsumer>\n");
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Name>%s</Name>\n",
		dqmPipe->getConsumerName().c_str());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<ID>%d</ID>\n", dqmPipe->getConsumerId());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Time>%d</Time>\n", 
		(int)dqmPipe->getLastEventRequestTime());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Host>%s</Host>\n", 
		dqmPipe->getHostName().c_str());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Events>%d</Events>\n", dqmPipe->getEvents());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Failed>%d</Failed>\n", 
		dqmPipe->getPushEventFailures());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Idle>%d</Idle>\n", dqmPipe->isIdle());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Disconnected>%d</Disconnected>\n", 
		dqmPipe->isDisconnected());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "<Ready>%d</Ready>\n", dqmPipe->isReadyForEvent());
	out->write(buffer,strlen(buffer));

	sprintf(buffer, "</DQMConsumer>\n");
	out->write(buffer,strlen(buffer));
      }
    }
    sprintf(buffer, "</ConsumerList>\n");
    out->write(buffer,strlen(buffer));
  }
  sprintf(buffer, "</Monitor>");
  out->write(buffer,strlen(buffer));
}

//////////////////// event server statistics web page //////////////////
void SMProxyServer::eventServerWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  // We should make the HTML header and the page banner common
  std::string url =
    getApplicationDescriptor()->getContextDescriptor()->getURL();
  std::string urn = getApplicationDescriptor()->getURN();

  // determine whether we're automatically updating the page
  // --> if the SM is not enabled, assume that users want updating turned
  // --> ON so that they don't A) think that is is ON (when it's not) and
  // --> B) wait forever thinking that something is wrong.
  //bool autoUpdate = true;
  // 11-Jun-2008, KAB - changed auto update default to OFF
  bool autoUpdate = false;
  if(fsm_.stateName()->toString() == "Enabled") {
    cgicc::Cgicc cgiWrapper(in);
    cgicc::const_form_iterator updateRef = cgiWrapper.getElement("update");
    if (updateRef != cgiWrapper.getElements().end()) {
      std::string updateString =
        boost::algorithm::to_lower_copy(updateRef->getValue());
      if (updateString == "off") {
        autoUpdate = false;
      }
      else {
        autoUpdate = true;
      }
    }
  }

  *out << "<html>" << std::endl;
  *out << "<head>" << std::endl;
  if (autoUpdate) {
    *out << "<meta http-equiv=\"refresh\" content=\"10\">" << std::endl;
  }
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"/" << urn << "/styles.css\"/>" << std::endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName()
       << " Instance " << getApplicationDescriptor()->getInstance()
       << "</title>" << std::endl;
  *out << "<style type=\"text/css\">" << std::endl;
  *out << "  .noBotMarg {margin-bottom:0px;}" << std::endl;
  *out << "</style>" << std::endl;
  *out << "</head><body>" << std::endl;

  *out << "<table border=\"1\" width=\"100%\">"                      << endl;
  *out << "<tr>"                                                     << endl;
  *out << "  <td align=\"left\">"                                    << endl;
  *out << "    <img"                                                 << endl;
  *out << "     align=\"middle\""                                    << endl;
  *out << "     src=\"/evf/images/smicon.jpg\""                      << endl;
  *out << "     alt=\"main\""                                        << endl;
  *out << "     width=\"64\""                                        << endl;
  *out << "     height=\"64\""                                       << endl;
  *out << "     border=\"\"/>"                                       << endl;
  *out << "    <b>"                                                  << endl;
  *out << getApplicationDescriptor()->getClassName() << " Instance "
       << getApplicationDescriptor()->getInstance();
  *out << ", State is " << fsm_.stateName()->toString()              << endl;
  *out << "    </b>"                                                 << endl;
  *out << "  </td>"                                                  << endl;
  *out << "  <td width=\"32\">"                                      << endl;
  *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << endl;
  *out << "      <img"                                               << endl;
  *out << "       align=\"middle\""                                  << endl;
  *out << "       src=\"/hyperdaq/images/HyperDAQ.jpg\""             << endl;
  *out << "       alt=\"HyperDAQ\""                                  << endl;
  *out << "       width=\"32\""                                      << endl;
  *out << "       height=\"32\""                                     << endl;
  *out << "       border=\"\"/>"                                     << endl;
  *out << "    </a>"                                                 << endl;
  *out << "  </td>"                                                  << endl;
  *out << "</tr>"                                                    << endl;
  if(fsm_.stateName()->value_ == "Failed")
  {
    *out << "<tr>"                                                   << endl;
    *out << " <td>"                                                  << endl;
    *out << "<textarea rows=" << 5 << " cols=60 scroll=yes";
    *out << " readonly title=\"Reason For Failed\">"                 << endl;
    *out << reasonForFailedState_                                    << endl;
    *out << "</textarea>"                                            << endl;
    *out << " </td>"                                                 << endl;
    *out << "</tr>"                                                  << endl;
  }
  *out << "</table>"                                                 << endl;

  if(fsm_.stateName()->toString() == "Enabled")
  {
    boost::shared_ptr<EventServer> eventServer;
    boost::shared_ptr<InitMsgCollection> initMsgCollection;
    if (dpm_.get() != NULL)
    {
      eventServer = dpm_->getEventServer();
      initMsgCollection = dpm_->getInitMsgCollection();
    }
    if (eventServer.get() != NULL && initMsgCollection.get() != NULL)
    {
      if (initMsgCollection->size() > 0)
      {
        int displayedConsumerCount = 0;
        double eventSum = 0.0;
        double eventRateSum = 0.0;
        double dataRateSum = 0.0;

        double now = ForeverCounter::getCurrentTime();
        *out << "<table border=\"0\" width=\"100%\">" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <td width=\"25%\" align=\"center\">" << std::endl;
        *out << "  </td>" << std::endl;
        *out << "    &nbsp;" << std::endl;
        *out << "  <td width=\"50%\" align=\"center\">" << std::endl;
        *out << "    <font size=\"+2\"><b>Event Server Statistics</b></font>"
             << std::endl;
        *out << "    <br/>" << std::endl;
        *out << "    Data rates are reported in MB/sec." << std::endl;
        *out << "    <br/>" << std::endl;
        *out << "    Maximum input event rate is "
             << eventServer->getMaxEventRate() << " Hz." << std::endl;
        *out << "    <br/>" << std::endl;
        *out << "    Maximum input data rate is "
             << eventServer->getMaxDataRate() << " MB/sec." << std::endl;
        *out << "    <br/>" << std::endl;
        *out << "    Consumer queue size is " << consumerQueueSize_
             << "." << std::endl;
        *out << "    <br/>" << std::endl;
        *out << "    Event request rate from SMs is "
             << maxEventRequestRate_ << " Hz." << std::endl;
        //*out << "    <br/>" << std::endl;
        //*out << "    Fair-share event serving is ";
        //if (fairShareES_) {
        //  *out << "ON." << std::endl;
        //}
        //else {
        //  *out << "OFF." << std::endl;
        //}
        *out << "    <br/>" << std::endl;
        *out << "    Selected HLT output module is "
             << esSelectedHLTOutputModule_.toString()
             << "." << std::endl;
        if ( ! esSelectedEventSelection_.empty() || !TriggerSelector_.toString().empty())
        {
            *out << "    <br/>" << std::endl;
            *out << "    Selected Event Selection is";

	    if (!TriggerSelector_.toString().empty()) *out << " " << TriggerSelector_.toString();
	    else
            for(unsigned int i = 0; i < esSelectedEventSelection_.elements(); ++i)
                *out << " " << esSelectedEventSelection_[i].toString();
            *out << "." << std::endl;
        }
        *out << "  </td>" << std::endl;
        *out << "  <td width=\"25%\" align=\"center\">" << std::endl;
        if (autoUpdate) {
          *out << "    <a href=\"" << url << "/" << urn
               << "/EventServerStats?update=off\">Turn updating OFF</a>"
               << std::endl;
        }
        else {
          *out << "    <a href=\"" << url << "/" << urn
               << "/EventServerStats?update=on\">Turn updating ON</a>"
               << std::endl;
        }
        *out << "    <br/><br/>" << std::endl;
        *out << "    <a href=\"" << url << "/" << urn
             << "\">Back to Proxy Status</a>"
             << std::endl;
        *out << "  </td>" << std::endl;
        *out << "</tr>" << std::endl;
        *out << "</table>" << std::endl;

        *out << "<h3>Event Server:</h3>" << std::endl;
        *out << "<h4 class=\"noBotMarg\">Input Events, Recent Results:</h4>" << std::endl;
        *out << "<font size=\"-1\">(Events can be double-counted if they are sent by multiple output modules.)</font><br/><br/>" << std::endl;
        *out << "<table border=\"1\" width=\"100%\">" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <th>HLT Output Module</th>" << std::endl;
        *out << "  <th>Event Count</th>" << std::endl;
        *out << "  <th>Event Rate</th>" << std::endl;
        *out << "  <th>Data Rate</th>" << std::endl;
        *out << "  <th>Duration (sec)</th>" << std::endl;
        *out << "</tr>" << std::endl;

        eventSum = 0.0;
        eventRateSum = 0.0;
        dataRateSum = 0.0;
        for (int idx = 0; idx < initMsgCollection->size(); ++idx) {
          InitMsgSharedPtr serializedProds = initMsgCollection->getElementAt(idx);
          InitMsgView initView(&(*serializedProds)[0]);
          uint32 outputModuleId = initView.outputModuleId();

          eventSum += eventServer->getEventCount(EventServer::SHORT_TERM_STATS,
                                                 EventServer::INPUT_STATS,
                                                 outputModuleId, now);
          eventRateSum += eventServer->getEventRate(EventServer::SHORT_TERM_STATS,
                                                    EventServer::INPUT_STATS,
                                                    outputModuleId, now);
          dataRateSum += eventServer->getDataRate(EventServer::SHORT_TERM_STATS,
                                                  EventServer::INPUT_STATS,
                                                  outputModuleId, now);

          *out << "<tr>" << std::endl;
          *out << "  <td align=\"center\">" << initView.outputModuleLabel()
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getEventCount(EventServer::SHORT_TERM_STATS,
                                             EventServer::INPUT_STATS,
                                             outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getEventRate(EventServer::SHORT_TERM_STATS,
                                            EventServer::INPUT_STATS,
                                            outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getDataRate(EventServer::SHORT_TERM_STATS,
                                           EventServer::INPUT_STATS,
                                           outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getDuration(EventServer::SHORT_TERM_STATS,
                                           EventServer::INPUT_STATS,
                                           outputModuleId, now)
               << "</td>" << std::endl;
          *out << "</tr>" << std::endl;
        }

        // add a row with the totals
        if (initMsgCollection->size() > 1) {
          *out << "<tr>" << std::endl;
          *out << "  <td align=\"center\">Totals</td>" << std::endl;
          *out << "  <td align=\"center\">" << eventSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">" << eventRateSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">" << dataRateSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
          *out << "</tr>" << std::endl;
        }
        *out << "</table>" << std::endl;

        *out << "<h4 class=\"noBotMarg\">Accepted Unique Events, Recent Results:</h4>" << std::endl;
        *out << "<font size=\"-1\">(Events can be double-counted if they are sent by multiple output modules.)</font><br/><br/>" << std::endl;
        *out << "<table border=\"1\" width=\"100%\">" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <th>HLT Output Module</th>" << std::endl;
        *out << "  <th>Event Count</th>" << std::endl;
        *out << "  <th>Event Rate</th>" << std::endl;
        *out << "  <th>Data Rate</th>" << std::endl;
        *out << "  <th>Duration (sec)</th>" << std::endl;
        *out << "</tr>" << std::endl;

        eventSum = 0.0;
        eventRateSum = 0.0;
        dataRateSum = 0.0;
        for (int idx = 0; idx < initMsgCollection->size(); ++idx) {
          InitMsgSharedPtr serializedProds = initMsgCollection->getElementAt(idx);
          InitMsgView initView(&(*serializedProds)[0]);
          uint32 outputModuleId = initView.outputModuleId();

          eventSum += eventServer->getEventCount(EventServer::SHORT_TERM_STATS,
                                                 EventServer::UNIQUE_ACCEPT_STATS,
                                                 outputModuleId, now);
          eventRateSum += eventServer->getEventRate(EventServer::SHORT_TERM_STATS,
                                                    EventServer::UNIQUE_ACCEPT_STATS,
                                                    outputModuleId, now);
          dataRateSum += eventServer->getDataRate(EventServer::SHORT_TERM_STATS,
                                                  EventServer::UNIQUE_ACCEPT_STATS,
                                                  outputModuleId, now);

          *out << "<tr>" << std::endl;
          *out << "  <td align=\"center\">" << initView.outputModuleLabel()
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getEventCount(EventServer::SHORT_TERM_STATS,
                                             EventServer::UNIQUE_ACCEPT_STATS,
                                             outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getEventRate(EventServer::SHORT_TERM_STATS,
                                            EventServer::UNIQUE_ACCEPT_STATS,
                                            outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getDataRate(EventServer::SHORT_TERM_STATS,
                                           EventServer::UNIQUE_ACCEPT_STATS,
                                           outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getDuration(EventServer::SHORT_TERM_STATS,
                                           EventServer::UNIQUE_ACCEPT_STATS,
                                           outputModuleId, now)
               << "</td>" << std::endl;
          *out << "</tr>" << std::endl;
        }

        // add a row with the totals
        if (initMsgCollection->size() > 1) {
          *out << "<tr>" << std::endl;
          *out << "  <td align=\"center\">Totals</td>" << std::endl;
          *out << "  <td align=\"center\">" << eventSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">" << eventRateSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">" << dataRateSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
          *out << "</tr>" << std::endl;
        }
        *out << "</table>" << std::endl;

        *out << "<h4 class=\"noBotMarg\">Accepted Events To All Consumers, Recent Results:</h4>" << std::endl;
        *out << "<font size=\"-1\">(Events can be double-counted if they are sent by multiple output modules or if they are sent to multiple consumers.)</font><br/><br/>" << std::endl;
        *out << "<table border=\"1\" width=\"100%\">" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <th>HLT Output Module</th>" << std::endl;
        *out << "  <th>Event Count</th>" << std::endl;
        *out << "  <th>Event Rate</th>" << std::endl;
        *out << "  <th>Data Rate</th>" << std::endl;
        *out << "  <th>Duration (sec)</th>" << std::endl;
        *out << "</tr>" << std::endl;

        eventSum = 0.0;
        eventRateSum = 0.0;
        dataRateSum = 0.0;
        for (int idx = 0; idx < initMsgCollection->size(); ++idx) {
          InitMsgSharedPtr serializedProds = initMsgCollection->getElementAt(idx);
          InitMsgView initView(&(*serializedProds)[0]);
          uint32 outputModuleId = initView.outputModuleId();

          eventSum += eventServer->getEventCount(EventServer::SHORT_TERM_STATS,
                                                 EventServer::OUTPUT_STATS,
                                                 outputModuleId, now);
          eventRateSum += eventServer->getEventRate(EventServer::SHORT_TERM_STATS,
                                                    EventServer::OUTPUT_STATS,
                                                    outputModuleId, now);
          dataRateSum += eventServer->getDataRate(EventServer::SHORT_TERM_STATS,
                                                  EventServer::OUTPUT_STATS,
                                                  outputModuleId, now);

          *out << "<tr>" << std::endl;
          *out << "  <td align=\"center\">" << initView.outputModuleLabel()
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getEventCount(EventServer::SHORT_TERM_STATS,
                                             EventServer::OUTPUT_STATS,
                                             outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getEventRate(EventServer::SHORT_TERM_STATS,
                                            EventServer::OUTPUT_STATS,
                                            outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getDataRate(EventServer::SHORT_TERM_STATS,
                                           EventServer::OUTPUT_STATS,
                                           outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getDuration(EventServer::SHORT_TERM_STATS,
                                           EventServer::OUTPUT_STATS,
                                           outputModuleId, now)
               << "</td>" << std::endl;
          *out << "</tr>" << std::endl;
        }

        // add a row with the totals
        if (initMsgCollection->size() > 1) {
          *out << "<tr>" << std::endl;
          *out << "  <td align=\"center\">Totals</td>" << std::endl;
          *out << "  <td align=\"center\">" << eventSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">" << eventRateSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">" << dataRateSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
          *out << "</tr>" << std::endl;
        }
        *out << "</table>" << std::endl;

        *out << "<h4 class=\"noBotMarg\">Input Events, Full Results:</h4>" << std::endl;
        *out << "<font size=\"-1\">(Events can be double-counted if they are sent by multiple output modules.)</font><br/><br/>" << std::endl;
        *out << "<table border=\"1\" width=\"100%\">" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <th>HLT Output Module</th>" << std::endl;
        *out << "  <th>Event Count</th>" << std::endl;
        *out << "  <th>Event Rate</th>" << std::endl;
        *out << "  <th>Data Rate</th>" << std::endl;
        *out << "  <th>Duration (sec)</th>" << std::endl;
        *out << "</tr>" << std::endl;

        eventSum = 0.0;
        eventRateSum = 0.0;
        dataRateSum = 0.0;
        for (int idx = 0; idx < initMsgCollection->size(); ++idx) {
          InitMsgSharedPtr serializedProds = initMsgCollection->getElementAt(idx);
          InitMsgView initView(&(*serializedProds)[0]);
          uint32 outputModuleId = initView.outputModuleId();

          eventSum += eventServer->getEventCount(EventServer::LONG_TERM_STATS,
                                                 EventServer::INPUT_STATS,
                                                 outputModuleId, now);
          eventRateSum += eventServer->getEventRate(EventServer::LONG_TERM_STATS,
                                                    EventServer::INPUT_STATS,
                                                    outputModuleId, now);
          dataRateSum += eventServer->getDataRate(EventServer::LONG_TERM_STATS,
                                                  EventServer::INPUT_STATS,
                                                  outputModuleId, now);

          *out << "<tr>" << std::endl;
          *out << "  <td align=\"center\">" << initView.outputModuleLabel()
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getEventCount(EventServer::LONG_TERM_STATS,
                                             EventServer::INPUT_STATS,
                                             outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getEventRate(EventServer::LONG_TERM_STATS,
                                            EventServer::INPUT_STATS,
                                            outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getDataRate(EventServer::LONG_TERM_STATS,
                                           EventServer::INPUT_STATS,
                                           outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getDuration(EventServer::LONG_TERM_STATS,
                                           EventServer::INPUT_STATS,
                                           outputModuleId, now)
               << "</td>" << std::endl;
          *out << "</tr>" << std::endl;
        }

        // add a row with the totals
        if (initMsgCollection->size() > 1) {
          *out << "<tr>" << std::endl;
          *out << "  <td align=\"center\">Totals</td>" << std::endl;
          *out << "  <td align=\"center\">" << eventSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">" << eventRateSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">" << dataRateSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
          *out << "</tr>" << std::endl;
        }
        *out << "</table>" << std::endl;

        *out << "<h4 class=\"noBotMarg\">Accepted Unique Events, Full Results:</h4>" << std::endl;
        *out << "<font size=\"-1\">(Events can be double-counted if they are sent by multiple output modules.)</font><br/><br/>" << std::endl;
        *out << "<table border=\"1\" width=\"100%\">" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <th>HLT Output Module</th>" << std::endl;
        *out << "  <th>Event Count</th>" << std::endl;
        *out << "  <th>Event Rate</th>" << std::endl;
        *out << "  <th>Data Rate</th>" << std::endl;
        *out << "  <th>Duration (sec)</th>" << std::endl;
        *out << "</tr>" << std::endl;

        eventSum = 0.0;
        eventRateSum = 0.0;
        dataRateSum = 0.0;
        for (int idx = 0; idx < initMsgCollection->size(); ++idx) {
          InitMsgSharedPtr serializedProds = initMsgCollection->getElementAt(idx);
          InitMsgView initView(&(*serializedProds)[0]);
          uint32 outputModuleId = initView.outputModuleId();

          eventSum += eventServer->getEventCount(EventServer::LONG_TERM_STATS,
                                                 EventServer::UNIQUE_ACCEPT_STATS,
                                                 outputModuleId, now);
          eventRateSum += eventServer->getEventRate(EventServer::LONG_TERM_STATS,
                                                    EventServer::UNIQUE_ACCEPT_STATS,
                                                    outputModuleId, now);
          dataRateSum += eventServer->getDataRate(EventServer::LONG_TERM_STATS,
                                                  EventServer::UNIQUE_ACCEPT_STATS,
                                                  outputModuleId, now);

          *out << "<tr>" << std::endl;
          *out << "  <td align=\"center\">" << initView.outputModuleLabel()
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getEventCount(EventServer::LONG_TERM_STATS,
                                             EventServer::UNIQUE_ACCEPT_STATS,
                                             outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getEventRate(EventServer::LONG_TERM_STATS,
                                            EventServer::UNIQUE_ACCEPT_STATS,
                                            outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getDataRate(EventServer::LONG_TERM_STATS,
                                           EventServer::UNIQUE_ACCEPT_STATS,
                                           outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getDuration(EventServer::LONG_TERM_STATS,
                                           EventServer::UNIQUE_ACCEPT_STATS,
                                           outputModuleId, now)
               << "</td>" << std::endl;
          *out << "</tr>" << std::endl;
        }

        // add a row with the totals
        if (initMsgCollection->size() > 1) {
          *out << "<tr>" << std::endl;
          *out << "  <td align=\"center\">Totals</td>" << std::endl;
          *out << "  <td align=\"center\">" << eventSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">" << eventRateSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">" << dataRateSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
          *out << "</tr>" << std::endl;
        }
        *out << "</table>" << std::endl;

        *out << "<h4 class=\"noBotMarg\">Accepted Events To All Consumers, Full Results:</h4>" << std::endl;
        *out << "<font size=\"-1\">(Events can be double-counted if they are sent by multiple output modules or if they are sent to multiple consumers.)</font><br/><br/>" << std::endl;
        *out << "<table border=\"1\" width=\"100%\">" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <th>HLT Output Module</th>" << std::endl;
        *out << "  <th>Event Count</th>" << std::endl;
        *out << "  <th>Event Rate</th>" << std::endl;
        *out << "  <th>Data Rate</th>" << std::endl;
        *out << "  <th>Duration (sec)</th>" << std::endl;
        *out << "</tr>" << std::endl;

        eventSum = 0.0;
        eventRateSum = 0.0;
        dataRateSum = 0.0;
        for (int idx = 0; idx < initMsgCollection->size(); ++idx) {
          InitMsgSharedPtr serializedProds = initMsgCollection->getElementAt(idx);
          InitMsgView initView(&(*serializedProds)[0]);
          uint32 outputModuleId = initView.outputModuleId();

          eventSum += eventServer->getEventCount(EventServer::LONG_TERM_STATS,
                                                 EventServer::OUTPUT_STATS,
                                                 outputModuleId, now);
          eventRateSum += eventServer->getEventRate(EventServer::LONG_TERM_STATS,
                                                    EventServer::OUTPUT_STATS,
                                                    outputModuleId, now);
          dataRateSum += eventServer->getDataRate(EventServer::LONG_TERM_STATS,
                                                  EventServer::OUTPUT_STATS,
                                                  outputModuleId, now);

          *out << "<tr>" << std::endl;
          *out << "  <td align=\"center\">" << initView.outputModuleLabel()
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getEventCount(EventServer::LONG_TERM_STATS,
                                             EventServer::OUTPUT_STATS,
                                             outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getEventRate(EventServer::LONG_TERM_STATS,
                                            EventServer::OUTPUT_STATS,
                                            outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getDataRate(EventServer::LONG_TERM_STATS,
                                           EventServer::OUTPUT_STATS,
                                           outputModuleId, now)
               << "</td>" << std::endl;
          *out << "  <td align=\"center\">"
               << eventServer->getDuration(EventServer::LONG_TERM_STATS,
                                           EventServer::OUTPUT_STATS,
                                           outputModuleId, now)
               << "</td>" << std::endl;
          *out << "</tr>" << std::endl;
        }

        // add a row with the totals
        if (initMsgCollection->size() > 1) {
          *out << "<tr>" << std::endl;
          *out << "  <td align=\"center\">Totals</td>" << std::endl;
          *out << "  <td align=\"center\">" << eventSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">" << eventRateSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">" << dataRateSum << "</td>" << std::endl;
          *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
          *out << "</tr>" << std::endl;
        }
        *out << "</table>" << std::endl;

        *out << "<h4>Timing:</h4>" << std::endl;
        *out << "<table border=\"1\" width=\"100%\">" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <th>&nbsp;</th>" << std::endl;
        *out << "  <th>CPU Time<br/>(sec)</th>" << std::endl;
        *out << "  <th>CPU Time<br/>Percent</th>" << std::endl;
        *out << "  <th>Real Time<br/>(sec)</th>" << std::endl;
        *out << "  <th>Real Time<br/>Percent</th>" << std::endl;
        *out << "  <th>Duration (sec)</th>" << std::endl;
        *out << "</tr>" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <td align=\"center\">Recent Results</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << eventServer->getInternalTime(EventServer::SHORT_TERM_STATS,
                                             EventServer::CPUTIME,
                                             now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << 100 * eventServer->getTimeFraction(EventServer::SHORT_TERM_STATS,
                                                   EventServer::CPUTIME,
                                                   now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << eventServer->getInternalTime(EventServer::SHORT_TERM_STATS,
                                             EventServer::REALTIME,
                                             now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << 100 * eventServer->getTimeFraction(EventServer::SHORT_TERM_STATS,
                                                   EventServer::REALTIME,
                                                   now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << eventServer->getTotalTime(EventServer::SHORT_TERM_STATS,
                                          EventServer::REALTIME,
                                          now)
             << "</td>" << std::endl;
        *out << "</tr>" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <td align=\"center\">Full Results</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << eventServer->getInternalTime(EventServer::LONG_TERM_STATS,
                                             EventServer::CPUTIME,
                                             now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << 100 * eventServer->getTimeFraction(EventServer::LONG_TERM_STATS,
                                                   EventServer::CPUTIME,
                                                   now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << eventServer->getInternalTime(EventServer::LONG_TERM_STATS,
                                             EventServer::REALTIME,
                                             now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << 100 * eventServer->getTimeFraction(EventServer::LONG_TERM_STATS,
                                                   EventServer::REALTIME,
                                                   now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << eventServer->getTotalTime(EventServer::LONG_TERM_STATS,
                                          EventServer::REALTIME,
                                          now)
             << "</td>" << std::endl;
        *out << "</tr>" << std::endl;
        *out << "</table>" << std::endl;

        *out << "<h3>Consumers:</h3>" << std::endl;
        std::map< uint32, boost::shared_ptr<ConsumerPipe> > consumerTable = 
          eventServer->getConsumerTable();
        if (consumerTable.size() == 0)
        {
          *out << "No consumers are currently registered with "
               << "this SMProxyServer instance.<br/>" << std::endl;
        }
        else
        {
          std::map< uint32, boost::shared_ptr<ConsumerPipe> >::const_iterator 
            consumerIter;

          // ************************************************************
          // * Consumer summary table
          // ************************************************************
          *out << "<h4>Summary:</h4>" << std::endl;
          *out << "<table border=\"1\" width=\"100%\">" << std::endl;
          *out << "<tr>" << std::endl;
          *out << "  <th>ID</th>" << std::endl;
          *out << "  <th>Name</th>" << std::endl;
          *out << "  <th>State</th>" << std::endl;
          *out << "  <th>Requested<br/>Rate</th>" << std::endl;
          *out << "  <th>Requested HLT<br/>Output Module</th>" << std::endl;
          *out << "  <th>Trigger<br/>Request</th>" << std::endl;
          *out << "</tr>" << std::endl;

          for (consumerIter = consumerTable.begin();
               consumerIter != consumerTable.end();
               consumerIter++)
          {
            boost::shared_ptr<ConsumerPipe> consPtr = consumerIter->second;
            *out << "<tr>" << std::endl;
            *out << "  <td align=\"center\">" << consPtr->getConsumerId()
                 << "</td>" << std::endl;

            *out << "  <td align=\"center\">";
            if (consPtr->isProxyServer()) {
              *out << "Proxy Server";
            }
            else {
              *out << consPtr->getConsumerName();
            }
            *out << "</td>" << std::endl;

            *out << "  <td align=\"center\">";
            if (consPtr->isDisconnected()) {
              *out << "Disconnected";
            }
            else if (consPtr->isIdle()) {
              *out << "Idle";
            }
            else {
              *out << "Active";
            }
            *out << "</td>" << std::endl;

            *out << "  <td align=\"center\">" << consPtr->getRateRequest()
                 << " Hz</td>" << std::endl;

            {
              std::string hltOut = consPtr->getHLTOutputSelection();
              if (hltOut.empty()) {
                *out << "  <td align=\"center\">&lt;none&gt;</td>" << std::endl;
              }
              else {
                *out << "  <td align=\"center\">" << hltOut
                     << "</td>" << std::endl;
              }
            }
	    Strings tESSet  = consPtr->getTriggerSelection();
	    std::string tTSSet = consPtr->getTriggerSelectionNew();

	    if (!tTSSet.empty()) 
		    *out << "  <td align=\"center\">"
			    << tTSSet
			    << "</td>" << std::endl;
	    else {
		    *out << "  <td align=\"center\">"
			    << InitMsgCollection::stringsToText(tESSet, 5)
			    << "</td>" << std::endl;
	    }
	    *out << "</tr>" << std::endl;
	  }
	  *out << "</table>" << std::endl;

          // ************************************************************
          // * Recent results for queued events
          // ************************************************************
          *out << "<h4>Queued Events, Recent Results:</h4>" << std::endl;
          *out << "<table border=\"1\" width=\"100%\">" << std::endl;
          *out << "<tr>" << std::endl;
          *out << "  <th>ID</th>" << std::endl;
          *out << "  <th>Name</th>" << std::endl;
          *out << "  <th>Event Count</th>" << std::endl;
          *out << "  <th>Event Rate</th>" << std::endl;
          *out << "  <th>Data Rate</th>" << std::endl;
          *out << "  <th>Duration<br/>(sec)</th>" << std::endl;
          *out << "  <th>Average<br/>Queue Size</th>" << std::endl;
          *out << "</tr>" << std::endl;

          displayedConsumerCount = 0;
          eventSum = 0.0;
          eventRateSum = 0.0;
          dataRateSum = 0.0;
          for (consumerIter = consumerTable.begin();
               consumerIter != consumerTable.end();
               consumerIter++)
          {
            boost::shared_ptr<ConsumerPipe> consPtr = consumerIter->second;
            if (consPtr->isDisconnected()) {continue;}

            ++displayedConsumerCount;
            eventSum += consPtr->getEventCount(ConsumerPipe::SHORT_TERM,
                                               ConsumerPipe::QUEUED_EVENTS,
                                               now);
            eventRateSum += consPtr->getEventRate(ConsumerPipe::SHORT_TERM,
                                                  ConsumerPipe::QUEUED_EVENTS,
                                                  now);
            dataRateSum += consPtr->getDataRate(ConsumerPipe::SHORT_TERM,
                                                ConsumerPipe::QUEUED_EVENTS,
                                                now);

            *out << "<tr>" << std::endl;
            *out << "  <td align=\"center\">" << consPtr->getConsumerId()
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">";
            if (consPtr->isProxyServer()) {
              *out << "Proxy Server";
            }
            else {
              *out << consPtr->getConsumerName();
            }
            *out << "</td>" << std::endl;

            *out << "  <td align=\"center\">"
                 << consPtr->getEventCount(ConsumerPipe::SHORT_TERM,
                                           ConsumerPipe::QUEUED_EVENTS,
                                           now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getEventRate(ConsumerPipe::SHORT_TERM,
                                          ConsumerPipe::QUEUED_EVENTS,
                                          now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getDataRate(ConsumerPipe::SHORT_TERM,
                                         ConsumerPipe::QUEUED_EVENTS,
                                         now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getDuration(ConsumerPipe::SHORT_TERM,
                                         ConsumerPipe::QUEUED_EVENTS,
                                         now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getAverageQueueSize(ConsumerPipe::SHORT_TERM,
                                                 ConsumerPipe::QUEUED_EVENTS,
                                                 now)
                 << "</td>" << std::endl;
            *out << "</tr>" << std::endl;
          }

          // add a row with the totals
          if (displayedConsumerCount > 1) {
            *out << "<tr>" << std::endl;
            *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
            *out << "  <td align=\"center\">Totals</td>" << std::endl;
            *out << "  <td align=\"center\">" << eventSum << "</td>" << std::endl;
            *out << "  <td align=\"center\">" << eventRateSum << "</td>" << std::endl;
            *out << "  <td align=\"center\">" << dataRateSum << "</td>" << std::endl;
            *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
            *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
            *out << "</tr>" << std::endl;
          }
          *out << "</table>" << std::endl;

          // ************************************************************
          // * Recent results for served events
          // ************************************************************
          *out << "<h4>Served Events, Recent Results:</h4>" << std::endl;
          *out << "<table border=\"1\" width=\"100%\">" << std::endl;
          *out << "<tr>" << std::endl;
          *out << "  <th>ID</th>" << std::endl;
          *out << "  <th>Name</th>" << std::endl;
          *out << "  <th>Event Count</th>" << std::endl;
          *out << "  <th>Event Rate</th>" << std::endl;
          *out << "  <th>Data Rate</th>" << std::endl;
          *out << "  <th>Duration (sec)</th>" << std::endl;
          *out << "</tr>" << std::endl;

          displayedConsumerCount = 0;
          eventSum = 0.0;
          eventRateSum = 0.0;
          dataRateSum = 0.0;
          for (consumerIter = consumerTable.begin();
               consumerIter != consumerTable.end();
               consumerIter++)
          {
            boost::shared_ptr<ConsumerPipe> consPtr = consumerIter->second;
            if (consPtr->isDisconnected()) {continue;}

            ++displayedConsumerCount;
            eventSum += consPtr->getEventCount(ConsumerPipe::SHORT_TERM,
                                               ConsumerPipe::SERVED_EVENTS,
                                               now);
            eventRateSum += consPtr->getEventRate(ConsumerPipe::SHORT_TERM,
                                                  ConsumerPipe::SERVED_EVENTS,
                                                  now);
            dataRateSum += consPtr->getDataRate(ConsumerPipe::SHORT_TERM,
                                                ConsumerPipe::SERVED_EVENTS,
                                                now);

            *out << "<tr>" << std::endl;
            *out << "  <td align=\"center\">" << consPtr->getConsumerId()
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">";
            if (consPtr->isProxyServer()) {
              *out << "Proxy Server";
            }
            else {
              *out << consPtr->getConsumerName();
            }
            *out << "</td>" << std::endl;

            *out << "  <td align=\"center\">"
                 << consPtr->getEventCount(ConsumerPipe::SHORT_TERM,
                                           ConsumerPipe::SERVED_EVENTS,
                                           now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getEventRate(ConsumerPipe::SHORT_TERM,
                                          ConsumerPipe::SERVED_EVENTS,
                                          now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getDataRate(ConsumerPipe::SHORT_TERM,
                                         ConsumerPipe::SERVED_EVENTS,
                                         now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getDuration(ConsumerPipe::SHORT_TERM,
                                         ConsumerPipe::SERVED_EVENTS,
                                         now)
                 << "</td>" << std::endl;
            *out << "</tr>" << std::endl;
          }

          // add a row with the totals
          if (displayedConsumerCount > 1) {
            *out << "<tr>" << std::endl;
            *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
            *out << "  <td align=\"center\">Totals</td>" << std::endl;
            *out << "  <td align=\"center\">" << eventSum << "</td>" << std::endl;
            *out << "  <td align=\"center\">" << eventRateSum << "</td>" << std::endl;
            *out << "  <td align=\"center\">" << dataRateSum << "</td>" << std::endl;
            *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
            *out << "</tr>" << std::endl;
          }
          *out << "</table>" << std::endl;

          // ************************************************************
          // * Full results for queued events
          // ************************************************************
          *out << "<h4>Queued Events, Full Results:</h4>" << std::endl;
          *out << "<table border=\"1\" width=\"100%\">" << std::endl;
          *out << "<tr>" << std::endl;
          *out << "  <th>ID</th>" << std::endl;
          *out << "  <th>Name</th>" << std::endl;
          *out << "  <th>Event Count</th>" << std::endl;
          *out << "  <th>Event Rate</th>" << std::endl;
          *out << "  <th>Data Rate</th>" << std::endl;
          *out << "  <th>Duration<br/>(sec)</th>" << std::endl;
          *out << "  <th>Average<br/>Queue Size</th>" << std::endl;
          *out << "</tr>" << std::endl;

          displayedConsumerCount = 0;
          eventSum = 0.0;
          eventRateSum = 0.0;
          dataRateSum = 0.0;
          for (consumerIter = consumerTable.begin();
               consumerIter != consumerTable.end();
               consumerIter++)
          {
            boost::shared_ptr<ConsumerPipe> consPtr = consumerIter->second;
            if (consPtr->isDisconnected()) {continue;}

            ++displayedConsumerCount;
            eventSum += consPtr->getEventCount(ConsumerPipe::LONG_TERM,
                                               ConsumerPipe::QUEUED_EVENTS,
                                               now);
            eventRateSum += consPtr->getEventRate(ConsumerPipe::LONG_TERM,
                                                  ConsumerPipe::QUEUED_EVENTS,
                                                  now);
            dataRateSum += consPtr->getDataRate(ConsumerPipe::LONG_TERM,
                                                ConsumerPipe::QUEUED_EVENTS,
                                                now);

            *out << "<tr>" << std::endl;
            *out << "  <td align=\"center\">" << consPtr->getConsumerId()
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">";
            if (consPtr->isProxyServer()) {
              *out << "Proxy Server";
            }
            else {
              *out << consPtr->getConsumerName();
            }
            *out << "</td>" << std::endl;

            *out << "  <td align=\"center\">"
                 << consPtr->getEventCount(ConsumerPipe::LONG_TERM,
                                           ConsumerPipe::QUEUED_EVENTS,
                                           now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getEventRate(ConsumerPipe::LONG_TERM,
                                          ConsumerPipe::QUEUED_EVENTS,
                                          now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getDataRate(ConsumerPipe::LONG_TERM,
                                         ConsumerPipe::QUEUED_EVENTS,
                                         now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getDuration(ConsumerPipe::LONG_TERM,
                                         ConsumerPipe::QUEUED_EVENTS,
                                         now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getAverageQueueSize(ConsumerPipe::LONG_TERM,
                                                 ConsumerPipe::QUEUED_EVENTS,
                                                 now)
                 << "</td>" << std::endl;
            *out << "</tr>" << std::endl;
          }

          // add a row with the totals
          if (displayedConsumerCount > 1) {
            *out << "<tr>" << std::endl;
            *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
            *out << "  <td align=\"center\">Totals</td>" << std::endl;
            *out << "  <td align=\"center\">" << eventSum << "</td>" << std::endl;
            *out << "  <td align=\"center\">" << eventRateSum << "</td>" << std::endl;
            *out << "  <td align=\"center\">" << dataRateSum << "</td>" << std::endl;
            *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
            *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
            *out << "</tr>" << std::endl;
          }
          *out << "</table>" << std::endl;

          // ************************************************************
          // * Full results for served events
          // ************************************************************
          *out << "<h4>Served Events, Full Results:</h4>" << std::endl;
          *out << "<table border=\"1\" width=\"100%\">" << std::endl;
          *out << "<tr>" << std::endl;
          *out << "  <th>ID</th>" << std::endl;
          *out << "  <th>Name</th>" << std::endl;
          *out << "  <th>Event Count</th>" << std::endl;
          *out << "  <th>Event Rate</th>" << std::endl;
          *out << "  <th>Data Rate</th>" << std::endl;
          *out << "  <th>Duration (sec)</th>" << std::endl;
          *out << "</tr>" << std::endl;

          displayedConsumerCount = 0;
          eventSum = 0.0;
          eventRateSum = 0.0;
          dataRateSum = 0.0;
          for (consumerIter = consumerTable.begin();
               consumerIter != consumerTable.end();
               consumerIter++)
          {
            boost::shared_ptr<ConsumerPipe> consPtr = consumerIter->second;
            if (consPtr->isDisconnected ()) {continue;}

            ++displayedConsumerCount;
            eventSum += consPtr->getEventCount(ConsumerPipe::LONG_TERM,
                                               ConsumerPipe::SERVED_EVENTS,
                                               now);
            eventRateSum += consPtr->getEventRate(ConsumerPipe::LONG_TERM,
                                                  ConsumerPipe::SERVED_EVENTS,
                                                  now);
            dataRateSum += consPtr->getDataRate(ConsumerPipe::LONG_TERM,
                                                ConsumerPipe::SERVED_EVENTS,
                                                now);

            *out << "<tr>" << std::endl;
            *out << "  <td align=\"center\">" << consPtr->getConsumerId()
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">";
            if (consPtr->isProxyServer()) {
              *out << "Proxy Server";
            }
            else {
              *out << consPtr->getConsumerName();
            }
            *out << "</td>" << std::endl;

            *out << "  <td align=\"center\">"
                 << consPtr->getEventCount(ConsumerPipe::LONG_TERM,
                                           ConsumerPipe::SERVED_EVENTS,
                                           now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getEventRate(ConsumerPipe::LONG_TERM,
                                          ConsumerPipe::SERVED_EVENTS,
                                          now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getDataRate(ConsumerPipe::LONG_TERM,
                                         ConsumerPipe::SERVED_EVENTS,
                                         now)
                 << "</td>" << std::endl;
            *out << "  <td align=\"center\">"
                 << consPtr->getDuration(ConsumerPipe::LONG_TERM,
                                         ConsumerPipe::SERVED_EVENTS,
                                         now)
                 << "</td>" << std::endl;
            *out << "</tr>" << std::endl;
          }

          // add a row with the totals
          if (displayedConsumerCount > 1) {
            *out << "<tr>" << std::endl;
            *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
            *out << "  <td align=\"center\">Totals</td>" << std::endl;
            *out << "  <td align=\"center\">" << eventSum << "</td>" << std::endl;
            *out << "  <td align=\"center\">" << eventRateSum << "</td>" << std::endl;
            *out << "  <td align=\"center\">" << dataRateSum << "</td>" << std::endl;
            *out << "  <td align=\"center\">&nbsp;</td>" << std::endl;
            *out << "</tr>" << std::endl;
          }
          *out << "</table>" << std::endl;
        }

        // ************************************************************
        // * HTTP POST timing
        // ************************************************************
        *out << "<h3>HTTP Timing:</h3>" << std::endl;
        *out << "<h4>Event Retrieval from Storage Manager(s):</h4>"
             << std::endl;
        *out << "<table border=\"1\" width=\"100%\">" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <th>&nbsp;</th>" << std::endl;
        *out << "  <th>Average Time per<br/>Request (sec)</th>" << std::endl;
        *out << "  <th>Number of<br/>Requests</th>" << std::endl;
        *out << "  <th>Measurement<br/>Duration (sec)</th>" << std::endl;
        *out << "</tr>" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <td align=\"center\">Recent Results</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << dpm_->getAverageValue(DataProcessManager::SHORT_TERM,
                                      DataProcessManager::EVENT_FETCH,
                                      now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << dpm_->getSampleCount(DataProcessManager::SHORT_TERM,
                                     DataProcessManager::EVENT_FETCH,
                                     now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << dpm_->getDuration(DataProcessManager::SHORT_TERM,
                                  DataProcessManager::EVENT_FETCH,
                                  now)
             << "</td>" << std::endl;
        *out << "</tr>" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <td align=\"center\">Full Results</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << dpm_->getAverageValue(DataProcessManager::LONG_TERM,
                                      DataProcessManager::EVENT_FETCH,
                                      now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << dpm_->getSampleCount(DataProcessManager::LONG_TERM,
                                     DataProcessManager::EVENT_FETCH,
                                     now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << dpm_->getDuration(DataProcessManager::LONG_TERM,
                                  DataProcessManager::EVENT_FETCH,
                                  now)
             << "</td>" << std::endl;
        *out << "</tr>" << std::endl;
        *out << "</table>" << std::endl;

        *out << "<h4>DQM Event Retrieval from Storage Manager(s):</h4>"
             << std::endl;
        *out << "<table border=\"1\" width=\"100%\">" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <th>&nbsp;</th>" << std::endl;
        *out << "  <th>Average Time per<br/>Request (sec)</th>" << std::endl;
        *out << "  <th>Number of<br/>Requests</th>" << std::endl;
        *out << "  <th>Measurement<br/>Duration (sec)</th>" << std::endl;
        *out << "</tr>" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <td align=\"center\">Recent Results</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << dpm_->getAverageValue(DataProcessManager::SHORT_TERM,
                                      DataProcessManager::DQMEVENT_FETCH,
                                      now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << dpm_->getSampleCount(DataProcessManager::SHORT_TERM,
                                     DataProcessManager::DQMEVENT_FETCH,
                                     now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << dpm_->getDuration(DataProcessManager::SHORT_TERM,
                                  DataProcessManager::DQMEVENT_FETCH,
                                  now)
             << "</td>" << std::endl;
        *out << "</tr>" << std::endl;
        *out << "<tr>" << std::endl;
        *out << "  <td align=\"center\">Full Results</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << dpm_->getAverageValue(DataProcessManager::LONG_TERM,
                                      DataProcessManager::DQMEVENT_FETCH,
                                      now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << dpm_->getSampleCount(DataProcessManager::LONG_TERM,
                                     DataProcessManager::DQMEVENT_FETCH,
                                     now)
             << "</td>" << std::endl;
        *out << "  <td align=\"center\">"
             << dpm_->getDuration(DataProcessManager::LONG_TERM,
                                  DataProcessManager::DQMEVENT_FETCH,
                                  now)
             << "</td>" << std::endl;
        *out << "</tr>" << std::endl;
        *out << "</table>" << std::endl;
      }
      else
      {
        *out << "<br/>Waiting for INIT messages from the Storage Managers...<br/>"
             << std::endl;
      }
    }
    else
    {
      *out << "<br/>The system is unable to fetch the Event Server "
           << "instance or the Init Message Collection instance. "
           << "This is a (very) unexpected error and could "
           << "be caused by an uninitialized DataProcessManager.<br/>"
           << std::endl;
    }
    if (dpm_.get()!=NULL) {
      if(dpm_->getInitMsgCollection().get() != NULL &&
	  dpm_->getInitMsgCollection()->size() > 0)
      {
	boost::shared_ptr<InitMsgCollection> initMsgCollection =
	  dpm_->getInitMsgCollection();
	*out << "<h3>HLT Trigger Paths:</h3>" << std::endl;
	*out << "<table border=\"1\" width=\"100%\">" << std::endl;

	{
	  InitMsgSharedPtr serializedProds = initMsgCollection->getLastElement();
	  InitMsgView initView(&(*serializedProds)[0]);
	  Strings triggerNameList;
	  initView.hltTriggerNames(triggerNameList);

	  *out << "<tr>" << std::endl;
	  *out << "  <td align=\"left\" valign=\"top\">"
	    << "Full Trigger List</td>" << std::endl;
	  *out << "  <td align=\"left\" valign=\"top\">"
	    << InitMsgCollection::stringsToText(triggerNameList, 0)
	    << "</td>" << std::endl;
	  *out << "</tr>" << std::endl;
	}

	for (int idx = 0; idx < initMsgCollection->size(); ++idx) {
	  InitMsgSharedPtr serializedProds = initMsgCollection->getElementAt(idx);
	  InitMsgView initView(&(*serializedProds)[0]);
	  Strings triggerSelectionList;
	  initView.hltTriggerSelections(triggerSelectionList);

	  *out << "<tr>" << std::endl;
	  *out << "  <td align=\"left\" valign=\"top\">"
	    << initView.outputModuleLabel()
	    << " Output Module</td>" << std::endl;
	  *out << "  <td align=\"left\" valign=\"top\">"
	    << InitMsgCollection::stringsToText(triggerSelectionList, 0)
	    << "</td>" << std::endl;
	  *out << "</tr>" << std::endl;
	}

	*out << "</table>" << std::endl;
      }
    }
  }
  else
  {
    *out << "<br/>Event server statistics are only available when the "
      << "SMProxyServer is in the Enabled state.<br/>" << std::endl;
  }

  *out << "<br/><hr/>" << std::endl;
  char timeString[64];
  time_t now = time(0);
  strftime(timeString, 60, "%d-%b-%Y %H:%M:%S %Z", localtime(&now));
  *out << "Last updated: " << timeString << std::endl;;
  *out << "</body>" << std::endl;
  *out << "</html>" << std::endl;
}

//////////// *** get DQMevent data web page //////////////////////////////////////////////////////////
void SMProxyServer::DQMeventdataWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  boost::mutex::scoped_lock ql(queue_lock_);
  // default the message length to zero
  int len=0;

  // determine the consumer ID from the event request
  // message, if it is available.
  unsigned int consumerId = 0;
  std::string lengthString = in->getenv("CONTENT_LENGTH");
  unsigned int contentLength = std::atol(lengthString.c_str());
  if (contentLength > 0) 
  {
    auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
    in->read(&(*bufPtr)[0], contentLength);
    OtherMessageView requestMessage(&(*bufPtr)[0]);
    // make the change below when a tag of IOPool/Streamer can be used without FW changes
    if (requestMessage.code() == Header::DQMEVENT_REQUEST)
    {
      uint8 *bodyPtr = requestMessage.msgBody();
      consumerId = convert32(bodyPtr);
    }
  }
  
  // first test if SMProxyServer is in Enabled state and this is a valid request
  // there must also be DQM data available
  if(fsm_.stateName()->toString() == "Enabled" && consumerId != 0)
  {
    boost::shared_ptr<DQMEventServer> eventServer;
    if (dpm_.get() != NULL)
    {
      eventServer = dpm_->getDQMEventServer();
    }
    if (eventServer.get() != NULL)
    {
      boost::shared_ptr< std::vector<char> > bufPtr =
        eventServer->getDQMEvent(consumerId);
      if (bufPtr.get() != NULL)
      {
        DQMEventMsgView msgView(&(*bufPtr)[0]);

        // what if mybuffer_ is used in multiple threads? Can it happen?
        unsigned char* from = msgView.startAddress();
        unsigned int dsize = msgView.size();
        if(dsize > mybuffer_.capacity() ) mybuffer_.resize(dsize);
        unsigned char* pos = (unsigned char*) &mybuffer_[0];

        copy(from,from+dsize,pos);
        len = dsize;
        FDEBUG(10) << "sending update at event " << msgView.eventNumberAtUpdate() << std::endl;
        ++sentDQMEvents_;
        addOutMeasurement(len);
      }
    }
    
    // check if zero length is sent when there is no valid data
    // i.e. on getDQMEvent, can already send zero length if request is invalid
    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);
  } // else send DONE as reponse (could be end of a run)
  else
  {
    // not an event request or not in enabled state, just send DONE message
    OtherMessageBuilder othermsg(&mybuffer_[0],Header::DONE);
    len = othermsg.size();
      
    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);
  }
  
}

////////////////////////////// DQM consumer registration web page ////////////////////////////
void SMProxyServer::DQMconsumerWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  boost::mutex::scoped_lock ql(queue_lock_);
  if(fsm_.stateName()->toString() == "Enabled")
  { // We need to be in the enabled state

    std::string consumerName = "None provided";
    std::string consumerRequest = "*";
    std::string consumerHost = in->getenv("REMOTE_HOST");

    // read the consumer registration message from the http input stream
    std::string lengthString = in->getenv("CONTENT_LENGTH");
    unsigned int contentLength = std::atol(lengthString.c_str());
    if (contentLength > 0)
    {
      auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
      in->read(&(*bufPtr)[0], contentLength);
      ConsRegRequestView requestMessage(&(*bufPtr)[0]);
      consumerName = requestMessage.getConsumerName();
      // for DQM consumers top folder name is stored in the "parameteSet"
      std::string reqFolder = requestMessage.getRequestParameterSet();
      if (reqFolder.size() >= 1) consumerRequest = reqFolder;
    }

    // create the buffer to hold the registration reply message
    const int BUFFER_SIZE = 100;
    char msgBuff[BUFFER_SIZE];

    // fetch the DQMevent server
    // (it and/or the job controller may not have been created yet
    //  if not in the enabled state)
    boost::shared_ptr<DQMEventServer> eventServer;
    if (dpm_.get() != NULL)
    {
      eventServer = dpm_->getDQMEventServer();
    }

    // if no event server, tell the consumer that we're not ready
    if (eventServer.get() == NULL)
    {
      // build the registration response into the message buffer
      ConsRegResponseBuilder respMsg(msgBuff, BUFFER_SIZE,
                                     ConsRegResponseBuilder::ES_NOT_READY, 0);
      // debug message so that compiler thinks respMsg is used
      FDEBUG(20) << "Registration response size =  " <<
        respMsg.size() << std::endl;
    }
    else
    {
      // create the local consumer interface and add it to the event server
      boost::shared_ptr<DQMConsumerPipe>
        consPtr(new DQMConsumerPipe(consumerName, "normal",
                                    DQMactiveConsumerTimeout_.value_,
                                    DQMidleConsumerTimeout_.value_,
                                    consumerRequest, consumerHost,
                                    DQMconsumerQueueSize_));
      eventServer->addConsumer(consPtr);

      // initialize it straight away (should later pass in the folder name to
      // optionally change the selection on a register?
      consPtr->initializeSelection();

      // build the registration response into the message buffer
      ConsRegResponseBuilder respMsg(msgBuff, BUFFER_SIZE,
                                     0, consPtr->getConsumerId());
      // debug message so that compiler thinks respMsg is used
      FDEBUG(20) << "Registration response size =  " <<
        respMsg.size() << std::endl;
    }

    // send the response
    ConsRegResponseView responseMessage(msgBuff);
    unsigned int len = responseMessage.size();
    if(len > mybuffer_.capacity() ) mybuffer_.resize(len);
    for (int i=0; i<(int)len; i++) mybuffer_[i]=msgBuff[i];

    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);

  } else { // is this the right thing to send?
   // In wrong state for this message - return zero length stream, should return Msg NOTREADY
   int len = 0;
   out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
   out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
   out->write((char*) &mybuffer_[0],len);
  }

}

////////////////////////////// receive event data from SM web page ////////////////////////////
void SMProxyServer::receiveEventWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  bool haveHeaderAlready = false;
  if(dpm_.get() != NULL) haveHeaderAlready = dpm_->haveHeader();
  if(fsm_.stateName()->toString() == "Enabled" && haveHeaderAlready)
  { // can only receive data if enabled and registered and have header

    // read the event message from the http input stream
    std::string lengthString = in->getenv("CONTENT_LENGTH");
    unsigned long contentLength = std::atol(lengthString.c_str());
    if (contentLength > 0)
    {
      // we need to make a copy of this event that sticks around until
      // all consumers have got sent a copy (So cannot use mybuffer_)
      auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
      in->read(&(*bufPtr)[0], contentLength);
      EventMsgView eventView(&(*bufPtr)[0]);

     if (!runNumber_) runNumber_=eventView.run();
      
     if (dropOldLumisectionEvents_) {
    
	uint32 lumi = eventView.lumi();
	if (lumi<currentLumiSection_) {
		//drop event and bail out
		++receivedEvents_;
		addMeasurement(contentLength);
		int len = 0;
		out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
		out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
		out->write((char*) &mybuffer_[0],len);
		return;
	}
	else if (lumi>currentLumiSection_) {
		currentLumiSection_=lumi;
	}
     }

     boost::shared_ptr<EventServer> eventServer;
     if (dpm_.get() != NULL)
     {
	     eventServer = dpm_->getEventServer();
	     if(eventServer.get() != NULL) {
		     eventServer->processEvent(eventView);
	     }
     }
     ++receivedEvents_;
     addMeasurement(contentLength);
    }

    // do we have to send a response? Will the SM hang/timeout if not?
    // we want the SM to keep running after a data push
    int len = 0;
    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);
  } else {
    // in wrong state
    int len = 0;
    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);
  }

}

////////////////////////////// receive DQM data from SM web page ////////////////////////////
void SMProxyServer::receiveDQMEventWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  bool haveHeaderAlready = false;
  if(dpm_.get() != NULL) haveHeaderAlready = dpm_->haveHeader();
  if(fsm_.stateName()->toString() == "Enabled" && haveHeaderAlready)
  { // can only receive data if enabled and registered

    // read the DQMevent message from the http input stream
    std::string lengthString = in->getenv("CONTENT_LENGTH");
    unsigned long contentLength = std::atol(lengthString.c_str());
    if (contentLength > 0)
    {
      // we need to make a copy of this event that sticks around until
      // all consumers have got sent a copy (So cannot use mybuffer_)
      auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
      in->read(&(*bufPtr)[0], contentLength);
      DQMEventMsgView dqmEventView(&(*bufPtr)[0]);
      //boost::shared_ptr<DQMEventServer> DQMeventServer;
      //if (dpm_.get() != NULL)
      //{
      //  DQMeventServer = dpm_->getDQMEventServer();
      //  if(DQMeventServer.get() != NULL) {
      //    DQMeventServer->processDQMEvent(dqmEventView);
      //  }
      //}
      boost::shared_ptr<stor::DQMServiceManager> dqmManager;
      if (dpm_.get() != NULL)
      {
        dqmManager = dpm_->getDQMServiceManager();
        if(dqmManager.get() != NULL) {
          dqmManager->manageDQMEventMsg(dqmEventView);
        }
      }
      ++receivedDQMEvents_;
      addMeasurement(contentLength);
    }

    // do we have to send a response? Will the SM hang/timeout if not?
    // we want the SM to keep running after a data push
    int len = 0;
    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);
  } else {
    // in wrong state
    int len = 0;
    out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
    out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
    out->write((char*) &mybuffer_[0],len);
  }


}

//------------------------------------------------------------------------------
// Everything that has to do with the flash list goes here
// 
// - setupFlashList()                  - setup variables and initialize them
// - actionPerformed(xdata::Event &e)  - update values in flash list
//------------------------------------------------------------------------------
void SMProxyServer::setupFlashList()
{
  //----------------------------------------------------------------------------
  // Setup the header variables
  //----------------------------------------------------------------------------
  class_    = getApplicationDescriptor()->getClassName();
  instance_ = getApplicationDescriptor()->getInstance();
  std::string url;
  url       = getApplicationDescriptor()->getContextDescriptor()->getURL();
  url      += "/";
  url      += getApplicationDescriptor()->getURN();
  url_      = url;

  //----------------------------------------------------------------------------
  // Create/Retrieve an infospace which can be monitored
  //----------------------------------------------------------------------------
  std::ostringstream oss;
  oss << "urn:xdaq-monitorable:" << class_.value_ << ":" << instance_.value_;
  toolbox::net::URN urn = this->createQualifiedInfoSpace(oss.str());
  xdata::InfoSpace *is = xdata::getInfoSpaceFactory()->get(urn.toString());

  //----------------------------------------------------------------------------
  // Publish monitor data in monitorable info space -- Head
  //----------------------------------------------------------------------------
  is->fireItemAvailable("class",                &class_);
  is->fireItemAvailable("instance",             &instance_);
  is->fireItemAvailable("runNumber",            &runNumber_);
  is->fireItemAvailable("url",                  &url_);
  // Body
  is->fireItemAvailable("storedDQMEvents",         &storedDQMEvents_);
  is->fireItemAvailable("sentEvents",           &sentEvents_);
  is->fireItemAvailable("sentDQMEvents",        &sentDQMEvents_);
  is->fireItemAvailable("storedVolume",         &storedVolume_);
  is->fireItemAvailable("instantBandwidth",     &instantBandwidth_);
  is->fireItemAvailable("instantRate",          &instantRate_);
  is->fireItemAvailable("instantLatency",       &instantLatency_);
  is->fireItemAvailable("maxBandwidth",         &maxBandwidth_);
  is->fireItemAvailable("minBandwidth",         &minBandwidth_);
  is->fireItemAvailable("duration",             &duration_);
  is->fireItemAvailable("totalSamples",         &totalSamples_);
  is->fireItemAvailable("meanBandwidth",        &meanBandwidth_);
  is->fireItemAvailable("meanRate",             &meanRate_);
  is->fireItemAvailable("meanLatency",          &meanLatency_);
  is->fireItemAvailable("stateName",            fsm_.stateName());
  is->fireItemAvailable("progressMarker",       &progressMarker_);
  is->fireItemAvailable("connectedSMs",         &connectedSMs_);
  is->fireItemAvailable("collateDQM",           &collateDQM_);
  is->fireItemAvailable("archiveDQM",           &archiveDQM_);
  is->fireItemAvailable("archiveIntervalDQM",   &archiveIntervalDQM_);
  is->fireItemAvailable("purgeTimeDQM",         &purgeTimeDQM_);
  is->fireItemAvailable("readyTimeDQM",         &readyTimeDQM_);
  is->fireItemAvailable("filePrefixDQM",        &filePrefixDQM_);
  is->fireItemAvailable("useCompressionDQM",    &useCompressionDQM_);
  is->fireItemAvailable("compressionLevelDQM",  &compressionLevelDQM_);
  //is->fireItemAvailable("nLogicalDisk",         &nLogicalDisk_);
  //is->fireItemAvailable("fileCatalog",          &fileCatalog_);
  is->fireItemAvailable("maxESEventRate",       &maxESEventRate_);
  is->fireItemAvailable("maxESDataRate",        &maxESDataRate_);
  is->fireItemAvailable("DQMmaxESEventRate",    &DQMmaxESEventRate_);
  is->fireItemAvailable("maxEventRequestRate",&maxEventRequestRate_);
  is->fireItemAvailable("maxDQMEventRequestRate",&maxDQMEventRequestRate_);
  is->fireItemAvailable("activeConsumerTimeout",&activeConsumerTimeout_);
  is->fireItemAvailable("idleConsumerTimeout",  &idleConsumerTimeout_);
  is->fireItemAvailable("consumerQueueSize",    &consumerQueueSize_);
  is->fireItemAvailable("esSelectedHLTOutputModule",&esSelectedHLTOutputModule_);
  is->fireItemAvailable("esSelectedEventSelection",&esSelectedEventSelection_);
  is->fireItemAvailable("TriggerSelector",&TriggerSelector_);
  is->fireItemAvailable("selectionFromClient",&selectionFromClient_);
  is->fireItemAvailable("allowMissingSM",       &allowMissingSM_);
  is->fireItemAvailable("dropOldLumisectionEvents",       &dropOldLumisectionEvents_);
  is->fireItemAvailable("enableDQMSM",       &enableDQMSM_);
  is->fireItemAvailable("maxConnectionRetries",       &maxConnectionRetries_);
  //is->fireItemAvailable("fairShareES",          &fairShareES_);

  //----------------------------------------------------------------------------
  // Attach listener to myCounter_ to detect retrieval event
  //----------------------------------------------------------------------------
  is->addItemRetrieveListener("class",                this);
  is->addItemRetrieveListener("instance",             this);
  is->addItemRetrieveListener("runNumber",            this);
  is->addItemRetrieveListener("url",                  this);
  // Body
  is->addItemRetrieveListener("storedDQMEvents",      this);
  is->addItemRetrieveListener("sentEvents",           this);
  is->addItemRetrieveListener("sentDQMEvents",        this);
  is->addItemRetrieveListener("storedVolume",         this);
  is->addItemRetrieveListener("instantBandwidth",     this);
  is->addItemRetrieveListener("instantRate",          this);
  is->addItemRetrieveListener("instantLatency",       this);
  is->addItemRetrieveListener("maxBandwidth",         this);
  is->addItemRetrieveListener("minBandwidth",         this);
  is->addItemRetrieveListener("duration",             this);
  is->addItemRetrieveListener("totalSamples",         this);
  is->addItemRetrieveListener("meanBandwidth",        this);
  is->addItemRetrieveListener("meanRate",             this);
  is->addItemRetrieveListener("meanLatency",          this);
  is->addItemRetrieveListener("stateName",            this);
  is->addItemRetrieveListener("progressMarker",       this);
  is->addItemRetrieveListener("connectedSMs",         this);
  is->addItemRetrieveListener("collateDQM",           this);
  is->addItemRetrieveListener("archiveDQM",           this);
  is->addItemRetrieveListener("archiveIntervalDQM",   this);
  is->addItemRetrieveListener("purgeTimeDQM",         this);
  is->addItemRetrieveListener("readyTimeDQM",         this);
  is->addItemRetrieveListener("filePrefixDQM",        this);
  is->addItemRetrieveListener("useCompressionDQM",    this);
  is->addItemRetrieveListener("compressionLevelDQM",  this);
  //is->addItemRetrieveListener("nLogicalDisk",         this);
  //is->addItemRetrieveListener("fileCatalog",          this);
  is->addItemRetrieveListener("maxESEventRate",       this);
  is->addItemRetrieveListener("maxESDataRate",        this);
  is->addItemRetrieveListener("DQMmaxESEventRate",    this);
  is->addItemRetrieveListener("maxEventRequestRate",  this);
  is->addItemRetrieveListener("maxDQMEventRequestRate",this);
  is->addItemRetrieveListener("activeConsumerTimeout",this);
  is->addItemRetrieveListener("idleConsumerTimeout",  this);
  is->addItemRetrieveListener("consumerQueueSize",    this);
  is->addItemRetrieveListener("esSelectedHLTOutputModule",this);
  is->addItemRetrieveListener("esSelectedEventSelection",this);
  is->addItemRetrieveListener("TriggerSelector",this);
  is->addItemRetrieveListener("selectionFromClient",this);
  is->addItemRetrieveListener("allowMissingSM",       this);
  is->addItemRetrieveListener("dropOldLumisectionEvents",       this);
  is->addItemRetrieveListener("enableDQMSM",       this);
  is->addItemRetrieveListener("maxConnectionRetries",       this);
  
  //is->addItemRetrieveListener("fairShareES",          this);
  //----------------------------------------------------------------------------
}


void SMProxyServer::actionPerformed(xdata::Event& e)  
{
  if (e.type() == "ItemRetrieveEvent") {
    std::ostringstream oss;
    oss << "urn:xdaq-monitorable:" << class_.value_ << ":" << instance_.value_;
    xdata::InfoSpace *is = xdata::InfoSpace::get(oss.str());

    is->lock();
    std::string item = dynamic_cast<xdata::ItemRetrieveEvent&>(e).itemName();
    // Only update those locations which are not always up to date
    if      (item == "connectedSMs")
      connectedSMs_   = smsenders_.size();
    else if (item == "storedVolume")
    {
      if (dpm_.get() != NULL)
        storedVolume_   = dpm_->totalvolumemb();
      else
        storedVolume_   = 0;
    }
    //else if (item == "progressMarker")
    //  progressMarker_ = ProgressMarker::instance()->status();
    is->unlock();
  } 
}



bool SMProxyServer::configuring(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start configuring ...");
    
    // check output locations and scripts before we continue
    if((bool)archiveDQM_) {
      try {
        checkDirectoryOK(filePrefixDQM_.toString());
      }
      catch(cms::Exception& e)
      {
        reasonForFailedState_ = e.explainSelf();
        fsm_.fireFailed(reasonForFailedState_,this);
        return false;
      }
    }

    // the poll rate is set by maxESEventRate_ and we poll for both events
    // and DQM events at the same time!
    
    if (maxESEventRate_ < 0.0)
      maxESEventRate_ = 0.0;
    if (maxESDataRate_ < 0.0)
      maxESDataRate_ = 0.0;
    if (DQMmaxESEventRate_ < 0.0)
      DQMmaxESEventRate_ = 0.0;

    outpmeter_->init(samples_, period4samples_);
    
    // TODO fixme: determine these two parameters properly
    xdata::Integer cutoff(20);
    xdata::Integer mincutoff(10);
    if (consumerQueueSize_ > cutoff)
      consumerQueueSize_ = cutoff;
    if (DQMconsumerQueueSize_ > cutoff)
      DQMconsumerQueueSize_ = cutoff;
    if (consumerQueueSize_ < mincutoff)
      consumerQueueSize_ = mincutoff;
    if (DQMconsumerQueueSize_ < mincutoff)
      DQMconsumerQueueSize_ = mincutoff;

    // set the urn as the consumer name to register with to SM
    std::string url = getApplicationDescriptor()->getContextDescriptor()->getURL();
    std::string urn = getApplicationDescriptor()->getURN();
    consumerName_ = url + "/" + urn + "/pushEventData";
    DQMconsumerName_ = url + "/" + urn + "/pushDQMEventData";

    //create static queue with static configuration, else wait for consumer to connect
    boost::mutex::scoped_lock ql(queue_lock_);

    if (!selectionFromClient_) { 
      if (!createQueue()) return false;
    }
    
    LOG4CPLUS_INFO(getApplicationLogger(),"Finished configuring!");
   
    fsm_.fireEvent("ConfigureDone",this);
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "configuring FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }

  return false;
}

bool SMProxyServer::createQueue() {
  // start a work loop that can process commands (do we need it in push mode?)
  // TODO fixme: use a pushmode variable to decide to change consumer names
  //             and not get events on push mode in work loop
  //
  try {
    dpm_.reset(new stor::DataProcessManager());
    dpm_->setHLTOutputModule(esSelectedHLTOutputModule_);
    std::vector<std::string> tmpVector;
    tmpVector.resize(esSelectedEventSelection_.elements());
    for(unsigned int i = 0; i < esSelectedEventSelection_.elements(); ++i)
      tmpVector[i] = static_cast<std::string>(esSelectedEventSelection_[i]);
    dpm_->setEventSelection(tmpVector);
    dpm_->setEventSelection(TriggerSelector_.toString());
    dpm_->setAllowMissingSM(allowMissingSM_);
    dpm_->setMaxConnectionRetries(maxConnectionRetries_);

    boost::shared_ptr<EventServer>
      eventServer(new EventServer(maxESEventRate_, maxESDataRate_,
	    esSelectedHLTOutputModule_,
	    fairShareES_));
    dpm_->setEventServer(eventServer);
    boost::shared_ptr<DQMEventServer>
      DQMeventServer(new DQMEventServer(DQMmaxESEventRate_));
    dpm_->setDQMEventServer(DQMeventServer);
    boost::shared_ptr<InitMsgCollection>
      initMsgCollection(new InitMsgCollection());
    dpm_->setInitMsgCollection(initMsgCollection);
    dpm_->setMaxEventRequestRate(maxEventRequestRate_);
    dpm_->setMaxDQMEventRequestRate(maxDQMEventRequestRate_);
    dpm_->setCollateDQM(collateDQM_);
    dpm_->setArchiveDQM(archiveDQM_);
    dpm_->setArchiveIntervalDQM(archiveIntervalDQM_);
    dpm_->setPurgeTimeDQM(purgeTimeDQM_);
    dpm_->setReadyTimeDQM(readyTimeDQM_);
    dpm_->setFilePrefixDQM(filePrefixDQM_);
    dpm_->setUseCompressionDQM(useCompressionDQM_);
    dpm_->setCompressionLevelDQM(compressionLevelDQM_);
    dpm_->setSamples(samples_);
    dpm_->setPeriod4Samples(period4samples_);

    // If we are in pull mode, we need to know which Storage Managers to
    // poll for events and DQM events
    // Only add the StorageManager URLs at this configuration stage
    dpm_->setConsumerName(consumerName_.toString());
    dpm_->setDQMConsumerName(DQMconsumerName_.toString());
    unsigned int rsize = (unsigned int)smRegList_.size();
    dpm_->setExpectedUpdatesDQM(rsize);
    for(unsigned int i = 0; i < rsize; ++i)
    {
      std::cout << "add to register list num = " << i << " url = "
	<< smRegList_.elementAt(i)->toString() << std::endl;
      dpm_->addSM2Register(smRegList_.elementAt(i)->toString());
      if ( enableDQMSM_ )
      dpm_->addDQMSM2Register(smRegList_.elementAt(i)->toString());
      smsenders_.insert(std::make_pair(smRegList_.elementAt(i)->toString(), false));
    }
  }
  catch(cms::Exception& e)
  {
    //XCEPT_RAISE (toolbox::fsm::exception::Exception, e.explainSelf());
    reasonForFailedState_ = e.explainSelf();
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }
  catch(std::exception& e)
  {
    //XCEPT_RAISE (toolbox::fsm::exception::Exception, e.what());
    reasonForFailedState_  = e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }
  catch(...)
  {
    //XCEPT_RAISE (toolbox::fsm::exception::Exception, "Unknown Exception");
    reasonForFailedState_  = "Unknown Exception while configuring";
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }
  queueCreated_=true;
  queueInactive_=false;

  //timeout _only_ if SM connection is setup from client
  if (!selectionFromClient_) return true;

  timeoutCounter_=queueTimeout_;
  if (!timeoutWorkLoop_->isActive()) timeoutWorkLoop_->activate();
  return true;
}

void SMProxyServer::destroyQueue() {

  queueCreated_=false;

  boost::shared_ptr<stor::DQMServiceManager> dqmManager;
  if (dpm_.get() != NULL)
  {
    dqmManager = dpm_->getDQMServiceManager();
    if(dqmManager.get() != NULL) {
      dqmManager->stop();
    }
    // clear out events from queues
    boost::shared_ptr<EventServer> eventServer;
    boost::shared_ptr<DQMEventServer> dqmeventServer;
    eventServer = dpm_->getEventServer();
    dqmeventServer = dpm_->getDQMEventServer();
    if (eventServer.get() != NULL) eventServer->clearQueue();
    if (dqmeventServer.get() != NULL) dqmeventServer->clearQueue();
    // do not stop dpm_ as we don't want to register again and get the header again
    // need to redo if we switch to polling for events
    // switched to polling for events
    if (dpm_.get() != NULL) {
      dpm_->stop();
      dpm_->join();
    }
  }

}

bool SMProxyServer::enabling(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start enabling ...");
    
    //fileList_.clear();
    //eventsInFile_.clear();
    //fileSize_.clear();
    storedDQMEvents_ = 0;
    sentEvents_   = 0;
    sentDQMEvents_   = 0;
    receivedEvents_ = 0;
    receivedDQMEvents_ = 0;
    currentLumiSection_ = 0;

    { 
      boost::mutex::scoped_lock ql(queue_lock_);
      if (queueCreated_) dpm_->start();

      LOG4CPLUS_INFO(getApplicationLogger(),"Finished enabling!");
      fsm_.fireEvent("EnableDone",this);
    }
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "enabling FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }
  
  return false;
}


bool SMProxyServer::stopping(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start stopping :) ...");

    // only write out DQM data if needed
    {
      boost::mutex::scoped_lock ql(queue_lock_);

      destroyQueue();

      // should tell StorageManager applications we are stopping in which
      // case we need to register again

      LOG4CPLUS_INFO(getApplicationLogger(),"Finished stopping!");
      fsm_.fireEvent("StopDone",this);
    }
    if( timeoutWorkLoop_->isActive()) timeoutWorkLoop_->cancel();
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "stopping FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }

  return false;
}


bool SMProxyServer::halting(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start halting ...");
    {
      boost::mutex::scoped_lock ql(queue_lock_);

      if (dpm_.get()) {
        dpm_->stop();
        dpm_->join();
      }
      smsenders_.clear();
      connectedSMs_ = 0;
      /* maybe we want to see these statistics after a halt 
      storedDQMEvents_ = 0;
      sentEvents_   = 0;
      sentDQMEvents_   = 0;
      receivedEvents_ = 0;
      receivedDQMEvents_ = 0;
      */
    
      {
        boost::mutex::scoped_lock sl(halt_lock_);
	if (dpm_.get()) dpm_.reset();
      }
    
      LOG4CPLUS_INFO(getApplicationLogger(),"Finished halting!");
      fsm_.fireEvent("HaltDone",this);
    }
    if( timeoutWorkLoop_->isActive()) timeoutWorkLoop_->cancel();
  }
  catch (xcept::Exception &e) {
    reasonForFailedState_ = "halting FAILED: " + (string)e.what();
    fsm_.fireFailed(reasonForFailedState_,this);
    return false;
  }
  
  return false;
}

bool SMProxyServer::queueTimeout(toolbox::task::WorkLoop* wl)
 {
	if ((unsigned int) queueTimeout_ == 0) return false;
 	::sleep(1); //sleep one second
	{
	  boost::mutex::scoped_lock ql(queue_lock_);
	  if (timeoutCounter_-- <= 0) { 
		  queueInactive_=true;
		  if (queueCreated_) destroyQueue();
	  }
	}
	return true;
 }

void SMProxyServer::checkDirectoryOK(std::string path)
{
  struct stat buf;

  int retVal = stat(path.c_str(), &buf);
  if(retVal !=0 )
  {
    edm::LogError("SMProxyServer") << "Directory or file " << path
                                    << " does not exist. Error=" << errno ;
    throw cms::Exception("SMProxyServer","checkDirectoryOK")
            << "Directory or file " << path << " does not exist. Error=" << errno << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
xoap::MessageReference SMProxyServer::fsmCallback(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  return fsm_.commandCallback(msg);
}


////////////////////////////////////////////////////////////////////////////////
// *** Provides factory method for the instantiation of SM applications
// should probably use the MACRO? Could a XDAQ version change cause problems?
extern "C" xdaq::Application
*instantiate_SMProxyServer(xdaq::ApplicationStub * stub)
{
  std::cout << "Going to construct a SMProxyServer instance "
	    << std::endl;
  return new stor::SMProxyServer(stub);
}

