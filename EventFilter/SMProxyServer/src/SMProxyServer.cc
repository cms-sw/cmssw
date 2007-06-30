// $Id: SMProxyServer.cc,v 1.4 2007/05/16 22:57:45 hcheung Exp $

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <sys/statfs.h>

#include "EventFilter/SMProxyServer/interface/SMProxyServer.h"
#include "EventFilter/StorageManager/interface/ConsumerPipe.h"
#include "EventFilter/StorageManager/interface/ProgressMarker.h"
//#include "EventFilter/StorageManager/interface/Configurator.h"
//#include "EventFilter/StorageManager/interface/Parameter.h"

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
#include "IOPool/Streamer/interface/StreamDeserializer.h"

#include "xcept/tools.h"

#include "xgi/Method.h"

#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xoap/domutils.h"

using namespace edm;
using namespace std;
using namespace stor;

SMProxyServer::SMProxyServer(xdaq::ApplicationStub * s)
  throw (xdaq::exception::Exception) :
  xdaq::Application(s),
  fsm_(this), 
  ah_(0), 
  collateDQM_(false),
  archiveDQM_(false),
  filePrefixDQM_("/tmp/DQM"),
  purgeTimeDQM_(DEFAULT_PURGE_TIME),
  readyTimeDQM_(DEFAULT_READY_TIME),
  useCompressionDQM_(true),
  compressionLevelDQM_(1),
  receivedEvents_(0),
  receivedDQMEvents_(0),
  mybuffer_(7000000),
  connectedSMs_(0), 
  storedDQMEvents_(0), 
  sentEvents_(0),
  sentDQMEvents_(0), 
  storedVolume_(0.),
  progressMarker_(ProgressMarker::instance()->idle())
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
  xgi::bind(this,&SMProxyServer::DQMeventdataWebPage,  "getDQMeventdata");
  xgi::bind(this,&SMProxyServer::DQMconsumerWebPage,   "registerDQMConsumer");

  xgi::bind(this,&SMProxyServer::receiveEventWebPage,     "pushEventData");
  xgi::bind(this,&SMProxyServer::receiveDQMEventWebPage,  "pushDQMEventData");

  ispace->fireItemAvailable("collateDQM",     &collateDQM_);
  ispace->fireItemAvailable("archiveDQM",     &archiveDQM_);
  ispace->fireItemAvailable("purgeTimeDQM",   &purgeTimeDQM_);
  ispace->fireItemAvailable("readyTimeDQM",   &readyTimeDQM_);
  ispace->fireItemAvailable("filePrefixDQM",  &filePrefixDQM_);
  ispace->fireItemAvailable("useCompressionDQM",  &useCompressionDQM_);
  ispace->fireItemAvailable("compressionLevelDQM",  &compressionLevelDQM_);
  //nLogicalDisk_   = 0;

  //ispace->fireItemAvailable("nLogicalDisk", &nLogicalDisk_);

  //boost::shared_ptr<stor::Parameter> smParameter_ = stor::Configurator::instance()->getParameter();
  //closeFileScript_    = smParameter_ -> closeFileScript();
  //notifyTier0Script_  = smParameter_ -> notifyTier0Script();
  //insertFileScript_   = smParameter_ -> insertFileScript();  
  //fileCatalog_        = smParameter_ -> fileCatalog(); 

  //ispace->fireItemAvailable("closeFileScript",    &closeFileScript_);
  //ispace->fireItemAvailable("notifyTier0Script",  &notifyTier0Script_);
  //ispace->fireItemAvailable("insertFileScript",   &insertFileScript_);
  //ispace->fireItemAvailable("fileCatalog",        &fileCatalog_);

  // added for Event Server
  maxESEventRate_ = 1.0;  // hertz
  ispace->fireItemAvailable("maxESEventRate",&maxESEventRate_);
  activeConsumerTimeout_ = 300;  // seconds
  ispace->fireItemAvailable("activeConsumerTimeout",&activeConsumerTimeout_);
  idleConsumerTimeout_ = 600;  // seconds
  ispace->fireItemAvailable("idleConsumerTimeout",&idleConsumerTimeout_);
  consumerQueueSize_ = 5;
  ispace->fireItemAvailable("consumerQueueSize",&consumerQueueSize_);
  DQMmaxESEventRate_ = 1.0;  // hertz
  ispace->fireItemAvailable("DQMmaxESEventRate",&DQMmaxESEventRate_);
  DQMactiveConsumerTimeout_ = 300;  // seconds
  ispace->fireItemAvailable("DQMactiveConsumerTimeout",&DQMactiveConsumerTimeout_);
  DQMidleConsumerTimeout_ = 600;  // seconds
  ispace->fireItemAvailable("DQMidleConsumerTimeout",&DQMidleConsumerTimeout_);
  DQMconsumerQueueSize_ = 5;
  ispace->fireItemAvailable("DQMconsumerQueueSize",&DQMconsumerQueueSize_);

  // for performance measurements
  samples_          = 100; // measurements every 25MB (about)
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

  pmeter_ = new stor::SMPerformanceMeter();
  pmeter_->init(samples_);
  outpmeter_ = new stor::SMPerformanceMeter();
  outpmeter_->init(samples_);

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
  delete pmeter_;
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
  // for bandwidth performance measurements
  if ( pmeter_->addSample(size) )
  {
    // Copy measurements for our record
    instantBandwidth_ = pmeter_->bandwidth();
    instantRate_      = pmeter_->rate();
    instantLatency_   = pmeter_->latency();
    totalSamples_     = pmeter_->totalsamples();
    duration_         = pmeter_->duration();
    meanBandwidth_    = pmeter_->meanbandwidth();
    meanRate_         = pmeter_->meanrate();
    meanLatency_      = pmeter_->meanlatency();

    // Determine minimum and maximum instantaneous bandwidth
    if (instantBandwidth_ > maxBandwidth_)
      maxBandwidth_ = instantBandwidth_;
    if (instantBandwidth_ < minBandwidth_)
      minBandwidth_ = instantBandwidth_;
  }
}

void SMProxyServer::addOutMeasurement(unsigned long size)
{
  // for bandwidth performance measurements
  if ( outpmeter_->addSample(size) )
  {
    // Copy measurements for our record
    outinstantBandwidth_ = outpmeter_->bandwidth();
    outinstantRate_      = outpmeter_->rate();
    outinstantLatency_   = outpmeter_->latency();
    outtotalSamples_     = outpmeter_->totalsamples();
    outduration_         = outpmeter_->duration();
    outmeanBandwidth_    = outpmeter_->meanbandwidth();
    outmeanRate_         = outpmeter_->meanrate();
    outmeanLatency_      = outpmeter_->meanlatency();

    // Determine minimum and maximum instantaneous bandwidth
    if (outinstantBandwidth_ > outmaxBandwidth_)
      outmaxBandwidth_ = outinstantBandwidth_;
    if (outinstantBandwidth_ < outminBandwidth_)
      outminBandwidth_ = outinstantBandwidth_;
  }
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

  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << endl;
  *out << "<colgroup> <colgroup align=\"right\">"                    << endl;
        *out << "<tr>" << endl;
        *out << "<th >" << endl;
        *out << "State" << endl;
        *out << "</th>" << endl;
        *out << "<th>" << endl;
        *out << fsm_.stateName()->toString() << endl;
        *out << "</th>" << endl;
        *out << "</tr>" << endl;
        *out << "<tr>" << endl;
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Input and Output Statistics"                  << endl;
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
    *out << "      " << "Input Performance for last " << samples_ << " HTTP posts"<< endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << instantBandwidth_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Rate (Frames/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << instantRate_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Latency (us/frame)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << instantLatency_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Maximum Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << maxBandwidth_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Minimum Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << minBandwidth_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
// mean performance statistics for whole run
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Mean Performance for " << totalSamples_ << " posts, duration "
         << duration_ << " seconds" << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << meanBandwidth_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Rate (Frames/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << meanRate_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Latency (us/frame)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << meanLatency_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
// performance statistics
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Output Performance for last " << samples_ << " HTTP posts"<< endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outinstantBandwidth_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Rate (Frames/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outinstantRate_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Latency (us/frame)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outinstantLatency_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Maximum Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outmaxBandwidth_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Minimum Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outminBandwidth_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
// mean performance statistics for whole run
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Mean Performance for " << outtotalSamples_ << " posts, duration "
         << outduration_ << " seconds" << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Bandwidth (MB/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outmeanBandwidth_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Rate (Frames/s)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outmeanRate_ << endl;
          *out << "</td>" << endl;
        *out << "  </tr>" << endl;
        *out << "<tr>" << endl;
          *out << "<td >" << endl;
          *out << "Latency (us/frame)" << endl;
          *out << "</td>" << endl;
          *out << "<td align=right>" << endl;
          *out << outmeanLatency_ << endl;
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
    *out << "      " << "FU Sender Information"                            << endl;
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

  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;
}


//////////// *** fusender web page //////////////////////////////////////////////////////////
void SMProxyServer::smsenderWebPage(xgi::Input *in, xgi::Output *out)
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
          if(si->second) 
            *out << "Yes" << endl;
          else
            *out << "No" << endl;
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

  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;
}


//////////// *** get event data web page //////////////////////////////////////////////////////////
void SMProxyServer::eventdataWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
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
  
}


//////////// *** callback for header request (registry) web page ////////////////////////////////////////
void SMProxyServer::headerdataWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
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
  bool haveHeaderAlready = false;
  if(dpm_.get() != NULL) haveHeaderAlready = dpm_->haveHeader();
  if(fsm_.stateName()->toString() == "Enabled" && haveHeaderAlready)
    {
      if(!dpm_->haveHeader()) // should not get here! (except for threading...)
	{ // not available yet - return zero length stream, should return MsgCode NOTREADY
	  int len = 0;
	  out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
	  out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
	  out->write((char*) &mybuffer_[0],len);
	} 
      else 
	{
	  // overlay an INIT message view on the serialized
	  // products array so that we can initialize the consumer event selection
	  unsigned int len = dpm_->headerSize();
          std::vector<unsigned char> serialized_prods = dpm_->getHeader();
	  InitMsgView initView(&serialized_prods[0]);
	  if (dpm_.get() != NULL)
	    {
	      boost::shared_ptr<EventServer> eventServer = dpm_->getEventServer();
	      if (eventServer.get() != NULL)
		{
		  boost::shared_ptr<ConsumerPipe> consPtr =
		    eventServer->getConsumer(consumerId);
		  if (consPtr.get() != NULL)
		    {
		      consPtr->initializeSelection(initView);
		    }
		}
	    }
          if(len > mybuffer_.capacity() ) mybuffer_.resize(len);
	  for (int i=0; i<(int)len; i++) mybuffer_[i]=serialized_prods[i];
	  
	  out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
	  out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
	  out->write((char*) &mybuffer_[0],len);
	}
    } 
  else 
    {
      // In wrong state for this message - return zero length stream, should return Msg NOTREADY
      int len = 0;
      out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
      out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
      out->write((char*) &mybuffer_[0],len);
    }
}


////////////////////////////// consumer registration web page ////////////////////////////
void SMProxyServer::consumerWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  if(fsm_.stateName()->toString() == "Enabled")
  { // what is the right place for this?

  std::string consumerName = "None provided";
  std::string consumerPriority = "normal";
  std::string consumerRequest = "<>";

  // read the consumer registration message from the http input stream
  std::string lengthString = in->getenv("CONTENT_LENGTH");
  unsigned long contentLength = std::atol(lengthString.c_str());
  if (contentLength > 0)
  {
    auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
    in->read(&(*bufPtr)[0], contentLength);
    ConsRegRequestView requestMessage(&(*bufPtr)[0]);
    consumerName = requestMessage.getConsumerName();
    consumerPriority = requestMessage.getConsumerPriority();
    std::string reqString = requestMessage.getRequestParameterSet();
    if (reqString.size() >= 2) consumerRequest = reqString;
  }

  // create the buffer to hold the registration reply message
  const int BUFFER_SIZE = 100;
  char msgBuff[BUFFER_SIZE];

  // fetch the event server
  // (it and/or the job controller may not have been created yet)
  boost::shared_ptr<EventServer> eventServer;
  if (dpm_.get() != NULL)
  {
    eventServer = dpm_->getEventServer();
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
    // construct a parameter set from the consumer request
    boost::shared_ptr<edm::ParameterSet>
      requestParamSet(new edm::ParameterSet(consumerRequest));

    // create the local consumer interface and add it to the event server
    boost::shared_ptr<ConsumerPipe>
      consPtr(new ConsumerPipe(consumerName, consumerPriority,
                               activeConsumerTimeout_.value_,
                               idleConsumerTimeout_.value_,
                               requestParamSet));
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

  } else { // is this the right thing to send?
   // In wrong state for this message - return zero length stream, should return Msg NOTREADY
   int len = 0;
   out->getHTTPResponseHeader().addHeader("Content-Type", "application/octet-stream");
   out->getHTTPResponseHeader().addHeader("Content-Transfer-Encoding", "binary");
   out->write((char*) &mybuffer_[0],len);
  }

}

//////////// *** get DQMevent data web page //////////////////////////////////////////////////////////
void SMProxyServer::DQMeventdataWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
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
    //if (requestMessage.code() == Header::DQMEVENT_REQUEST)
    if (requestMessage.code() == Header::EVENT_REQUEST)
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
  if(fsm_.stateName()->toString() == "Enabled")
  { // We need to be in the enabled state

    std::string consumerName = "None provided";
    std::string consumerPriority = "normal";
    std::string consumerRequest = "*";

    // read the consumer registration message from the http input stream
    std::string lengthString = in->getenv("CONTENT_LENGTH");
    unsigned int contentLength = std::atol(lengthString.c_str());
    if (contentLength > 0)
    {
      auto_ptr< vector<char> > bufPtr(new vector<char>(contentLength));
      in->read(&(*bufPtr)[0], contentLength);
      ConsRegRequestView requestMessage(&(*bufPtr)[0]);
      consumerName = requestMessage.getConsumerName();
      consumerPriority = requestMessage.getConsumerPriority();
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
        consPtr(new DQMConsumerPipe(consumerName, consumerPriority,
                                 DQMactiveConsumerTimeout_.value_,
                                 DQMidleConsumerTimeout_.value_,
                                 consumerRequest));
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
  xdata::InfoSpace *is = xdata::InfoSpace::get(oss.str());

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
  is->fireItemAvailable("purgeTimeDQM",         &purgeTimeDQM_);
  is->fireItemAvailable("readyTimeDQM",         &readyTimeDQM_);
  is->fireItemAvailable("filePrefixDQM",        &filePrefixDQM_);
  is->fireItemAvailable("useCompressionDQM",    &useCompressionDQM_);
  is->fireItemAvailable("compressionLevelDQM",  &compressionLevelDQM_);
  //is->fireItemAvailable("nLogicalDisk",         &nLogicalDisk_);
  //is->fireItemAvailable("fileCatalog",          &fileCatalog_);
  is->fireItemAvailable("maxESEventRate",       &maxESEventRate_);
  is->fireItemAvailable("activeConsumerTimeout",&activeConsumerTimeout_);
  is->fireItemAvailable("idleConsumerTimeout",  &idleConsumerTimeout_);
  is->fireItemAvailable("consumerQueueSize",    &consumerQueueSize_);

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
  is->addItemRetrieveListener("purgeTimeDQM",         this);
  is->addItemRetrieveListener("readyTimeDQM",         this);
  is->addItemRetrieveListener("filePrefixDQM",        this);
  is->addItemRetrieveListener("useCompressionDQM",    this);
  is->addItemRetrieveListener("compressionLevelDQM",  this);
  //is->addItemRetrieveListener("nLogicalDisk",         this);
  //is->addItemRetrieveListener("fileCatalog",          this);
  is->addItemRetrieveListener("maxESEventRate",       this);
  is->addItemRetrieveListener("activeConsumerTimeout",this);
  is->addItemRetrieveListener("idleConsumerTimeout",  this);
  is->addItemRetrieveListener("consumerQueueSize",    this);
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
      storedVolume_   = pmeter_->totalvolumemb();
    else if (item == "progressMarker")
      progressMarker_ = ProgressMarker::instance()->status();
    is->unlock();
  } 
}



bool SMProxyServer::configuring(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start configuring ...");
    
    // the poll rate is set by maxESEventRate_ and we poll for both events
    // and DQM events at the same time!
    
    if (maxESEventRate_ < 0.0)
      maxESEventRate_ = 0.0;
    if (DQMmaxESEventRate_ < 0.0)
      DQMmaxESEventRate_ = 0.0;
    
    // consumer queues are not yet implemented, only one slot is available
    xdata::Integer cutoff(1);
    if (consumerQueueSize_ < cutoff)
      consumerQueueSize_ = cutoff;
    if (DQMconsumerQueueSize_ < cutoff)
      DQMconsumerQueueSize_ = cutoff;

    // set the urn as the consumer name to register with to SM
    std::string url = getApplicationDescriptor()->getContextDescriptor()->getURL();
    std::string urn = getApplicationDescriptor()->getURN();
    consumerName_ = url + "/" + urn + "/pushEventData";
    DQMconsumerName_ = url + "/" + urn + "/pushDQMEventData";
    // start a work loop that can process commands (do we need it in push mode?)
    try {
      dpm_.reset(new stor::DataProcessManager());
      
      boost::shared_ptr<EventServer>
        eventServer(new EventServer(maxESEventRate_));
      dpm_->setEventServer(eventServer);
      boost::shared_ptr<DQMEventServer>
        DQMeventServer(new DQMEventServer(DQMmaxESEventRate_));
      dpm_->setDQMEventServer(DQMeventServer);

      dpm_->setCollateDQM(collateDQM_);
      dpm_->setArchiveDQM(archiveDQM_);
      dpm_->setPurgeTimeDQM(purgeTimeDQM_);
      dpm_->setReadyTimeDQM(readyTimeDQM_);
      dpm_->setFilePrefixDQM(filePrefixDQM_);
      dpm_->setUseCompressionDQM(useCompressionDQM_);
      dpm_->setCompressionLevelDQM(compressionLevelDQM_);

      // If we are in pull mode, we need to know which Storage Managers to
      // poll for events and DQM events
      // Only add the StorageManager URLs at this configuration stage
      dpm_->setConsumerName(consumerName_.toString());
      dpm_->setDQMConsumerName(DQMconsumerName_.toString());
      unsigned int rsize = (unsigned int)smRegList_.size();
      for(unsigned int i = 0; i < rsize; ++i)
      {
        std::cout << "add to register list num = " << i << " url = " 
                  << smRegList_.elementAt(i)->toString() << std::endl;
        dpm_->addSM2Register(smRegList_.elementAt(i)->toString());
        dpm_->addDQMSM2Register(smRegList_.elementAt(i)->toString());
        smsenders_.insert(std::make_pair(smRegList_.elementAt(i)->toString(), false));
      }
    
    }
    catch(cms::Exception& e)
      {
	XCEPT_RAISE (toolbox::fsm::exception::Exception, 
		     e.explainSelf());
      }
    catch(std::exception& e)
      {
	XCEPT_RAISE (toolbox::fsm::exception::Exception, 
		     e.what());
      }
    catch(...)
      {
	XCEPT_RAISE (toolbox::fsm::exception::Exception, 
		     "Unknown Exception");
      }
    
    
    LOG4CPLUS_INFO(getApplicationLogger(),"Finished configuring!");
    
    fsm_.fireEvent("ConfigureDone",this);
  }
  catch (xcept::Exception &e) {
    string msg = "configuring FAILED: " + (string)e.what();
    fsm_.fireFailed(msg,this);
  }

  return false;
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
    // need this to register, get header and if we pull (poll) for events
    dpm_->start();

    LOG4CPLUS_INFO(getApplicationLogger(),"Finished enabling!");
    
    fsm_.fireEvent("EnableDone",this);
  }
  catch (xcept::Exception &e) {
    string msg = "enabling FAILED: " + (string)e.what();
    fsm_.fireFailed(msg,this);
  }
  
  return false;
}


bool SMProxyServer::stopping(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start stopping :) ...");

    // only write out DQM data if needed
    boost::shared_ptr<stor::DQMServiceManager> dqmManager;
    if (dpm_.get() != NULL)
    {
      dqmManager = dpm_->getDQMServiceManager();
      if(dqmManager.get() != NULL) {
        dqmManager->stop();
      }
    }
    // clear out events from queues
    boost::shared_ptr<EventServer> eventServer;
    boost::shared_ptr<DQMEventServer> dqmeventServer;
    if (dpm_.get() != NULL)
    {
      eventServer = dpm_->getEventServer();
      dqmeventServer = dpm_->getDQMEventServer();
    }
    if (eventServer.get() != NULL) eventServer->clearQueue();
    if (dqmeventServer.get() != NULL) dqmeventServer->clearQueue();
    // do not stop dpm_ as we don't want to register again and get the header again
    // need to redo if we switch to polling for events

    // should tell StorageManager applications we are stopping in which
    // case we need to register again

    LOG4CPLUS_INFO(getApplicationLogger(),"Finished stopping!");
    
    fsm_.fireEvent("StopDone",this);
  }
  catch (xcept::Exception &e) {
    string msg = "stopping FAILED: " + (string)e.what();
    fsm_.fireFailed(msg,this);
  }
  
  return false;
}


bool SMProxyServer::halting(toolbox::task::WorkLoop* wl)
{
  try {
    LOG4CPLUS_INFO(getApplicationLogger(),"Start halting ...");

    dpm_->stop();
    dpm_->join();
    
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
      dpm_.reset();
    }
    
    LOG4CPLUS_INFO(getApplicationLogger(),"Finished halting!");
    
    fsm_.fireEvent("HaltDone",this);
  }
  catch (xcept::Exception &e) {
    string msg = "halting FAILED: " + (string)e.what();
    fsm_.fireFailed(msg,this);
  }
  
  return false;
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

