#include "iDie.h"

#include "xdaq/NamespaceURI.h"

#include "xdata/InfoSpaceFactory.h"
#include "toolbox/task/TimerFactory.h"

#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xoap/domutils.h"

#include <boost/tokenizer.hpp>

#include <netinet/in.h>
#include <sstream>
#include <errno.h>
#include <iomanip>
#include <algorithm>

#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
//#include <sys/dir.h>
#include <time.h>
#include <math.h>

#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/FormFile.h"
#include "cgicc/HTMLClasses.h"

#include "EventFilter/Utilities/interface/DebugUtils.h"

//#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
//#undef HAVE_STAT
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "EventFilter/Utilities/interface/ParameterSetRetriever.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/src/DQMService.h"
using namespace evf;

#define ROLL 20
#define PASTUPDATES 4

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
iDie::iDie(xdaq::ApplicationStub *s) 
  : xdaq::Application(s)
  , log_(getApplicationLogger())
  , dqmState_("Null")
  , instance_(0)
  , runNumber_(0)
  , lastRunNumberSet_(0)
  , runActive_(false)
  , runTS_(0)
  , latencyTS_(0)
  , dqmCollectorHost_()
  , dqmCollectorPort_()
  , totalCores_(0)
  , nstates_(0)
  , cpustat_(std::vector<std::vector<int> >(0))
  , last_ls_(0)
  , f_(0)
  , t_(0)
  , b_(0)
  , b1_(0)
  , b2_(0)
  , b3_(0)
  , b4_(0)
  , datap_(0)
  , trppriv_(0)
  , nModuleLegendaMessageReceived_(0)
  , nPathLegendaMessageReceived_(0)
  , nModuleLegendaMessageWithDataReceived_(0)
  , nPathLegendaMessageWithDataReceived_(0)
  , nModuleHistoMessageReceived_(0)
  , nPathHistoMessageReceived_(0)
  , nDatasetLegendaMessageReceived_(0)
  , nDatasetLegendaMessageWithDataReceived_(0)
  , evtProcessor_(0)
  , meInitialized_(false)
  , meInitializedStreams_(false)
  , meInitializedDatasets_(false)
  , dqmService_(nullptr)
  , dqmStore_(nullptr)
  , dqmEnabled_(false)
  , debugMode_(false)
  , saveLsInterval_(10)
  , ilumiprev_(0)
  , dqmSaveDir_("")
  , dqmFilesWritable_(true)
  , topLevelFolder_("DAQ")
  , savedForLs_(0)
  , reportingStart_(0)
  , dsMismatch(0)
{
  // initialize application info
  url_     =
    getApplicationDescriptor()->getContextDescriptor()->getURL()+"/"+
    getApplicationDescriptor()->getURN();
  class_   =getApplicationDescriptor()->getClassName();
  instance_=getApplicationDescriptor()->getInstance();
  hostname_=getApplicationDescriptor()->getContextDescriptor()->getURL();
  getApplicationDescriptor()->setAttribute("icon", "/evf/images/idieapp.jpg");

  //soap interface
  xoap::bind(this,&evf::iDie::fsmCallback,"Configure",XDAQ_NS_URI);
  xoap::bind(this,&evf::iDie::fsmCallback,"Enable",   XDAQ_NS_URI);
  xoap::bind(this,&evf::iDie::fsmCallback,"Stop",     XDAQ_NS_URI);
  xoap::bind(this,&evf::iDie::fsmCallback,"Halt",     XDAQ_NS_URI);

  // web interface
  xgi::bind(this,&evf::iDie::defaultWeb,               "Default");
  xgi::bind(this,&evf::iDie::summaryTable,             "summary");
  xgi::bind(this,&evf::iDie::detailsTable,             "details");
  xgi::bind(this,&evf::iDie::dumpTable,                "dump"   );
  xgi::bind(this,&evf::iDie::updater,                  "updater");
  xgi::bind(this,&evf::iDie::iChoke,                   "iChoke" );
  xgi::bind(this,&evf::iDie::iChokeMiniInterface,      "iChokeMiniInterface" );
  xgi::bind(this,&evf::iDie::spotlight,                "Spotlight" );
  xgi::bind(this,&evf::iDie::postEntry,                "postEntry");
  xgi::bind(this,&evf::iDie::postEntryiChoke,          "postChoke");
  //  gui_->setSmallAppIcon("/evf/images/Hilton.gif");
  //  gui_->setLargeAppIcon("/evf/images/Hilton.gif");

  xdata::InfoSpace *ispace = getApplicationInfoSpace();
  ispace->fireItemAvailable("runNumber",            &runNumber_                   );
  getApplicationInfoSpace()->addItemChangedListener("runNumber",              this);
  ispace->fireItemAvailable("dqmCollectorHost",         &dqmCollectorHost_        );
  ispace->fireItemAvailable("dqmCollectorPort",         &dqmCollectorPort_        );
  ispace->fireItemAvailable("saveLsInterval",           &saveLsInterval_          );
  ispace->fireItemAvailable("dqmSaveDir",               &dqmSaveDir_              );
  ispace->fireItemAvailable("dqmFilesWritableByAll",    &dqmFilesWritable_        );
  ispace->fireItemAvailable("dqmTopLevelFolder",        &topLevelFolder_          );
  ispace->fireItemAvailable("dqmEnabled",               &dqmEnabled_              );
  ispace->fireItemAvailable("debugMode",                &debugMode_               );

  // timestamps
  lastModuleLegendaMessageTimeStamp_.tv_sec=0;
  lastModuleLegendaMessageTimeStamp_.tv_usec=0;
  lastPathLegendaMessageTimeStamp_.tv_sec=0;
  lastPathLegendaMessageTimeStamp_.tv_usec=0;
  lastDatasetLegendaMessageTimeStamp_.tv_sec=0;
  lastDatasetLegendaMessageTimeStamp_.tv_usec=0;
  runStartDetectedTimeStamp_.tv_sec=0;
  runStartDetectedTimeStamp_.tv_usec=0;

  //dqm python configuration
  configString_= "import FWCore.ParameterSet.Config as cms\n";
  configString_+="process = cms.Process(\"iDieDQM\")\n";
  configString_+="process.source = cms.Source(\"EmptySource\")\n";
  configString_+="process.DQMStore = cms.Service(\"DQMStore\",\n";
  configString_+="  referenceFileName = cms.untracked.string(''),\n";
  configString_+="  verbose = cms.untracked.int32(0),\n";
  configString_+="  verboseQT = cms.untracked.int32(0),\n";
  configString_+="  collateHistograms = cms.untracked.bool(False))\n";
  configString_+="process.DQM = cms.Service(\"DQM\",\n";
  configString_+="  debug = cms.untracked.bool(False),\n";
  configString_+="  publishFrequency = cms.untracked.double(1.0),\n";
  configString_+="  collectorPort = cms.untracked.int32(EMPTYPORT),\n";
  configString_+="  collectorHost = cms.untracked.string('EMPTYHOST'),\n";
  configString_+="  filter = cms.untracked.string(''),\n";
  configString_+="  verbose = cms.untracked.bool(False))\n";
  configString_+="process.p = cms.Path()\n";

  epInstances   =     {7,    8,     12,  16, 24,  32};
  epMax         =     {8,    8,     24,  32, 24,  32};
  HTscaling     =     {1,    1,   0.28,0.28, 0.28,0.28};
  machineWeight =     {91.6, 91.6, 253, 352, 253, 352};
  machineWeightInst = {80.15,91.6, 196, 275, 253, 352};

  for (unsigned int i=0;i<epInstances.size();i++) {
    currentLs_.push_back(0);
    nbSubsList[epInstances[i]]=i;
    nbSubsListInv[i]=epInstances[i];
    std::map<unsigned int, unsigned int> mptmp;
    occupancyNameMap.push_back(mptmp);
  }
  nbSubsClasses = epInstances.size();
  lsHistory = new std::deque<lsStat*>[nbSubsClasses];
  //umask for setting permissions of created directories

  //flashlists
  flashRunNumber_=0;
  cpuLoadLastLs_=0;
  cpuLoadSentLs_=0;
  std::string cpuInfoSpaceName="filterFarmUsageAndTiming";
  toolbox::net::URN urn = this->createQualifiedInfoSpace(cpuInfoSpaceName);
  cpuInfoSpace_ = xdata::getInfoSpaceFactory()->get(urn.toString());
  cpuInfoSpace_->fireItemAvailable("runNumber",&flashRunNumber_);
  cpuInfoSpace_->fireItemAvailable("lumiSection",&flashLoadLs_);
  cpuInfoSpace_->fireItemAvailable("hltCPULoad",&flashLoad_);
  cpuInfoSpace_->fireItemAvailable("systemCPULoad",&flashLoadPS_);
  cpuInfoSpace_->fireItemAvailable("eventTime7EP",&flashLoadTime7_);
  cpuInfoSpace_->fireItemAvailable("eventTime8EP",&flashLoadTime8_);
  cpuInfoSpace_->fireItemAvailable("eventTime12EP",&flashLoadTime12_);
  cpuInfoSpace_->fireItemAvailable("eventTime16EP",&flashLoadTime16_);
  cpuInfoSpace_->fireItemAvailable("eventTime24EP",&flashLoadTime24_);
  cpuInfoSpace_->fireItemAvailable("eventTime32EP",&flashLoadTime32_);
  cpuInfoSpace_->fireItemAvailable("hltProcessingRate",&flashLoadRate_);

  monNames_.push_back("runNumber");
  monNames_.push_back("lumiSection");
  monNames_.push_back("hltCPULoad");
  monNames_.push_back("systemCPULoad");
  monNames_.push_back("eventTime7EP");
  monNames_.push_back("eventTime8EP");
  monNames_.push_back("eventTime12EP");
  monNames_.push_back("eventTime16EP");
  monNames_.push_back("eventTime24EP");
  monNames_.push_back("eventTime32EP");
  monNames_.push_back("hltProcessingRate");

  //be permissive for written files
  umask(000);

  //start flashlist updater timer
  try {
   toolbox::task::Timer * timer = toolbox::task::getTimerFactory()->createTimer("xmas-iDie-updater");
   toolbox::TimeInterval timerInterval;
   timerInterval.fromString("PT15S");
   toolbox::TimeVal timerStart;
   timerStart = toolbox::TimeVal::gettimeofday();
   //timer->start();
   timer->scheduleAtFixedRate( timerStart, this, timerInterval, 0, "xmas-iDie-producer" );
  }
  catch (xdaq::exception::Exception& e) {
    LOG4CPLUS_WARN(getApplicationLogger(), e.what());
  }
}


//______________________________________________________________________________
iDie::~iDie()
{
}

//______________________________________________________________________________
void iDie::actionPerformed(xdata::Event& e)
{
  
  if (e.type()=="ItemChangedEvent" ) {
    std::string item = dynamic_cast<xdata::ItemChangedEvent&>(e).itemName();
    
    if ( item == "runNumber") {
      LOG4CPLUS_WARN(getApplicationLogger(),
		     "New Run was started - iDie will reset");
      reset();
      runActive_=true;
      setRunStartTimeStamp();

      dqmState_ = "Prepared";
      if (dqmEnabled_.value_) { 
	if (!evtProcessor_) initFramework();
	if (!meInitialized_) initMonitorElements();
	doFlush();
      }
    }
    
  }
}

//______________________________________________________________________________
xoap::MessageReference iDie::fsmCallback(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  
  xoap::SOAPPart     part    =msg->getSOAPPart();
  xoap::SOAPEnvelope env     =part.getEnvelope();
  xoap::SOAPBody     body    =env.getBody();
  DOMNode           *node    =body.getDOMNode();
  DOMNodeList       *bodyList=node->getChildNodes();
  DOMNode           *command =0;
  std::string             commandName;
  
  for (unsigned int i=0;i<bodyList->getLength();i++) {
    command = bodyList->item(i);
    if(command->getNodeType() == DOMNode::ELEMENT_NODE) {
      commandName = xoap::XMLCh2String(command->getLocalName());
      break;
    }
  }
  
  if (commandName.empty()) {
    XCEPT_RAISE(xoap::exception::Exception,"Command not found.");
  }
  
  // fire appropriate event and create according response message
  try {

    // response string
    xoap::MessageReference reply = xoap::createMessage();
    xoap::SOAPEnvelope envelope  = reply->getSOAPPart().getEnvelope();
    xoap::SOAPName responseName  = envelope.createName(commandName+"Response",
						       "xdaq",XDAQ_NS_URI);
    xoap::SOAPBodyElement responseElem =
      envelope.getBody().addBodyElement(responseName);
    
    // generate correct return state string
    std::string state;
    if(commandName == "Configure") {dqmState_ = "Ready"; state = "Ready";}
    else if(commandName == "Enable" || commandName == "Start") {
      dqmState_ = "Enabled"; state = "Enabled";
      setRunStartTimeStamp();

    }
    else if(commandName == "Stop" || commandName == "Halt") {
      runActive_=false;
      //EventInfo:reset timestamps
      runTS_=0.;
      latencyTS_=0;
      //cleanup flashlist data
      cpuLoadLastLs_=0;
      cpuLoadSentLs_=0;
      //remove histograms
      std::cout << " Stopping/Halting iDie. command=" << commandName << " initialized=" << meInitialized_ << std::endl;
      if (meInitialized_) {
        dqmState_ = "Removed";
        usleep(10000);//propagating dqmState to caches
        meInitialized_=false;
        meInitializedStreams_=false;
        meInitializedDatasets_=false;
        sleep(1);//making sure that any running ls update finishes

        dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/Layouts/Streams/");
        dqmStore_->removeContents();
        dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/Layouts/Datasets/");
        dqmStore_->removeContents();
        dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/Layouts/Modules/");
        dqmStore_->removeContents();
        dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/Layouts/Tables/");
        dqmStore_->removeContents();
        dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/Layouts/");
        dqmStore_->removeContents();
        dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/EventInfo/");
        dqmStore_->removeContents();
        doFlush(); 
      }
      if (reportingStart_) delete reportingStart_;
      reportingStart_=0;
      state = "Ready";
    }
    //else if(commandName == "Halt") state = "Halted";
    else state = "BOH";

    xoap::SOAPName    stateName     = envelope.createName("state",
							  "xdaq",XDAQ_NS_URI);
    xoap::SOAPElement stateElem     = responseElem.addChildElement(stateName);
    xoap::SOAPName    attributeName = envelope.createName("stateName",
							  "xdaq",XDAQ_NS_URI);
    stateElem.addAttribute(attributeName,state);
    
    return reply;
  }
  catch (toolbox::fsm::exception::Exception & e) {
    XCEPT_RETHROW(xoap::exception::Exception,"invalid command.",e);
  }	
  


}

//______________________________________________________________________________
void iDie::defaultWeb(xgi::Input *in,xgi::Output *out)
  throw (xgi::exception::Exception)
{
  cgicc::Cgicc cgi(in);
  std::string method = cgi.getEnvironment().getRequestMethod();
  if(method == "POST"){
    unsigned int run = 0;
    std::vector<cgicc::FormEntry> el1 = cgi.getElements();
    cgi.getElement("run",el1);
    if(el1.size()!=0){
      run = el1[0].getIntegerValue();
      if(run > runNumber_.value_ || runNumber_.value_==0){
	runNumber_.value_ = run;
	runActive_=true;
	if(runNumber_.value_!=0) 
	  {
	    reset();
	    if(f_ == 0)
	      {
		std::ostringstream ost;
		ost << "microReport"<<runNumber_<<".root";
		f_ = new TFile(ost.str().c_str(),"RECREATE","microreport");
	      }
	  }
      }
    }
    internal::fu fuinstance;

    fuinstance.ccount = 0;
    std::string hostname = cgi.getEnvironment().getRemoteHost();
    std::transform(hostname.begin(), hostname.end(),
		   hostname.begin(), ::toupper);
    fus_[hostname] = fuinstance;
  }
  else{
    *out << "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\">" 
	 << "<html><head><title>" << getApplicationDescriptor()->getClassName()
	 << getApplicationDescriptor()->getInstance() << "</title>"
	 << "<meta http-equiv=\"REFRESH\" content=\"0;url=/evf/html/idiePage.html\">"
	 << "</head></html>";
  }
}

//______________________________________________________________________________
void iDie::updater(xgi::Input *in,xgi::Output *out)
  throw (xgi::exception::Exception)
{
  *out << runNumber_.value_ << std::endl;
}
//______________________________________________________________________________
void iDie::summaryTable(xgi::Input *in,xgi::Output *out)
  throw (xgi::exception::Exception)
{
    *out << "<tr><td>"<<fus_.size()<<"</td><td>" << totalCores_ 
	 << "</td><td></td></tr>" << std::endl;
}

//______________________________________________________________________________
void iDie::detailsTable(xgi::Input *in,xgi::Output *out)
  throw (xgi::exception::Exception)
{
  timeval tv;
  gettimeofday(&tv,0);
  time_t now = tv.tv_sec;
  for(ifmap i = fus_.begin(); i != fus_.end(); i++)
    if((*i).second.ccount != 0){
      *out << "<tr><td " 
	   << (now-(*i).second.tstamp<300 ? "style=\"background-color:red\"" : "")
	   << ">"<<(*i).first<<"</td><td>" 
	   << (*i).second.ccount << "</td>"
	   << "<td onClick=loaddump(\'" << url_.value_ << "/dump?name="
	   << (*i).first << "\')>" << (*i).second.cpids.back()
	   << "</td><td>" <<(*i).second.signals.back() 
	   << "</td></tr>" << std::endl;
    }
}

//______________________________________________________________________________
void iDie::dumpTable(xgi::Input *in,xgi::Output *out)
  throw (xgi::exception::Exception)
{
  cgicc::Cgicc cgi(in); 

  std::vector<cgicc::FormEntry> el1;
  cgi.getElement("name",el1);
  if(el1.size()!=0){
    std::string hostname = el1[0].getValue();
    std::transform(hostname.begin(), hostname.end(),
		   hostname.begin(), ::toupper);
    ifmap fi = fus_.find(hostname);    
    if(fi!=fus_.end()){
      *out << (*fi).second.stacktraces.back() << std::endl;
    }
    else{ 
      for(fi=fus_.begin(); fi != fus_.end(); fi++) 
	std::cout << "known hosts: " << (*fi).first << std::endl;
    }
  }
}

//______________________________________________________________________________
void iDie::iChokeMiniInterface(xgi::Input *in,xgi::Output *out)
  throw (xgi::exception::Exception)
{
  unsigned int i = 0;

  if(last_ls_==0) return; //wait until at least one complete cycle so we have all arrays sized correctly !!!
  *out << "<div id=\"cls\">" << last_ls_ << "</div>" 
       << "<div id=\"clr\">" << cpuentries_[last_ls_-1] << "</div>" << std::endl;
  sorted_indices tmp(cpustat_[last_ls_-1]);
  //  std::sort(tmp.begin(),tmp.end());// figure out how to remap indices of legenda
  *out << "<tbody id=\"cpue\">";
  while(i<nstates_){
    if(tmp[i]!=0) *out << "<tr><td>" << mapmod_[tmp.ii(i)] << "</td>" << "<td>" 
		       << float(tmp[i])/float(cpuentries_[last_ls_-1]) << "</td></tr>";
    i++;
  }
  *out << "</tbody>\n";
  *out << "<tbody id=\"cpui\"><tr><td></td>";
  unsigned int begin = last_ls_<5 ? 0 : last_ls_-5;
  for(i=begin; i < last_ls_; i++)
    *out << "<td>" << i +1 << "</td>";
  *out << "</tr><tr><td></td>";
  for(i=begin; i < last_ls_; i++)
    *out << "<td>" << float(cpustat_[i][2])/float(cpuentries_[i]) << "</td>";
  *out << "</tr></tbody>\n";

  *out << "<tbody id=\"rate\"><tr><td></td>";
  begin = last_ls_<5 ? 0 : last_ls_-5;
  for(i=begin; i < last_ls_; i++)
    *out << "<td>" << float(trp_[i].eventSummary.totalEventsPassed)/float(trp_[i].eventSummary.totalEvents) << "</td>"; 
  *out << "</tr>\n<tr><td></td>";
  for(i=begin; i < last_ls_; i++)
    *out << "<td>" << trp_[i].eventSummary.totalEvents << "</td>"; 
  *out << "</tr>\n<tr><td></td>";
  for(int j = 0; j < trp_[0].trigPathsInMenu; j++)
    {
      *out << "<tr><td></td>";
      for(i=begin; i < last_ls_; i++)
	*out << "<td>" << trp_[i].trigPathSummaries[j].timesPassed << "("
	     << trp_[i].trigPathSummaries[j].timesPassedL1 << ")("
	     << trp_[i].trigPathSummaries[j].timesPassedPs << ")</td>";
      *out << "<td>" << mappath_[j] << "</td>";
      *out << "</tr>\n";
    }
  for(int j = 0; j < trp_[0].endPathsInMenu; j++)
    {
      *out << "<tr><td></td>";
      for(i=begin; i < last_ls_; i++)
	*out << "<td>" << trp_[i].endPathSummaries[j].timesPassed << "</td>";
      *out << "<td>" << mappath_[j+trp_[last_ls_-1].trigPathsInMenu] << "</td>";
      *out << "</tr>\n";
    }
  *out << "</tbody>\n";
}

//______________________________________________________________________________
void iDie::iChoke(xgi::Input *in,xgi::Output *out)
  throw (xgi::exception::Exception)
{
    *out << "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\">" 
	 << "<html><head><title>" << getApplicationDescriptor()->getClassName()
	 << getApplicationDescriptor()->getInstance() << "</title>"
	 << "<meta http-equiv=\"REFRESH\" content=\"0;url=/evf/html/ichokePage.html\">"
	 << "</head></html>";



}

//______________________________________________________________________________
void iDie::postEntry(xgi::Input*in,xgi::Output*out)
  throw (xgi::exception::Exception)
{

  timeval tv;
  gettimeofday(&tv,0);
  time_t now = tv.tv_sec;

  try {
    cgicc::Cgicc cgi(in); 
    unsigned int run = 0;
    pid_t cpid = 0;
    /*  cgicc::CgiEnvironment cgie(in);
	cout << "query = "  << cgie.getContentLength() << endl;
	*/
    std::vector<cgicc::FormEntry> el1;
    el1 = cgi.getElements();
    //   for(unsigned int i = 0; i < el1.size(); i++)
    //     std::cout << "name="<<el1[i].getName() << std::endl;
    el1.clear();
    cgi.getElement("run",el1);
    if(el1.size()!=0)
    {
      run =  el1[0].getIntegerValue();
    }
    el1.clear();
    cgi.getElement("stacktrace",el1);
    if(el1.size()!=0)
    {
      cpid = run;
      //      std::cout << "=============== stacktrace =============" << std::endl;
      //      std::cout << el1[0].getValue() << std::endl;
      if(el1[0].getValue().find("Dead")==0){

	std::string host = cgi.getEnvironment().getRemoteHost();
	std::transform(host.begin(), host.end(),
		       host.begin(), ::toupper);
	ifmap fi = fus_.find(host);
	if(fi!=fus_.end()){
	  fus_.erase(fi);
	}
	if(fus_.size()==0) { //close the root file if we know the run is over

	  if(f_!=0){
	    f_->cd();
	    f_->Write();
	  }
	  if(t_ != 0) {
	    delete t_;
	    t_ = 0;
	  }
	  if(f_!=0){
	    f_->Close();
	    delete f_; f_ = 0;
	  }
	}
      }
      else{
	totalCores_++;
	std::string st = el1[0].getValue();
	std::string sig; 
	size_t psig = st.find("signal");
	if(psig != std::string::npos)
	  sig = st.substr(psig,9);
	std::string host = cgi.getEnvironment().getRemoteHost();
	std::transform(host.begin(), host.end(),
		       host.begin(), ::toupper);
	ifmap fi = fus_.find(host);
	if(fi!=fus_.end()){
	  (*fi).second.tstamp = now;
	  (*fi).second.ccount++;
	  (*fi).second.cpids.push_back(cpid);
	  (*fi).second.signals.push_back(sig);
	  (*fi).second.stacktraces.push_back(st);
	}
      }
    }
    el1.clear();
    cgi.getElement("legenda",el1);
    if(el1.size()!=0)
    {
      parsePathLegenda(el1[0].getValue());
    }
    cgi.getElement("LegendaAux",el1);
    if (el1.size()!=0)
    {
      parseDatasetLegenda(el1[0].getValue());
    }
    cgi.getElement("trp",el1);
    if(el1.size()!=0)
    {
      unsigned int lsid = run;
      parsePathHisto((unsigned char*)(el1[0].getValue().c_str()),lsid);
    }
    el1.clear();
  }
  catch (edm::Exception &e) {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Caught edm exception in postEntry: " << e.what());
  }
  catch (cms::Exception &e) {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Caught cms exception in postEntry: " << e.what());
  }
  catch (std::exception &e) {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Caught std exception in postEntry: " << e.what());
  }
  catch (...) {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Caught unknown exception in postEntry");
  }

}

//______________________________________________________________________________
void iDie::postEntryiChoke(xgi::Input*in,xgi::Output*out)
  throw (xgi::exception::Exception)
{
  //  std::cout << "postEntryiChoke " << std::endl;
 
  if (dqmEnabled_.value_) {
    if (!evtProcessor_) initFramework();
    if (!meInitialized_) {
      if (dqmState_!="Removed")	initMonitorElements();
    }
  }


  unsigned int lsid = 0;
  try {
    cgicc::Cgicc cgi(in); 
    /*  cgicc::CgiEnvironment cgie(in);
	cout << "query = "  << cgie.getContentLength() << endl;
	*/
    std::vector<cgicc::FormEntry> el1;
    el1 = cgi.getElements();
    //   for(unsigned int i = 0; i < el1.size(); i++)
    //     std::cout << "name="<<el1[i].getName() << std::endl;
    el1.clear();
    cgi.getElement("run",el1);
    if(el1.size()!=0)
    {
      lsid =  el1[0].getIntegerValue();
    }
    el1.clear();

    //with the first message for the new lsid, resize all containers so 
    // a web access won't address an invalid location in case it interleaves between 
    // the first cpustat update and the first scalers update or viceversa
    if(lsid!=0){
      if(lsid>cpustat_.size()){
	cpustat_.resize(lsid,std::vector<int>(nstates_,0));
	cpuentries_.resize(lsid,0);
      }
      if(lsid>trp_.size()){
	trp_.resize(lsid);
	funcs::reset(&trp_[lsid-1]);
	trpentries_.resize(lsid,0);
      }
      if(last_ls_ < lsid) {
	last_ls_ = lsid; 
	funcs::reset(&trp_[lsid-1]);
	if(t_ && (last_ls_%10==0)) t_->Write();
      } 
    }

    cgi.getElement("legenda",el1);
    if(el1.size()!=0)
    {
      parseModuleLegenda(el1[0].getValue());
    }
    cgi.getElement("trp",el1);
    if(el1.size()!=0)
    {
      parseModuleHisto(el1[0].getValue().c_str(),lsid);
    }
    el1.clear();
  }

  catch (edm::Exception &e) {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Caught edm exception in postEntryiChoke: " << e.what());
  }
  catch (cms::Exception &e) {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Caught cms exception in postEntryiChoke: " << e.what());
  }
  catch (std::exception &e) {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Caught std exception in postEntryiChoke: " << e.what());
  }
  catch (...) {
    LOG4CPLUS_ERROR(getApplicationLogger(),"Caught unknown exception in postEntryiChoke");
  }
}


void iDie::reset()
{
  fus_.erase(fus_.begin(),fus_.end());
  totalCores_=0;
  last_ls_ = 0;
  trp_.clear();
  trpentries_.clear();
  cpustat_.clear();
  cpuentries_.clear();

  if(f_!=0){
    f_->cd();
    f_->Write();
  }

  if(t_ != 0)
  {
    delete t_; t_=0;
  }

  if(f_!=0){
    f_->Close();
    delete f_; f_ = 0;
  }
  if(datap_ != 0)
    {delete datap_; datap_ = 0;}
  b_=0; b1_=0; b2_=0; b3_=0; b4_=0;

}

void iDie::parseModuleLegenda(std::string leg)
{
  nModuleLegendaMessageReceived_++;
  if(leg=="") return;
  gettimeofday(&lastModuleLegendaMessageTimeStamp_,0);
  nModuleLegendaMessageWithDataReceived_++;
  mapmod_.clear();
  //  if(cpustat_) delete cpustat_;
  boost::char_separator<char> sep(",");
  boost::tokenizer<boost::char_separator<char> > tokens(leg, sep);
  for (boost::tokenizer<boost::char_separator<char> >::iterator tok_iter = tokens.begin();
       tok_iter != tokens.end(); ++tok_iter){
    mapmod_.push_back((*tok_iter));
  }
  nstates_ = mapmod_.size();
  //  cpustat_ = new int[nstates_];
//   for(int i = 0; i < nstates_; i++)
//     cpustat_[i]=0;	
//   cpuentries_ = 0;
}

void iDie::parseModuleHisto(const char *crp, unsigned int lsid)
{
  if(lsid==0) return;
  nModuleHistoMessageReceived_++;
  int *trp = (int*)crp;
  if(t_==0 && f_!=0){
    datap_ = new int[nstates_+5];
    std::ostringstream ost;
    ost<<mapmod_[0]<<"/I";
    for(unsigned int i = 1; i < nstates_; i++)
      ost<<":"<<mapmod_[i];
    ost<<":nsubp:instance:nproc:ncpubusy";//
    f_->cd();
    t_ = new TTree("microReport","microstate report tree");
    t_->SetAutoSave(500000);
    b_ = t_->Branch("microstates",datap_,ost.str().c_str());
    b1_ = t_->Branch("ls",&lsid,"ls/I");

  }

  memcpy(datap_,trp,(nstates_+5)*sizeof(int));
  //check ls for subprocess type
  unsigned int datapLen_ = nstates_+5;
  unsigned int nbsubs_ = datap_[datapLen_-5];
  unsigned int nbproc_ = datap_[datapLen_-3];
  unsigned int ncpubusy_ = datap_[datapLen_-2];
  unsigned int deltaTms_ = datap_[datapLen_-1];

  //find index number
  int nbsIdx = -1;

  /* debugging test
  unsigned int randls = 0;
  unsigned int randslot = 0;
  if (lsid>3) {
    randslot = rand();
    if (randslot%2) nbsubs_=7;
    else nbsubs_=8;
    randls = rand();
    randls%=3;
    lsid-=randls;
  }
  */

  if (meInitialized_ && nbSubsList.find(nbsubs_)!=nbSubsList.end() && lsid) {
     nbsIdx = nbSubsList[nbsubs_];
    if (currentLs_[nbsIdx]<lsid) {//new lumisection for this ep class
      if (currentLs_[nbsIdx]!=0) {
        if (lsHistory[nbsIdx].size()) {
	  
	  //refresh run/lumi number and timestamp
          runId_->Fill(runNumber_.value_);
	  lumisecId_->Fill(currentLs_[nbsIdx]);
	  struct timeval now;
	  gettimeofday(&now, 0);
	  double time_now = now.tv_sec + 1e-6*now.tv_usec;
	  eventTimeStamp_->Fill( time_now );

	  //check if run timestamp is set
	  double runTS = runTS_;
	  if (runTS==0.)
            runTS_ = time_now;

	  runStartTimeStamp_->Fill(runTS);

	  processLatencyMe_->Fill(time_now-latencyTS_);
	  latencyTS_=time_now;
	  processTimeStampMe_->Fill(time_now);

	  //do histogram updates for the lumi
	  lsStat * lst = lsHistory[nbsIdx].back();
	  fillDQMStatHist(nbsIdx,currentLs_[nbsIdx]);
	  fillDQMModFractionHist(nbsIdx,currentLs_[nbsIdx],lst->getNSampledNonIdle(),
	      lst->getOffendersVector());
	  doFlush();
	  perLumiFileSaver(currentLs_[nbsIdx]);
	  perTimeFileSaver();
	}
      }

      currentLs_[nbsIdx]=lsid;

      //add elements for new lumisection, fill the gap if needed
      unsigned int lclast = commonLsHistory.size() ? commonLsHistory.back()->ls_:0;
      for (unsigned int newls=lclast+1;newls<=lsid;newls++) {
          commonLsHistory.push_back(new commonLsStat(newls,epInstances.size()));
      }

      unsigned int lhlast = lsHistory[nbsIdx].size() ? lsHistory[nbsIdx].back()->ls_:0;
      for (size_t newls=lhlast+1;newls<=lsid;newls++) {
        lsHistory[nbsIdx].push_back(new lsStat(newls,nbsubs_,nModuleLegendaMessageReceived_,nstates_));
      }

      //remove old elements from queues
      while (commonLsHistory.size()>ROLL) {delete commonLsHistory.front(); commonLsHistory.pop_front();}
      while (lsHistory[nbsIdx].size()>ROLL) {delete lsHistory[nbsIdx].front(); lsHistory[nbsIdx].pop_front();}
    }
    if (currentLs_[nbsIdx]>=lsid) { // update for current or previous lumis
      unsigned int qsize=lsHistory[nbsIdx].size();
      unsigned int delta = currentLs_[nbsIdx]-lsid;
      if (qsize>delta && delta<ROLL) {
        lsStat * lst = (lsHistory[nbsIdx])[qsize-delta-1];
	unsigned int cumulative_ = 0;
	auto fillvec = lst->getModuleSamplingPtr();
	for (unsigned int i=0;i<nstates_;i++) {
	  cumulative_+=datap_[i];
	  if (fillvec) {
	    fillvec[i].second+=datap_[i];
	  }
	}
	unsigned int busyCounts = cumulative_-datap_[2];
	lst->update(busyCounts,datap_[2],nbproc_,ncpubusy_,deltaTms_);
      }
    }
  }
  else {
    //no predefined plots for this number of sub processes
  }

  if(t_!=0){
    t_->SetEntries(t_->GetEntries()+1); b_->Fill(); b1_->Fill();
  }

  for(unsigned int i=0;i<nstates_; i++)
    {
      cpustat_[lsid-1][i] += trp[i];
      cpuentries_[lsid-1] += trp[i];
    }
}


void iDie::parsePathLegenda(std::string leg)
{
  nPathLegendaMessageReceived_++;
  if(leg=="")return;
  gettimeofday(&lastPathLegendaMessageTimeStamp_,0);
  nPathLegendaMessageWithDataReceived_++;
  mappath_.clear();
  boost::char_separator<char> sep(",");
  boost::tokenizer<boost::char_separator<char> > tokens(leg, sep);
  endPathNames_.clear();
  for (boost::tokenizer<boost::char_separator<char> >::iterator tok_iter = tokens.begin();
       tok_iter != tokens.end(); ++tok_iter){
      mappath_.push_back((*tok_iter));

      if (std::string(*tok_iter).find("Output")!=std::string::npos) {
	std::string path_token = *tok_iter;
	if (path_token.find("=")!=std::string::npos)
          endPathNames_.push_back(path_token.substr(path_token.find("=")+1));
	else
          endPathNames_.push_back(*tok_iter);
      }
  }
  //look for daqval-type menu if no "Output" endpaths found
  if (!endPathNames_.size()) {
	  
    for (boost::tokenizer<boost::char_separator<char> >::iterator tok_iter = tokens.begin();
	tok_iter != tokens.end(); ++tok_iter){

      if (std::string(*tok_iter).find("output")!=std::string::npos) {
	std::string path_token = *tok_iter;
	if (path_token.find("=")!=std::string::npos)
	  endPathNames_.push_back(path_token.substr(path_token.find("=")+1));
	else
	  endPathNames_.push_back(*tok_iter);
      }
    }
  }
}

void iDie::parseDatasetLegenda(std::string leg)
{
  nDatasetLegendaMessageReceived_++;
  datasetNames_.clear();
  dsMismatch=0;
  if(leg=="")return;
  gettimeofday(&lastDatasetLegendaMessageTimeStamp_,0);
  nDatasetLegendaMessageWithDataReceived_++;
  try {
    boost::char_separator<char> sep(",");
    boost::tokenizer<boost::char_separator<char> > tokens(leg, sep);
    for (boost::tokenizer<boost::char_separator<char> >::iterator tok_iter = tokens.begin();
	tok_iter != tokens.end(); ++tok_iter) {
      datasetNames_.push_back((*tok_iter));
    }
  }
  catch (...) {}
}

void iDie::parsePathHisto(const unsigned char *crp, unsigned int lsid)
{
  if(lsid==0) return;
  nPathHistoMessageReceived_++;
//   if(lsid>=trp_.size()){
//     trp_.resize(lsid);
//     funcs::reset(&trp_[lsid-1]);
//     trpentries_.resize(lsid,0);
//   }
  trppriv_ = (TriggerReportStatic*)crp;
  for( int i=0; i< trppriv_->trigPathsInMenu; i++)
    {
      r_.ptimesRun[i] = trppriv_->trigPathSummaries[i].timesRun;
      r_.ptimesPassedPs[i] = trppriv_->trigPathSummaries[i].timesPassedPs;
      r_.ptimesPassedL1[i] = trppriv_->trigPathSummaries[i].timesPassedL1;
      r_.ptimesPassed[i] = trppriv_->trigPathSummaries[i].timesPassed;
      r_.ptimesFailed[i] = trppriv_->trigPathSummaries[i].timesFailed;
      r_.ptimesExcept[i] = trppriv_->trigPathSummaries[i].timesExcept;
    }
  //find |common ls history" object for current ls
  commonLsStat * cst = 0;
  if (meInitialized_) {
    if (commonLsHistory.size()) cst=commonLsHistory.back();
    if (cst && cst->ls_>=lsid) {
      unsigned int countback=commonLsHistory.size()-1;
      while (cst->ls_>lsid && countback) {
	countback--;
	cst=commonLsHistory[countback];
      }
    }
  }

  for( int i=0; i< trppriv_->endPathsInMenu; i++)
    {
      r_.etimesRun[i] = trppriv_->endPathSummaries[i].timesRun;
      r_.etimesPassedPs[i] = trppriv_->endPathSummaries[i].timesPassedPs;
      r_.etimesPassedL1[i] = trppriv_->endPathSummaries[i].timesPassedL1;
      r_.etimesPassed[i] = trppriv_->endPathSummaries[i].timesPassed;
      r_.etimesFailed[i] = trppriv_->endPathSummaries[i].timesFailed;
      r_.etimesExcept[i] = trppriv_->endPathSummaries[i].timesExcept;
      if (cst) {
        if ((unsigned)i < cst->endPathCounts_.size()) cst->endPathCounts_[i]+=r_.etimesPassed[i];
        else cst->endPathCounts_.push_back(r_.etimesPassed[i]);
      }
    }

  //mismatch in expected and reported dataset number
  if (trppriv_->datasetsInMenu!=(int)datasetNames_.size())
  {
    dsMismatch++;
    if  (!(dsMismatch%100) || dsMismatch<10) {
      LOG4CPLUS_WARN(getApplicationLogger(),"mismatch in number of datasets! " 
	  << trppriv_->datasetsInMenu << " in report, " << datasetNames_.size() 
	  << " from legend! received legends:"<< nDatasetLegendaMessageWithDataReceived_);
    }
  }

  for( int i=0; i< trppriv_->datasetsInMenu; i++)
  {
    if (cst) {
      if ((unsigned)i < cst->datasetCounts_.size()) cst->datasetCounts_[i]+=trppriv_->datasetSummaries[i].timesPassed;
      else cst->datasetCounts_.push_back(trppriv_->datasetSummaries[i].timesPassed);
    }
  }

  r_.nproc = trppriv_->eventSummary.totalEvents;
  r_.nsub = trppriv_->nbExpected;
  r_.nrep = trppriv_->nbReporting;


  if(t_!=0 && f_!=0 && b2_==0){

    b2_ = t_->Branch("rate",&r_,"nproc/I:nsub:nrep");
    std::ostringstream ost1;
    ost1 << "p_nrun[" << trppriv_->trigPathsInMenu
	 << "]/I:p_npps[" << trppriv_->trigPathsInMenu
	 << "]:p_npl1[" << trppriv_->trigPathsInMenu
	 << "]:p_npp[" << trppriv_->trigPathsInMenu 
	 << "]:p_npf[" << trppriv_->trigPathsInMenu
	 << "]:p_npe[" << trppriv_->trigPathsInMenu <<"]";

    b3_ = t_->Branch("paths",r_.ptimesRun,ost1.str().c_str());
    std::ostringstream ost2;
    ost2 << "ep_nrun[" << trppriv_->endPathsInMenu
	 << "]/I:en_npps[" << trppriv_->endPathsInMenu
	 << "]:ep_npl1[" << trppriv_->endPathsInMenu
	 << "]:ep_npp[" << trppriv_->endPathsInMenu
	 << "]:ep_npf[" << trppriv_->endPathsInMenu
	 << "]:ep_npe[" << trppriv_->endPathsInMenu << "]";
    b4_ = t_->Branch("endpaths",r_.etimesRun,ost2.str().c_str());
  }
  if(b2_!=0) b2_->Fill();
  if(b3_!=0) b3_->Fill();
  if(b4_!=0) b4_->Fill();

  funcs::addToReport(&trp_[lsid-1],trppriv_,lsid);
  trpentries_[lsid-1]++;

}

// web pages

void iDie::spotlight(xgi::Input *in,xgi::Output *out)
  throw (xgi::exception::Exception)
{

  std::string urn = getApplicationDescriptor()->getURN();

  *out << "<!-- base href=\"/" <<  urn
       << "\"> -->" << std::endl;
  *out << "<html>"                                                   << std::endl;
  *out << "<head>"                                                   << std::endl;
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"/evf/html/styles.css\"/>"                   << std::endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() 
       << getApplicationDescriptor()->getInstance() 
       << " MAIN</title>"     << std::endl;
  *out << "</head>"                                                  << std::endl;
  *out << "<body>"                                                   << std::endl;
  *out << "<table border=\"0\" width=\"100%\">"                      << std::endl;
  *out << "<tr>"                                                     << std::endl;
  *out << "  <td align=\"left\">"                                    << std::endl;
  *out << "    <img"                                                 << std::endl;
  *out << "     align=\"middle\""                                    << std::endl;
  *out << "     src=\"/evf/images/spoticon.jpg\""			     << std::endl;
  *out << "     alt=\"main\""                                        << std::endl;
  *out << "     width=\"64\""                                        << std::endl;
  *out << "     height=\"64\""                                       << std::endl;
  *out << "     border=\"\"/>"                                       << std::endl;
  *out << "    <b>"                                                  << std::endl;
  *out << getApplicationDescriptor()->getClassName() 
       << getApplicationDescriptor()->getInstance()                  << std::endl;
  *out << "    </b>"                                                 << std::endl;
  *out << "  </td>"                                                  << std::endl;
  *out << "  <td width=\"32\">"                                      << std::endl;
  *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << std::endl;
  *out << "      <img"                                               << std::endl;
  *out << "       align=\"middle\""                                  << std::endl;
  *out << "       src=\"/hyperdaq/images/HyperDAQ.jpg\""             << std::endl;
  *out << "       alt=\"HyperDAQ\""                                  << std::endl;
  *out << "       width=\"32\""                                      << std::endl;
  *out << "       height=\"32\""                                     << std::endl;
  *out << "       border=\"\"/>"                                     << std::endl;
  *out << "    </a>"                                                 << std::endl;
  *out << "  </td>"                                                  << std::endl;
  *out << "  <td width=\"32\">"                                      << std::endl;
  *out << "  </td>"                                                  << std::endl;
  *out << "  <td width=\"32\">"                                      << std::endl;
  *out << "    <a href=\"/" << urn << "/\">"                         << std::endl;
  *out << "      <img"                                               << std::endl;
  *out << "       align=\"middle\""                                  << std::endl;
  *out << "       src=\"/evf/images/idieapp.jpg\""		     << std::endl;
  *out << "       alt=\"main\""                                      << std::endl;
  *out << "       width=\"32\""                                      << std::endl;
  *out << "       height=\"32\""                                     << std::endl;
  *out << "       border=\"\"/>"                                     << std::endl;
  *out << "    </a>"                                                 << std::endl;
  *out << "  </td>"                                                  << std::endl;
  *out << "</tr>"                                                    << std::endl;
  *out << "</table>"                                                 << std::endl;
  *out << "<hr/>"                                                    << std::endl;
  *out << "<table><tr><th>Parameter</th><th>Value</th></tr>"         << std::endl;
  *out << "<tr><td>module legenda messages received</td><td>" 
       << nModuleLegendaMessageReceived_ << "</td></tr>"      << std::endl;
  *out << "<tr><td>path legenda messages received</td><td>" 
       << nPathLegendaMessageReceived_ << "</td></tr>"        << std::endl;
  *out << "<tr><td>module legenda messages with data</td><td>" 
       << nModuleLegendaMessageWithDataReceived_ << "</td></tr>"      << std::endl;
  *out << "<tr><td>path legenda messages with data</td><td>" 
       << nPathLegendaMessageWithDataReceived_ << "</td></tr>"        << std::endl;
  *out << "<tr><td>dataset legenda messages with data</td><td>" 
       << nDatasetLegendaMessageWithDataReceived_ << "</td></tr>"        << std::endl;
  *out << "<tr><td>module histo messages received</td><td>" 
       << nModuleHistoMessageReceived_<< "</td></tr>"        << std::endl;
  *out << "<tr><td>path histo messages received</td><td>" 
       << nPathHistoMessageReceived_<< "</td></tr>"        << std::endl;
  tm *uptm = localtime(&lastPathLegendaMessageTimeStamp_.tv_sec);
  char datestring[256];
  strftime(datestring, sizeof(datestring),"%c", uptm);
  *out << "<tr><td>time stamp of last path legenda with data</td><td>" 
       << datestring << "</td></tr>"        << std::endl;
  uptm = localtime(&lastModuleLegendaMessageTimeStamp_.tv_sec);
  strftime(datestring, sizeof(datestring),"%c", uptm);
  *out << "<tr><td>time stamp of last module legenda with data</td><td>" 
       << datestring << "</td></tr>"        << std::endl;
  *out << "</table></body>" << std::endl;

}

void iDie::initFramework()
{

  //ParameterSetRetriever pr(configString_);
  //std::string configuration_ = pr.getAsString();

  std::string configuration_ = configString_;
  configuration_.replace(configuration_.find("EMPTYHOST"),9,dqmCollectorHost_.value_);

  //check if port is a number
  {
    std::string & s = dqmCollectorPort_.value_;
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    if (it != s.end() || s.empty()) dqmCollectorPort_="0";
  }
  configuration_.replace(configuration_.find("EMPTYPORT"),9,dqmCollectorPort_.value_);

  PythonProcessDesc ppdesc = PythonProcessDesc(configuration_);
  boost::shared_ptr<edm::ProcessDesc> pdesc;
  std::vector<std::string> defaultServices = {"InitRootHandlers"};
  pdesc = ppdesc.processDesc();
  pdesc->addServices(defaultServices);

  if (!pServiceSets_) {
    pServiceSets_ = pdesc->getServicesPSets();
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }
  try {
    edm::PresenceFactory *pf = edm::PresenceFactory::get();
    if(pf != 0) {
      pf->makePresence("MessageServicePresence").release();
    }
    else {
    LOG4CPLUS_WARN(getApplicationLogger(),"Unable to create message service presence");
    }
  } 
  catch(edm::Exception e) {
    LOG4CPLUS_WARN(getApplicationLogger(),e.what());
  }

  catch(cms::Exception e) {
    LOG4CPLUS_WARN(getApplicationLogger(),e.what());
  }
 
  catch(std::exception e) {
    LOG4CPLUS_WARN(getApplicationLogger(),e.what());
  }
  catch(...) {
    LOG4CPLUS_WARN(getApplicationLogger(),"Unknown Exception (Message Presence)");
  }

  try {
  serviceToken_ = edm::ServiceRegistry::createSet(*pServiceSets_);
  }
  catch (...) {
    LOG4CPLUS_WARN(getApplicationLogger(),"Failed creation of service token ");
    dqmEnabled_.value_=false;
  }
  edm::ServiceRegistry::Operate operate(serviceToken_);

  evtProcessor_ = new edm::EventProcessor(pdesc,
      serviceToken_,
      edm::serviceregistry::kTokenOverrides);

  try{
    if(edm::Service<DQMStore>().isAvailable())
      dqmStore_ = edm::Service<DQMStore>().operator->();
  }
  catch(...) {
    LOG4CPLUS_WARN(getApplicationLogger(),"exception when trying to get service DQMStore");
    dqmEnabled_.value_=false;
  }
  try{
    if(edm::Service<DQMService>().isAvailable())
      dqmService_ = edm::Service<DQMService>().operator->();
  }
  catch(...) {
    LOG4CPLUS_WARN(getApplicationLogger(),"exception when trying to get service DQMService");
    dqmEnabled_.value_=false;
  }
  if (!dqmEnabled_.value_) LOG4CPLUS_ERROR(getApplicationLogger(),"Failed to initialize DQMService/DQMStore");

  if (dqmState_!="Removed")
    initMonitorElements();

}

void iDie::initMonitorElements()
{
  if (!evtProcessor_) return;
  dqmStore_->cd();

  meVecRate_.clear();
  meVecTime_.clear();
  meVecOffenders_.clear();
  for (unsigned int i=0;i<epInstances.size();i++) {
	  currentLs_[i]=0;
  }
  ilumiprev_ = 0;
  savedForLs_=0;
  summaryLastLs_ = 0;
  pastSavedFiles_.clear();
  
  dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/EventInfo/");
  runId_     = dqmStore_->bookInt("iRun");
  runId_->Fill(-1);
  lumisecId_ = dqmStore_->bookInt("iLumiSection");
  lumisecId_->Fill(-1);
  eventId_ = dqmStore_->bookInt("iEvent");
  eventId_->Fill(-1);
  eventTimeStamp_ = dqmStore_->bookFloat("eventTimeStamp");
  runStartTimeStamp_ = dqmStore_->bookFloat("runStartTimeStamp");
  initDQMEventInfo();

  for (unsigned int i=0;i<nbSubsClasses;i++) {
    std::ostringstream str;
    str << nbSubsListInv[i];
    dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/Layouts/");
    meVecRate_.push_back(dqmStore_->book1D("EVENT_RATE_"+TString(str.str().c_str()),
	  "Average event rate for nodes with " + TString(str.str().c_str()) + " EP instances",
	  4000,1.,4001));
    meVecTime_.push_back(dqmStore_->book1D("EVENT_TIME_"+TString(str.str().c_str()),
	  "Average event processing time for nodes with " + TString(str.str().c_str()) + " EP instances",
	  4000,1.,4001));
    dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/Layouts/Modules/");
    meVecOffenders_.push_back(dqmStore_->book2D("MODULE_FRACTION_"+TString(str.str().c_str()),
	  "Module processing time fraction_"+ TString(str.str().c_str()),
	  ROLL,1.,1.+ROLL,MODNAMES,0,MODNAMES));
    //fill 1 in underrflow bin
    meVecOffenders_[i]->Fill(0,1);
    occupancyNameMap[i].clear();
  }

  //tables
  dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/Layouts/Tables");
  rateSummary_   = dqmStore_->book2D("00_RATE_SUMMARY","Rate Summary (Hz)",ROLL,0,ROLL,epInstances.size()+1,0,epInstances.size()+1);
  reportPeriodSummary_   = dqmStore_->book2D("00_REPORT_PERIOD_SUMMARY","Average report period (s)",ROLL,0,ROLL,epInstances.size()+1,0,epInstances.size()+1);
  timingSummary_ = dqmStore_->book2D("01_TIMING_SUMMARY","Event Time Summary (ms)",ROLL,0,ROLL,epInstances.size()+1,0,epInstances.size()+1);
  busySummary_ = dqmStore_->book2D("02_BUSY_SUMMARY","Busy fraction ",ROLL,0,ROLL,epInstances.size()+2,0,epInstances.size()+2);
  busySummary2_ = dqmStore_->book2D("03_BUSY_SUMMARY_PROCSTAT","Busy fraction from /proc/stat",ROLL,0,ROLL,epInstances.size()+2,0,epInstances.size()+2);
  busySummaryUncorr1_ = dqmStore_->book2D("04_BUSY_SUMMARY_UNCORR","Busy fraction (uncorrected)",
      ROLL,0,ROLL,epInstances.size()+2,0,epInstances.size()+2);
  busySummaryUncorr2_ = dqmStore_->book2D("05_BUSY_SUMMARY_UNCORR_PROCSTAT","Busy fraction from /proc/stat(uncorrected)",
      ROLL,0,ROLL,epInstances.size()+2,0,epInstances.size()+2);
  fuReportsSummary_ = dqmStore_->book2D("06_EP_REPORTS_SUMMARY","Number of reports received",ROLL,0,ROLL,epInstances.size()+1,0,epInstances.size()+1);

  //everything else goes into layouts folder
  dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/Layouts/");
  std::ostringstream busySummaryTitle;
  busySummaryTitle << "DAQ HLT Farm busy (%) for run "<< runNumber_.value_;
  lastRunNumberSet_ = runNumber_.value_;
  daqBusySummary_ = dqmStore_->book1D("00 reportSummaryMap",busySummaryTitle.str(),4000,1,4001.);
  daqBusySummary2_ = dqmStore_->book1D("00 reportSummaryMap_PROCSTAT","DAQ HLT Farm busy (%) from /proc/stat",4000,1,4001.);
  daqTotalRateSummary_ = dqmStore_->book1D("00 reportSummaryMap_TOTALRATE","DAQ HLT Farm input rate",4000,1,4001.);

  for (size_t i=1;i<=ROLL;i++) {
    std::ostringstream ostr;
    ostr << i;
    rateSummary_->setBinLabel(i,ostr.str(),1);
    reportPeriodSummary_->setBinLabel(i,ostr.str(),1);
    timingSummary_->setBinLabel(i,ostr.str(),1);
    busySummary_->setBinLabel(i,ostr.str(),1);
    busySummary2_->setBinLabel(i,ostr.str(),1);
    busySummaryUncorr1_->setBinLabel(i,ostr.str(),1);
    busySummaryUncorr2_->setBinLabel(i,ostr.str(),1);
    fuReportsSummary_->setBinLabel(i,ostr.str(),1);
  }
  for (size_t i=1;i<epInstances.size()+1;i++) {
    std::ostringstream ostr;
    ostr << epInstances[i-1];
    rateSummary_->setBinLabel(i,ostr.str(),2);
    reportPeriodSummary_->setBinLabel(i,ostr.str(),2);
    timingSummary_->setBinLabel(i,ostr.str(),2);
    busySummary_->setBinLabel(i,ostr.str(),2);
    busySummary2_->setBinLabel(i,ostr.str(),2);
    busySummaryUncorr1_->setBinLabel(i,ostr.str(),2);
    busySummaryUncorr2_->setBinLabel(i,ostr.str(),2);
    fuReportsSummary_->setBinLabel(i,ostr.str(),2);
  }
  rateSummary_->setBinLabel(epInstances.size()+1,"All",2);
  //timingSummary_->setBinLabel(i,"Avg",2);
  busySummary_->setBinLabel(epInstances.size()+1,"%Conf",2);
  busySummary_->setBinLabel(epInstances.size()+2,"%Max",2);
  busySummary2_->setBinLabel(epInstances.size()+1,"%Conf",2);
  busySummary2_->setBinLabel(epInstances.size()+2,"%Max",2);
  fuReportsSummary_->setBinLabel(epInstances.size()+1,"All",2);

  //wipe out all ls history
  for (size_t i=0;i<epInstances.size();i++) {
    while (lsHistory[i].size()) {
      delete lsHistory[i].front();
      lsHistory[i].pop_front();
    }
  }
  while (commonLsHistory.size()) {
    delete commonLsHistory.front();
    commonLsHistory.pop_front();
  }
  meInitialized_=true;

}

void iDie::initMonitorElementsStreams() {
  if (!dqmEnabled_.value_ || !evtProcessor_) return;
  if (meInitializedStreams_) return;

  //add OUTPUT Stream histograms
  endPathRates_.clear();
  dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/Layouts/Streams/");
  for (size_t i=0;i<endPathNames_.size();i++) {
    endPathRates_.push_back(dqmStore_->book1D(endPathNames_[i]+"_RATE",endPathNames_[i]+" events/s",4000,1,4001.));
  }
  meInitializedStreams_=true;
}


void iDie::initMonitorElementsDatasets() {
  if (!dqmEnabled_.value_ || !evtProcessor_) return;
  if (meInitializedDatasets_) return;

  //add OUTPUT Stream histograms
  datasetRates_.clear();
  dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/Layouts/Datasets/");
  for (size_t i=0;i<datasetNames_.size();i++) {
    datasetRates_.push_back(dqmStore_->book1D(datasetNames_[i]+"_RATE",datasetNames_[i]+" events/s",4000,1,4001.));
  }
  meInitializedDatasets_=true;
}


void iDie::deleteFramework()
{
  if (evtProcessor_) delete evtProcessor_;
}

void iDie::fillDQMStatHist(unsigned int nbsIdx, unsigned int lsid)
{
  if (!evtProcessor_ || lsid==0) return;
  unsigned int qsize = lsHistory[nbsIdx].size();
  //may be larger size
  unsigned int cqsize = lsHistory[nbsIdx].size();

  //update lumis
  if (qsize) {
    for (int i =(int)qsize-1;i>=0 && i>=(int)qsize-PASTUPDATES;i--) {
      unsigned int qpos=(unsigned int) i;
      unsigned int forls = lsid - (qsize-1-i);
      lsStat * lst = (lsHistory[nbsIdx])[qpos];
      unsigned int clsPos = unsigned((int)qpos+ (int)cqsize - (int)qsize);
      commonLsStat * clst = commonLsHistory[unsigned((int)qpos+ (int)cqsize - (int)qsize)];

      meVecRate_[nbsIdx]->setBinContent(forls,lst->getRatePerMachine());
      meVecRate_[nbsIdx]->setBinError(forls,lst->getRateErrPerMachine());
      meVecTime_[nbsIdx]->setBinContent(forls>2? forls:0,lst->getEvtTime()*1000);//msec
      meVecTime_[nbsIdx]->setBinError(forls>2? forls:0,lst->getEvtTimeErr()*1000);//msec
      updateRollingHistos(nbsIdx, forls,lst,clst,i==(int)qsize-1);
      commonLsStat * prevclst = clsPos>0 ? commonLsHistory[clsPos-1]:nullptr;
      updateStreamHistos(forls,clst,prevclst);
      updateDatasetHistos(forls,clst,prevclst);
    }
  }
}

void iDie::updateRollingHistos(unsigned int nbsIdx, unsigned int lsid, lsStat * lst, commonLsStat  * clst, bool roll) {
  unsigned int lsidBin;
  if (roll) {
    if (lsid>ROLL) {
      lsidBin=ROLL;
      if (lsid>summaryLastLs_) { //last ls in plots isn't up to date
	unsigned int lsdiff = lsid-summaryLastLs_;
	for (unsigned int i=1;i<=ROLL;i++) {
	  if (i<ROLL) {
	    bool emptyBin=false;
	    if (i>ROLL-lsdiff) emptyBin=true;
	    for (unsigned int j=1;j<=epInstances.size()+1;j++) {
	      rateSummary_->setBinContent(i,j,emptyBin? 0 : rateSummary_->getBinContent(i+lsdiff,j));
	      reportPeriodSummary_->setBinContent(i,j,emptyBin? 0 : reportPeriodSummary_->getBinContent(i+lsdiff,j));
	      timingSummary_->setBinContent(i,j,emptyBin ? 0 : timingSummary_->getBinContent(i+lsdiff,j));
	      busySummary_->setBinContent(i,j,emptyBin ? 0 : busySummary_->getBinContent(i+lsdiff,j));
	      busySummary2_->setBinContent(i,j,emptyBin ? 0 : busySummary2_->getBinContent(i+lsdiff,j));
	      busySummaryUncorr1_->setBinContent(i,j,emptyBin ? 0 : busySummaryUncorr1_->getBinContent(i+lsdiff,j));
	      busySummaryUncorr2_->setBinContent(i,j,emptyBin ? 0 : busySummaryUncorr2_->getBinContent(i+lsdiff,j));
	      fuReportsSummary_->setBinContent(i,j,emptyBin ? 0 : fuReportsSummary_->getBinContent(i+lsdiff,j));
	    }
	    busySummary_->setBinContent(i,epInstances.size()+2,emptyBin ? 0 : busySummary2_->getBinContent(i+lsdiff,epInstances.size()+2));
	    busySummary2_->setBinContent(i,epInstances.size()+2,emptyBin ? 0 : busySummary2_->getBinContent(i+lsdiff,epInstances.size()+2));
	  }

	  std::ostringstream ostr;
	  ostr << lsid-ROLL+i;
	  rateSummary_->setBinLabel(i,ostr.str(),1);
	  reportPeriodSummary_->setBinLabel(i,ostr.str(),1);
	  timingSummary_->setBinLabel(i,ostr.str(),1);
	  busySummary_->setBinLabel(i,ostr.str(),1);
	  busySummary2_->setBinLabel(i,ostr.str(),1);
	  busySummaryUncorr1_->setBinLabel(i,ostr.str(),1);
	  busySummaryUncorr2_->setBinLabel(i,ostr.str(),1);
	  fuReportsSummary_->setBinLabel(i,ostr.str(),1);

	}
	summaryLastLs_=lsid;
      }
      else if (lsid<summaryLastLs_) {
	if (summaryLastLs_-lsid>=ROLL) return;//very old
	lsidBin=ROLL-(summaryLastLs_-lsid);
      }
    }
    else if (lsid) {lsidBin=lsid;summaryLastLs_=lsid;} else return;
  }
  else {// previous lumisection updates
    unsigned int roll_pos = ROLL-(summaryLastLs_-lsid);
    lsidBin=lsid > roll_pos ? roll_pos : lsid;
  }

  //how busy is it with current setup
  float busyCorr = lst->getFracBusy() * (float)epInstances[nbsIdx]/epMax[nbsIdx];
  //max based on how much is configured and max possible
  float fracMax  = 0.5 + (std::max(epInstances[nbsIdx]-epMax[nbsIdx]/2.,0.)/(epMax[nbsIdx])) *HTscaling[nbsIdx];
  if (epInstances[nbsIdx]<epMax[nbsIdx]/2) {
	  fracMax = epInstances[nbsIdx]/((double)epMax[nbsIdx]);
  }

  //corrections for the HT effect
  float busyFr=0;
  float busyCPUFr=0;
  float busyFrTheor=0;
  float busyFrCPUTheor=0;

  //microstates based calculation
  if (busyCorr>0.5) {//take into account HT scaling for the busy fraction
    busyFr=(0.5 + (busyCorr-0.5)*HTscaling[nbsIdx])/fracMax;
    busyFrTheor = (0.5+(busyCorr-0.5)*HTscaling[nbsIdx])/ (0.5+0.5*HTscaling[nbsIdx]);
  }
  else {//below the HT threshold
    busyFr=busyCorr / fracMax;
    busyFrTheor = busyCorr / (0.5+0.5*HTscaling[nbsIdx]);
  }

  //proc/stat based calculation
  float busyCorr_CPU = lst->getFracCPUBusy();
  if (busyCorr_CPU>0.5) {
    busyCPUFr=(0.5 + (busyCorr_CPU-0.5)*HTscaling[nbsIdx])/fracMax;
    busyFrCPUTheor = (0.5+(busyCorr_CPU-0.5)*HTscaling[nbsIdx])/ (0.5+0.5*HTscaling[nbsIdx]);
  }
  else {
    busyCPUFr=busyCorr_CPU / fracMax;
    busyFrCPUTheor = busyCorr_CPU / (0.5+0.5*HTscaling[nbsIdx]);
  }

  //average
  clst->setBusyForClass(nbsIdx,lst->getRate(),busyFr,busyFrTheor,busyCPUFr,busyFrCPUTheor,lst->getReports());
  float busyAvg = clst->getBusyTotalFrac(false,machineWeightInst);
  float busyAvgCPU = clst->getBusyTotalFrac(true,machineWeightInst);

  //rounding
  busyFr=fround(busyFr,0.001f);
  busyCPUFr=fround(busyCPUFr,0.001f);
  busyFrTheor=fround(busyFrTheor,0.001f);
  busyFrCPUTheor=fround(busyFrCPUTheor,0.001f);
  busyAvg=fround(busyAvg,0.001f);

  //flashlist per-lumi values
  if (lsid>2)
    while (cpuLoadLastLs_<lsid-2) {
      if (cpuLoadLastLs_>=4000-1) break;
      cpuLoad_[cpuLoadLastLs_]=daqBusySummary_->getBinContent(cpuLoadLastLs_+1)*0.01;
      cpuLoadPS_[cpuLoadLastLs_]=daqBusySummary2_->getBinContent(cpuLoadLastLs_+1)*0.01;
      cpuLoadTime7_[cpuLoadLastLs_]=meVecTime_[0]->getBinContent(cpuLoadLastLs_+1)*0.001;
      cpuLoadTime8_[cpuLoadLastLs_]=meVecTime_[1]->getBinContent(cpuLoadLastLs_+1)*0.001;
      cpuLoadTime12_[cpuLoadLastLs_]=meVecTime_[2]->getBinContent(cpuLoadLastLs_+1)*0.001;
      cpuLoadTime16_[cpuLoadLastLs_]=meVecTime_[3]->getBinContent(cpuLoadLastLs_+1)*0.001;
      cpuLoadTime24_[cpuLoadLastLs_]=meVecTime_[4]->getBinContent(cpuLoadLastLs_+1)*0.001;
      cpuLoadTime32_[cpuLoadLastLs_]=meVecTime_[5]->getBinContent(cpuLoadLastLs_+1)*0.001;
      cpuLoadRate_[cpuLoadLastLs_]=daqTotalRateSummary_->getBinContent(cpuLoadLastLs_+1);
      cpuLoadLastLs_++;
  }

  //filling plots
  daqBusySummary_->setBinContent(lsid,busyAvg*100.);
  daqBusySummary_->setBinError(lsid,0);
  daqBusySummary2_->setBinContent(lsid,busyAvgCPU*100.);
  daqBusySummary2_->setBinError(lsid,0);

  daqTotalRateSummary_->setBinContent(lsid,clst->getTotalRate());
  daqTotalRateSummary_->setBinError(lsid,0);

  //"rolling" histograms
  rateSummary_->setBinContent(lsidBin,nbsIdx+1,lst->getRate());
  reportPeriodSummary_->setBinContent(lsidBin,nbsIdx+1,lst->getDt());
  timingSummary_->setBinContent(lsidBin,nbsIdx+1,lst->getEvtTime()*1000);
  fuReportsSummary_->setBinContent(lsidBin,nbsIdx+1,lst->getReports());
  busySummary_->setBinContent(lsidBin,nbsIdx+1,fround(busyFr,0.001f));
  busySummary2_->setBinContent(lsidBin,nbsIdx+1,fround(busyCPUFr,0.001f));
  busySummaryUncorr1_->setBinContent(lsidBin,nbsIdx+1,fround(lst->getFracBusy(),0.001f));
  busySummaryUncorr2_->setBinContent(lsidBin,nbsIdx+1,fround(lst->getFracCPUBusy(),0.001f));

  rateSummary_->setBinContent(lsidBin,epInstances.size()+1,clst->getTotalRate());
  fuReportsSummary_->setBinContent(lsidBin,epInstances.size()+1,clst->getNReports());

  busySummary_->setBinContent(lsidBin,epInstances.size()+1,fround(busyAvg,0.001f));
  busySummary2_->setBinContent(lsidBin,epInstances.size()+1,fround(busyAvgCPU,0.001f));
  busySummary_->setBinContent(lsidBin,epInstances.size()+2,fround(clst->getBusyTotalFracTheor(false,machineWeight),0.001f));
  busySummary2_->setBinContent(lsidBin,epInstances.size()+2,fround(clst->getBusyTotalFracTheor(true,machineWeight),0.001f));

}

void iDie::updateStreamHistos(unsigned int forls, commonLsStat *clst, commonLsStat *prevclst)
{
  if (endPathRates_.size()!=endPathNames_.size()) meInitializedStreams_=false; 
  initMonitorElementsStreams();//reinitialize (conditionally)
  for (size_t i=0;i<endPathRates_.size();i++) {
    unsigned int count_current=0;
    unsigned int count_last=0;
    if (clst->endPathCounts_.size()>i) {
      count_current=clst->endPathCounts_[i];
    }
    endPathRates_[i]->setBinContent(forls,(count_current-count_last)/23.1);//approx ls
  } 
}


void iDie::updateDatasetHistos(unsigned int forls, commonLsStat *clst, commonLsStat *prevclst)
{
  if (datasetRates_.size()!=datasetNames_.size()) meInitializedDatasets_=false; 
  initMonitorElementsDatasets();//reinitialize (conditionally)
  for (size_t i=0;i<datasetRates_.size();i++) {
    unsigned int count_current=0;
    unsigned int count_last=0;
    if (clst->datasetCounts_.size()>i) {
      count_current=clst->datasetCounts_[i];
    }
    datasetRates_[i]->setBinContent(forls,(count_current-count_last)/23.1);//approx ls
  } 
}


void iDie::fillDQMModFractionHist(unsigned int nbsIdx, unsigned int lsid, unsigned int nonIdle, std::vector<std::pair<unsigned int,unsigned int>> offenders)
{
  if (!evtProcessor_) return;
  MonitorElement * me = meVecOffenders_[nbsIdx];
  //shift bin names by 1
  unsigned int xBinToFill=lsid;
  if (lsid>ROLL) {
    for (unsigned int i=1;i<=ROLL;i++) {
      for (unsigned int j=1;j<=MODNAMES;j++) {
	if (i<ROLL)
	  me->setBinContent(i,j,me->getBinContent(i+1,j));
	else
	  me->setBinContent(i,j,0);
      }
      std::ostringstream ostr;
      ostr << lsid-ROLL+i;
      me->setBinLabel(i,ostr.str(),1);
    }
    std::ostringstream ostr;
    ostr << lsid;
    xBinToFill=ROLL;
  }
  float nonIdleInv=0.;
  if (nonIdle>0)nonIdleInv=1./(double)nonIdle;
  //1st pass (there are free bins left)
  for (unsigned int i=0;i<offenders.size();i++) {
    unsigned int x=offenders[i].first;
    float percentageUsed=offenders[i].second*nonIdleInv;
    if (percentageUsed>0.02) {//2% threshold
      if (occupancyNameMap[nbsIdx].count(x)==0) {//new element
	unsigned int y=occupancyNameMap[nbsIdx].size();
	if (y<MODNAMES) {
	  (occupancyNameMap[nbsIdx])[x]=y;
	  me->setBinContent(xBinToFill,y+1,fround(percentageUsed,0.001f));
	  me->setBinLabel(y+1,mapmod_[x],2);
	}
	else break;
      }
    }
  }
  //2nd pass (beyond available bins)
  for (unsigned int i=0;i<offenders.size();i++) {
    unsigned int x=offenders[i].first;
    float percentageUsed=offenders[i].second*nonIdleInv;
    if (percentageUsed>0.02) {//2% threshold
      if (occupancyNameMap[nbsIdx].count(x)==0) {
	unsigned int y=occupancyNameMap[nbsIdx].size();
	if (y>=MODNAMES && xBinToFill>1) {
	  //filled up, replace another one
	  float minbinval=1.;
	  unsigned int toReplace=0;
	  for (size_t j=1;j<=MODNAMES;j++) {
	    //decide based on the smallest value
	    float bin=me->getBinContent(xBinToFill,j);
	    if (bin<minbinval) {toReplace=j;minbinval=bin;}
	  }
	  if (percentageUsed>minbinval && toReplace) {
	    int key=-1;
	    for (auto it = occupancyNameMap[nbsIdx].begin(); it != occupancyNameMap[nbsIdx].end(); ++it) {
	      if (it->second == toReplace-1) {
		key = it->first;
		break;
	      }
	    }
	    if (key>-1) {
	      //erase old
	      occupancyNameMap[nbsIdx].erase(key);
	      //add new
	      (occupancyNameMap[nbsIdx])[x]=toReplace-1;
	      //fill histogram
	      me->setBinContent(xBinToFill,toReplace,fround(percentageUsed,0.001f));
	      me->setBinLabel(toReplace,mapmod_[x],2);
	      //reset fields for previous lumis
	      unsigned qsize = lsHistory[nbsIdx].size();
	      for (size_t k=1;k<xBinToFill;k++) {
                if (xBinToFill-k+1<qsize) {
                  float fr = (lsHistory[nbsIdx])[qsize-xBinToFill+k-1]->getOffenderFracAt(x);
		  if (fr>0.02) me->setBinContent(k,toReplace,fround(fr,0.001f));
		}
		else
                  me->setBinContent(k,toReplace,0);
	      }
	    }
	  }
	}
      }
      else {
	unsigned int y=(occupancyNameMap[nbsIdx])[x];
	me->setBinContent(xBinToFill,y+1,fround(percentageUsed,0.001f));
      }
    }
  }
}

void iDie::doFlush() {
    if (dqmEnabled_.value_)
      dqmService_->flushStandalone();
}

void iDie::timeExpired(toolbox::task::TimerEvent& e)
{
  //bool pushUpdate=false;
  if (debugMode_.value_)
    std::cout << "debug - runNumber:" << runNumber_ << " run active:" << runActive_ << std::endl;
  if (!runActive_) return;
  if (!runNumber_) return;
  try
  {
    if (runNumber_>flashRunNumber_ && flashRunNumber_>0)
    {
      cpuInfoSpace_->lock();
      flashRunNumber_=runNumber_;
      cpuInfoSpace_->unlock();
    }

    if (debugMode_.value_) std::cout << " checking per-lumi flashlist" << std::endl;

    if (cpuLoadSentLs_>cpuLoadLastLs_) cpuLoadSentLs_=0;
    if (cpuLoadSentLs_<cpuLoadLastLs_ && cpuLoadLastLs_<=4000)
    {
      unsigned int toSend = cpuLoadLastLs_;
      if (toSend) {
        toSend--;
        cpuInfoSpace_->lock();
        flashLoadLs_=toSend+1;
        flashLoad_=cpuLoad_[toSend];
	flashLoadPS_=cpuLoadPS_[toSend];
	flashLoadTime7_=cpuLoadTime7_[toSend];
	flashLoadTime8_=cpuLoadTime8_[toSend];
	flashLoadTime12_=cpuLoadTime12_[toSend];
	flashLoadTime16_=cpuLoadTime16_[toSend];
	flashLoadTime24_=cpuLoadTime24_[toSend];
	flashLoadTime32_=cpuLoadTime32_[toSend];
	flashLoadRate_=cpuLoadRate_[toSend];
	cpuLoadSentLs_++;
	cpuInfoSpace_->unlock();
	if (cpuLoadSentLs_<=cpuLoadLastLs_) {

	  if (debugMode_.value_)
	    std::cout << "debug - updated lumi flashlist with values "
	      << flashLoadLs_ << " " << flashLoad_ << " " << flashLoadPS_
	      << " t:" << flashLoadTime7_ << " " << flashLoadTime8_ << " " << flashLoadTime12_  << " "
	      << flashLoadTime16_ << " " << flashLoadTime24_ << flashLoadTime32_ << " r:" << flashLoadRate_ << std::endl;

	  cpuInfoSpace_->fireItemGroupChanged(monNames_, this);

	}
      }
    }
  }
  catch (xdata::exception::Exception& xe)
  {
    LOG4CPLUS_WARN(getApplicationLogger(), xe.what() );
  }
  catch (std::exception& se)
  {
    std::string msg = "Caught standard exception while trying to collect: ";
    msg += se.what();
    LOG4CPLUS_WARN(getApplicationLogger(), msg );
  }
  catch (...)
  {
    std::string msg = "Caught unknown exception while trying to collect";
    LOG4CPLUS_WARN(getApplicationLogger(), msg );
  }
}

void iDie::perLumiFileSaver(unsigned int lsid)
{

  //make sure that run number is updated before saving
  if (lastRunNumberSet_<runNumber_) {
    if (meInitialized_) {
      std::ostringstream busySummaryTitle;
      busySummaryTitle << "DAQ HLT Farm busy (%) for run "<< runNumber_.value_;
      daqBusySummary_->setTitle(busySummaryTitle.str());
      lastRunNumberSet_ = runNumber_.value_;
    }
  }

  if (dqmSaveDir_.value_=="") return;
  //try to create directory if not there

  if (savedForLs_==0)
  {
    struct stat st;
    if (stat((dqmSaveDir_.value_+"/output").c_str(),&st) != 0) {
      if (mkdir((dqmSaveDir_.value_+"/output").c_str(), 0777) != 0) {
        LOG4CPLUS_ERROR(getApplicationLogger(),"iDie could not find nor create DQM \"output\" directory. DQM archiving -> Off.");
	dqmSaveDir_.value_="";//reset parameter
        return;
      }
    }
    if (stat((dqmSaveDir_.value_+"/done").c_str(),&st) != 0) {
      if (mkdir((dqmSaveDir_.value_+"/done").c_str(), 0777) != 0) {
        LOG4CPLUS_WARN(getApplicationLogger(),"iDie could not find nor create DQM \"done\" directory. DQM archiving might fail.");
      }
    }
    //static filename part
    char version[8];
    sprintf(version, "_V%04d_", int(1));
    version[7]='\0';
    std::string sDir = dqmSaveDir_.value_;
    if (sDir[sDir.size()-1]!='/') sDir+="/";
    sDir+="output/";
    fileBaseName_ = sDir + "DQM" + version;

    //checking if directory is there
    if ( access( sDir.c_str(), 0 ) == 0 )
    {
      struct stat status;
      stat( sDir.c_str(), &status );

      if ( status.st_mode & S_IFDIR ) writeDirectoryPresent_=true;
      else writeDirectoryPresent_=false;
    }
  }

  if (lsid > 0 && (lsid%saveLsInterval_.value_)==0  && lsid>savedForLs_ && writeDirectoryPresent_)
  {
    savedForLs_=lsid;
    char suffix[64];
    char rewrite[128];
    sprintf(suffix, "_R%09d_L%06d", runNumber_.value_, lsid);
    sprintf(rewrite, "\\1Run %d/\\2/By Lumi Section %d-%d", runNumber_.value_, ilumiprev_, lsid);

    std::vector<std::string> systems = {topLevelFolder_.value_};

    for (size_t i = 0, e = systems.size(); i != e; ++i) {
      std::string filename = fileBaseName_ + systems[i] + suffix + ".root";
      try {
	dqmStore_->save(filename, systems[i] , "^(Reference/)?([^/]+)",
	    rewrite, (DQMStore::SaveReferenceTag) DQMStore::SaveWithReference, dqm::qstatus::STATUS_OK);
	pastSavedFiles_.push_back(filename);
	if (dqmFilesWritable_.value_)
	  chmod(filename.c_str(),0777);//allow deletion by dqm script
	//if (pastSavedFiles_.size() > 500)
	//{
	  //remove(pastSavedFiles_.front().c_str());
	  //pastSavedFiles_.pop_front();
	//}
      }
      catch (...) {
	LOG4CPLUS_ERROR(getApplicationLogger(),"iDie could not create root file " << filename);
      }
    }

    ilumiprev_ = lsid;

    //cd() to micro report root file
    if (f_)
      f_->cd();
  }
}



void iDie::perTimeFileSaver()
{

  //make sure that run number is updated before saving
  if (lastRunNumberSet_<runNumber_) {
    if (meInitialized_) {
      std::ostringstream busySummaryTitle;
      busySummaryTitle << "DAQ HLT Farm busy (%) for run "<< runNumber_.value_;
      daqBusySummary_->setTitle(busySummaryTitle.str());
      lastRunNumberSet_ = runNumber_.value_;
    }
  }
  
  if (dqmSaveDir_.value_=="") return;

  //save interval (+9 every minutes after initial)
  std::vector<unsigned int> minutes = {4,8,12,20};
  
  //directory should already be there
  //first invocation - just record time
  if (!reportingStart_) {
    reportingStart_ = new timeval;
    gettimeofday(reportingStart_,0);
    lastSavedForTime_=0;
    return;
  }
  timeval new_ts;
  gettimeofday(&new_ts,0);

  unsigned int dT = (new_ts.tv_sec - reportingStart_->tv_sec) / 60;

  unsigned int willSaveForTime = 0;

  for (size_t i=0;i<minutes.size();i++) {
    if (dT>=minutes[i]) {
      if (lastSavedForTime_ < minutes[i]) {
        willSaveForTime=dT;
	lastSavedForTime_=dT;
	break;
      }
    }
  }

  //in periodic part
  unsigned int lastMinutesTime = minutes[minutes.size()-1];
  if (!willSaveForTime && dT>lastMinutesTime)
  {
    if (lastSavedForTime_<lastMinutesTime || (dT-lastMinutesTime)/9 > (lastSavedForTime_-lastMinutesTime)/9) {
      willSaveForTime=dT;
      lastSavedForTime_=dT;
    }
  }
  if (willSaveForTime && writeDirectoryPresent_)
  {
    char suffix[64];
    char rewrite[128];
    //sprintf(suffix, "_R%09d_T%08d", runNumber_.value_, willSaveForTime);
    sprintf(suffix, "_R%09d", runNumber_.value_);
    sprintf(rewrite, "\\1Run %d/\\2/Run summary", runNumber_.value_);

    std::vector<std::string> systems = {topLevelFolder_.value_};

    for (size_t i = 0, e = systems.size(); i != e; ++i) {
      std::string filename = fileBaseName_ + systems[i] + suffix + ".root";
      try {
	dqmStore_->save(filename, systems[i] , "^(Reference/)?([^/]+)",
	    rewrite, (DQMStore::SaveReferenceTag) DQMStore::SaveWithReference, dqm::qstatus::STATUS_OK);
	if (dqmFilesWritable_.value_)
	  chmod(filename.c_str(),0777);//allow deletion by dqm script
      }
      catch (...) {
	LOG4CPLUS_ERROR(getApplicationLogger(),"iDie could not create root file " << filename);
      }
    }

    //cd() to micro report root file
    if (f_)
      f_->cd();
  }
}


void iDie::initDQMEventInfo()
{
  struct timeval now;
  gettimeofday(&now, 0);
  double time_now = now.tv_sec + 1e-6*now.tv_usec;

  dqmStore_->setCurrentFolder(topLevelFolder_.value_ + "/EventInfo/");
  runId_     = dqmStore_->bookInt("iRun");
  runId_->Fill(-1);
  lumisecId_ = dqmStore_->bookInt("iLumiSection");
  lumisecId_->Fill(-1);
  eventId_ = dqmStore_->bookInt("iEvent");
  eventId_->Fill(-1);
  eventTimeStamp_ = dqmStore_->bookFloat("eventTimeStamp");

  runStartTimeStamp_ = dqmStore_->bookFloat("runStartTimeStamp");

  processTimeStampMe_ = dqmStore_->bookFloat("processTimeStamp");
  processTimeStampMe_->Fill(time_now);
  processLatencyMe_ = dqmStore_->bookFloat("processLatency");
  processLatencyMe_->Fill(-1);
  processEventsMe_ = dqmStore_->bookInt("processedEvents");
  processEventsMe_->Fill(0);
  processEventRateMe_ = dqmStore_->bookFloat("processEventRate");
  processEventRateMe_->Fill(-1); 
  nUpdatesMe_= dqmStore_->bookInt("processUpdates");
  nUpdatesMe_->Fill(-1);
  processIdMe_= dqmStore_->bookInt("processID"); 
  processIdMe_->Fill(getpid());
  processStartTimeStampMe_ = dqmStore_->bookFloat("processStartTimeStamp");
  processStartTimeStampMe_->Fill(time_now);
  hostNameMe_= dqmStore_->bookString("hostName","cmsidie");
  processNameMe_= dqmStore_->bookString("processName","iDie");
  workingDirMe_= dqmStore_->bookString("workingDir","/tmp");
  cmsswVerMe_= dqmStore_->bookString("CMSSW_Version",edm::getReleaseVersion());
}

void iDie::setRunStartTimeStamp()
{
  struct timeval now;
  gettimeofday(&now, 0);
  double time_now = now.tv_sec + 1e-6*now.tv_usec;
  runTS_ = time_now;
}

////////////////////////////////////////////////////////////////////////////////
// xdaq instantiator implementation macro
////////////////////////////////////////////////////////////////////////////////

XDAQ_INSTANTIATOR_IMPL(iDie)
