#include "iDie.h"

#include "xdaq/NamespaceURI.h"

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


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
iDie::iDie(xdaq::ApplicationStub *s) 
  : xdaq::Application(s)
  , log_(getApplicationLogger())
  , instance_(0)
  , runNumber_(0)
//  , dqmCollectorHost_()
//  , dqmCollectorPort_()
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
  , evtProcessor_(0)
  , meInitialized_(false)
  , dqmDisabled_(false)
  , saveLsInterval_(10)
  , ilumiprev_(0)
  , dqmSaveDir_("")
  , dqmFilesWritable_(false)
  , savedForLs_(0)
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
      //
  // timestamps
  lastModuleLegendaMessageTimeStamp_.tv_sec=0;
  lastModuleLegendaMessageTimeStamp_.tv_usec=0;
  lastPathLegendaMessageTimeStamp_.tv_sec=0;
  lastPathLegendaMessageTimeStamp_.tv_usec=0;
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

  epInstances   =     {7,    8,     12,  16,  22,  24,  32};
  epMax         =     {8,    8,     24,  32,  24,  24,  32};
  HTscaling     =     {1,    1,   0.28,0.28,0.28,0.28,0.28};
  nbMachines    =     {0,    0,      0,   0,   0,   0,   0};
  machineWeight =     {91.6, 91.6, 253, 352, 253, 253, 352};
  machineWeightInst = {80.15,91.6, 196, 352, 237, 253, 352};

  for (unsigned int i=0;i<epInstances.size();i++) {
    currentLs_.push_back(0);
    nbSubsList[epInstances[i]]=i;
    nbSubsListInv[i]=epInstances[i];
  }
  nbSubsClasses = epInstances.size();
  lsHistory = new std::queue<lsStat>[nbSubsClasses];
  //mask for permissions of created files
  umask(000);

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
    
    std::string state;
    // generate correct return state string
    if(commandName == "Configure") state = "Ready";
    else if(commandName == "Enable") state = "Enabled";
    else if(commandName == "Stop") {
      //remove histograms
      dqmStore_->setCurrentFolder("DAQ/EventInfo/");
      dqmStore_->removeContents();
      dqmStore_->setCurrentFolder("DAQ/Layouts/");
      dqmStore_->removeContents();
      meInitialized_=false;
      doFlush(); 
      state = "Ready";
    }
    else if(commandName == "Halt") state = "Halted";
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

  if (!evtProcessor_ && !dqmDisabled_) initFramework();
  else if (evtProcessor_ && !meInitialized_) initMonitorElements();

  timeval tv;
  gettimeofday(&tv,0);
  time_t now = tv.tv_sec;
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
  cgi.getElement("trp",el1);
  if(el1.size()!=0)
    {
      unsigned int lsid = run;
      parsePathHisto((unsigned char*)(el1[0].getValue().c_str()),lsid);
    }
  el1.clear();


}

//______________________________________________________________________________
void iDie::postEntryiChoke(xgi::Input*in,xgi::Output*out)
  throw (xgi::exception::Exception)
{
  //  std::cout << "postEntryiChoke " << std::endl;
  unsigned int lsid = 0;
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

  if (!evtProcessor_ && !dqmDisabled_) {
    initFramework();
  }
  else if (evtProcessor_) initMonitorElements();
  doFlush();

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
    datap_ = new int[nstates_+4];
    std::ostringstream ost;
    ost<<mapmod_[0]<<"/I";
    for(unsigned int i = 1; i < nstates_; i++)
      ost<<":"<<mapmod_[i];
    ost<<":nsubp:instance:nproc:ncpubusy";//:nprocstat1k
    f_->cd();
    t_ = new TTree("microReport","microstate report tree");
    t_->SetAutoSave(500000);
    b_ = t_->Branch("microstates",datap_,ost.str().c_str());
    b1_ = t_->Branch("ls",&lsid,"ls/I");

  }

  memcpy(datap_,trp,(nstates_+4)*sizeof(int));
  //check ls for subprocess type
  unsigned int datapLen_ = nstates_+4;
  unsigned int nbsubs_ = datap_[datapLen_-4];
  unsigned int nbproc_ = datap_[datapLen_-2];
  unsigned int ncpubusy_ = datap_[datapLen_-1];

  //find index number
  int nbsIdx = -1;
  if (nbSubsList.find(nbsubs_)!=nbSubsList.end()) {
     nbsIdx = nbSubsList[nbsubs_];
    if (currentLs_[nbsIdx]<lsid) {
      if (currentLs_[nbsIdx]!=0) {
        if (lsHistory[nbsIdx].size()) {
	  //push update
	  lsStat & lst = lsHistory[nbsIdx].back();

	  fillDQMStatHist(nbsIdx,currentLs_[nbsIdx],lst.getRate(),lst.getEvtTime(),
			  lst.getFracBusy(),lst.getFracCPUBusy(),
			  lst.getRateErr(),lst.getEvtTimeErr());

	  fillDQMModFractionHist(nbsIdx,currentLs_[nbsIdx],lst.getNSampledNonIdle(),
			  lst.getOffendersVector());

	  lsHistory[nbsIdx].back().deleteModuleSamplingPtr();//clear
	  doFlush();
	  perLumiFileSaver(currentLs_[nbsIdx]);
	}
      }
      currentLs_[nbsIdx]=lsid;
      if (!commonLsHistory.size() || commonLsHistory.back().ls_<lsid)
        commonLsHistory.push(commonLsStat(lsid,epInstances.size()));
      lsHistory[nbsIdx].push(lsStat(lsid,nbsubs_,nModuleLegendaMessageReceived_,nstates_));
      nbMachines[nbsIdx]=0;
    }

    //reporting machines
    nbMachines[nbsIdx]++;

    unsigned int cumulative_ = 0;
    std::pair<unsigned int,unsigned int> * fillptr = lsHistory[nbsIdx].back().getModuleSamplingPtr();
    for (unsigned int i=0;i<nstates_;i++) {
      cumulative_+=datap_[i];
      if (fillptr) {
        fillptr[i].second+=datap_[i];
      }
    }
    lsHistory[nbsIdx].back().update(cumulative_-datap_[2],datap_[2],nbproc_,ncpubusy_);
  }
  else {
   //no defined plots for this number of sub processes
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
  for (boost::tokenizer<boost::char_separator<char> >::iterator tok_iter = tokens.begin();
       tok_iter != tokens.end(); ++tok_iter){
      mappath_.push_back((*tok_iter));
  }
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
  for( int i=0; i< trppriv_->endPathsInMenu; i++)
    {
      r_.etimesRun[i] = trppriv_->endPathSummaries[i].timesRun;
      r_.etimesPassedPs[i] = trppriv_->endPathSummaries[i].timesPassedPs;
      r_.etimesPassedL1[i] = trppriv_->endPathSummaries[i].timesPassedL1;
      r_.etimesPassed[i] = trppriv_->endPathSummaries[i].timesPassed;
      r_.etimesFailed[i] = trppriv_->endPathSummaries[i].timesFailed;
      r_.etimesExcept[i] = trppriv_->endPathSummaries[i].timesExcept;
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
  if (!dqmCollectorHost_.value_.size() || !dqmCollectorPort_.value_.size()) {
    dqmDisabled_=true;
    std::cout << " DQM connection parameters not present. Disabling DQM histograms" << std::endl;
    return;
  }

  std::string configuration_ = configString_;
  configuration_.replace(configuration_.find("EMPTYHOST"),9,dqmCollectorHost_.value_);
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
      std::cout << "SLAVE: Unable to create message service presence "<<std::endl;
    }
  } 
  catch(edm::Exception e) {
    std::cout << "edm::Exception: "<< e.what() << std::endl;
  }

  catch(cms::Exception e) {
    std::cout << "cms::Exception: "<< e.what() << std::endl;
  }
 
  catch(std::exception e) {
    std::cout << "std::exception: "<< e.what() << std::endl;
  }
  catch(...) {
    std::cout <<"SLAVE: Unknown Exception (Message Presence)"<<std::endl;
  }

  try {
  serviceToken_ = edm::ServiceRegistry::createSet(*pServiceSets_);
  }
  catch (...) {std::cout << "Failed creation of service token "<<std::endl;
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
  }
  try{
    if(edm::Service<DQMService>().isAvailable())
      dqmService_ = edm::Service<DQMService>().operator->();
  }
  catch(...) {
    LOG4CPLUS_WARN(getApplicationLogger(),"exception when trying to get service DQMServic");
  }

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
          nbMachines[i]=0;
  }
  ilumiprev_ = 0;
  savedForLs_=0;
  pastSavedFiles_.clear();
 
  dqmStore_->setCurrentFolder("DAQ/Layouts/");
  for (unsigned int i=0;i<nbSubsClasses;i++) {
    std::ostringstream str;
    str << nbSubsListInv[i];
    meVecRate_.push_back(dqmStore_->book1D("EVENT_RATE_"+TString(str.str().c_str()),
		         "Average event rate for nodes with " + TString(str.str().c_str()) + " EP instances",
		         4000,1.,4001));
    meVecTime_.push_back(dqmStore_->book1D("EVENT_TIME_"+TString(str.str().c_str()),
		         "Average event processing time for nodes with " + TString(str.str().c_str()) + " EP instances",
		         4000,1.,4001));
    meVecOffenders_.push_back(dqmStore_->book2D("MODULE_FRACTION_"+TString(str.str().c_str()),
			                        "Module processing time fraction_"+ TString(str.str().c_str()),
		                                MODLZSIZELUMI,1.,1.+MODLZSIZELUMI,MODLZSIZE,0,MODLZSIZE));
    //fill 1 in underrflow bin
    meVecOffenders_[i]->Fill(0,1);
  }
  occupancyNameMap.clear();
  rateSummary_   = dqmStore_->book2D("00_RATE_SUMMARY","Rate Summary (Hz)",20,0,20,epInstances.size()+1,0,epInstances.size()+1);
  timingSummary_ = dqmStore_->book2D("01_TIMING_SUMMARY","Event Time Summary (ms)",20,0,20,epInstances.size()+1,0,epInstances.size()+1);
  busySummary_ = dqmStore_->book2D("02_BUSY_SUMMARY","Busy fraction ",20,0,20,epInstances.size()+2,0,epInstances.size()+2);
  busySummary2_ = dqmStore_->book2D("02_BUSY_SUMMARY_PROCSTAT","Busy fraction from /proc/stat",20,0,20,epInstances.size()+2,0,epInstances.size()+2);
  dqmStore_->setCurrentFolder("DAQ/EventInfo/");
  daqBusySummary_ = dqmStore_->book1D("reportSummaryMap","DAQ HLT Farm busy (%)",4000,1,4001.);
  summaryLastLs_ = 0;
  for (size_t i=1;i<=20;i++) {
    std::ostringstream ostr;
    ostr << i;
    rateSummary_->setBinLabel(i,ostr.str(),1);
    timingSummary_->setBinLabel(i,ostr.str(),1);
    busySummary_->setBinLabel(i,ostr.str(),1);
    busySummary2_->setBinLabel(i,ostr.str(),1);
  }
  for (size_t i=1;i<epInstances.size()+1;i++) {
    std::ostringstream ostr;
    ostr << epInstances[i-1];
    rateSummary_->setBinLabel(i,ostr.str(),2);
    timingSummary_->setBinLabel(i,ostr.str(),2);
    busySummary_->setBinLabel(i,ostr.str(),2);
    busySummary2_->setBinLabel(i,ostr.str(),2);
  }
  rateSummary_->setBinLabel(epInstances.size()+1,"All",2);
  //timingSummary_->setBinLabel(i,"Avg",2);
  busySummary_->setBinLabel(epInstances.size()+1,"%Conf",2);
  busySummary_->setBinLabel(epInstances.size()+2,"%Max",2);
  busySummary2_->setBinLabel(epInstances.size()+1,"%Conf",2);
  busySummary2_->setBinLabel(epInstances.size()+2,"%Max",2);

  for (size_t i=0;i<epInstances.size();i++) {
    lsHistory[i]=std::queue<lsStat>();
    commonLsHistory=std::queue<commonLsStat>();
  }

  meInitialized_=true;

}

void iDie::deleteFramework()
{
  if (evtProcessor_) delete evtProcessor_;
}

void iDie::fillDQMStatHist(int nbsIdx,unsigned int lsid,float rate, float time, float busy, float busyCPU, float rateErr, float timeErr)
{
  if (!evtProcessor_) return;
  meVecRate_[nbsIdx]->setBinContent(lsid,rate);
  meVecRate_[nbsIdx]->setBinError(lsid,rateErr);
  meVecTime_[nbsIdx]->setBinContent(lsid,epInstances[nbsIdx]*time*1000);//msec
  meVecTime_[nbsIdx]->setBinError(lsid,epInstances[nbsIdx]*timeErr*1000);//msec
  updateRollingHistos(lsid,rate,time*1000,busy,busyCPU,nbsIdx);
}

void iDie::updateRollingHistos(unsigned int lsid,unsigned int rate, float ms, float busy, float busyCPU, unsigned int nbsIdx) {
  unsigned int lsidBin;
  if (lsid>20) {
    if (lsid!=summaryLastLs_) {//see if plots aren't up to date
      for (unsigned int i=1;i<=20;i++) {
	if (i<20) {
          for (unsigned int j=1;j<=epInstances.size()+1;j++) {
	    rateSummary_->setBinContent(i,j,rateSummary_->getBinContent(i+1,j));
	    timingSummary_->setBinContent(i,j,timingSummary_->getBinContent(i+1,j));
	    busySummary_->setBinContent(i,j,busySummary_->getBinContent(i+1,j));
	    busySummary2_->setBinContent(i,j,busySummary2_->getBinContent(i+1,j));
          }
	  busySummary_->setBinContent(i,epInstances.size()+2,busySummary2_->getBinContent(i+1,epInstances.size()+2));
	  busySummary2_->setBinContent(i,epInstances.size()+2,busySummary2_->getBinContent(i+1,epInstances.size()+2));
	}

	std::ostringstream ostr;
	ostr << lsid-20+i;
	rateSummary_->setBinLabel(i,ostr.str(),1);
	timingSummary_->setBinLabel(i,ostr.str(),1);
	busySummary_->setBinLabel(i,ostr.str(),1);
	busySummary2_->setBinLabel(i,ostr.str(),1);

      }
    }
    lsidBin=20;
  }
  else if (lsid) {lsidBin=lsid;} else return;

  rateSummary_->setBinContent(lsidBin,nbsIdx+1,rate);
  timingSummary_->setBinContent(lsidBin,nbsIdx+1,epInstances[nbsIdx]*ms);

  //float epMaxInv=1/epMax[nbsIdx];
  float busyCorr = busy * (float)epInstances[nbsIdx]/epMax[nbsIdx];//really how busy is machine (uncorrected)
  ///busyCPU *= (float)epInstances[nbsIdx]/epMax[nbsIdx];
  //max based on how much is configured and max possible
  float fracMax  = 0.5 + (std::max(epInstances[nbsIdx]-epMax[nbsIdx]/2.,0.)/(epMax[nbsIdx])) *HTscaling[nbsIdx];

  //std::cout << "busy:" << busy << " busyCorr:" << busyCorr << " fracMax:" << fracMax << " ht scaling:" << HTscaling[nbsIdx] << std::endl; 
  float busyFr=0;
  float busyCPUFr=0;
  float busyFrTheor=0;
  float busyFrCPUTheor=0;
  if (busyCorr>0.5) {//take into account HT scaling for the busy fraction
    busyFr=(0.5 + (busyCorr-0.5)*HTscaling[nbsIdx])/fracMax;
    busyCPUFr=(0.5 + (busyCPU-0.5)*HTscaling[nbsIdx])/fracMax;
    busyFrTheor = (0.5+(busyCorr-0.5)*HTscaling[nbsIdx])/ (0.5+HTscaling[nbsIdx]);
    busyFrCPUTheor = (0.5+(busyCPU-0.5)*HTscaling[nbsIdx])/ (0.5+HTscaling[nbsIdx]);
  }
  else {//below the HT threshold
    busyFr=busyCorr / fracMax;
    busyCPUFr=busyCPU / fracMax;
    busyFrTheor = busyCorr / (0.5+HTscaling[nbsIdx]);
    busyFrCPUTheor = busyCPU / (0.5+HTscaling[nbsIdx]);
  }
  busySummary_->setBinContent(lsidBin,nbsIdx+1,busyFr);//"corrected" cpu busy fraction
  busySummary2_->setBinContent(lsidBin,nbsIdx+1,busyCPUFr);//"corrected" cpu busy fraction
  //std::cout << " b: " << busyFr << " " << busyFrTheor << " " << nbMachines[nbsIdx] << std::endl;
  commonLsHistory.back().setBusyForClass(nbsIdx,rate,busyFr,busyFrTheor,busyCPUFr,busyFrCPUTheor,nbMachines[nbsIdx]);
  rateSummary_->Fill(lsid,epInstances.size()+1,commonLsHistory.back().getTotalRate());
  busySummary_->setBinContent(lsid,epInstances.size()+1,commonLsHistory.back().getBusyTotalFrac(false,machineWeightInst));
  busySummary2_->setBinContent(lsid,epInstances.size()+1,commonLsHistory.back().getBusyTotalFrac(true,machineWeightInst));
  busySummary_->setBinContent(lsid,epInstances.size()+2,commonLsHistory.back().getBusyTotalFracTheor(false,machineWeight));
  busySummary2_->setBinContent(lsid,epInstances.size()+2,commonLsHistory.back().getBusyTotalFracTheor(true,machineWeight));
  daqBusySummary_->setBinContent(lsid,busyFr*100);
  daqBusySummary_->setBinError(lsid,0);

}

void iDie::fillDQMModFractionHist(int nbsIdx, unsigned int lsid, unsigned int nonIdle, std::vector<std::pair<unsigned int,unsigned int>> offenders)
{
  if (!evtProcessor_) return;
  MonitorElement * me = meVecOffenders_[nbsIdx];
  //shift bin names by 1
  unsigned int xBinToFill=lsid;
  if (lsid>MODLZSIZELUMI) {
    for (unsigned int i=1;i<=MODLZSIZELUMI;i++) {
      for (unsigned int j=1;j<=MODLZSIZE;j++) {
	if (i<MODLZSIZELUMI)
	  me->setBinContent(i,j,me->getBinContent(i+1,j));
	else
	  me->setBinContent(i,j,0);
      }
      std::ostringstream ostr;
      ostr << lsid-MODLZSIZELUMI+i;
      me->setBinLabel(i,ostr.str(),1);
    }
    std::ostringstream ostr;
    ostr << lsid;
    //me->setBinLabel(MODLZSIZELUMI,ostr.str(),1);
    //me->setBinContent(MODLZSIZELUMI,ostr.str(),1);
    xBinToFill=MODLZSIZELUMI;
  }
  float nonIdleInv=0.;
  if (nonIdle>0.)nonIdleInv=1./nonIdle;
  for (unsigned int i=0;i<offenders.size();i++) {
    unsigned int x=offenders[i].first;
    float percentageUsed=offenders[i].second*nonIdleInv;
    if (percentageUsed>0.03) {//3% threshold
      if (occupancyNameMap.count(x)==0) {
	unsigned int y=occupancyNameMap.size();
	if (y<MODLZSIZE) {
	  occupancyNameMap[x]=y;
	  me->setBinContent(xBinToFill,y+1,((int)(1000.*percentageUsed))/1000.);
	  me->setBinLabel(y+1,mapmod_[x],2);
	}
      }
    }
  }
  for (unsigned int i=0;i<offenders.size();i++) {
    unsigned int x=offenders[i].first;
    float percentageUsed=offenders[i].second*nonIdleInv;
    if (percentageUsed>0.03) {//3% threshold
      if (occupancyNameMap.count(x)==0) {
	unsigned int y=occupancyNameMap.size();
	if (y>=MODLZSIZE && xBinToFill>1) {
	  //filled up, replace another one
	  float minbinval=1.;
	  unsigned int toReplace=0;
	  for (size_t j=1;j<=MODLZSIZE;j++) {
	    //decide based on the smallest value
	    float bin=me->getBinContent(xBinToFill,j);
	    if (bin<minbinval) {toReplace=j;minbinval=bin;}
	  }
	  if (percentageUsed>minbinval && toReplace) {
	    int key=-1;
	    for (auto it = occupancyNameMap.begin(); it != occupancyNameMap.end(); ++it) {
	      if (it->second == toReplace-1) {
		key = it->first;
		break;
	      }
	    }
	    if (key>-1) {
	      //erase old
	      occupancyNameMap.erase(key);
	      //add new
	      occupancyNameMap[x]=toReplace-1;
	      //fill histogram
	      me->setBinContent(xBinToFill,toReplace,((int)(100.*percentageUsed))/100.);
	      me->setBinLabel(toReplace,mapmod_[x],2);
	      //reset fields for previous lumis
	      for (size_t k=1;k<xBinToFill;k++)
                me->setBinContent(k,toReplace,0);
	    }
	  }
	}
      }
      else {
	unsigned int y=occupancyNameMap[x];
	me->setBinContent(xBinToFill,y,((int)(100.*percentageUsed))/100.);
      }
    }
  }
}

void iDie::doFlush() {
    dqmService_->flushStandalone();
}

void iDie::perLumiFileSaver(unsigned int lsid)
{
 
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

    std::vector<std::string> systems = {"DAQ"};

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
	LOG4CPLUS_ERROR(getApplicationLogger(),"iDie could not create root file");
      }
    }

    ilumiprev_ = lsid;

    //cd() to micro report root file
    if (f_)
      f_->cd();
  }
}


////////////////////////////////////////////////////////////////////////////////
// xdaq instantiator implementation macro
////////////////////////////////////////////////////////////////////////////////

XDAQ_INSTANTIATOR_IMPL(iDie)
