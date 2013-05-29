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
#include <time.h>

#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/FormFile.h"
#include "cgicc/HTMLClasses.h"

#include "EventFilter/Utilities/interface/DebugUtils.h"

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
  ispace->fireItemAvailable("parameterSet",         &configString_                );
  ispace->fireItemAvailable("runNumber",            &runNumber_                   );
  getApplicationInfoSpace()->addItemChangedListener("runNumber",              this);

  // timestamps
  lastModuleLegendaMessageTimeStamp_.tv_sec=0;
  lastModuleLegendaMessageTimeStamp_.tv_usec=0;
  lastPathLegendaMessageTimeStamp_.tv_sec=0;
  lastPathLegendaMessageTimeStamp_.tv_usec=0;
  runStartDetectedTimeStamp_.tv_sec=0;
  runStartDetectedTimeStamp_.tv_usec=0;
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
    else if(commandName == "Stop") state = "Ready";
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
	if(fus_.size()==0) //close the root file if we know the run is over 
	  if(f_!=0){
	    f_->cd();
	    f_->Write();
	    f_->Close();
	    t_ = 0;
	    delete f_; f_ = 0;
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
    f_->Close();
    delete f_; f_ = 0;
  }
  if(t_ != 0)
    {delete t_; t_=0;}
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
    datap_ = new int[nstates_+3];
    std::ostringstream ost;
    ost<<mapmod_[0]<<"/I";
    for(unsigned int i = 1; i < nstates_; i++)
      ost<<":"<<mapmod_[i];
    ost<<":nsubp:instance:nproc";
    f_->cd();
    t_ = new TTree("microReport","microstate report tree");
    t_->SetAutoSave(500000);
    b_ = t_->Branch("microstates",datap_,ost.str().c_str());
    b1_ = t_->Branch("ls",&lsid,"ls/I");
  }
  memcpy(datap_,trp,(nstates_+3)*sizeof(int));
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


////////////////////////////////////////////////////////////////////////////////
// xdaq instantiator implementation macro
////////////////////////////////////////////////////////////////////////////////

XDAQ_INSTANTIATOR_IMPL(iDie)
