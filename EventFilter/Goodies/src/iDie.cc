#include "iDie.h"

#include "EventFilter/Utilities/interface/CurlUtils.h"

#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xoap/domutils.h"

#include <boost/tokenizer.hpp>

#include <netinet/in.h>
#include <sstream>
#include <errno.h>

#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"
#include "TH1F.h"
#include "TCanvas.h"


using namespace std;
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
{
  // initialize application info
  url_     =
    getApplicationDescriptor()->getContextDescriptor()->getURL()+"/"+
    getApplicationDescriptor()->getURN();
  class_   =getApplicationDescriptor()->getClassName();
  instance_=getApplicationDescriptor()->getInstance();
  hostname_=getApplicationDescriptor()->getContextDescriptor()->getURL();
  
  // web interface
  xgi::bind(this,&evf::iDie::defaultWeb,"Default"    );
  xgi::bind(this,&evf::iDie::summaryTable,"summary"  );
  xgi::bind(this,&evf::iDie::detailsTable,"details"  );
  xgi::bind(this,&evf::iDie::dumpTable,"dump"        );
  xgi::bind(this,&evf::iDie::updater,"updater"       );

  xgi::bind(this,&evf::iDie::postEntry, "postEntry"  );
  //  gui_->setSmallAppIcon("/evf/images/Hilton.gif");
  //  gui_->setLargeAppIcon("/evf/images/Hilton.gif");
  

}


//______________________________________________________________________________
iDie::~iDie()
{
}



//______________________________________________________________________________
void iDie::defaultWeb(xgi::Input *in,xgi::Output *out)
  throw (xgi::exception::Exception)
{
  cgicc::Cgicc cgi(in);
  std::string method = cgi.getEnvironment().getRequestMethod();
  if(method == "POST"){
    int run;
    std::vector<cgicc::FormEntry> el1 = cgi.getElements();
    std::cout << "dump of post to defaultWeb" << std::endl;
    for(unsigned int i = 0; i < el1.size(); i++)
      std::cout << el1[i].getValue() << std::endl;
    cgi.getElement("run",el1);
    if(el1.size()!=0){
      std::cout << "got runnumber " << el1[0].getIntegerValue() << std::endl;
      run = el1[0].getIntegerValue();
      if(run>runNumber_.value_ || runNumber_.value_==0){
	if(runNumber_.value_!=0) reset();
	runNumber_.value_ = run;
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
	   << "</td></tr>" << std::endl;
    }
}

//______________________________________________________________________________
void iDie::dumpTable(xgi::Input *in,xgi::Output *out)
  throw (xgi::exception::Exception)
{
  cgicc::Cgicc cgi(in); 
  std::cout << "QueryString " << cgi.getEnvironment().getQueryString() 
	    << std::endl;

  std::vector<cgicc::FormEntry> el1;
  cgi.getElement("name",el1);
  if(el1.size()!=0){
    std::string hostname = el1[0].getValue();
    std::transform(hostname.begin(), hostname.end(),
		   hostname.begin(), ::toupper);
    std::cout << "looking for "<< hostname << std::endl;
    ifmap fi = fus_.find(hostname);    
    if(fi!=fus_.end()){
      std::cout << " found " << el1[0].getValue() << std::endl;
      *out << (*fi).second.stacktraces.back() << std::endl;
    }
    else{ 
      for(fi=fus_.begin(); fi != fus_.end(); fi++) 
	std::cout << "known hosts: " << (*fi).first << std::endl;
    }
  }
}

//______________________________________________________________________________
void iDie::postEntry(xgi::Input*in,xgi::Output*out)
  throw (xgi::exception::Exception)
{
  timeval tv;
  gettimeofday(&tv,0);
  time_t now = tv.tv_sec;
  cgicc::Cgicc cgi(in); 
  pid_t cpid = 0;
  /*  cgicc::CgiEnvironment cgie(in);
  cout << "query = "  << cgie.getContentLength() << endl;
  */
  std::cout << "QueryString " << cgi.getEnvironment().getQueryString() 
	    << std::endl;
  std::vector<cgicc::FormEntry> el1;
  cgi.getElement("name",el1);
  if(el1.size()!=0)
    std::cout << "entry from node " << el1[0].getValue() << std::endl;
  el1.clear();
  cgi.getElement("run",el1);
  if(el1.size()!=0)
    {
      std::cout << "entry for run " << el1[0].getIntegerValue() << std::endl;
      cpid =  el1[0].getIntegerValue();
    }
  el1.clear();
  cgi.getElement("stacktrace",el1);
  if(el1.size()!=0)
    {
      //      std::cout << "=============== stacktrace =============" << std::endl;
      //      std::cout << el1[0].getValue() << std::endl;
      if(el1[0].getValue().find("Dead")==0){

	std::string host = cgi.getEnvironment().getRemoteHost();
	std::cout << "Received Dead signal from " << host << std::endl;
	std::transform(host.begin(), host.end(),
		       host.begin(), ::toupper);
	ifmap fi = fus_.find(host);
	if(fi!=fus_.end()){
	  fus_.erase(fi);
	}
      }
      else{
	totalCores_++;
	std::string st = el1[0].getValue();
	std::cout << "postEntry string " << st << std::endl;
	std::string host = cgi.getEnvironment().getRemoteHost();
	std::transform(host.begin(), host.end(),
		       host.begin(), ::toupper);
	ifmap fi = fus_.find(host);
	if(fi!=fus_.end()){
	  (*fi).second.tstamp = now;
	  (*fi).second.ccount++;
	  (*fi).second.cpids.push_back(cpid);
	  (*fi).second.stacktraces.push_back(st);
	}
      }
    }

}

void iDie::reset()
{
  fus_.erase(fus_.begin(),fus_.end());
  totalCores_=0;
}

////////////////////////////////////////////////////////////////////////////////
// xdaq instantiator implementation macro
////////////////////////////////////////////////////////////////////////////////

XDAQ_INSTANTIATOR_IMPL(iDie)
