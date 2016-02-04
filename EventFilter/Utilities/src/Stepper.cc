

#include "EventFilter/Utilities/interface/Stepper.h"

#include "xgi/Input.h"
#include "xgi/Output.h"
#include "xgi/Utils.h"

#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"

#include <iostream>

#include <string>


namespace evf{

  Stepper::Stepper(const edm::ParameterSet& iPS, 
		   edm::ActivityRegistry& reg) : ServiceWeb("Stepper"), rid_(0), eid_(0), step_(false)
  {
  
    reg.watchPostBeginJob(this,&Stepper::postBeginJob);
    reg.watchPostEndJob(this,&Stepper::postEndJob);
  
    reg.watchPreProcessEvent(this,&Stepper::preEventProcessing);
    reg.watchPostProcessEvent(this,&Stepper::postEventProcessing);
    reg.watchPreSource(this,&Stepper::preSource);
    reg.watchPostSource(this,&Stepper::postSource);
  
    reg.watchPreModule(this,&Stepper::preModule);
    reg.watchPostModule(this,&Stepper::postModule);
    epstate_ = "BJ";
    modulename_ = "NOT YET";
    modulelabel_ = "INIT";
    pthread_mutex_init(&mutex_,0);
    pthread_cond_init(&cond_,0);
  }


  Stepper::~Stepper()
  {
  }

  void Stepper::postBeginJob()
  {
    //    wait_on_signal();
    epstate_ = "BJD";
  }

  void Stepper::postEndJob()
  {
    wait_on_signal();
    epstate_ = "EJ";
    modulelabel_ = "done";
  }

  void Stepper::preEventProcessing(const edm::EventID& iID,
					     const edm::Timestamp& iTime)
  {
    rid_ = iID.run();
    eid_ = iID.event();
    wait_on_signal();
    epstate_ = "PRO";
  }

  void Stepper::postEventProcessing(const edm::Event& e, const edm::EventSetup&)
  {
  }
  void Stepper::preSource()
  {
    wait_on_signal();
    modulename_ = "source";
    modulelabel_ = "IN";
  }

  void Stepper::postSource()
  {
    wait_on_signal();
    modulelabel_ = "IND";
  }

  void Stepper::preModule(const edm::ModuleDescription& desc)
  {
    wait_on_signal();
    modulename_ = desc.moduleName();
    modulelabel_ = desc.moduleLabel();
  }

  void Stepper::postModule(const edm::ModuleDescription& desc)
  {
  }
  
  void Stepper::defaultWebPage(xgi::Input *in, xgi::Output *out)
  {

    std::string path;
    std::string urn;
    std::string mname;
    try 
      {
	cgicc::Cgicc cgi(in);
	if ( xgi::Utils::hasFormElement(cgi,"service") )
	  mname = xgi::Utils::getFormElement(cgi, "service")->getValue();
	if ( xgi::Utils::hasFormElement(cgi,"step") )
	  {
	    pthread_mutex_lock(&mutex_);
	    
	    pthread_cond_signal(&cond_);
	    
	    pthread_mutex_unlock(&mutex_);
	  }
	cgicc::CgiEnvironment cgie(in);
	if(original_referrer_ == "")
	  original_referrer_ = cgie.getReferrer();
	path = cgie.getPathInfo() + "?" + cgie.getQueryString();
	
      }
      catch (const std::exception & e) 
	{
	  // don't care if it did not work
	}
      
      
      using std::endl;
      *out << "<html>"                                                   << endl;
      *out << "<head>"                                                   << endl;


      *out << "<STYLE type=\"text/css\"> #T1 {border-width: 2px; border: solid blue; text-align: center} </STYLE> "                                      << endl; 
      *out << "<link type=\"text/css\" rel=\"stylesheet\"";
      *out << " href=\"/" <<  urn
	   << "/styles.css\"/>"                   << endl;

      *out << "<title>" << serviceName_
	   << " MAIN</title>"                                            << endl;

      *out << "</head>"                                                  << endl;
      *out << "<body onload=\"loadXMLDoc()\">"                           << endl;
      *out << "<table border=\"0\" width=\"100%\">"                      << endl;
      *out << "<tr>"                                                     << endl;
      *out << "  <td align=\"left\">"                                    << endl;
      *out << "    <img"                                                 << endl;
      *out << "     align=\"middle\""                                    << endl;
      *out << "     src=\"/evf/images/stepper.jpg\""			 << endl;
      *out << "     alt=\"main\""                                        << endl;
      *out << "     width=\"90\""                                        << endl;
      *out << "     height=\"64\""                                       << endl;
      *out << "     border=\"\"/>"                                       << endl;
      *out << "    <b>"                                                  << endl;
      *out <<             serviceName_                                   << endl;
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
      *out << "  <td width=\"32\">"                                      << endl;
      *out << "  </td>"                                                  << endl;
      *out << "  <td width=\"32\">"                                      << endl;
      *out << "    <a href=\"" << original_referrer_  << "\">"           << endl;
      *out << "      <img"                                               << endl;
      *out << "       align=\"middle\""                                  << endl;
      *out << "       src=\"/evf/images/epicon.jpg\""			 << endl;
      *out << "       alt=\"main\""                                      << endl;
      *out << "       width=\"32\""                                      << endl;
      *out << "       height=\"32\""                                     << endl;
      *out << "       border=\"\"/>"                                     << endl;
      *out << "    </a>"                                                 << endl;
      *out << "  </td>"                                                  << endl;
      *out << "</tr>"                                                    << endl;
      *out << "</table>"                                                 << endl;

      *out << "<hr/>"                                                    << endl;
      
      *out << "run number        " << rid_ << "<br>" << endl;
      *out << "event number      " << eid_ << "<br>" << endl;

      *out << "event processor   " << epstate_ << "<br>" << endl;

      *out << "next module type  " << modulename_ << "<br>" << endl;
      *out << "next module label " << modulelabel_ << "<br>" << endl;

      *out << "<hr/>"                                                    << endl;
  
      *out << cgicc::form().set("method","GET").set("action", path ) 
	   << std::endl;
      *out << cgicc::input().set("type","hidden").set("name","service").set("value", mname) 
	   << std::endl;
      *out << cgicc::input().set("type","hidden").set("name","step").set("value", "yes") 
	   << std::endl;
      *out << cgicc::input().set("type","submit").set("value","Step")      << std::endl;
      *out << cgicc::form()						   << std::endl;  

      *out << "</body>"                                                  << endl;
      *out << "</html>"                                                  << endl;
  }
} //end namespace evf

