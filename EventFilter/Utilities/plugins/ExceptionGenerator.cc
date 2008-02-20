#include "ExceptionGenerator.h"

#include <iostream>
#include <typeinfo>
#include <map>
#include <sstream>

#include "xgi/Method.h"
#include "xgi/Utils.h"

#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"

#include <stdio.h>

using namespace std;
namespace evf{

    const std::string ExceptionGenerator::menu[menu_items] =  {"SleepForever","Exit with error","Abort", "Unknown Exception", "Endless loop" };

    ExceptionGenerator::ExceptionGenerator( const edm::ParameterSet& pset) : 
      ModuleWeb("ExceptionGenerator"), actionRequired_(false), actionId_(-1)
    {
      

    }
    void ExceptionGenerator::analyze(const edm::Event & e, const edm::EventSetup& c)
    {
      if(actionRequired_) 
	{
	  int ind = 0; 
	  int step = 1; 
	  switch(actionId_)
	    {
	    case 0:
	      ::sleep(0xFFFFFFF);
	      break;
	    case 1:
	      exit(-1);
	      break;
	    case 2:
	      abort();
	      break;
	    case 3:
	      throw "pippo";
	      break;
	    case 4:
	      while(1){ind+=step; if(ind>1000000) step = -1; if(ind==0) step = 1;}
	    }
	}
    }
    
    void ExceptionGenerator::endLuminosityBlock(edm::LuminosityBlock const &lb, edm::EventSetup const &es)
    {

    }
    
    void ExceptionGenerator::defaultWebPage(xgi::Input *in, xgi::Output *out)
    {

      std::string path;
      std::string mname;
      try 
	{
	  cgicc::Cgicc cgi(in);
	  if ( xgi::Utils::hasFormElement(cgi,"exceptiontype") )
	    {
	      actionId_ = xgi::Utils::getFormElement(cgi, "exceptiontype")->getIntegerValue();
	      actionRequired_ = true;
	    }
	  if ( xgi::Utils::hasFormElement(cgi,"module") )
	    mname = xgi::Utils::getFormElement(cgi, "module")->getValue();
	  cgicc::CgiEnvironment cgie(in);
	  path = cgie.getPathInfo() + "?" + cgie.getQueryString();
	  
	}
      catch (const std::exception & e) 
	{
	  // don't care if it did not work
	}
      
      
      using std::endl;
      *out << "<html>"                                                   << endl;
      *out << "<head>"                                                   << endl;
      
      *out << "<title>" << typeid(ExceptionGenerator).name()
	   << " MAIN</title>"                                            << endl;
      
      *out << "</head>"                                                  << endl;
      *out << "<body>"                                                   << endl;
      
      *out << cgicc::form().set("method","GET").set("action", path ) 
	   << std::endl;
      *out << cgicc::input().set("type","hidden").set("name","module").set("value", mname) 
	   << std::endl;
      *out << cgicc::select().set("name","exceptiontype")     << std::endl;
      char istring[2];

      for(int i = 0; i < menu_items; i++)
	{
	  sprintf(istring,"%d",i);
	  *out << cgicc::option().set("value",istring) << menu[i] << cgicc::option()       << std::endl;
	}
      *out << cgicc::select() 	     << std::endl;
      *out << cgicc::input().set("type","submit").set("value","Do It !")  	     << std::endl;
      *out << cgicc::form()						   << std::endl;  

      *out << "</body>"                                                  << endl;
      *out << "</html>"                                                  << endl;
    }

    }
