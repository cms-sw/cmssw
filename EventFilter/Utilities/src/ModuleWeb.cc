#include "EventFilter/Utilities/interface/ModuleWeb.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "xgi/Input.h"
#include "xgi/Output.h"
#include "xgi/Utils.h"

#include <string>


namespace evf{


ModuleWeb::ModuleWeb(const std::string &moduleName) : moduleName_(moduleName)
{
  if(edm::Service<ModuleWebRegistry>())
    edm::Service<ModuleWebRegistry>()->registerWeb(moduleName_, this);
}

void ModuleWeb::publishForkInfo(moduleweb::ForkInfoObj * forkInfoObj)
{
  return;
}


void ModuleWeb::defaultWebPage(xgi::Input *in, xgi::Output *out)
{
  using std::endl;
  std::string path;
  std::string urn;
  std::string mname;
  try 
    {
      cgicc::Cgicc cgi(in);
      if ( xgi::Utils::hasFormElement(cgi,"module") )
	mname = xgi::Utils::getFormElement(cgi, "module")->getValue();
      cgicc::CgiEnvironment cgie(in);
      urn = cgie.getReferrer();
      path = cgie.getPathInfo() + "?" + cgie.getQueryString();
      
    }
  catch (const std::exception & e) 
    {
      // don't care if it did not work
    }
  

  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;

  *out << "<STYLE type=\"text/css\"> #T1 {border-width: 2px; border: solid blue; text-align: center} </STYLE> "
       << endl; 
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"/" <<  urn
       << "/styles.css\"/>"                   << endl;

  *out << "<title>" << moduleName_
       << " MAIN</title>"                                            << endl;

  *out << "</head>"                                                  << endl;
  *out << "<body onload=\"loadXMLDoc()\">"                           << endl;
  *out << "<table border=\"0\" width=\"100%\">"                      << endl;
  *out << "<tr>"                                                     << endl;
  *out << "  <td align=\"left\">"                                    << endl;
  *out << "    <img"                                                 << endl;
  *out << "     align=\"middle\""                                    << endl;
  *out << "     src=\"/evf/images/epicon.jpg\"" 		     << endl;
  *out << "     alt=\"main\""                                        << endl;
  *out << "     width=\"64\""                                        << endl;
  *out << "     height=\"64\""                                       << endl;
  *out << "     border=\"\"/>"                                       << endl;
  *out << "    <b>"                                                  << endl;
  *out << "<title>" << moduleName_
       << " MAIN</title>"                                            << endl;
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
  *out << "    <a href=\"/" << urn 
       << "\">"                                                      << endl;
  *out << "      <img"                                               << endl;
  *out << "       align=\"middle\""                                  << endl;
  *out << "       src=\"/evf/images/epicon.jpg\""		     << endl;
  *out << "       alt=\"main\""                                      << endl;
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
} //end namespace evf
