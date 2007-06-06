#include "EventFilter/Utilities/interface/ModuleWeb.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "xgi/Input.h"
#include "xgi/Output.h"

#include <string>


namespace evf{

ModuleWeb::ModuleWeb(const std::string &moduleName) : moduleName_(moduleName)
{
  if(edm::Service<ModuleWebRegistry>())
    edm::Service<ModuleWebRegistry>()->registerWeb(moduleName_, this);
}

void ModuleWeb::defaultWebPage(xgi::Input *in, xgi::Output *out)
{
  using std::endl;
  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;

  *out << "<title>" << moduleName_
       << " MAIN</title>"                                            << endl;
  *out << "</head>"                                                  << endl;
  *out << "<body>"                                                   << endl;
  *out << moduleName_ << " Default web page "                        << endl;
  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;
}
} //end namespace evf
