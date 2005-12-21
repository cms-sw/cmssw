#include "EventFilter/Utilities/interface/ModuleWeb.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <string>

using namespace std;
using namespace evf;

ModuleWeb::ModuleWeb(const string &moduleName) : moduleName_(moduleName)
{
  if(edm::Service<ModuleWebRegistry>())
    edm::Service<ModuleWebRegistry>()->registerWeb(moduleName_, this);
}

#include "xgi/include/xgi/Input.h"
#include "xgi/include/xgi/Output.h"

void ModuleWeb::defaultWebPage(xgi::Input *in, xgi::Output *out)
{
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
