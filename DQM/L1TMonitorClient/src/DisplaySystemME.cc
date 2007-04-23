#include "DQM/L1TMonitorClient/interface/DisplaySystemME.h"


void DisplaySystemME::printHTML(xgi::Output * out)
{
//  
  std::string open_command = "makeMeListRequest(\'" + name + "\')"; 
  std::string position = "position:absolute; left:" + get_pix_left() + "; top:" + get_pix_top();
  
  *out << cgicc::br() << std::endl;
  *out << cgicc::div().set("style", position.c_str()) << std::endl;
  *out << cgicc::input().set("type", "button").set("name", "button").set("value", name).set("style","width: 100px").set("onClick", open_command) << std::endl;
  *out << cgicc::div()   << std::endl;
}

/*
void DisplaySystemME::printSelectHTML(xgi::Output * out, std::string name, std::string onchange)
{
  if (name == "Open") 
    {
        *out << cgicc::body().set("onload", onchange) << std::endl;
    }
  else if (name != "Open")
    {
//       
//      Problem handler here....!
//      
   }
}

*/
