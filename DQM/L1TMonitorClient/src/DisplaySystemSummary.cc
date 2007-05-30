#include "DQM/L1TMonitorClient/interface/DisplaySystemSummary.h"


void DisplaySystemSummary::printHTML(xgi::Output * out)
{
  //  
  std::string position = "position:absolute; left:" + get_pix_left() 
    + "; top:" + get_pix_top();
  
  *out << cgicc::br() << std::endl;
  *out << cgicc::div().set("style", position.c_str()) << std::endl;
  *out << cgicc::input().
    set("type", "button").
    set("name", "button").
    set("value", name).
    set("style","width: 100px").
    set("onClick", command) << std::endl;
  *out << cgicc::div()   << std::endl;
}

