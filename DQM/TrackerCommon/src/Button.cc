#include "DQM/TrackerCommon/interface/Button.h"


void Button::printHTML(xgi::Output *out)
{
  std::string position = "position:absolute; left:" + get_pix_left() + "; top:" + get_pix_top();
  *out << cgicc::div().set("style", position.c_str()) << std::endl;

  std::string js_command = "makeRequest('" + get_url() + "/" + "Request?RequestID=" + requestID + "', dummy)";
  *out << cgicc::input().set("type", "button")
    .set("value", name.c_str())
    .set("onclick", js_command.c_str()) << std::endl;

  *out << cgicc::div() << std::endl;
}
