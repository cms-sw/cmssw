#include "DQM/TrackerCommon/interface/HTMLLink.h"


void HTMLLink::printHTML(xgi::Output *out)
{
  std::string position = "position:absolute; left:" + get_pix_left() + "; top:" + get_pix_top();
  *out << cgicc::div().set("style", position.c_str()) << std::endl;

  *out << cgicc::a().set("href", link_url.c_str()) << text.c_str() << cgicc::a() << std::endl;

  *out << cgicc::div() << std::endl;
}
