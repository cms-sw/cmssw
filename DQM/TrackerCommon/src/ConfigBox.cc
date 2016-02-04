#include "DQM/TrackerCommon/interface/ConfigBox.h"

void ConfigBox::printHTML(xgi::Output * out)
{
  std::string position = "position:absolute; left:" + get_pix_left() + "; top:" + get_pix_top();
  *out << cgicc::div().set("style", position.c_str()) << std::endl;
  *out << cgicc::form().set("name", "ConfigurationForm") << std::endl;
  *out << cgicc::table().set("border", "0") << std::endl;
  *out << cgicc::tr() 
       << cgicc::td() << "Hostname:" << cgicc::td() 
       << cgicc::td() << cgicc::input().set("type", "text").set("name", "Hostname") << cgicc::td()
       << cgicc::tr() << std::endl;
  *out << cgicc::tr() 
       << cgicc::td() << "Port:" << cgicc::td()
       << cgicc::td() << cgicc::input().set("type", "text").set("name", "Port") << cgicc::td()
       << cgicc::tr() << std::endl;
  *out << cgicc::tr()
       << cgicc::td() << "Client Name:" << cgicc::td()
       << cgicc::td() << cgicc::input().set("type", "text").set("name", "Name") << cgicc::td()
       << cgicc::tr() << std::endl;

  std::string js_command = "submitConfigure('" + get_url() + "', form)";
  *out << cgicc::tr() 
       << cgicc::td() << cgicc::input().set("type", "button").set("value", "(Re)configure!").set("onClick", js_command) << cgicc::td()
       << cgicc::tr() << std::endl;
  *out << cgicc::table() << std::endl;
  *out << cgicc::form()  << std::endl;
  *out << cgicc::div()   << std::endl;
}

