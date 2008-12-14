#include "DQM/TrackerCommon/interface/Select.h"


void Select::printHTML(xgi::Output * out)
{
  std::string position = "position:absolute; left:" + get_pix_left() + "; top:" + get_pix_top();

  // the javascript function makeSelectRequest expects the url of the request and the id of the widget
  // Example: makeSelectRequest('http://[ApplicationURL]/Request?RequestID=MyCustomRequest', 'select menu 1');
  std::string applicationURL = get_url();
  std::string requestName    = "RequestID=" + requestID;
  std::string js_command = "makeSelectRequest('" + applicationURL + "/Request?" + requestName + "'" + 
    ", '" + name + "')";

  *out << cgicc::div().set("style", position.c_str()) << std::endl;

  // The id of the form is the name of the select menu + " Form".
  *out << cgicc::form().set("name", name + " Form").set("id", name + " Form") << std::endl;

  *out << cgicc::table() << std::endl;
  *out << cgicc::tr() << std::endl
       << cgicc::td() << std::endl
       << name << ":" << std::endl
       << cgicc::td() << std::endl;


  *out << cgicc::td() << std::endl;
  *out << cgicc::select().set("name", name).set("id", name).set("onchange", js_command) << std::endl; 
  *out << cgicc::option().set("value", "").set("selected") << cgicc::option() << std::endl;
  for (std::vector<std::string>::iterator it = options_v.begin(); it != options_v.end(); it++)
    {
      *out <<  cgicc::option().set("value", *it) << *it << cgicc::option() << std::endl;
    }
  *out << cgicc::select() << std::endl;
  *out << cgicc::td() << std::endl;

  *out << cgicc::tr() << std::endl;

  *out << cgicc::table() << std::endl;

  *out << cgicc::form() << std::endl;
  *out << cgicc::div() << std::endl;
}


