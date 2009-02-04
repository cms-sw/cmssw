#include "DQM/TrackerCommon/interface/GifDisplay.h"

void GifDisplay::printHTML(xgi::Output *out)
{
  std::string position = "position:absolute; left:" + get_pix_left() + "; top:" + get_pix_top();

  *out << cgicc::div().set("style", position.c_str()) << std::endl;

  *out << cgicc::iframe()
    .set("name", name)
    .set("id", name)
    .set("src", "")
    .set("height", height).set("width", width);
  *out << cgicc::iframe() << std::endl;
  *out << cgicc::br() << std::endl;
  *out << cgicc::input().set("type", "button").set("value", "start viewer").set("onclick", "startViewing('" + name + "')");
  *out << std::endl;
  *out << cgicc::input().set("type", "button").set("value", "stop viewer").set("onclick", "stopViewing('" + name + "')");
  *out << std::endl;
  *out << cgicc::input().set("type", "button").set("value", "make current").set("onclick", "makeCurrent('" + name + "')");
  *out << std::endl;

  *out << cgicc::div() << std::endl;
}
