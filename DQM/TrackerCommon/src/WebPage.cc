#include "DQM/TrackerCommon/interface/WebPage.h"

WebPage::WebPage(std::string the_url)
{
  url = the_url;
}

void WebPage::add(std::string name, WebElement * element_p)
{
  element_map[name] = element_p;
}

void WebPage::remove(const std::string name)
{
  element_map.erase(name);
}

void WebPage::clear()
{
  element_map.clear();
}


/************************************************/
// Print out the web page elements

void WebPage::printHTML(xgi::Output * out)
{
  std::map<std::string, WebElement *>::iterator it;

   *out << cgicc::body().set("onload", "fillDisplayList()") << std::endl;

  for (it = element_map.begin(); it != element_map.end(); it++)
    {
      it->second->printHTML(out);
    }

  *out << cgicc::body() << std::endl;
}
