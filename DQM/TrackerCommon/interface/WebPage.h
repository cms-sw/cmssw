#ifndef _WebPage_h_
#define _WebPage_h_

#include "xgi/Method.h"
#include "cgicc/HTMLClasses.h"

#include "DQM/TrackerCommon/interface/WebElement.h"


class WebPage
{
 private:

  std::string url;
  std::map<std::string, WebElement *> element_map;

 protected:

 public:

  WebPage(std::string the_url);

  void add(std::string, WebElement *);
  void remove(std::string);
  void clear();
  
  void printHTML(xgi::Output * out);
};

#endif
