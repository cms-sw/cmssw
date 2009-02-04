#ifndef _DQM_TrackerCommon_HTMLLink_h_
#define _DQM_TrackerCommon_HTMLLink_h_

#include "xgi/Method.h"
#include "cgicc/HTMLClasses.h"
#include "DQM/TrackerCommon/interface/WebElement.h"


class HTMLLink : public WebElement
{
 private:
  /// the text of the link
  std::string text; 
  std::string link_url;

 public:

  HTMLLink(std::string the_url, std::string top, std::string left, std::string the_text, std::string the_link_url)
    : WebElement(the_url, top, left)
    {
      text = the_text;
      link_url = the_link_url;
    }

  ~HTMLLink()
    {
    }

  void printHTML(xgi::Output *out);
};

#endif
