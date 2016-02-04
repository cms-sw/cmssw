#ifndef _ContentViewer_h_
#define _ContentViewer_h_

/** 
   This is the class that should be instantiated in case the
   user wants to have a select menu, the elements of which, 
   submit a request when clicked on. 
*/

#include "xgi/Method.h"
#include "cgicc/HTMLClasses.h"
#include "DQM/TrackerCommon/interface/WebElement.h"

class ContentViewer : public WebElement
{
 private:

 public:

  ContentViewer(std::string the_url ,         /// url of the application
		std::string top, std::string left) /// position of the widget
    : WebElement(the_url, top, left)
    {
    }
  ~ContentViewer()
    {
    }

  void printHTML(xgi::Output * out);
  void printSelectHTML(xgi::Output * out, std::string name, std::string onchange);
};


#endif
