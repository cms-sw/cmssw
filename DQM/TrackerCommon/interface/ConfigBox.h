#ifndef _ConfigBox_h_
#define _ConfigBox_h_

/** 
   This is the class that should be instantiated in case the
   user wants to have a box that resets the configuration of 
   the DQM client. Submitting the information of this box
   should result in an attempt to connect to a new collector
   according to the information submitted. 
*/

#include "xgi/Method.h"
#include "cgicc/HTMLClasses.h"
#include "DQM/TrackerCommon/interface/WebElement.h"

class ConfigBox : public WebElement
{
 private:

  std::string callback;

 public:

  ConfigBox(std::string the_url, std::string top, std::string left) : WebElement(the_url, top, left)
    {
      callback = "Configure";
    }
  ~ConfigBox()
    {
    }

  void printHTML(xgi::Output * out);

};


#endif
