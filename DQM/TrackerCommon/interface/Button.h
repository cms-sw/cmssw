#ifndef _DQM_TrackerCommon_Button_h_
#define _DQM_TrackerCommon_Button_h_

/** 
   This is the class that should be instantiated in case the 
   user wants to have a button on the web page, which is 
   connected to a function defined in your web client. 
   Pressing this button will result in the javascript function 
   makeRequest being called and the request 
   "/Request?RequestID=[Your request name]" being submitted to the server.
*/

#include "xgi/Method.h"
#include "cgicc/HTMLClasses.h"
#include "DQM/TrackerCommon/interface/WebElement.h"


class Button : public WebElement
{
 private:
  /// the name that will appear on the button  
   std::string name;     
   /// the string connected to the callback, eg "Default"
     std::string requestID; 

 public:
  
  Button(std::string the_url, std::string top, std::string left, std::string the_requestID, std::string the_name) 
    : WebElement(the_url, top, left)
    {
      name = the_name;
      requestID = the_requestID;
    }
  
  ~Button()
    {
    }

  void printHTML(xgi::Output *out);
};


#endif
