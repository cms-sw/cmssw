#ifndef DISPLAY_SYSTEM_SUMMARY
#define DISPLAY_SYSTEM_SUMMARY


#include "xgi/Method.h"
#include "cgicc/HTMLClasses.h"
#include "DQMServices/WebComponents/interface/WebElement.h"

class DisplaySystemSummary : public WebElement
{


 public:

  DisplaySystemSummary(std::string the_url,         /// url of the application
		       std::string _top, std::string _left, 
		       std::string _name,
		       std::string _command = "makeSummary()") 
    :  WebElement(the_url, _top, _left) {
      name = _name;
      top = _top;
      left = _left;
      command = _command;
    }

    ~DisplaySystemSummary() {
    }

    void printHTML(xgi::Output * out);
 private:

    std::string name;
    std::string top;
    std::string left;  
    std::string command;  

};


#endif
