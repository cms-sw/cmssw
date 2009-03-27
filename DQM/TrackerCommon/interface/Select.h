#ifndef _DQM_TrackerCommon_Select_h_
#define _DQM_TrackerCommon_Select_h_

/** 
   This is the class that should be instantiated in case the
   user wants to have a select menu, the elements of which, 
   submit a request when clicked on. 
*/

#include "xgi/Method.h"
#include "cgicc/HTMLClasses.h"
#include "DQM/TrackerCommon/interface/WebElement.h"

class Select : public WebElement
{
 private:

  std::string name;      // the title over the menu
  std::string requestID; // the string connected to the callback function

  std::vector<std::string> options_v;

 public:

  Select(std::string the_url, std::string top, std::string left,
	 std::string the_requestID, std::string the_name)
    : WebElement(the_url, top, left)
    {
      name = the_name;
      requestID = the_requestID;
    }

  ~Select()
    {
    }

  void setOptionsVector(std::vector<std::string> the_options_v)
    {
      options_v = the_options_v;
    }

  void printHTML(xgi::Output * out);
};


#endif
