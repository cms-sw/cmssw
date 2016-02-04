#ifndef _GifDisplay_h_
#define _GifDisplay_h_

/** 
   This is the class that should be instantiated and
   added to the web page in order to present a display
   for the histograms on the browser screen
*/

#include "xgi/Method.h"
#include "cgicc/HTMLClasses.h"
#include "DQM/TrackerCommon/interface/WebElement.h"


class GifDisplay : public WebElement
{
 private:
  
  std::string height;
  std::string width;
  std::string name;

 public:
  
  GifDisplay(std::string the_url, std::string top, std::string left, 
	     std::string the_height, std::string the_width, std::string the_name) 
    : WebElement(the_url, top, left)
    {
      height = the_height;
      width = the_width;
      name = the_name;
    }
  
  ~GifDisplay()
    {
    }

  void printHTML(xgi::Output *out);
};


#endif
