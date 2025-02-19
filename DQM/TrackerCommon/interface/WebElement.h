#ifndef _WebElement_
#define _WebElement_

#include "xgi/Utils.h"
#include "xgi/Method.h"


class WebElement
{
 private:

  /// The url of the application
   std::string url; 
   /// position in pixels (from top)
   std::string pix_top; 
   /// position in pixels (from left)
   std::string pix_left; 

 public:

  WebElement(std::string the_url, std::string top, std::string left)
    {
      url = the_url;
      pix_top = top;
      pix_left = left;
    }

  virtual ~WebElement()
    {
    }

  std::string get_url()
    {
      return url;
    }

  std::string get_pix_top() { return pix_top; }
  std::string get_pix_left() { return pix_left; }
  
  virtual void printHTML(xgi::Output *out) = 0;

};

#endif
