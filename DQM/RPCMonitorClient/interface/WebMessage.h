#ifndef WebMessage_h
#define WebMessage_h

#include "xgi/Method.h"
#include "cgicc/HTMLClasses.h"
#include "DQMServices/WebComponents/interface/WebElement.h"


class WebMessage : public WebElement
{
 private:
  
  std::string name;     // the message that will appear on the web page
  std::string colour;    // message colour

 public:
  
 WebMessage(std::string the_url, std::string top, std::string left,  std::string the_name,  std::string the_colour) 
    : WebElement(the_url, top, left)
    {
      name = the_name;
      colour= the_colour;
    }
  
  ~WebMessage()
    {
    }

  void printHTML(xgi::Output *out);
};

void  WebMessage::printHTML(xgi::Output *out)
{
  std::string position = "position:absolute; left:" + get_pix_left() + "; top:" + get_pix_top();
  *out << cgicc::div().set("style", position.c_str()) << std::endl;

  *out << "<H3><font color=\""<<colour<<"\">"<< name << "</H3></font>"<<std::endl;

  *out << cgicc::div()   << std::endl;
}

#endif
