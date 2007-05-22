#ifndef DISPLAY_SYSTEM_ME
#define DISPLAY_SYSTEM_ME 


#include "xgi/Method.h"
#include "cgicc/HTMLClasses.h"
#include "DQMServices/WebComponents/interface/WebElement.h"

class DisplaySystemME : public WebElement
{


public:

DisplaySystemME(std::string the_url,         /// url of the application
	    std::string _top, std::string _left, std::string _name) :  WebElement(the_url, _top, _left)
{
name = _name;
top = _top;
left = _left;
}

~DisplaySystemME()
{
}

void printHTML(xgi::Output * out);
//void printSelectHTML(xgi::Output * out, std::string name, std::string onchange);


private:

std::string name;
std::string top;
std::string left;  

};


#endif
