#ifndef EVENTFILTER_CSS_H 
#define EVENTFILTER_CSS_H 

#include "xgi/Input.h"
#include "xgi/Output.h"
#include "xgi/exception/Exception.h"



namespace evf{

  class Css
    {
    public:
      void css(xgi::Input  *in,
	       xgi::Output *out) throw (xgi::exception::Exception)
	{
	  using std::endl;
	  out->getHTTPResponseHeader().addHeader("Content-Type", "text/css");
	  
	  *out << "body"                                   << endl;
	  *out << "{"                                      << endl;
	  *out << "background-color: white;"               << endl;
	  *out << "font-family: Arial;"                    << endl;
	  *out << "}"                                      << endl;
	  *out                                             << endl;
	  *out << "table.modules"                          << endl;
	  *out << "{"                                      << endl;
	  *out << "font-family: Arial;"                    << endl;
	  *out << "border: thin dotted;"                   << endl;
	  *out << "}"                                      << endl;
	  *out                                             << endl;
	  *out << "table.modules colgroup"                 << endl;
	  *out << "{"                                      << endl;
	  *out << "width: 30%;"                            << endl;
	  *out << "border: solid;"                         << endl;
	  *out << "}"                                      << endl;
	  *out                                             << endl;
	  *out << "table.modules th"                       << endl;
	  *out << "{"                                      << endl;
	  *out << "color: white;"                          << endl;
	  *out << "background-color: #63F;"                << endl;
	  *out << "}"                                      << endl;
	  *out << "table.modules tr.special"               << endl;
	  *out << "{"                                      << endl;
	  *out << "color: white;"                          << endl;
	  *out << "background-color: #000;"                << endl;
	  *out << "}"                                      << endl;
	  
	  *out << "table.states"                           << endl;
	  *out << "{"                                      << endl;
	  *out << "font-family: Arial;"                    << endl;
	  *out << "border: thin dotted;"                   << endl;
	  *out << "}"                                      << endl;
	  *out                                             << endl;
	  *out << "table.states th"                        << endl;
	  *out << "{"                                      << endl;
	  *out << "color: white;"                          << endl;
	  *out << "background-color: #63F;"                << endl;
	  *out << "}"                                      << endl;
	  *out << "table.states tr.special"                << endl;
	  *out << "{"                                      << endl;
	  *out << "color: white;"                          << endl;
	  *out << "background-color: #000;"                << endl;
	  *out << "}"                                      << endl;
	  *out << "table.states .special a"                << endl;
	  *out << "{"                                      << endl;
	  *out << "color: white;"                          << endl;
	  *out << "}"                                      << endl;
	  
	  *out                                             << endl;
	  
	  
	  
	}

    private: 
    };
}
#endif
