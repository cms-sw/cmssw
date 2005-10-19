#ifndef EVENTFILTER_CSS_H 
#define EVENTFILTER_CSS_H 

#include "xgi/include/xgi/Input.h"
#include "xgi/include/xgi/Output.h"
#include "xgi/include/xgi/exception/Exception.h"

using namespace std;

namespace evf{

  class Css
    {
    public:
      void css(xgi::Input  *in,
	       xgi::Output *out) throw (xgi::exception::Exception)
	{
	      out->getHTTPResponseHeader().addHeader("Content-Type", "text/css");

	      *out << "body"                                   << endl;
	      *out << "{"                                      << endl;
	      *out << "background-color: white;"               << endl;
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

	      *out                                             << endl;



	}

    private: 
    };
}
#endif
