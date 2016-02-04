#ifndef _CgiInterface_h
#define _CgiReader_h

#include <string>

#include "xgi/Utils.h"
#include "xgi/Method.h"

#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTTPHTMLHeader.h"
#include "cgicc/HTTPRedirectHeader.h"
#include "cgicc/HTMLClasses.h"


class CgiReader
{
 protected:

  std::string url;

  xgi::Input *in;
  xgi::Output *out;

 public:

  CgiReader(xgi::Input *the_in)
    {
      in  = the_in;
    }

  ~CgiReader(){}

  void read_form(std::multimap<std::string, std::string> &form_info);
  std::string read_cookie(std::string name);

};


#endif
