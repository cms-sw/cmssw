#include "xgi/Utils.h"
#include "xgi/Method.h"
#include "cgicc/HTMLClasses.h"

class CgiWriter
{
 private:

  xgi::Output * out;
  std::string contextURL;

 public:

  CgiWriter(xgi::Output * the_out, std::string the_contextURL)
    {
      out = the_out;
      contextURL = the_contextURL;
      std::cout << "Created a CgiWriter! ContextURL=" << contextURL << std::endl;
    }

  ~CgiWriter(){}

  void output_preamble();
  void output_head();
  void output_finish();
};
