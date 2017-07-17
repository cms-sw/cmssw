#ifndef CommonToolsUtils_popenCPP_H
#define CommonToolsUtils_popenCPP_H
#include<memory>
#include<string>
#include <sstream>


namespace reco {
  namespace exprEvalDetails {

    std::unique_ptr<std::istream> popenCPP(const std::string &cmdline);

    inline
    std::string execSysCommand(const std::string &cmdline){
     std::ostringstream n1;
     {
       auto s1 = popenCPP(cmdline+" 2>&1");
       n1 << s1->rdbuf();
     }
     return n1.str();
    }

  }
}


#endif // CommonToolsUtils__popenCPP_H

