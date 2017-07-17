#include "CommonTools/Utils/src/popenCPP.h"
#include "FWCore/Utilities/interface/Exception.h"
#include<iostream>
#include<string>


using namespace reco::exprEvalDetails;

int main() {

  try { 
    char c;
    {
    auto ss = popenCPP("c++ -v 2>&1");
    while (ss->get(c)) std::cout << c;
    std::cout << std::endl;
    }
    

    {
    auto n1 = execSysCommand( "c++ -v");
    std::cout << "\n|" << n1 << '|' << std::endl;
    }

    std::cout << "\n|" <<  execSysCommand("uuidgen | sed 's/-//g'") << '|' << std::endl;

    std::cout << "\n|" <<  execSysCommand("notexisting 2>&1") << '|' << std::endl;

  }catch( cms::Exception const & e) {
    std::cout << e.what()  << std::endl;
  }catch(...) {
    std::cout << "unknown error...." << std::endl;
  }

  return 0;

}
