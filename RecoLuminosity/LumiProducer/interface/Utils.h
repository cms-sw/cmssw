#ifndef RecoLuminosity_LumiProducer_Utils_H
#define RecoLuminosity_LumiProducer_Utils_H
#include <string>
#include <sstream>
#include <iostream>
namespace lumi{
  /**convert string to numeric type
   **/
  template <class T> bool from_string(T& t, 
				      const std::string& s, 
				      std::ios_base& (*f)(std::ios_base&)){
    std::istringstream iss(s);
    return !(iss >> f >> t).fail();
  }
}
#endif
