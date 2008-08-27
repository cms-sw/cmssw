#ifndef PARAMETERSETRETRIEVER_H
#define PARAMETERSETRETRIEVER_H

#include <string>

namespace evf {
  
  class ParameterSetRetriever
  {
  public:
    ParameterSetRetriever(const std::string& in);
    std::string getAsString() const; 
    std::string getHostString(const std::string &in) const; 
  private:
    std::string pset;
    static const std::string fileheading;
    static const std::string dbheading;  
    static const std::string webheading;

  };

} // evf

#endif
