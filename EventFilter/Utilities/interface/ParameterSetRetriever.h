#ifndef PARAMETERSETRETRIEVER_H
#define PARAMETERSETRETRIEVER_H

#include <string>

namespace evf {
  
  class ParameterSetRetriever
  {
  public:
    ParameterSetRetriever(const std::string& in);
    std::string getAsString() const; 

  private:
    std::string pset;
  };

} // evf

#endif
