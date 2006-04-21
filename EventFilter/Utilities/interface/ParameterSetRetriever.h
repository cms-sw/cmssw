#ifndef PARAMETERSETRETRIEVER_H
#define PARAMETERSETRETRIEVER_H

#include <string>

namespace evf{
  
  class ParameterSetRetriever{
    
  public:
    ParameterSetRetriever(std::string &);
    std::string getAsString() const; 
  private:
    std::string pset;

  };
}
#endif
