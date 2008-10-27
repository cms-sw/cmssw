#ifndef DBCommon_TokenInterpreter_h
#define DBCommon_TokenInterpreter_h
#include <string>
//
// Package:    CondCore/DBCommon
// Class:      TokenInterpreter
//
/**\class TokenInterpreter TokenInterpreter.h CondCore/DBCommon/interface/TokenInterpreter.h
   Description: Utility class desect token and return original information. It is implicitly assumed that the dictionary containing the class is loaded.
*/
namespace cond{
   /// check if the token is valid
  bool validToken(const std::string& tokenString);
 
  class TokenInterpreter{
  public:
  public:
    explicit TokenInterpreter(const std::string& tokenString);
    ~TokenInterpreter();
    /// check if the token is valid
    bool isValid() const;
    /// return the container name
    std::string containerName() const;
    /// return the true class name
    std::string className() const;
  private:
    std::string m_tokenstr;
    std::string m_containerName;
    std::string m_className;
  };
}//ns cond
#endif
