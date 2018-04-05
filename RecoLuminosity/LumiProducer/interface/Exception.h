#ifndef RecoLuminosity_LumiProducer_Exception_H
#define RecoLuminosity_LumiProducer_Exception_H
#include <string>
#include <exception>
namespace lumi{
  class Exception : public std::exception{
  public:
    Exception( const std::string& message,
	       const std::string& methodname,
	       const std::string& moduleName);
    ~Exception() throw() override{}
    char const* what() const throw() override{
      return m_message.c_str();
    }
  private:
    std::string m_message;
  };

  class nonCollisionException : public lumi::Exception{
  public:
    nonCollisionException(const std::string& methodname,
			 const std::string& moduleName);
    ~nonCollisionException() throw() override{}
  };

  class invalidDataException : public lumi::Exception{
  public:
    invalidDataException(const std::string& message,
			 const std::string& methodname,
			 const std::string& moduleName);
    ~invalidDataException() throw() override{}
  };

  class noStableBeamException : public lumi::Exception{
  public:
    noStableBeamException(const std::string& message,
			 const std::string& methodname,
			 const std::string& moduleName);
    ~noStableBeamException() throw() override{}
  };
  
  class duplicateRunInDataTagException : public lumi::Exception{
  public:
    duplicateRunInDataTagException(const std::string& message,
				       const std::string& methodname,
				       const std::string& moduleName);
    ~duplicateRunInDataTagException() throw() override{}
  };
}//ns lumi
#endif
