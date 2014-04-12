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
    virtual ~Exception() throw(){}
    virtual char const* what() const throw(){
      return m_message.c_str();
    }
  private:
    std::string m_message;
  };

  class nonCollisionException : public lumi::Exception{
  public:
    nonCollisionException(const std::string& methodname,
			 const std::string& moduleName);
    virtual ~nonCollisionException() throw(){}
  };

  class invalidDataException : public lumi::Exception{
  public:
    invalidDataException(const std::string& message,
			 const std::string& methodname,
			 const std::string& moduleName);
    virtual ~invalidDataException() throw(){}
  };

  class noStableBeamException : public lumi::Exception{
  public:
    noStableBeamException(const std::string& message,
			 const std::string& methodname,
			 const std::string& moduleName);
    virtual ~noStableBeamException() throw(){}
  };
  
  class duplicateRunInDataTagException : public lumi::Exception{
  public:
    duplicateRunInDataTagException(const std::string& message,
				       const std::string& methodname,
				       const std::string& moduleName);
    virtual ~duplicateRunInDataTagException() throw(){}
  };
}//ns lumi
#endif
