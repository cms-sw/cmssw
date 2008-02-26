#ifndef CondCore_IOVService_IOVNames_h
#define CondCore_IOVService_IOVNames_h
#include <string>
namespace cond{
  class IOVNames{
  public:
    static std::string container() {
      return std::string("cond::IOV"); 
    }
    static std::string iovTableName() {
      return std::string("IOV"); 
    }
    static std::string iovDataTableName() {
      return std::string("IOV_DATA"); 
    }
    static std::string iovMappingVersion() {
      return std::string("CONDIOV_2.0");
    }
    static std::string const & iovMappingXML(){
      static const std::string buffer = 
	std::string("<?xml version='1.0' encoding=\"UTF-8\"?>\n")+
	std::string("<!DOCTYPE Mapping SYSTEM \"InMemory\">\n")+
	std::string("<Mapping version=\"CONDIOV_3.0\" >\n")+
	std::string("<Class table=\"IOV\" id_columns=\"ID\" name=\"cond::IOV\" >\n")+
    	std::string("<Primitive column=\"FIRSTSINCE\" name=\"firstsince\" />\n")+
	std::string("<Primitive column=\"TIMETYPE\" name=\"timetype\" />\n")+
        std::string("<Container table=\"IOV_DATA\" id_columns=\"ID\" name=\"iov\" position_column=\"POS\" >\n")+
	std::string("<Object table=\"IOV_DATA\" id_columns=\"POS ID\" name=\"value_type\" >\n")+
        std::string("<Primitive column=\"IOV_TOKEN\" name=\"first\" />\n")+
        std::string("<Primitive column=\"IOV_TIME\" name=\"second\" />\n")+
	std::string("</Object >\n")+
	std::string("</Container >\n")+
	std::string("</Class >\n")+
	std::string("</Mapping >\n");
      return buffer;
    }
  };
}//ns pool
#endif
