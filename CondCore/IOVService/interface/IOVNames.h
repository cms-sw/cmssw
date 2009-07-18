#ifndef CondCore_IOVService_IOVNames_h
#define CondCore_IOVService_IOVNames_h
#include <string>
namespace cond{
  class IOVNames{
  public:
    static std::string container() {
      return std::string("cond::IOVSequence"); 
    }
    static std::string iovTableName() {
      return std::string("IOVSequence"); 
    }
    static std::string iovDataTableName() {
      return std::string("IOV_DATA"); 
    }
    static std::string iovMappingVersion() {
      return std::string("CONDIOV_4.0");
    }
    static std::string const & iovMappingXML(){
      static const std::string buffer = 
	std::string("<?xml version='1.0' encoding=\"UTF-8\"?>\n")+
	std::string("<!DOCTYPE PoolDatabase SYSTEM \"InMemory\">\n")+
        std::string("<PoolDatabase >\n")+
        std::string("<PoolContainer name=\"cond::IOVSequence\" >\n")+
	std::string("<Class table=\"IOV\" id_columns=\"ID\" name=\"cond::IOVSequence\" mapping_version=\"CONDIOV_4.0\" >\n")+
    	std::string("<Primitive column=\"LASTTILL\" name=\"m_lastTill\" />\n")+
	std::string("<Primitive column=\"TIMETYPE\" name=\"m_timetype\" />\n")+
	std::string("<Primitive column=\"NOTORDERED\" name=\"m_notOrdered\" />\n")+
	std::string("<Primitive column=\"METADATA\" name=\"m_metadata\" />\n")+
        std::string("<PoolArray table=\"IOV_DATA\" id_columns=\"ID\" name=\"m_iovs\" position_column=\"POS\" >\n")+
	std::string("<Object name=\"value_type\" >\n")+
        std::string("<Primitive column=\"IOV_TIME\" name=\"m_sinceTime\" />\n")+
        std::string("<Primitive column=\"IOV_TOKEN\" name=\"m_wrapper\" />\n")+
	std::string("</Object >\n")+
	std::string("</PoolArray >\n")+
	std::string("</Class >\n")+
	std::string("</PoolContainer >\n")+
	std::string("</PoolDatabase >\n");
      return buffer;
    }
  };
}//ns pool
#endif
