#ifndef RecoLuminosity_LumiProducer_idDealer_H
#define RecoLuminosity_LumiProducer_idDealer_H
#include <string>
namespace coral{
  class ISchema;
}
namespace lumi{
  class idDealer{
  public:
    explicit idDealer( coral::ISchema& schema);
    unsigned long long getIDforTable( const std::string& tableName );
    unsigned long long generateNextIDForTable( const std::string& tableName, unsigned int interval=1);
  private:
    coral::ISchema& m_schema;
    std::string m_idtablecolumnName;
    std::string m_idtablecolumnType;
  };//cs IdDealer
}//ns lumi
#endif
