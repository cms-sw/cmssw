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
    unsigned int getIDforTable( const std::string& tableName );
    unsigned int generateNextIDForTable( const std::string& tableName );
  private:
    coral::ISchema& m_schema;
    std::string m_idtablecolumnName;
    std::string m_idtablecolumnType;
  };//cs IdDealer
}//ns lumi
#endif
