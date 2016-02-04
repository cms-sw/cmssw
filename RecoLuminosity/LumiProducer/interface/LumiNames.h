#ifndef RECOLUMINOSITY_LUMIPRODUCER_H
#define RECOLUMINOSITY_LUMIPRODUCER_H
#include <string>
namespace lumi{
  class LumiNames{
  public:
    static const std::string cmsrunsummaryTableName();
    static const std::string lumirunsummaryTableName();
    static const std::string lumisummaryTableName();
    static const std::string lumidetailTableName();
    static const std::string trgTableName();
    static const std::string hltTableName();
    static const std::string trghltMapTableName();
    static const std::string lumiresultTableName();
    static const std::string lumihltresultTableName();
    static const std::string idTableName( const std::string& dataTableName);
    static const std::string idTableColumnName();
    static const std::string idTableColumnType();
  };
}
#endif
