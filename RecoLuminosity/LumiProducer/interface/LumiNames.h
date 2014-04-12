#ifndef RECOLUMINOSITY_LUMIPRODUCER_H
#define RECOLUMINOSITY_LUMIPRODUCER_H
#include <string>
namespace lumi{
  class LumiNames{
  public:
    static const std::string cmsrunsummaryTableName();
    static const std::string lumidataTableName();
    static const std::string lumirunsummaryTableName();
    static const std::string lumisummaryTableName();
    static const std::string lumisummaryv2TableName();
    static const std::string lumidetailTableName();
    static const std::string luminormTableName();
    static const std::string luminormv2TableName();
    static const std::string luminormv2dataTableName();
    static const std::string trgdataTableName();
    static const std::string lstrgTableName();
    static const std::string trgTableName();
    static const std::string hltTableName();
    static const std::string hltdataTableName();
    static const std::string lshltTableName();
    static const std::string tagRunsTableName();
    static const std::string tagsTableName();
    static const std::string trghltMapTableName();
    static const std::string intglumiTableName();
    static const std::string intglumiv2TableName();
    static const std::string lumiresultTableName();
    static const std::string lumihltresultTableName();
    static const std::string idTableName( const std::string& dataTableName);
    static const std::string idTableColumnName();
    static const std::string idTableColumnType();
    static const std::string revisionTableName();    
    static const std::string revmapTableName(const std::string& datatablename );
    static const std::string entryTableName(const std::string& datatablename );
  };
}
#endif
