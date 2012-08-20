#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
const std::string lumi::LumiNames::cmsrunsummaryTableName(){
  return "CMSRUNSUMMARY";
}
const std::string lumi::LumiNames::lumidataTableName(){
  return "LUMIDATA";
}
const std::string lumi::LumiNames::lumirunsummaryTableName(){
  return "CMSRUNSUMMARY";
}
const std::string lumi::LumiNames::lumisummaryTableName(){
  return "LUMISUMMARY";
}
const std::string lumi::LumiNames::lumisummaryv2TableName(){
  return "LUMISUMMARYV2";
}
const std::string lumi::LumiNames::lumidetailTableName(){
  return "LUMIDETAIL";
}
const std::string lumi::LumiNames::luminormTableName(){
  return "LUMINORMS";
}
const std::string lumi::LumiNames::luminormv2TableName(){
  return "LUMINORMSV2";
}
const std::string lumi::LumiNames::luminormv2dataTableName(){
  return "LUMINORMSV2DATA";
}
const std::string lumi::LumiNames::trgdataTableName(){
  return "TRGDATA";
}
const std::string lumi::LumiNames::lstrgTableName(){
  return "LSTRG";
}
const std::string lumi::LumiNames::trgTableName(){
  return "TRG";
}
const std::string lumi::LumiNames::hltTableName(){
  return "HLT";
}
const std::string lumi::LumiNames::hltdataTableName(){
  return "HLTDATA";
}
const std::string lumi::LumiNames::lshltTableName(){
  return "LSHLT";
}
const std::string lumi::LumiNames::tagRunsTableName(){
  return "TAGRUNS";
}
const std::string lumi::LumiNames::tagsTableName(){
  return "TAGS";
}
const std::string lumi::LumiNames::trghltMapTableName(){
  return "TRGHLTMAP";
}
const std::string lumi::LumiNames::intglumiTableName(){
  return "INTGLUMI";
}
const std::string lumi::LumiNames::intglumiv2TableName(){
  return "INTGLUMIV2";
}
const std::string lumi::LumiNames::lumiresultTableName(){
  return "INTLUMI";
}
const std::string lumi::LumiNames::lumihltresultTableName(){
  return "INTLUMIHLT";
}
const std::string lumi::LumiNames::idTableName(const std::string& dataTableName){
  return dataTableName+"_ID";
}
const std::string lumi::LumiNames::idTableColumnName(){
  return "NEXTID";
}
const std::string lumi::LumiNames::idTableColumnType(){
  return "unsigned long long";
}
const std::string lumi::LumiNames::revisionTableName(){
  return "REVISIONS";
}
const std::string lumi::LumiNames::revmapTableName(const std::string& datatablename ){
  return datatablename+"_REV";
}
const std::string lumi::LumiNames::entryTableName(const std::string& datatablename ){
  return datatablename+"_ENTRIES";
}
