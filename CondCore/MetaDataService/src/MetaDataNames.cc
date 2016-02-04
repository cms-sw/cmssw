#include "CondCore/MetaDataService/interface/MetaDataNames.h"
const std::string& cond::MetaDataNames::metadataTable(){
  static const std::string s_metadataTable("METADATA");
  return s_metadataTable;
}
const std::string& cond::MetaDataNames::tagColumn(){
  static const std::string s_tagColumn("NAME");
  return s_tagColumn;
}
const std::string& cond::MetaDataNames::tokenColumn(){
  static const std::string s_tokenColumn("TOKEN");
  return s_tokenColumn;
}
const std::string& cond::MetaDataNames::timetypeColumn(){
  static const std::string s_timetypeColumn("TIMETYPE");
  return s_timetypeColumn;
}
