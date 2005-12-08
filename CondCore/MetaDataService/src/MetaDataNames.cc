#include "CondCore/MetaDataService/interface/MetaDataNames.h"
const std::string& cond::MetaDataNames::metadataTable(){
  static const std::string s_metadataTable("METADATA");
  return s_metadataTable;
}
const std::string& cond::MetaDataNames::tagColumn(){
  static const std::string s_tagColumn("name");
  return s_tagColumn;
}
const std::string& cond::MetaDataNames::tokenColumn(){
  static const std::string s_tokenColumn("token");
  return s_tokenColumn;
}
