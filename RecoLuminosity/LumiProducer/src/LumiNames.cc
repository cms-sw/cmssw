#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
const std::string lumi::LumiNames::cmsrunsummaryTableName(){
  return "CMSRUNSUMMARY";
}
const std::string lumi::LumiNames::lumirunsummaryTableName(){
  return "LUMIRUNSUMMARY";
}
const std::string lumi::LumiNames::lumisummaryTableName(){
  return "LUMISUMMARY";
}
const std::string lumi::LumiNames::lumidetailTableName(){
  return "LUMIDETAIL";
}
const std::string lumi::LumiNames::trgTableName(){
  return "TRG";
}
const std::string lumi::LumiNames::hltTableName(){
  return "HLT";
}
const std::string lumi::LumiNames::trghltMapTableName(){
  return "TRGHLTMAP";
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
