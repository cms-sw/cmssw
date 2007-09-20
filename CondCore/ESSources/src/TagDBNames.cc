#include "TagDBNames.h"
const std::string& 
cond::TagDBNames::tagTreeTable(){
  static const std::string s_tagTreeTable("TAGTREE_TABLE");
  return s_tagTreeTable;
}
const std::string& 
cond::TagDBNames::tagInventoryTable(){
  static const std::string s_tagInventoryTable("TAGINVENTORY_TABLE");
  return s_tagInventoryTable;
}

