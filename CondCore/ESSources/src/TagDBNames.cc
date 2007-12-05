#include "TagDBNames.h"
const std::string& 
cond::TagDBNames::tagTreeTablePrefix(){
  static const std::string s_tagTreeTablePrefix("TAGTREE_TABLE");
  return s_tagTreeTablePrefix;
}
const std::string& 
cond::TagDBNames::tagInventoryTable(){
  static const std::string s_tagInventoryTable("TAGINVENTORY_TABLE");
  return s_tagInventoryTable;
}

