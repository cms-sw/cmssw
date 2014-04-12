#ifndef INCLUDE_ORA_RECORDMANIP_H
#define INCLUDE_ORA_RECORDMANIP_H

#include "CondCore/ORA/interface/Record.h"
#include "CoralBase/Attribute.h"

namespace {
  void newRecordFromAttributeList( ora::Record & rec, const coral::AttributeList& data ){
    for( size_t i=0;i<data.size();i++ ){
      rec.set( i, const_cast<void*>(data[i].addressOfData()) );
    }
  }
  
  void newAttributeListFromRecord( coral::AttributeList& alist, const ora::Record& data ){
    for( size_t i=0;i<data.size();i++ ){
      alist[i].setValueFromAddress( data.get(i) );
    }
  }
  
}

#endif  


