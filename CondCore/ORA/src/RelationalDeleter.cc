#include "RelationalDeleter.h"
#include "RelationalOperation.h"
#include "RelationalBuffer.h"
#include "MappingElement.h"
#include "ContainerSchema.h"
// externals
#include "CoralBase/Attribute.h"

ora::RelationalDeleter::RelationalDeleter( MappingElement& dataMapping ):
  m_mappings(),
  m_operations(){
  m_mappings.push_back( &dataMapping );
}

ora::RelationalDeleter::RelationalDeleter( const std::vector<MappingElement>& mappingList ):
  m_mappings(),
  m_operations(){
  for(  std::vector<MappingElement>::const_iterator iMe = mappingList.begin();
        iMe != mappingList.end(); ++iMe ){
    m_mappings.push_back( &(*iMe) );
  } 
}

ora::RelationalDeleter::~RelationalDeleter(){
  clear();
}

void ora::RelationalDeleter::clear(){
  m_operations.clear();
}

void ora::RelationalDeleter::build( RelationalBuffer& buffer ){
  m_operations.clear();
  for( std::vector<const MappingElement*>::iterator iMe = m_mappings.begin();
       iMe != m_mappings.end(); iMe++  ){
    std::vector<std::pair<std::string,std::string> > tableHierarchy = (*iMe)->tableHierarchy();
    size_t sz = tableHierarchy.size();
    for( size_t i=1; i<sz+1; i++ ){
      DeleteOperation& delOperation = buffer.newDelete( tableHierarchy[sz-i].first, i==sz );
      delOperation.addWhereId( tableHierarchy[sz-i].second );
      m_operations.push_back(&delOperation);      
    }
  }
}

void ora::RelationalDeleter::erase( int itemId ){
  for( std::vector<DeleteOperation*>::const_iterator iDel = m_operations.begin();
       iDel != m_operations.end(); ++iDel ){
    coral::AttributeList& whereData = (*iDel)->whereData();
    whereData.begin()->data<int>() = itemId;
  }
}

