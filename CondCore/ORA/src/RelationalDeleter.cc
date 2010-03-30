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
    for( std::vector<std::pair<std::string,std::string> >::const_reverse_iterator iT = tableHierarchy.rbegin();
         iT != tableHierarchy.rend(); ++iT ){
      DeleteOperation& delOperation = buffer.newDelete( iT->first );
      delOperation.addWhereId( iT->second );
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

