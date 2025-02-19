#ifndef INCLUDE_ORA_RELATIONALDELETER_H
#define INCLUDE_ORA_RELATIONALDELETER_H

#include "MappingElement.h"

namespace ora {
  
  class MappingElement;
  class RelationalBuffer;
  class DeleteOperation;

  class RelationalDeleter {
    
    public:

    explicit RelationalDeleter( MappingElement& dataMapping );
    explicit RelationalDeleter( const std::vector<MappingElement>& mappingList  );
    
      /// Destructor
    virtual ~RelationalDeleter();

    void build( RelationalBuffer& buffer );

    void clear();

    void erase( int itemId );

    private:
    std::vector<const MappingElement*> m_mappings;
    std::vector<DeleteOperation*> m_operations;
      
  };
}
  
#endif

    
