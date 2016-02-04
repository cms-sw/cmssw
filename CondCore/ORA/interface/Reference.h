#ifndef INCLUDE_ORA_REFERENCE_H
#define INCLUDE_ORA_REFERENCE_H

#include "OId.h"

namespace ora {
  
  class Reference {
    public:
    Reference( );
    explicit Reference( const OId& oid );
    Reference( const Reference& rhs );
    virtual ~Reference();
    Reference& operator=( const Reference& rhs );
    void set( const OId& oid );
    OId oid() const;
    private:
    int m_containerId;
    int m_itemId;
  };
  
}

#endif
  
    
    
