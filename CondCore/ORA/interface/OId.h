#ifndef INCLUDE_ORA_OID_H
#define INCLUDE_ORA_OID_H

#include <string>

namespace ora {

  class OId {
    public:
    OId();
    OId( int contId, int itemId );
    OId( const OId& rhs );
    OId& operator=( const OId& rhs );
    bool operator==( const OId& rhs );
    bool operator!=( const OId& rhs );
    int containerId() const;
    int itemId() const;
    std::string toString();
    void fromString( const std::string& s );
    private:
    int m_containerId;
    int m_itemId;
  };
  
  
}

#endif


