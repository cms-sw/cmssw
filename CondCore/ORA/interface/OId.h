#ifndef INCLUDE_ORA_OID_H
#define INCLUDE_ORA_OID_H

#include <string>

namespace ora {

  class OId {

    public:
    static bool isOId( const std::string& input );
  
    public:

    OId();

    explicit OId( const std::pair<int,int>& oidPair );

    OId( int contId, int itemId );

    OId( const OId& rhs );

    OId& operator=( const OId& rhs );

    bool operator==( const OId& rhs ) const;

    bool operator!=( const OId& rhs ) const;

    int containerId() const;

    int itemId() const;

    bool fromString( const std::string& s );

    std::string toString() const;

    void toOutputStream( std::ostream& os ) const;     

    void reset();

    bool isInvalid() const;

    std::pair<int,int> toPair() const;

    private:
    int m_containerId;
    int m_itemId;
  };

  class ITokenParser {
    public:
    virtual ~ITokenParser(){
    }

    virtual OId parse( const std::string& poolToken ) = 0;
    virtual std::string className( const std::string& poolToken ) = 0;
  };

  class ITokenWriter {
    public:
    virtual ~ITokenWriter(){
    }

    virtual std::string write( const OId& oid ) = 0;
  };

}
 
inline std::ostream& operator << (std::ostream& os, const ora::OId& oid ){
  oid.toOutputStream(os);
  return os;
}

#endif


