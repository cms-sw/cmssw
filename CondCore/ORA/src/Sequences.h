#ifndef INCLUDE_ORA_SEQUENCES_H
#define INCLUDE_ORA_SEQUENCES_H

//
#include <map>
#include <string>

namespace ora {

  class IDatabaseSchema;

  class Sequences {
    public:
    explicit Sequences( IDatabaseSchema& dbSchema );
    virtual ~Sequences();
    int getNextId( const std::string& sequenceName, bool sinchronize = false );
    void sinchronize( const std::string& sequenceName );
    void sinchronizeAll();
    void erase( const std::string& sequenceName );
    void clear();
    private:
    std::map<std::string, int> m_lastIds;
    IDatabaseSchema& m_schema;
  };

  class NamedSequence {
    public:
    NamedSequence( const std::string& sequenceName, IDatabaseSchema& dbSchema );
    virtual ~NamedSequence();
    int getNextId( bool sinchronize = false );
    void sinchronize();
    void erase();
    void clear();
    private:
    std::string m_name;
    Sequences m_sequences;
  };
    
    
}

#endif
  
    
