#ifndef INCLUDE_ORA_SEQUENCEMANAGER_H
#define INCLUDE_ORA_SEQUENCEMANAGER_H

#include "Exception.h"
//
#include <boost/shared_ptr.hpp>

namespace coral {
  class ISchema;
}

namespace ora {

  class Sequences;
  class OraSequenceTable;

  class SequenceManager {
    public:

    // 
    SequenceManager( const std::string& tableName, coral::ISchema& schema );

    //
    SequenceManager( const SequenceManager& rhs );
    
    /// 
    virtual ~SequenceManager();

    /// 
    SequenceManager& operator=( const SequenceManager& rhs );

    ///
    std::string tableName();

    ///
    void create( const std::string& sequenceName );

    ///
    int getNextId( const std::string& sequenceName, bool sinchronize = false );

    ///
    void sinchronize( const std::string& sequenceName );

    ///
    void sinchronizeAll();

    ///
    void erase( const std::string& sequenceName );

    ///
    void clear();

    private:

    boost::shared_ptr<OraSequenceTable> m_table;
    boost::shared_ptr<Sequences> m_impl;
    
  };

}

#endif
