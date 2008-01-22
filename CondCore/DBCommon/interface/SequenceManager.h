#ifndef CondCore_DBCommon_SequenceManager_h
#define CondCore_DBCommon_SequenceManager_h

#include <string>
#include <map>
//
// Package:     DBCommon
// Class  :     SequenceManager
// 
/**\class SequenceManager SequenceManager.h CondCore/DBCommon/interface/SequenceManager.h
   Description: utility class. Handle sequences. Universal to all DB backend.
*/
//
// Author:      Zhen Xie
//
namespace coral{
  class ISchema;
  class AttributeList;  
}
namespace cond {  
  class CoralTransaction;
  /**Utility class to manage sequences in a schema
   */
  class SequenceManager {
  public:
    /// Constructor
    SequenceManager(cond::CoralTransaction& coraldb,
		    const std::string& sequenceTableName);

    /// Destructor
    ~SequenceManager();

    /// Increments and returns a new valid oid for a table
    unsigned long long incrementId( const std::string& reftableName );

    /// Updates the last used id
    void updateId( const std::string& reftableName, 
		   unsigned long long lastId );

    /// Clears the internal state
    void clear();
    
  private:
    /// Whether sequence table exists
    bool existSequencesTable();

    /// Creates the table holding the sequences
    void createSequencesTable();

    /// Locks an entry in the table
    bool lockEntry( const std::string& reftableName,
		    unsigned long long& lastId );

    private:
      /// The schema in use
      coral::ISchema& m_schema;

      /// Sequence table name
      std::string m_sequenceTableName;

      /// Map of ids used.
      std::map< std::string, unsigned long long > m_tableToId;
      
      /// Flag indicating whether the sequence table exists
      bool m_sequenceTableExists;

      /// The where clause pinning a sequence entry
      std::string m_whereClause;

      /// The data for the where clause
      coral::AttributeList* m_whereData;

      /// The set clause for updating a sequence entry
      std::string m_setClause;
    };
}//ns cond
#endif
