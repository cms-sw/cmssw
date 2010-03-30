#ifndef INCLUDE_ORA_TABLENAMERATIONALISER_H
#define INCLUDE_ORA_TABLENAMERATIONALISER_H

//
#include <string>
#include <set>
#include <map>

namespace coral {

  class ISchema;
  
}

namespace ora {
  
  class MappingElement;

  /**
     Utility class to provide valid table names taking into account the maximum size allowed for a table name.
   */

  class TableNameRationalizer {
  public:
    /// The default maximum size of a table name
    static const unsigned int defaultMaximumSizeForTableName = 20;

    /// Constructor using the default values 
    explicit TableNameRationalizer( coral::ISchema& schema );

    /// Constructor
    TableNameRationalizer( coral::ISchema& schema,
                           unsigned int maximumSizeForTableName );

    /// Destructor
    ~TableNameRationalizer();

    /// Returns a valid name for a new table given a suggestion
    std::string validNameForNewTable( const std::string& suggestion );


    /// Rationalizes the table entries in a mapping element recursively
    void rationalizeMappingElement( MappingElement& element );

  private:
    /// Reference to the working schema
    coral::ISchema& m_schema;

    /// The maximum size of a table name
    unsigned int m_maximumSizeForTableName;

    /// The set of the valid table names so far
    std::set< std::string > m_validTableNames;

    /// The suggested to valid name resolutions
    std::map< std::string, std::string > m_suggestedToValidName;
  };

}

#endif
