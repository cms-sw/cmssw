#ifndef INCLUDE_ORA_MAPPINGTOSCHEMA_H
#define INCLUDE_ORA_MAPPINGTOSCHEMA_H

//
#include <vector>
#include <string>
#include <map>
#include <set>
#include <memory>

namespace coral {

  class ISchema;
  class ITableDescription;
  class ITable;
}

namespace ora {
      
  class MappingTree;
  class MappingElement;
  class TableDescriptionStack;

    /**
     * Helper class which is used for the creation of the tables which
     * hold the object data, conforming to a given object/relational mapping.
     */
  class MappingToSchema {
    public:
    /// Constructor
    explicit MappingToSchema( coral::ISchema& schema );

    /// Destructor
    ~MappingToSchema();

    /// Main method to materialize a mapping
    bool createOrAlter( const MappingTree& mapping, bool evolve, bool dryRun=false );

    /// Main method to materialize a mapping
    bool createOrAlter( const MappingElement& mappingElement, bool evolve, bool dryRun=false );

    const std::set<std::string>& operationList() const;

    private:
    /// Reference to the schema in use
    coral::ISchema& m_schema;

    /// dry run flag
    bool m_dryRun;

    /// A table description stack
    std::auto_ptr<TableDescriptionStack> m_tableDescriptionStack;

    /// The column list of the involved tables
    std::map<std::string, std::set<std::string> > m_tableMap;

    bool m_mappingToProcess;

    std::set<std::string> m_operationList;

    private:

    //
    bool processMapping(const MappingTree& mapping, bool evolve);
    //
    bool processMapping(const MappingElement& mappingElement, bool evolve);

    bool isTableEmpty(coral::ITable& table);
      
    /// Processes a mapping element
    bool processElement( const MappingElement& element,
                         bool evolve );

    /// Processes a mapping sub-element
    bool processSubElement( const MappingElement& parentElement,
                            const MappingElement& element,
                            bool evolve );
    /// Checks if a unique index of a primary key exists
    bool existsUniqueIndex( const coral::ITableDescription& description,
                            const std::vector< std::string >& columnNames ) const;

    /// Processes an element of an object type
    bool processObjectElement( const MappingElement& element,
                               bool evolve );

    /// Processes an element of a dependent object type
    bool processDependentObjectElement( const MappingElement& element,
                                        bool evolve );

    /// Processes the most inner elements of the mapping tree
    bool processLeafElement( const std::string& tableName,
                             const std::string& columnName,
                             const std::string& variableType,
                             bool evolve );

    /// Processes an element of a primitive type
    bool processPrimitiveElement( const MappingElement& element,
                                  bool evolve );

    /// Processes an element of a bolb type
    bool processBlobElement( const MappingElement& element,
                             bool evolve );

    /// Processes an element of an array type
    bool processArrayElement( const MappingElement& parentElement,
                              const MappingElement& element,
                              bool evolve );

    /// Processes an element of a POOL reference type
    bool processOraReferenceElement( const MappingElement& element,
                                     bool evolve );

    /// Processes an element of an pool pointer type
    bool processOraPointerElement( const MappingElement& element,
                                   bool evolve );

    /// Processes an element of an pool pointer type
    bool processUniqueReferenceElement( const MappingElement& element,
                                        bool evolve );
      
      /// Processes an element of a pointer/reference type
      //bool processPointerElement( const pool::ObjectRelationalMappingElement& element,
      //                            bool evolve );

      /// Processes an object
    bool processObject( const std::string& tableName,
                        const std::vector< std::string >& columnNames,
                        bool isDependent,
                        bool evolve );

    /// Processes an object
    bool processDependentObject( const std::string& tableName,
                                 const std::vector< std::string >& columnNames,
                                 const std::string& parentTableName,
                                 const std::vector< std::string >& parentColumnNames,
                                 bool evolve );

    /// Checks or sets (if not found) a column
    bool checkOrSetColumn(coral::ITable& table,
                          const std::string& columnName,
                          const std::string& columnType,
                          bool& exists,
                          bool evolve);

    /// Checks or sets (if not found) a foreign key constraint.
    bool checkOrSetForeignKey( const std::string& tableName,
                               const std::vector< std::string >& columnNames,
                               const std::string& referencedTable,
                               const std::vector< std::string >& referencedColumnNames,
                               bool evolve );

    /// Checks for the existence of a foreign key constraint
    bool existsForeignKey( const coral::ITableDescription& description,
                           const std::vector< std::string >& columnNames,
                           const std::string& referencedTable,
                           const std::vector< std::string >& referencedColumnNames );
    void logOperation(const std::string& message);
  };

}

#endif
