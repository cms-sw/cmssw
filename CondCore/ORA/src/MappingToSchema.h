#ifndef INCLUDE_ORA_MAPPINGTOSCHEMA_H
#define INCLUDE_ORA_MAPPINGTOSCHEMA_H

namespace coral {

  class ISchema;

}

namespace ora {
      
  class MappingTree;
  class TableInfo;

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

    void create( const MappingTree& mapping );

    void alter( const MappingTree& mapping );

    private:

    void createTable( const TableInfo& tableInfo );

    private:
    /// Reference to the schema in use
    coral::ISchema& m_schema;

  };

}

#endif
