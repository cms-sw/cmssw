#ifndef INCLUDE_ORA_MAPPINGTREE_H
#define INCLUDE_ORA_MAPPINGTREE_H

#include "MappingElement.h"
//
#include <set>

namespace ora {

  struct TableInfo {
      TableInfo():
        m_dependency( false ),
        m_tableName(""),
        m_idColumns(),
        m_dataColumns(),
        m_parentTableName(""),
        m_refColumns(){
      }
      TableInfo( const TableInfo& rhs ):
        m_dependency( rhs.m_dependency ),
        m_tableName( rhs.m_tableName ),
        m_idColumns( rhs.m_idColumns ),
        m_dataColumns( rhs.m_dataColumns ),
        m_parentTableName(rhs.m_parentTableName),
        m_refColumns(rhs.m_refColumns){
      }
      TableInfo& operator=( const TableInfo& rhs ){
        m_dependency = rhs.m_dependency;
        m_tableName = rhs.m_tableName;
        m_idColumns = rhs.m_idColumns;
        m_dataColumns = rhs.m_dataColumns;
        m_parentTableName = rhs.m_parentTableName;
        m_refColumns = rhs.m_refColumns;
        return *this;
      }
      bool m_dependency;
      std::string m_tableName;
      std::vector<std::string> m_idColumns;
      std::map<std::string,std::string> m_dataColumns;
      std::string m_parentTableName;
      std::vector<std::string> m_refColumns;
  };
  

  /**
   * The structure holding an object/relational mapping.
   */
  
  class MappingTree
  {
    
    /** Public methods: */
  public:
    /// Constructor
    MappingTree();

    /// Constructor
    explicit MappingTree( const std::string& version );

    /// Destructor
    ~MappingTree() {}

    void setVersion( const std::string& version);
    
    /**
     * Returns the version of the mapping
     */
    const std::string& version() const;

    /**
     * Appends the element to the structure
     * @param className The class name of the new element
     * @param tableName The table name of the new element
     * @param elementType The type code of the new element
     */
    MappingElement& setTopElement( const std::string& className,
                                   const std::string& tableName,
                                   bool isDependent = false );

    void setDependency( const MappingTree& parentTree );

    /**
     * Returns the main mapping element
     */
    const MappingElement& topElement() const;
    /**
     * Returns the main mapping element
     */
    MappingElement& topElement();
    
    /**
     * Returns the main class 
     */
    const std::string& className() const;

    /// replace present data with the provided source
    void override(const MappingTree& source);

    std::vector<TableInfo> tables() const;
    
  private:
    /**
     * The mapping version
     */
    std::string m_version;

    /**
     * The main tree
     */
    MappingElement m_element;

    std::auto_ptr<TableInfo> m_parentTable;

  };
}

inline const std::string&
ora::MappingTree::version() const {
  return m_version;
}

inline void
ora::MappingTree::setVersion( const std::string& version ){
  m_version = version;
}

inline const std::string&
ora::MappingTree::className() const {
  return m_element.variableName();
}

inline const ora::MappingElement& 
ora::MappingTree::topElement() const {
  return m_element;
}

inline ora::MappingElement& 
ora::MappingTree::topElement(){
  return m_element;
}

#endif 
