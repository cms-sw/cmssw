#ifndef INCLUDE_ORA_MAPPINGTREE_H
#define INCLUDE_ORA_MAPPINGTREE_H

#include "MappingElement.h"

namespace ora {

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
                                   bool isDependency=false );

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

  private:
    /**
     * The mapping version
     */
    std::string m_version;

    /**
     * The main tree
     */
    MappingElement m_element;

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
