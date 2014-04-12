#ifndef INCLUDE_ORA_MAPPINGELEMENT_H
#define INCLUDE_ORA_MAPPINGELEMENT_H

//
#include <map>
#include <vector>
#include <string>
#include <memory>

namespace ora {

  /**
   * @class MappingElement MappingElement.h 
   *
   * Class describing an element of the object-relational mapping structure.
   */

  class MappingElement
  {
  public:
    typedef enum { Undefined = -1,
                   Class,
                   Object,
                   Dependency,
                   Primitive,
                   Array,
                   CArray,
                   InlineCArray,
                   Pointer,
                   Reference,
                   OraReference,
                   OraPointer,
                   UniqueReference,
                   OraArray,
                   Blob,
                   NamedReference } ElementType;

    /// Returns the name of the class mapping element type
    static std::string classMappingElementType();
    /// Returns the name of the object mapping element type
    static std::string objectMappingElementType();
    /// Returns the name of the dependent class mapping element type
    static std::string dependencyMappingElementType();
    /// Returns the name of the primitive mapping element type
    static std::string primitiveMappingElementType();
    /// Returns the name of the array mapping element type
    static std::string arrayMappingElementType();
    /// Returns the name of the array mapping element type
    static std::string CArrayMappingElementType();
    /// Returns the name of the inline array mapping element type
    static std::string inlineCArrayMappingElementType();
    /// Returns the name of the ORA reference mapping element type
    static std::string OraReferenceMappingElementType();
    /// Returns the name of the ORA pointer mapping element type
    static std::string OraPointerMappingElementType();
    /// Returns the name of the ORA polymorphic pointer mapping element type
    static std::string uniqueReferenceMappingElementType();
    /// Returns the name of the ORA array mapping element type
    static std::string OraArrayMappingElementType();
    /// Returns the name of the pointer mapping element type
    static std::string pointerMappingElementType();
    /// Returns the name of the reference mapping element type
    static std::string referenceMappingElementType();
    /// Returns the name of the blob mapping element type
    static std::string blobMappingElementType();
    /// Returns the name of the named reference element type
    static std::string namedReferenceMappingElementType();
    /// Checks if the provided element type is valid
    static bool isValidMappingElementType( const std::string& elementType );

    /// Converts the enumeration type to a string
    static std::string elementTypeAsString( ElementType elementType );

    /// Converts a string into an element type
    static ElementType elementTypeFromString( const std::string& elementType );

    public:
    /// Iterator definition
    typedef std::map< std::string, MappingElement >::iterator iterator;
    typedef std::map< std::string, MappingElement >::const_iterator const_iterator;

    public:

    /// Empty Constructor
    MappingElement();

    /// Constructor
    MappingElement( const std::string& elementType,
                    const std::string& variableName,
                    const std::string& variableType,
                    const std::string& tableName );

    // Copy constructor
    MappingElement( const MappingElement& element);

    // Assignment operator
    MappingElement& operator=(const MappingElement& element);

    /// Destructor
    ~MappingElement();

    /**
     * Returns the element type
     */
    ElementType elementType() const;

    std::string elementTypeString() const;

    /**
     * Returns the dependent flag
     */
    bool isDependent() const;

    /**
     * Returns the parent class mapping element
     */
    const MappingElement& parentClassMappingElement() const;
    
    /**
     * Returns the scope name
     */
    const std::string& scopeName() const;
    /**
     * Returns the variable name
     */
    const std::string& variableName() const;

    /**
     * Returns the variable name for the column-table name generation
     */
    const std::string& variableNameForSchema() const;
    /**
     * Returns the variable C++ type
     */
    const std::string& variableType() const;

    /**
     * Returns the associated table name
     */
    const std::string& tableName() const;
    
    /**
     * Returns the associated columns
     */
    const std::vector< std::string >& columnNames() const;

    std::string idColumn() const;

    std::string pkColumn() const;

    std::vector<std::string> recordIdColumns() const;

    std::string posColumn() const;

    /**
     * Returns the table names and their id columns  
     */
    std::vector<std::pair<std::string,std::string> > tableHierarchy() const;
     
    /**
     * Changes the type of the element and propagates the changes to the sub-elements accordingly
     * Note: It should be called before a call to alterTableName and setColumnNames.
     * @param elementType The new type
     */
    void alterType( const std::string& elementType );
    
    /**
     * Changes the name of the associated table and propagates the new information to the sub-elements.
     * Note: It should be called before a call to setColumnNames.
     * @param tableName The new table name
     */
    void alterTableName( const std::string& tableName );

    /**
     * Sets the column names. It is propagated to objects having the same associated table.
     * @param columns The names of the associated columns.
     */
    void setColumnNames( const std::vector< std::string >& columns );

    /// Returns an iterator in the beginning of the sequence
    iterator begin();
    const_iterator begin() const;

    /// Returns an iterator in the end of the sequence
    iterator end();
    const_iterator end() const;

    /// Retrieves a sub-element
    iterator find( const std::string& key );
    const_iterator find( const std::string& key ) const;

    /// Remove a sub-element
    bool removeSubElement( const std::string& key );

    /**
     * Appends a new sub-element.
     * In case the current element is not an object or an array, an ObjectRelationalException is thrown.
     *
     * @param elementType The element type
     * @param variableName The variable name
     * @param variableType The variable type
     * @param tableName The name of the associated table
     */
    MappingElement& appendSubElement( const std::string& elementType,
                                      const std::string& variableName,
                                      const std::string& variableType,
                                      const std::string& tableName );

    /*
     * replace present data with the provided source
     */
    void override(const MappingElement& source);

    void printXML( std::ostream& outputStream, std::string indentation="" ) const;

  private:
    /**
     * The type of this mapping element.
     */
    ElementType m_elementType;
    /**
     * Flag to recursive mark the mapping trees of depending objects
     */
    bool m_isDependentTree;    
    /**
     * The scope (parent of the given element). In case of top level classes it is an
     * empty string. In case of a field under a top level class it is the fully
     * qualified class name. If the element of the field defines a sub-element then
     * their scope name is the "scope name of the parent" + "::" + "the field name"
     * (m_variableName).
     */
    std::string m_scopeName;
    /**
     * For a top level class this is the same as m_variableType. For a field in a class it is the name of the
     * field itself. For an element in an array, it is the same as the m_variableType.
     */
    std::string m_variableName;
    /**
     * The variable name used for the column-table name generation.
     */
    std::string m_variableNameForSchema;
    /**
     * The C++ type of the element.
     */
    std::string m_variableType;
    /**
     * The name of the associated table. For an object it is where it's identity columns
     * are defined. For a primitive and a ORA reference, a reference or a pointer it is the table of
     * the parent object. For an array it is the table where the array elements are stored.
     */
    std::string m_tableName;
    /**
     * The array of the names of the associated columns. For an object they are the ones defining
     * its identity. A primitive is associated to a single column. For an ORA reference they are two columns corresponding
     * to the two fields of the OID.
     * For a pointer or a reference the first column is the referenced
     * table and the rest refer to the identity columns of the target object. For an array the first column is the "position"
     * column and the rest are the columns forming the foreign key constraint w.r.t. the parent table.
     */
    std::vector<std::string> m_columnNames;
    
    /**
     * The map of sub-elements.
     */
    std::map< std::string, MappingElement > m_subElements;

  };
}

inline 
ora::MappingElement::MappingElement():
  m_elementType(ora::MappingElement::Undefined),
  m_isDependentTree(false),
  m_scopeName(""),
  m_variableName(""),
  m_variableNameForSchema(""),
  m_variableType(""),
  m_tableName(""),
  m_columnNames(),
  m_subElements(){
}

inline 
ora::MappingElement::MappingElement( const MappingElement& element):
  m_elementType(element.m_elementType),
  m_isDependentTree(element.m_isDependentTree),
  m_scopeName(element.m_scopeName),
  m_variableName(element.m_variableName),
  m_variableNameForSchema(element.m_variableNameForSchema),
  m_variableType(element.m_variableType),
  m_tableName(element.m_tableName),
  m_columnNames(element.m_columnNames),
  m_subElements(element.m_subElements){
}

inline ora::MappingElement& 
ora::MappingElement::operator=(const MappingElement& element){
  if(this != &element){
     m_elementType = element.m_elementType;
     m_isDependentTree = element.m_isDependentTree;
     m_scopeName = element.m_scopeName;
     m_variableName = element.m_variableName;
     m_variableNameForSchema = element.m_variableNameForSchema;
     m_variableType = element.m_variableType;
     m_tableName = element.m_tableName;
     m_columnNames = element.m_columnNames;
     m_subElements = element.m_subElements;
  }
  return *this;
}


inline ora::MappingElement::iterator
ora::MappingElement::begin()
{
  return m_subElements.begin();
}

inline ora::MappingElement::const_iterator
ora::MappingElement::begin() const
{
  return m_subElements.begin();
}

inline ora::MappingElement::iterator
ora::MappingElement::end()
{
  return m_subElements.end();
}

inline ora::MappingElement::const_iterator
ora::MappingElement::end() const
{
  return m_subElements.end();
}

inline ora::MappingElement::iterator
ora::MappingElement::find( const std::string& key )
{
  return m_subElements.find( key );
}

inline ora::MappingElement::const_iterator
ora::MappingElement::find( const std::string& key ) const
{
  return m_subElements.find( key );
}

inline ora::MappingElement::ElementType
ora::MappingElement::elementType() const
{
  return m_elementType;
}

inline std::string
ora::MappingElement::elementTypeString() const
{
  return elementTypeAsString( m_elementType );
}

inline bool 
ora::MappingElement::isDependent() const {
  return m_isDependentTree;
}

inline const std::string&
ora::MappingElement::scopeName() const
{
  return m_scopeName;
}

inline const std::string&
ora::MappingElement::variableName() const
{
  return m_variableName;
}

inline const std::string&
ora::MappingElement::variableNameForSchema() const
{
  return m_variableNameForSchema;
}

inline const std::string&
ora::MappingElement::variableType() const
{
  return m_variableType;
}

inline const std::string&
ora::MappingElement::tableName() const
{
  return m_tableName;
}

inline const std::vector< std::string >&
ora::MappingElement::columnNames() const
{
  return m_columnNames;
}

#endif
