#include "MappingToSchema.h"
#include "MappingTree.h"
#include "MappingRules.h"
//
#include <algorithm>
#include <sstream>
#include <memory>
// externals
#include "CoralBase/Blob.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/AttributeListSpecification.h"
#include "CoralBase/MessageStream.h"
#include "RelationalAccess/IColumn.h"
#include "RelationalAccess/IForeignKey.h"
#include "RelationalAccess/IIndex.h"
#include "RelationalAccess/IUniqueConstraint.h"
#include "RelationalAccess/IPrimaryKey.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IView.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/SchemaException.h"
#include "RelationalAccess/TableDescription.h"

namespace ora {

  class TableDescriptionStack {
    public:
      TableDescriptionStack();
      ~TableDescriptionStack();
      coral::TableDescription& description( const std::string& tableName );
      const coral::TableDescription& description( const std::string& tableName ) const;
      typedef std::vector< std::string >::const_iterator const_iterator;
      const_iterator begin() const;
      const_iterator end() const;
    private:
      std::vector< std::string > m_tableNames;
      std::map< std::string, coral::TableDescription* > m_descriptions;
  };
}


ora::TableDescriptionStack::TableDescriptionStack():
  m_tableNames(),
  m_descriptions(){}


ora::TableDescriptionStack::~TableDescriptionStack() {
  for ( std::map< std::string, coral::TableDescription* >::iterator iDescription = m_descriptions.begin();
        iDescription != m_descriptions.end(); ++iDescription ) {
    delete iDescription->second;
  }
}

coral::TableDescription&
ora::TableDescriptionStack::description( const std::string& tableName ){
  std::map< std::string, coral::TableDescription* >::iterator iDescription = m_descriptions.find( tableName );
  if ( iDescription != m_descriptions.end() ) {
    return *( iDescription->second );
  }
  coral::TableDescription* newDescription = new coral::TableDescription("ORA");
  newDescription->setName(tableName);
  m_tableNames.push_back( tableName );
  m_descriptions.insert( std::make_pair( tableName, newDescription ) );
  return *newDescription;
}

const coral::TableDescription&
ora::TableDescriptionStack::description( const std::string& tableName ) const {
  return *( m_descriptions.find( tableName )->second );
}

ora::TableDescriptionStack::const_iterator
ora::TableDescriptionStack::begin() const {
  return m_tableNames.begin();
}

ora::TableDescriptionStack::const_iterator
ora::TableDescriptionStack::end() const {
  return m_tableNames.end();
}


ora::MappingToSchema::MappingToSchema( coral::ISchema& schema ):
  m_schema( schema ),
  m_dryRun(false),
  m_tableDescriptionStack( 0 ),
  m_tableMap(),
  m_mappingToProcess(true),
  m_operationList(){}

ora::MappingToSchema::~MappingToSchema(){
}

bool
ora::MappingToSchema::createOrAlter( const MappingTree& mapping,
                                     bool evolve,
                                     bool dryRun )
{
  m_dryRun = dryRun;
  
  m_tableDescriptionStack.reset( new ora::TableDescriptionStack );

  if(!this->processMapping(mapping, evolve)) return false;
  
    // check if the existing columns in the tables are all mapped to the object attributes 
  for(std::map<std::string,std::set<std::string> >::const_iterator iT=m_tableMap.begin();
      iT!=m_tableMap.end();++iT){
    if( m_schema.existsTable( iT->first )){

      coral::ITable& table = m_schema.tableHandle( iT->first );
      coral::MessageStream log( "ORA" );
      log << coral::Debug << "Checking existing columns in table \"" << iT->first <<"\"."
          << coral::MessageStream::endmsg;
      int nColumns = table.description().numberOfColumns();
      for(int i=0;i<nColumns;++i){
        const coral::IColumn& column = table.description().columnDescription(i);
        std::set< std::string >::const_iterator iC = iT->second.find(column.name());
        if(iC==iT->second.end()){
          log << coral::Info << "The column named \"" << column.name() << "\" in table \"" << iT->first
              << "\" is not mapped to any class attribute."
              << coral::MessageStream::endmsg;
          if(column.isNotNull()){
            if ( ! evolve ) {
              log << coral::Error << "The column named \"" << column.name() << "\" in table \"" << iT->first
                  << "\", has been found with a NOT NULL constraint, and schema evolution has not been enabled."
                  << coral::MessageStream::endmsg;
              return false;
            }
            else {
              std::ostringstream message;
              message << "Unsetting NOT NULL constraint on column \"" << column.name() << "\" in table \"" << iT->first <<"\"";
              this->logOperation( message.str() );
              if(!m_dryRun) table.schemaEditor().setNotNullConstraint( column.name(), false );
            }
          }
        }
      }
    }
  }

  // create the tables with the given description set.
  for ( ora::TableDescriptionStack::const_iterator iTableName = m_tableDescriptionStack->begin();
        iTableName != m_tableDescriptionStack->end(); ++iTableName ) {
    std::ostringstream message;
    message << "Creating table \"" << *iTableName <<"\"";
    this->logOperation( message.str() );
    if(!m_dryRun) m_schema.createTable( m_tableDescriptionStack->description( *iTableName ) ).privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
        
  }
  
  m_tableDescriptionStack.reset();

  if(m_mappingToProcess){
    if(!this->processMapping(mapping, evolve)) return false;
  }
  
  return true;
}

bool
ora::MappingToSchema::createOrAlter( const MappingElement& mappingElement,
                                     bool evolve,
                                     bool dryRun )
{
  m_dryRun = dryRun;
  
  m_tableDescriptionStack.reset( new ora::TableDescriptionStack );

  if(!this->processMapping(mappingElement, evolve)) return false;
  
    // check if the existing columns in the tables are all mapped to the object attributes 
  for(std::map<std::string,std::set<std::string> >::const_iterator iT=m_tableMap.begin();
      iT!=m_tableMap.end();++iT){
    if( m_schema.existsTable( iT->first )){

      coral::ITable& table = m_schema.tableHandle( iT->first );
      coral::MessageStream log( "ORA" );
      log << coral::Debug << "Checking existing columns in table \"" << iT->first <<"\"."
          << coral::MessageStream::endmsg;
      int nColumns = table.description().numberOfColumns();
      for(int i=0;i<nColumns;++i){
        const coral::IColumn& column = table.description().columnDescription(i);
        std::set< std::string >::const_iterator iC = iT->second.find(column.name());
        if(iC==iT->second.end()){
          log << coral::Info << "The column named \"" << column.name() << "\" in table \"" << iT->first
              << "\" is not mapped to any class attribute."
              << coral::MessageStream::endmsg;
          if(column.isNotNull()){
            if ( ! evolve ) {
              log << coral::Error << "The column named \"" << column.name() << "\" in table \"" << iT->first
                  << "\", has been found with a NOT NULL constraint, and schema evolution has not been enabled."
                  << coral::MessageStream::endmsg;
              return false;
            }
            else {
              std::ostringstream message;
              message << "Unsetting NOT NULL constraint on column \"" << column.name() << "\" in table \"" << iT->first <<"\"";
              this->logOperation( message.str() );
              if(!m_dryRun) table.schemaEditor().setNotNullConstraint( column.name(), false );
            }
          }
        }
      }
    }
  }

  // create the tables with the given description set.
  for ( ora::TableDescriptionStack::const_iterator iTableName = m_tableDescriptionStack->begin();
        iTableName != m_tableDescriptionStack->end(); ++iTableName ) {
    std::ostringstream message;
    message << "Creating table \"" << *iTableName <<"\"";
    this->logOperation( message.str() );
    if(!m_dryRun) m_schema.createTable( m_tableDescriptionStack->description( *iTableName ) ).privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
        
  }
  
  m_tableDescriptionStack.reset();

  if(m_mappingToProcess){
    if(!this->processMapping(mappingElement, evolve)) return false;
  }
  
  return true;
}


const std::set<std::string>&
ora::MappingToSchema::operationList() const {
  return m_operationList;
}

bool
ora::MappingToSchema::processMapping(const MappingTree& mapping, bool evolve){
  m_mappingToProcess = false;
  // Loop over the elements of the mapping.
  bool ok = true;
  if ( ! this->processElement( mapping.topElement(), evolve ) ) {
    ok = false;
  }
  if(!ok){
    m_tableDescriptionStack.reset();
    return false;
  }
  return true;
}

bool
ora::MappingToSchema::processMapping(const MappingElement& mappingElement,
                                     bool evolve){
  m_mappingToProcess = false;
  if ( ! this->processElement( mappingElement, evolve ) ) {
    m_tableDescriptionStack.reset();
    return false;
  }
  return true;
}

bool
ora::MappingToSchema::isTableEmpty(coral::ITable& table){
  std::auto_ptr< coral::IQuery > query( table.newQuery() );
  query->limitReturnedRows( 1, 0 );
  coral::ICursor& cursor = query->execute();
  if ( cursor.next() ) return false;
  else return true;
}

bool
ora::MappingToSchema::processElement( const MappingElement& element,
                                      bool evolve )
{
  if( element.elementType()== MappingElement::Dependency ){
    if ( ! ( this->processDependentObjectElement( element, evolve ) ) ) return false;
  } else {
    if ( ! ( this->processObjectElement( element, evolve ) ) ) {
      return false;
    }
  }

  //  compile the table map
  std::map<std::string, std::set<std::string> >::iterator iT = m_tableMap.find(element.tableName());
  if(iT==m_tableMap.end()) {
    std::set<std::string> cols;
    iT = m_tableMap.insert(std::make_pair(element.tableName(),cols)).first;
  }
  for(std::vector<std::string>::const_iterator iC=element.columnNames().begin();
      iC!=element.columnNames().end();++iC){
    iT->second.insert(*iC);
  }

  //  loop over sub-elements.
  for ( MappingElement::const_iterator iElement = element.begin();
        iElement != element.end(); ++iElement ) {
    const MappingElement& subElement = iElement->second;
    if ( ! this->processSubElement( element, subElement, evolve ) ) return false;
  }

  return true;
}

bool
ora::MappingToSchema::processSubElement( const MappingElement& parentElement,
                                         const MappingElement& element,
                                         bool evolve )
{
  MappingElement::ElementType elementType = element.elementType();
  switch ( elementType ) {
  case MappingElement::Object :
    if ( !( this->processObjectElement( element, evolve ) ) ) return false;
    break;
  case MappingElement::Dependency :
    if ( !( this->processDependentObjectElement( element, evolve ) ) ) return false;
    break;
  case MappingElement::Primitive :
    if ( !( this->processPrimitiveElement( element, evolve ) ) ) return false;
    break;
  case MappingElement::Blob :
    if ( !( this->processBlobElement( element, evolve ) ) ) return false;
    break;
  case MappingElement::Array :
    if ( !( this->processArrayElement( parentElement, element, evolve ) ) ) {
      return false;
    }
    break;
  case MappingElement::OraArray :
    if ( !( this->processArrayElement( parentElement, element, evolve ) ) ) return false;
    break;
  case MappingElement::CArray :
    if ( !( this->processArrayElement( parentElement, element, evolve ) ) ) return false;
    break;
  case MappingElement::InlineCArray :
    if ( !( this->processObjectElement( element, evolve ) ) ) return false;
    break;
  case MappingElement::OraReference :
    if ( !( this->processOraReferenceElement( element, evolve ) ) ) return false;
    break;
  case MappingElement::OraPointer :
    if ( !( this->processOraPointerElement( element, evolve ) ) ) {
      return false;
    }
    break;
  case MappingElement::UniqueReference :
    if ( !( this->processUniqueReferenceElement( element, evolve ) ) ) return false;
    break;
  case MappingElement::Class :
  case MappingElement::Reference :
  case MappingElement::Pointer :
    //if ( !( this->processPointerElement( element, evolve ) ) ) return false;
    break;
  case MappingElement::Undefined :
    // This should never appear
    break;
  };
  
  //  compile the table map
  std::map<std::string, std::set<std::string> >::iterator iT = m_tableMap.find(element.tableName());
  if(iT==m_tableMap.end()) {
    std::set<std::string> cols;
    iT = m_tableMap.insert(std::make_pair(element.tableName(),cols)).first;
  }
  for(std::vector<std::string>::const_iterator iC=element.columnNames().begin();
      iC!=element.columnNames().end();++iC){
    iT->second.insert(*iC);
  }

  //  loop over sub-elements.
  for ( MappingElement::const_iterator iElement = element.begin();
        iElement != element.end(); ++iElement ) {
    const MappingElement& subElement = iElement->second;
    if ( ! this->processSubElement( element, subElement, evolve ) ) return false;
  }

  return true;
}

bool
ora::MappingToSchema::existsUniqueIndex( const coral::ITableDescription& description,
                                                                         const std::vector< std::string >& columnNames ) const
{
  if ( description.hasPrimaryKey() ) {
    const std::vector<std::string>& columns = description.primaryKey().columnNames();
    if ( columns.size() == columnNames.size() ) {
      bool same = false;
      for ( size_t i = 0; i < columns.size(); ++i ) {
        same = false;
        for( size_t j = 0; j < columnNames.size() && !same; j++) {
          if ( columns[i] == columnNames[j] ) {
            same = true;
          }
        }
        if(!same) break;
      }
      if ( same ) return true;
    }
  }

  int numberOfUniqueConstraints = description.numberOfUniqueConstraints();
  for ( int iConstraint = 0; iConstraint < numberOfUniqueConstraints; ++iConstraint ) {
    const std::vector<std::string>& columns = description.uniqueConstraint( iConstraint ).columnNames();
    if ( columns.size() == columnNames.size() ) {
      bool same = false;
      for ( size_t i = 0; i < columns.size(); ++i ) {
        same = false;
        for( size_t j = 0; j < columnNames.size() && !same; j++) {
          if ( columns[i] == columnNames[j] ) {
            same = true;
          }
        }
        if(!same) break;
      }
      if ( same ) return true;
    }
  }

  int numberOfIndices = description.numberOfIndices();
  for ( int iIndex = 0; iIndex < numberOfIndices; ++iIndex ) {
    if ( ! description.index( iIndex ).isUnique() ) continue;
    const std::vector<std::string>& columns = description.index( iIndex ).columnNames();
    if ( columns.size() == columnNames.size() ) {
      bool same = false;
      for ( size_t i = 0; i < columns.size(); ++i ) {
        same = false;
        for( size_t j = 0; j < columnNames.size() && !same; j++) {
          if ( columns[i] == columnNames[j] ) {
            same = true;
          }
        }
        if(!same) break;
      }
      if ( same ) return true;
    }
  }

  return false;
}


bool
ora::MappingToSchema::processObjectElement( const MappingElement& element,
                                                                            bool evolve )
{
  if ( ! this->processObject( element.tableName(), element.columnNames(), false, evolve ) ) {
    return false;
  }
  // Process the sub elements
  for ( MappingElement::const_iterator iElement = element.begin();
        iElement != element.end(); ++iElement ) {
    const MappingElement& subElement = iElement->second;
    if ( ! this->processSubElement( element, subElement, evolve ) ) {
      return false;
    }
  }

  return true;
}


bool
ora::MappingToSchema::processDependentObjectElement( const MappingElement& element,
                                                                                     bool evolve )
{
  if ( ! this->processDependentObject( element.tableName(), element.columnNames(),
                                       element.parentClassMappingElement()->tableName(),element.parentClassMappingElement()->columnNames(),
                                       evolve ) ) return false;
  // Process the sub elements
  for ( MappingElement::const_iterator iElement = element.begin();
        iElement != element.end(); ++iElement ) {
    const MappingElement& subElement = iElement->second;
    if ( ! this->processSubElement( element, subElement, evolve ) ) return false;
  }

  return true;
}

bool
ora::MappingToSchema::processLeafElement( const std::string& tableName,
                                                                          const std::string& columnName,
                                                                          const std::string& variableType,
                                                                          bool evolve )
{
  coral::MessageStream log( "ORA" );
  // Check if the table exists.
  if ( m_schema.existsTable( tableName ) ) {
    coral::ITable& table = m_schema.tableHandle( tableName );
    // Check if the column exists in the table
    try {      
      const coral::IColumn& column = table.description().columnDescription( columnName );
      if ( ! column.isNotNull() && isTableEmpty(table) ) {
        if ( ! evolve ) {
          log << coral::Error << "The column named \"" << columnName << "\" in table \""
              << tableName
              << "\" has been found without a NOT NULL constraint while processing a mapping element, and schema evolution has not been enabled."
              << coral::MessageStream::endmsg;
          return false;
        } else {
          std::ostringstream message;
          message << "Setting NOT NULL constraint on column \"" << columnName << "\" in table \"";
          this->logOperation( message.str() );
          if(!m_dryRun) table.schemaEditor().setNotNullConstraint( columnName );
        }
      }
    }
    catch (const coral::InvalidColumnNameException&) {
      if ( ! evolve ) {
        log << coral::Error << "A column named \"" << columnName << "\" has not been found in table \""
            << tableName << "\" while processing a mapping element, and schema evolution has not been enabled." << coral::MessageStream::endmsg;
        return false;
      } else {
        std::ostringstream message;
        message << "Inserting column \"" << columnName << "\" of type \""<< variableType<<"\" in table \"" << tableName <<"\"";
        this->logOperation( message.str() );
        if(!m_dryRun){
          table.schemaEditor().insertColumn( columnName,
                                             variableType );
        }
        if( isTableEmpty( table )){
          std::ostringstream message;
          message << "Setting NOT NULL constraint on column \"" << columnName <<"\" in table \"" << tableName <<"\"";
          this->logOperation( message.str() );
         if(!m_dryRun) table.schemaEditor().setNotNullConstraint( columnName );
        }
      }
    }
  } // Check if the view exists in the schema
  else if ( m_schema.existsView( tableName ) ) {
    coral::IView& view = m_schema.viewHandle( tableName );
    bool found = false;
    int numberOfColumns = view.numberOfColumns();
    for ( int iColumn = 0; iColumn < numberOfColumns; ++iColumn ) {
      const coral::IColumn& column = view.column( iColumn );
      if ( column.name() != columnName ) continue;
      found = true;
      if ( ! column.isNotNull() ) {
        log << coral::Error << "The column named \"" << columnName << "\" in view \""
            << tableName
            << "\" has been found without a NOT NULL constraint."
            << coral::MessageStream::endmsg;
        return false;
      }
      break;
    }
    if ( ! found ) {
       log << coral::Error << "A column named \"" << columnName << "\" has not been found in view \""
          << tableName << "\" while processing a mapping element." << coral::MessageStream::endmsg;
      return false;
    }
  }
  else {
    coral::TableDescription& description = m_tableDescriptionStack->description( tableName );
    try {
      description.columnDescription(columnName);
    }
    catch (const coral::InvalidColumnNameException&) {
      description.insertColumn( columnName,
                                variableType );
      description.setNotNullConstraint( columnName );
    }
  }

  return true;
}

bool
ora::MappingToSchema::processBlobElement( const MappingElement& element,
                                                                          bool evolve )
{
  const std::string& tableName = element.tableName();
  const std::string& columnName = element.columnNames()[0];
  std::string variableType = coral::AttributeSpecification::typeNameForId( typeid(coral::Blob) );
  return processLeafElement(tableName, columnName, variableType, evolve);
}

bool
ora::MappingToSchema::processPrimitiveElement( const MappingElement& element,
                                                                               bool evolve )
{
  const std::string& tableName = element.tableName();
  const std::string& columnName = element.columnNames()[0];
  const std::string variableType = element.variableType();
  return processLeafElement(tableName, columnName, variableType, evolve);
}

bool
ora::MappingToSchema::processArrayElement( const MappingElement& parentElement,
                                           const MappingElement& element,
                                           bool evolve )
{
  const std::string& parentTableName = parentElement.tableName();
  const std::string& tableName = element.tableName();
  // Retrieve the columns of the parent table.
  std::vector< std::string > parentIdColumns = parentElement.columnNames();

  const std::vector< std::string >& columnNames = element.columnNames();
  bool isDependent = element.isDependentTree();
  
  // To set up the array table.
  if ( ! ( this->processObject( tableName, columnNames, isDependent, evolve ) ) ) return false;
    
  // Process the sub elements
  for ( MappingElement::const_iterator iElement = element.begin();
        iElement != element.end(); ++iElement ) {
    const MappingElement& subElement = iElement->second;
    if ( ! this->processSubElement( element, subElement, evolve ) ) return false;
  }

  // Make sure that there is a foreign key constraint id columns
  std::vector< std::string > idColumns;
  size_t i = 0;
  //if(isDependent) {
  //  idColumns.push_back( columnNames[i] );
  //  i++;
  //}
  // skip the position column
  //i++;
  //for ( ; i < columnNames.size(); ++i ) idColumns.push_back( columnNames[i] );
  for ( ; i < columnNames.size()-1; ++i ) idColumns.push_back( columnNames[i] );
  
  // Reverse the columns.
  //std::reverse( parentIdColumns.begin(), parentIdColumns.end() );
  //std::reverse( idColumns.begin(), idColumns.end() );
  return this->checkOrSetForeignKey( tableName, idColumns, parentTableName, parentIdColumns, evolve );
}

bool
ora::MappingToSchema::processOraReferenceElement( const MappingElement& element,
                                                  bool evolve )
{
  
  const std::string& tableName = element.tableName();
  const std::string variableType = coral::AttributeSpecification::typeNameForId( typeid(int) );
  bool ok = true;
  for( unsigned int i=0;i<2 && ok ;i++){
    const std::string& columnName = element.columnNames()[i];
    ok = processLeafElement(tableName, columnName, variableType, evolve);
  }
  return ok;
}

bool
ora::MappingToSchema::processOraPointerElement( const MappingElement& element,
                                                bool evolve )
{
  const std::string& tableName = element.tableName();

  std::vector< std::string > columnNames = element.columnNames();
  bool isDependent = element.isDependentTree();
  
  // To set up the array table.
  if ( ! ( this->processObject( tableName, columnNames, isDependent, evolve ) ) ) return false;
    
  // Process the sub elements
  for ( MappingElement::const_iterator iElement = element.begin();
        iElement != element.end(); ++iElement ) {
    const MappingElement& subElement = iElement->second;
    if ( ! this->processSubElement( element, subElement, evolve ) ) return false;
  }
  return true;
}

bool
ora::MappingToSchema::processUniqueReferenceElement( const MappingElement& element,
                                                                                            bool evolve )
{
  const std::string& tableName = element.tableName();
  const std::string& columnName0 = element.columnNames()[0];
  const std::string variableType0 = coral::AttributeSpecification::typeNameForId(typeid(std::string));
  const std::string& columnName1 = element.columnNames()[1];
  const std::string variableType1 = coral::AttributeSpecification::typeNameForId(typeid(int));
  if(!processLeafElement(tableName, columnName0, variableType0, evolve)) return false;
  return processLeafElement(tableName, columnName1, variableType1, evolve);
}


bool
ora::MappingToSchema::checkOrSetColumn(coral::ITable& table,
                                                                       const std::string& columnName,
                                                                       const std::string& columnType,
                                                                       bool& exists,
                                                                       bool evolve){
  coral::MessageStream log( "ORA" );
  const std::string& tableName = table.description().name();
  try {
    const coral::IColumn&  column = table.description().columnDescription(columnName);
    if ( ! column.isNotNull() && isTableEmpty( table ) ) {
      if ( ! evolve ) {
        log << coral::Error << "The column named \"" << columnName << "\" in table \"" << tableName
            << "\" has been found without a NOT NULL constraint while processing a mapping element, and schema evolution has not been enabled."
            << coral::MessageStream::endmsg;
        return false;
      } else {
        std::ostringstream message;
        message << "Setting NOT NULL constraint on column \"" << columnName << "\" in table \"" << tableName << "\"";
        this->logOperation( message.str() );
        if(!m_dryRun) table.schemaEditor().setNotNullConstraint( columnName );
      }
    }
  }catch (const coral::InvalidColumnNameException&){
    if ( ! evolve ) {
      log << coral::Error << "A column named \"" << columnName << "\" has not been found in table \""
          << tableName << "\" while processing a mapping element, and automatic schema evolution has not been enabled." << coral::MessageStream::endmsg;
      return false;
    } else {
      std::ostringstream message;
      message << "Inserting column \"" << columnName << "\" of type \""
              << columnType
              <<"\" in table \"" << tableName <<"\"";
      this->logOperation( message.str() );
      if(!m_dryRun) {
        table.schemaEditor().insertColumn( columnName,columnType );
        // an integer type for the identity column by default
      }
      if( isTableEmpty( table )){
        std::ostringstream message;
        message << "Setting NOT NULL constraint on column \"" << columnName <<"\" in table \"" << tableName <<"\"";
        this->logOperation( message.str() );
        if(!m_dryRun) table.schemaEditor().setNotNullConstraint( columnName );
      }
      exists = false;
    }
  }
  return true;
}

bool
ora::MappingToSchema::processObject( const std::string& tableName,
                                     const std::vector< std::string >& columnNames,
                                     bool isDependent,
                                     bool evolve )
{
  coral::MessageStream log( "ORA" );
  // Check if the table exists.
  if ( m_schema.existsTable( tableName ) ) {
    coral::ITable& table = m_schema.tableHandle( tableName );
    
    bool allColumnsExist = true;
    size_t idCol = 0;
    if( isDependent ){
      if(!checkOrSetColumn( table, columnNames[0], coral::AttributeSpecification::typeNameForType<int>(), allColumnsExist, evolve )) return false;
      idCol++;
    }    
    for ( ; idCol < columnNames.size(); idCol++){
      bool exists = true;
      if(!checkOrSetColumn( table, columnNames[idCol], coral::AttributeSpecification::typeNameForType<int>(), exists, evolve )) return false;
      if(!exists) allColumnsExist = false;
    }

    std::vector<std::string> columnsForIndex = columnNames;

    // Reverse the order of the columns
    std::reverse( columnsForIndex.begin(), columnsForIndex.end() );

    if ( ! allColumnsExist ) { // define a primary key or a unique index
      if ( ! table.description().hasPrimaryKey() ) {
        std::ostringstream message;
        message << "Setting PRIMARY KEY constraint on columns ";
        size_t cols = columnsForIndex.size();
        unsigned int i=0;
        while(i<cols) {
          message << "\""<< columnsForIndex[i] << "\"";
          if(i<cols-1) message<<",";
          i++;
        }
        message <<"\" in table \"" << tableName <<"\"";
        this->logOperation( message.str() );
        if(!m_dryRun) table.schemaEditor().setPrimaryKey( columnsForIndex );
      } else {
        std::ostringstream message;
        message << "Creating INDEX \""<<MappingRules::indexNameForIdentity( tableName )<<"\""
            << " on columns ";
        size_t cols = columnsForIndex.size();
        unsigned int i=0;
        while(i<cols) {
          message << "\""<< columnsForIndex[i] << "\"";
          if(i<cols-1) message<<",";
          i++;
        }
        message <<"\" in table \"" << tableName <<"\"";
        this->logOperation( message.str() );
        if(!m_dryRun) table.schemaEditor().createIndex( MappingRules::indexNameForIdentity( tableName ),
                                                        columnsForIndex, true );
      }
    }
    else {
      if ( !( this->existsUniqueIndex( table.description(), columnsForIndex ) ) ) {
        if ( ! evolve ) {
          log << coral::Error << "A proper unique index on columns=[";
          size_t cols = columnsForIndex.size();
          unsigned int i=0;
          while(i<cols) {
            log << "\""<< columnsForIndex[i] << "\"";
            if(i<cols-1) log<<",";
            i++;
          }
          log << "] has not been found in table \""
              << tableName << "\" while processing a mapping element, and schema evolution has not been enabled." << coral::MessageStream::endmsg;
          return false;
        } else {
          if ( ! table.description().hasPrimaryKey() ) {
            std::ostringstream message;
            message << "Setting PRIMARY KEY constraint on columns ";
            size_t cols = columnsForIndex.size();
            unsigned int i=0;
            while(i<cols) {
              message << "\""<< columnsForIndex[i] << "\"";
              if(i<cols-1) message<<",";
              i++;
            }
            message <<"\" in table \"" << tableName <<"\"";
            this->logOperation( message.str() );
            if(!m_dryRun) table.schemaEditor().setPrimaryKey( columnsForIndex );
          } else {
            std::ostringstream message;
            message << "Creating INDEX \""<<MappingRules::indexNameForIdentity( tableName )<<"\""
                    << " on columns ";
            size_t cols = columnsForIndex.size();
            unsigned int i=0;
            while(i<cols) {
              message << "\""<< columnsForIndex[i] << "\"";
              if(i<cols-1) message<<",";
              i++;
            }
            message <<"\" in table \"" << tableName <<"\"";
            this->logOperation( message.str() );
            if(!m_dryRun) table.schemaEditor().createIndex( MappingRules::indexNameForIdentity( tableName ),
                                                            columnsForIndex, true );
          }
        }
      }
    }
  }
  else if ( m_schema.existsView( tableName ) ) { // Check if a corresponding view exists instead.

    coral::IView& view = m_schema.viewHandle( tableName );
    std::map< std::string, int > columnNameToIndex;
    int numberOfColumns = view.numberOfColumns();
    for ( int iColumn = 0; iColumn < numberOfColumns; ++iColumn ) {
      const coral::IColumn& column = view.column( iColumn );
      columnNameToIndex.insert( std::make_pair( column.name(), iColumn ) );
    }
    for ( std::vector< std::string >::const_iterator iColumnName = columnNames.begin();
          iColumnName != columnNames.end(); ++iColumnName ) {
      std::map< std::string, int >::const_iterator iColumnIndex = columnNameToIndex.find( *iColumnName );
      coral::MessageStream log( "ORA" );
      if ( iColumnIndex == columnNameToIndex.end() ) {
        log << coral::Error << "A column named \"" << *iColumnName << "\" has not been found in view \""
            << tableName << "\" while processing a mapping element." << coral::MessageStream::endmsg;
        return false;
      } else {
        if ( ! view.column( iColumnIndex->second ).isNotNull() ) {
          log << coral::Error << "The column named \"" << *iColumnName << "\" in view \"" << tableName
              << "\" has been found without a NOT NULL constraint while processing a mapping element."
              << coral::MessageStream::endmsg;
          return false;
        }
      }
    }

    // No check for index in this case...

  }
  else {
    coral::TableDescription& description = m_tableDescriptionStack->description( tableName );
    bool first = true;
    for ( std::vector< std::string >::const_iterator iColumnName = columnNames.begin();
          iColumnName != columnNames.end(); ++iColumnName ) {
      try
      {
        description.columnDescription(*iColumnName);
      }
      catch(const coral::InvalidColumnNameException&)
      {
        // an integer type for the identity column by default
        std::string columnType = coral::AttributeSpecification::typeNameForType<int>();
        description.insertColumn( *iColumnName,columnType ); 
        description.setNotNullConstraint( *iColumnName );
      }
      first = false;
    }

    std::vector<std::string> columnsForIndex = columnNames;
    
    // Reverse the order of the columns
    std::reverse( columnsForIndex.begin(), columnsForIndex.end() );

    if ( !( this->existsUniqueIndex( description, columnsForIndex ) ) ) {
      if ( ! description.hasPrimaryKey() ) {
        description.setPrimaryKey( columnsForIndex );
      }
      else {
        description.createIndex( MappingRules::indexNameForIdentity( tableName ),
                                 columnsForIndex, true );
      }
    }
  }

  return true;
}

bool
ora::MappingToSchema::processDependentObject( const std::string& tableName,
                                              const std::vector< std::string >& columnNames,
                                              const std::string& parentTableName,
                                              const std::vector< std::string >& parentColumnNames,
                                              bool evolve )
{
  coral::MessageStream log( "ORA" );
  if(columnNames.size()<2) {
    log << coral::Error << "Cannot process dependent object: number of columns provided is="<<columnNames.size()<<std::endl;
    return false;
  }

  if(!processObject( tableName, columnNames, true, evolve )) return false;

  // Make sure that there is a foreign key constraint id columns
  std::vector< std::string > idColumns;
  // the Id col only
  idColumns.push_back( columnNames[0] );
  
  std::vector< std::string > parentIdColumns( parentColumnNames );
  return this->checkOrSetForeignKey( tableName, idColumns, parentTableName, parentIdColumns, evolve );
}

bool
ora::MappingToSchema::checkOrSetForeignKey( const std::string& tableName,
                                            const std::vector< std::string >& columnNames,
                                            const std::string& referencedTable,
                                            const std::vector< std::string >& referencedColumnNames,
                                            bool evolve )
{
  coral::MessageStream log( "ORA" );
  std::string fkName = MappingRules::fkNameForIdentity( tableName );
  if ( m_schema.existsTable( tableName ) ) {
    if( !m_schema.existsTable( referencedTable )){
      // the processing has to be re-run after the creation of all of the tables
      m_mappingToProcess = true;
      return true;
    }
    
    const coral::ITableDescription& description = m_schema.tableHandle( tableName ).description();
    
    if ( this->existsForeignKey( description, columnNames, referencedTable, referencedColumnNames ) ) return true;

    if ( ! evolve ) {
      log << coral::Error
          << "A proper foreign key has not been found in table \"" << tableName
          << "\" referencing table \"" << referencedTable
          << "\" while processing a mapping element, and schema evolution has not been enabled." << coral::MessageStream::endmsg;
      return false;
    }
    
    // examine the existing foreign keys to drop the current ones
    int numberOfForeignKeys = description.numberOfForeignKeys();
    std::set<std::string> fkeys;
    for ( int iForeignKey = 0; iForeignKey < numberOfForeignKeys; ++iForeignKey ) {
      const coral::IForeignKey& foreignKey = description.foreignKey( iForeignKey );
      const std::vector< std::string >& fkColumns = foreignKey.columnNames();
      for(std::vector<std::string>::const_iterator ifk=fkColumns.begin();
            ifk!=fkColumns.end();++ifk){
        bool found = false;
        for(std::vector<std::string>::const_iterator iC=columnNames.begin();
              iC!=columnNames.end()  && !found ;++iC){
          if(*ifk==*iC) found = true;
        }
        if(found) fkeys.insert(foreignKey.name());
      }
      
    }
    for(std::set<std::string>::const_iterator ifk=fkeys.begin();
        ifk!=fkeys.end();++ifk){
      std::ostringstream message;
      message << "Dropping FOREIGN KEY constraint \""<<*ifk<<"\"";
      this->logOperation( message.str() );
      if(!m_dryRun) m_schema.tableHandle( tableName ).schemaEditor().dropForeignKey( *ifk );
    }
    

  try{
      std::ostringstream message;
      message << "Setting FOREIGN KEY constraint \""<<fkName<<"\" on columns ";
      size_t cols = columnNames.size();
      unsigned int i=0;
      while(i<cols) {
        message << "\""<< columnNames[i] << "\"";
        if(i<cols-1) message<<",";
        i++;
      }
      message <<"\" in table \"" << tableName <<"\""
              << " referencing columns ";
      cols = referencedColumnNames.size();
      i=0;
      while(i<cols) {
        message << "\""<< referencedColumnNames[i] << "\"";
        if(i<cols-1) message<<",";
        i++;
      }
      message <<"\" in table \"" << referencedTable << "\"";
      this->logOperation( message.str() );
      if(!m_dryRun) m_schema.tableHandle( tableName ).schemaEditor().createForeignKey( fkName, columnNames, referencedTable, referencedColumnNames );
      return true;
  }catch (const coral::InvalidForeignKeyIdentifierException& e){
      log << coral::Error << e.what()<<std::endl;
      return false;
    }
  }
  else if ( m_schema.existsView( tableName ) ) {
    return true;
  }
  else {
    if ( m_schema.existsView( referencedTable ) )
      return true;
    coral::TableDescription& description = m_tableDescriptionStack->description( tableName );
    if ( this->existsForeignKey( description, columnNames, referencedTable, referencedColumnNames ) ) {
      return true;
    }
    else {
      try
      {
        description.createForeignKey( fkName, columnNames, referencedTable, referencedColumnNames );
        return true;
      }
      catch (const coral::InvalidForeignKeyIdentifierException& e)
      {
        log << coral::Error << e.what()<<std::endl;
        return false;
      }
    }
  }
}

bool
ora::MappingToSchema::existsForeignKey( const coral::ITableDescription& description,
                                                                        const std::vector< std::string >& columnNames,
                                                                        const std::string& referencedTableName,
                                                                        const std::vector< std::string >& referencedColumnNames )
{
  int numberOfForeignKeys = description.numberOfForeignKeys();
  for ( int iForeignKey = 0; iForeignKey < numberOfForeignKeys; ++iForeignKey ) {
    const coral::IForeignKey& foreignKey = description.foreignKey( iForeignKey );
    if ( foreignKey.referencedTableName() == referencedTableName ) {
      const std::vector< std::string >& fkColumns = foreignKey.columnNames();
      const std::vector< std::string >& fkReferencedColumns = foreignKey.referencedColumnNames();
      if ( columnNames.size() == fkColumns.size() && referencedColumnNames.size() == fkReferencedColumns.size() ) {
        bool same = false;
        for ( size_t i = 0; i < columnNames.size(); ++i ) {
          same = false;
          for( size_t j = 0; j < fkColumns.size() && !same; j++) {
            if ( columnNames[i] == fkColumns[i] ) {
              same = true;
            }
          }
          if(!same) break;
        }
        if ( !same ) continue;
        for ( size_t i = 0; i < referencedColumnNames.size(); ++i ) {
          same = false;
          for( size_t j = 0; j <fkReferencedColumns.size() && !same; j++) {
            if ( referencedColumnNames[i] == fkReferencedColumns[i] ) {
              same = true;
            }
          }
          if(!same) break;
        }
        if ( same ) {
          return true;
        }
      }
    }
  }
  return false;
}

void
ora::MappingToSchema::logOperation(const std::string& message){
  coral::MessageStream log( "ORA" );
  if(!m_dryRun) {
    log << coral::Warning << message << coral::MessageStream::endmsg;
  } else {
    m_operationList.insert( message );
  }
}

