#include "MappingTree.h"
#include "CondCore/ORA/interface/Exception.h"
// externals
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Blob.h"

ora::MappingTree::MappingTree():
  m_version( "" ),
  m_element(),
  m_parentTable(){
}

ora::MappingTree::MappingTree( const std::string& version ):
  m_version( version ),
  m_element(),
  m_parentTable(){
}

ora::MappingTree::MappingTree( const MappingTree& rhs ):
  m_version( rhs.m_version ),
  m_element( rhs.m_element ),
  m_parentTable(){
  if( rhs.m_parentTable.get()) m_parentTable.reset( new TableInfo( *rhs.m_parentTable ) );
}

ora::MappingTree::~MappingTree(){
}

ora::MappingTree& ora::MappingTree::operator=( const MappingTree& rhs ){
  if( this != &rhs ){
    m_version = rhs.m_version;
    m_element = rhs.m_element;
    m_parentTable.reset();
    if( rhs.m_parentTable.get()) m_parentTable.reset( new TableInfo( *rhs.m_parentTable ) );
  }
  return *this;
}

ora::MappingElement&
ora::MappingTree::setTopElement( const std::string& className,
                                 const std::string& tableName,
                                 bool isDependency  ){
  std::string elementType = ora::MappingElement::classMappingElementType();
  if( isDependency ) elementType = ora::MappingElement::dependencyMappingElementType();
  m_element = ora::MappingElement( elementType,
                                   className,
                                   className,
                                   tableName );
  return m_element;
}

void ora::MappingTree::setDependency( const MappingTree& parentTree ){
  m_parentTable.reset( new TableInfo() );
  m_parentTable->m_tableName = parentTree.m_element.tableName();
  m_parentTable->m_idColumns = parentTree.m_element.columnNames();
}

void ora::MappingTree::override(const MappingTree& source)
{
  if( className() == source.className() ) m_element.override( source.m_element );
}

namespace ora {
  void scanElement( const MappingElement& element,
                    const TableInfo& currentTable,
                    bool isDependency,
                    std::map<std::string,TableInfo>& destination ){
    const std::string& tName = element.tableName();
    std::map<std::string,TableInfo>::iterator iT = destination.find( tName );
    if( iT == destination.end() ){
      iT = destination.insert( std::make_pair( tName, TableInfo() ) ).first;
      iT->second.m_dependency = isDependency;
      iT->second.m_tableName = tName;
      iT->second.m_idColumns = element.columnNames();
      iT->second.m_parentTableName = currentTable.m_tableName;
      iT->second.m_refColumns = currentTable.m_idColumns;
    } else {
      const std::vector<std::string>& dataCols = element.columnNames();
      MappingElement::ElementType elementType = element.elementType();
      switch ( elementType ) {
        case MappingElement::Primitive :
          iT->second.m_dataColumns.insert( std::make_pair( dataCols[0],
                                                           element.variableType() ));
          break;
        case MappingElement::Blob :
          iT->second.m_dataColumns.insert( std::make_pair( dataCols[0],
                                                           coral::AttributeSpecification::typeNameForId( typeid(coral::Blob) ) ));
          break;
        case MappingElement::OraReference :
          iT->second.m_dataColumns.insert( std::make_pair( dataCols[0],
                                                           coral::AttributeSpecification::typeNameForId( typeid(int) ) ));
          iT->second.m_dataColumns.insert( std::make_pair( dataCols[1],
                                                           coral::AttributeSpecification::typeNameForId( typeid(int) ) ));
          break;
        case MappingElement::UniqueReference :
          iT->second.m_dataColumns.insert( std::make_pair( dataCols[0],
                                                           coral::AttributeSpecification::typeNameForId( typeid(std::string) ) ));
          iT->second.m_dataColumns.insert( std::make_pair( dataCols[1],
                                                           coral::AttributeSpecification::typeNameForId( typeid(int) ) ));
          break;
        default:
          break;
      }
    }
    TableInfo currT = iT->second;
    for( MappingElement::const_iterator iM = element.begin();
         iM != element.end(); ++iM ){
      scanElement( iM->second, currT, isDependency, destination );
    }
  }
  
}

std::vector<ora::TableInfo> ora::MappingTree::tables() const {
  std::map<std::string,TableInfo> tableMap;
  TableInfo mainTable;
  bool isDependency = false;
  if( m_parentTable.get() ){
    isDependency = true;
    mainTable = *m_parentTable;
  }
  scanElement( m_element, mainTable, isDependency, tableMap );
  std::vector<TableInfo> ret;
  for( std::map<std::string,TableInfo>::const_iterator iT = tableMap.begin();
       iT != tableMap.end(); ++iT ){
    ret.push_back( iT->second );
  }
  return ret;
}


