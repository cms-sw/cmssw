#include "CondCore/ORA/interface/Exception.h"
#include "MappingGenerator.h"
#include "RelationalMapping.h"
#include "MappingTree.h"
#include "MappingRules.h"
// externals
#include "Reflex/Reflex.h"

ora::MappingGenerator::MappingGenerator( coral::ISchema& schema ):
  m_schema( schema ),
  m_tableRegister( schema ){
}

ora::MappingGenerator::~MappingGenerator(){}

void ora::MappingGenerator::createNewMapping( const std::string& containerName,
                                              const Reflex::Type& classDictionary,
                                              MappingTree& destination ){
  std::string className = classDictionary.Name(Reflex::SCOPED);

  size_t sz = RelationalMapping::sizeInColumns( classDictionary );
  if(sz > MappingRules::MaxColumnsPerTable){
    std::stringstream messg;
    messg << "Cannot process default mapping for class \"" << className+"\"";
    messg << " : number of columns ("<<sz<<") required exceedes maximum="<< MappingRules::MaxColumnsPerTable;
    throwException( messg.str(),"MappingGenerator::processClass");
  }

  std::string tableName = ora::MappingRules::tableNameForItem( containerName );
  if(m_tableRegister.checkTable(tableName)){
    throwException( "Table \"" +tableName+ "\" already assigned, cannot be allocated for the mapping of class \""+className+"\"",
                    "MappingGenerator::processClass");
  }
  m_tableRegister.insertTable(tableName);
  // Define the top level element
  MappingElement& topElement = destination.setTopElement( className, tableName );
  topElement.setColumnNames( std::vector< std::string >( 1, ora::MappingRules::columnNameForId() ) );
  m_tableRegister.insertColumns(tableName, topElement.columnNames() );
  RelationalMappingFactory mappingFactory( m_tableRegister );
  std::auto_ptr<IRelationalMapping> processor( mappingFactory.newProcessor( classDictionary ) );
  std::string nameForSchema = MappingRules::formatName( className, MappingRules::ClassNameLengthForSchema );
  std::string scope("");
  processor->process( topElement, className, nameForSchema, scope   );
}

void ora::MappingGenerator::createNewMapping( const std::string& containerName,
                                              const Reflex::Type& classDictionary,
                                              const MappingTree& baseMapping,
                                              MappingTree& destination ){
  createNewMapping( containerName, classDictionary, destination );
  if(baseMapping.className()!=destination.className()){
    throwException( "Mapping specified as base does not map the target class \"" + destination.className()+"\"",
                    "MappingGenerator::createNewMapping" );
  }
  destination.override( baseMapping );
}

void ora::MappingGenerator::createNewDependentMapping( const Reflex::Type& classDictionary,
                                                       const MappingTree& parentClassMapping,
                                                       MappingTree& destination ){
  std::string className = classDictionary.Name(Reflex::SCOPED);
  
  size_t sz = RelationalMapping::sizeInColumns( classDictionary );
  if(sz > MappingRules::MaxColumnsPerTable){
    std::stringstream messg;
    messg << "Cannot process default mapping for class \"" << className+"\"";
    messg << " : number of columns ("<<sz<<") required exceedes maximum="<< MappingRules::MaxColumnsPerTable;
    throwException( messg.str(),"MappingGenerator::processClass");
  }
  
  std::string mainTableName = parentClassMapping.topElement().tableName();
  std::string initialTable = mainTableName;
  std::string tableName = ora::MappingRules::newNameForDepSchemaObject( initialTable, 0, ora::MappingRules::MaxTableNameLength );
  unsigned int i=0;
  while(m_tableRegister.checkTable(tableName)){
    tableName = ora::MappingRules::newNameForDepSchemaObject( initialTable, i, ora::MappingRules::MaxTableNameLength );
    i++;
  }
  m_tableRegister.insertTable(tableName);

  destination.setDependency( parentClassMapping );
  ora::MappingElement& classElement = destination.setTopElement( className, tableName, true );
  // Set the id of the class
  std::vector<std::string> columns;
  columns.push_back( ora::MappingRules::columnNameForId() );
  columns.push_back( ora::MappingRules::columnNameForRefColumn() );
  classElement.setColumnNames( columns );
  m_tableRegister.insertColumns(tableName, columns );
  RelationalMappingFactory mappingFactory( m_tableRegister );
  std::auto_ptr<IRelationalMapping> processor( mappingFactory.newProcessor( classDictionary ) );
  std::string nameForSchema = MappingRules::formatName( className, MappingRules::ClassNameLengthForSchema );
  std::string scope("");
  processor->process( classElement,  className, nameForSchema, scope );
}

void ora::MappingGenerator::createNewDependentMapping( const Reflex::Type& classDictionary,
                                                       const MappingTree& parentClassMapping,
                                                       const MappingTree& dependentClassBaseMapping,
                                                       MappingTree& destination ){
  createNewDependentMapping( classDictionary, parentClassMapping, destination );
  if(dependentClassBaseMapping.className()!=destination.className()){
    throwException( "Mapping specified as base does not map the target class \"" + destination.className()+"\"",
                    "MappingGenerator::createNewDependentMapping" );
  }
  destination.override( dependentClassBaseMapping );
}

