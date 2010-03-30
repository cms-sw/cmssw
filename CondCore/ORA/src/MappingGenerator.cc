#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Reference.h"
#include "MappingGenerator.h"
#include "MappingRules.h"
#include "TableNameRationalizer.h"
#include "MappingTree.h"
#include "ClassUtils.h"
//
#include <sstream>
// externals
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/MessageStream.h"
#include "Reflex/Reflex.h"

namespace ora {
  struct ArrayMember{
      ArrayMember(const std::string& n, const std::string nForSchema,
                  const Reflex::Type& memberType, bool inBlob):
        name(n),nameForSchema(nForSchema),type(memberType),mappingInBlob(inBlob){
      }
      ArrayMember( const ArrayMember& rhs):
        name(rhs.name),nameForSchema(rhs.nameForSchema),type(rhs.type),mappingInBlob(rhs.mappingInBlob){
      }
      ArrayMember& operator=( const ArrayMember& rhs)
      {
        if(&rhs != this){
          name = rhs.name;
          nameForSchema = rhs.nameForSchema;
          type = rhs.type;
          mappingInBlob = rhs.mappingInBlob;
        }
        return *this;
      }
      std::string name;
      std::string nameForSchema;
      Reflex::Type type;
      bool mappingInBlob;
  };
  
      // process Base class
    bool processBaseClasses( MappingElement& mappingElement,
                             const Reflex::Type& classDictionary,
                             std::vector<ArrayMember>& carrays);

}

ora::MappingGenerator::MappingGenerator( coral::ISchema& schema ):
  m_schema( schema ),
  m_tableRegister(){
}

ora::MappingGenerator::~MappingGenerator(){}

bool ora::MappingGenerator::createNewMapping( const std::string& containerName,
                                              const Reflex::Type& classDictionary,
                                              MappingTree& destination ){
  bool ret = processClass( containerName, classDictionary, destination );
  if( ret ){
    TableNameRationalizer rationalizer( m_schema );
    rationalizer.rationalizeMappingElement( destination.topElement() );
  }
  return ret;
}

bool ora::MappingGenerator::createNewMapping( const std::string& containerName,
                                              const Reflex::Type& classDictionary,
                                              MappingTree& baseMapping,
                                              MappingTree& destination ){
  bool ret = processClass( containerName, classDictionary, destination );
  if( ret ){
    TableNameRationalizer rationalizer( m_schema );
    rationalizer.rationalizeMappingElement( destination.topElement() );
    if(baseMapping.className()!=destination.className()){
      throwException( "Mapping specified as base does not map the target class \"" + destination.className()+"\"",
                      "MappingGenerator::createNewMapping" );
    }
    destination.override( baseMapping );
  }
  return ret;
}

bool ora::MappingGenerator::createNewDependentMapping( const Reflex::Type& dependentClassDictionary,
                                                       MappingTree& parentClassMapping,
                                                       MappingTree& destination ){
  bool ret = processDependentClass( dependentClassDictionary, parentClassMapping, destination );
  if( ret ){
    TableNameRationalizer rationalizer( m_schema );
    rationalizer.rationalizeMappingElement( destination.topElement() );
  }
  return ret;
}

bool ora::MappingGenerator::createNewDependentMapping( const Reflex::Type& dependentClassDictionary,
                                                       MappingTree& parentClassMapping,
                                                       MappingTree& dependentClassBaseMapping,
                                                       MappingTree& destination ){
  bool ret = processDependentClass( dependentClassDictionary, parentClassMapping, destination );
  if( ret ){
    TableNameRationalizer rationalizer( m_schema );
    rationalizer.rationalizeMappingElement( destination.topElement() );
    if(dependentClassBaseMapping.className()!=destination.className()){
      throwException( "Mapping specified as base does not map the target class \"" + destination.className()+"\"",
                      "MappingGenerator::createNewDependentMapping" );
    }
    destination.override( dependentClassBaseMapping );
  }
  return ret;
}

bool
ora::MappingGenerator::processClass( const std::string& containerName,
                                     const Reflex::Type& classDictionary,
                                     MappingTree& destination ){
  std::string className = classDictionary.Name(Reflex::SCOPED);

  size_t sz = sizeInColumns( classDictionary );
  if(sz > MappingRules::MaxColumnsPerTable){
    std::stringstream messg;
    messg << "Cannot process default mapping for class \"" << className+"\"";
    messg << " : number of columns ("<<sz<<") required exceedes maximum="<< MappingRules::MaxColumnsPerTable;
    throwException( messg.str(),"MappingGenerator::processClass");
  }

  std::string tableName = ora::MappingRules::tableNameForClass( containerName );
  if(m_tableRegister.checkTable(tableName)){
    throwException( "Table \"" +tableName+ "\" already assigned, cannot be allocated for the mapping of class \""+className+"\"",
                    "MappingGenerator::processClass");
  }
  m_tableRegister.insertTable(tableName);
  
  // Define the top level element
  MappingElement& topElement = destination.setTopElement( className, tableName );
  topElement.setColumnNames( std::vector< std::string >( 1, ora::MappingRules::columnNameForId() ) );
  return this->processItem( topElement,
                            className,
                            className,
                            classDictionary,
                            false );
}

bool
ora::MappingGenerator::processDependentClass( const Reflex::Type& classDictionary,
                                              MappingTree& parentClassMapping,
                                              MappingTree& destination )
{
  std::string className = classDictionary.Name(Reflex::SCOPED);
  
  size_t sz = sizeInColumns( classDictionary );
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

  ora::MappingElement& classElement = destination.setTopElement( className, tableName, true );
  classElement.setDependency( parentClassMapping.topElement() );
  // Set the id of the class
  std::vector<std::string> columns;
  columns.push_back( ora::MappingRules::columnNameForId() );
  columns.push_back( ora::MappingRules::columnNameForRefColumn() );
  classElement.setColumnNames( columns );
  return this->processItem( classElement,
                            className,
                            className,
                            classDictionary,
                            false );
}

bool
ora::MappingGenerator::buildCArrayElementTree( MappingElement& topElement ){
  if( topElement.elementType()!=MappingElement::CArray ){
    return false;
  }
  Reflex::Type elementType = Reflex::Type::ByName(topElement.variableType());
  elementType = ClassUtils::resolvedType( elementType );
  Reflex::Type arrayElementType = elementType.ToType();
  std::string contentTypeName = arrayElementType.Name();
  std::string variableNameForSchema = ora::MappingRules::variableNameForArrayColumn( arrayElementType );
  return processItem( topElement, contentTypeName, variableNameForSchema, arrayElementType, false );
}

bool
ora::MappingGenerator::processPrimitive( ora::MappingElement& parentElement,
                                         const std::string& attributeName,
                                         const std::string& attributeNameForSchema,
                                         const Reflex::Type& attributeType )
{
  const std::type_info* attrType = &attributeType.TypeInfo();
  if(attributeType.IsEnum()) attrType = &typeid(int);
  if(ClassUtils::isTypeString( attributeType )) attrType = &typeid(std::string);
  std::string typeName = coral::AttributeSpecification::typeNameForId(*attrType);

  return processLeafElement(ora::MappingElement::primitiveMappingElementType(),
                            parentElement,
                            attributeName,
                            attributeNameForSchema,
                            typeName);
}

bool
ora::MappingGenerator::processBlob( ora::MappingElement& parentElement,
                                    const std::string& attributeName,
                                    const std::string& attributeNameForSchema,
                                    const Reflex::Type& attributeType )
{
  std::string className = attributeType.Name(Reflex::SCOPED);
  return processLeafElement(ora::MappingElement::blobMappingElementType(),
                            parentElement,
                            attributeName,
                            attributeNameForSchema,
                            className);
}

bool
ora::MappingGenerator::processLeafElement( const std::string& elementType,
                                           MappingElement& parentelement,
                                           const std::string& attributeName,
                                           const std::string& attributeNameForSchema,
                                           const std::string& typeName ){
  if(parentelement.find( attributeName )!=parentelement.end()){
    throwException("Attribute name \""+attributeName+"\" is already defined in the mapping element of variable \""+parentelement.variableName()+"\".","MappingGenerator::processLeafElement");
  }
  
  if(!m_tableRegister.checkTable( parentelement.tableName())){
    throwException("Table \""+parentelement.tableName()+"\" has not been allocated.",
                   "MappingGenerator::processLeafElement");
  }
  ora::MappingElement& me = parentelement.appendSubElement( elementType, attributeName, typeName, parentelement.tableName() );
  std::string inputCol = ora::MappingRules::columnNameForVariable( attributeNameForSchema, parentelement.variableNameForColumn() );
  std::string columnName(inputCol);
  unsigned int i=0;
  while(m_tableRegister.checkColumn(parentelement.tableName(),columnName)){
    columnName = ora::MappingRules::newNameForSchemaObject( inputCol, i, ora::MappingRules::MaxColumnNameLength );
    i++;
  }
  m_tableRegister.insertColumn(parentelement.tableName(),columnName);
  me.setColumnNames( std::vector< std::string >( 1, columnName ) );

  return true;  
}

bool 
ora::MappingGenerator::processOraReference( ora::MappingElement& parentelement,
                                            const std::string& attributeName,
                                            const std::string& attributeNameForSchema,
                                            const Reflex::Type& attributeType )
{
  std::string className = attributeType.Name(Reflex::SCOPED);
  std::string elementType = ora::MappingElement::OraReferenceMappingElementType();
  ora::MappingElement& me = parentelement.appendSubElement( elementType, attributeName, className, parentelement.tableName() );
  me.setVariableNameForSchema(parentelement.variableNameForSchema(),attributeNameForSchema, false);

  std::vector<std::string> cols;
  for(unsigned int j=0;j<2;j++){
    std::string inputCol = ora::MappingRules::columnNameForOID( attributeNameForSchema, parentelement.variableNameForColumn(), j );
    std::string columnName(inputCol);
    unsigned int i=0;
    while(m_tableRegister.checkColumn(parentelement.tableName(),columnName)){
      columnName = ora::MappingRules::newNameForSchemaObject( inputCol, i, ora::MappingRules::MaxColumnNameLength );
      i++;
    }
    m_tableRegister.insertColumn(parentelement.tableName(),columnName);
    cols.push_back( columnName );
  }
  me.setColumnNames( cols );
  return true;
}

bool 
ora::MappingGenerator::processEmbeddedClass( ora::MappingElement& parentelement,
                                             const std::string& attributeName,
                                             const std::string& attributeNameForSchema,
                                             const Reflex::Type& attributeType )
{
  std::string className = attributeType.Name(Reflex::SCOPED);
  std::string elementType = ora::MappingElement::objectMappingElementType();
  ora::MappingElement& me = parentelement.appendSubElement( elementType, attributeName, className, parentelement.tableName() );
  bool flag = (parentelement.elementType()==ora::MappingElement::Object);
  me.setVariableNameForSchema(parentelement.variableNameForSchema(),attributeNameForSchema, flag);

  std::vector< std::string > idColumns = parentelement.columnNames();
  
  me.setColumnNames( idColumns );
  return this->processObject( me, attributeType );
}


bool
ora::MappingGenerator::processBaseClasses( MappingElement& mappingElement,
                                           const Reflex::Type& objType,
                                           std::vector<ArrayMember>& carrays){
  std::string className = objType.Name(Reflex::SCOPED);
  for ( size_t i=0; i< objType.BaseSize(); i++){
    Reflex::Base base = objType.BaseAt(i);
    Reflex::Type baseType = ClassUtils::resolvedType( base.ToType() );
    if(!baseType){
      return false;
    }
    
    if(!processBaseClasses( mappingElement, baseType, carrays )) return false;
    for ( size_t j=0; j< baseType.DataMemberSize(); j++){
      Reflex::Member baseMember = baseType.DataMemberAt(j);
      // Skip the transient and the static ones
      if ( baseMember.IsTransient() || baseMember.IsStatic() ) continue;
      // Retrieve the data member type
      Reflex::Type type = ClassUtils::resolvedType( baseMember.TypeOf() );
      Reflex::Type declaringType = ClassUtils::resolvedType( baseMember.DeclaringType());
      std::string scope = declaringType.Name(Reflex::SCOPED);
      // Retrieve the field name
      std::string objectMemberName = ora::MappingRules::scopedVariableName( baseMember.Name(), scope );
      std::string objectNameForSchema = ora::MappingRules::scopedVariableForSchemaObjects( baseMember.Name(), scope );

      std::string mappingType("");
      Reflex::PropertyList memberProps = baseMember.Properties();
      if( memberProps.HasProperty(ora::MappingRules::mappingPropertyNameInDictionary())){
        mappingType = memberProps.PropertyAsString(ora::MappingRules::mappingPropertyNameInDictionary());
      }
      bool arraysInBlobs = ora::MappingRules::isMappedToBlob( mappingType );
      
      if ( type.IsArray() ){
        carrays.push_back( ArrayMember( objectMemberName, objectNameForSchema, type, arraysInBlobs ) );
      } else {
        if ( ! this->processItem( mappingElement, objectMemberName, objectNameForSchema, type, arraysInBlobs ) ) return false;
      }  
    }
    
  }
  return true;
}

bool
ora::MappingGenerator::processObject( ora::MappingElement& mappingElement,
                                      const Reflex::Type& objType )
{
  std::string className = objType.Name(Reflex::SCOPED);

  // resolve possible typedef chains
  Reflex::Type objectType = ClassUtils::resolvedType(objType);
  std::vector<ArrayMember > carraysMembers;
  // process base class data members
  if(!processBaseClasses( mappingElement, objectType, carraysMembers )) return false;
  // loop over the data members 
  for ( size_t i=0; i< objectType.DataMemberSize(); i++){

    Reflex::Member objectMember = objectType.DataMemberAt(i);
    // Skip the transient and the static ones
    if ( objectMember.IsTransient() || objectMember.IsStatic() ) continue;

    // Retrieve the field type
    Reflex::Type type = ClassUtils::resolvedType( objectMember.TypeOf() );
    // Check for the existence of the dictionary information
    if ( !type ){
      throwException( "Type for data member \""+objectMember.Name()+"\" of class \""+className+
                      "\" has not been found in the dictionary.",
                      "MappingGenerator::processObject");
    }
    
    // check if the member is from a class in the inheritance tree
    std::string scope("");
    Reflex::Type declaringType = ClassUtils::resolvedType( objectMember.DeclaringType());
    if( declaringType != objectType ){
      continue;
    }
    // Retrieve the field name
    std::string objectMemberName = ora::MappingRules::scopedVariableName( objectMember.Name(), scope );
    std::string objectNameForSchema = ora::MappingRules::scopedVariableForSchemaObjects( objectMember.Name(), scope );
    
    std::string mappingType("");
    Reflex::PropertyList memberProps = objectMember.Properties();
    if( memberProps.HasProperty(ora::MappingRules::mappingPropertyNameInDictionary())){
      mappingType = memberProps.PropertyAsString(ora::MappingRules::mappingPropertyNameInDictionary());
    }
    bool arraysInBlobs = ora::MappingRules::isMappedToBlob( mappingType );    
    
    if ( type.IsArray() ){
      carraysMembers.push_back( ArrayMember( objectMemberName, objectNameForSchema, type, arraysInBlobs ) );
    } else {
      if ( ! this->processItem( mappingElement, objectMemberName, objectNameForSchema, type, arraysInBlobs ) ) return false;      
    }  
  }
  
  // C-arrays are processed separately and at the end.
  for( size_t i=0; i< carraysMembers.size(); i++){
    if ( ! this->processItem( mappingElement, carraysMembers[i].name, carraysMembers[i].nameForSchema, carraysMembers[i].type, carraysMembers[i].mappingInBlob ) ) return false; 
  }
  
  return true;
}

bool
ora::MappingGenerator::processItem( ora::MappingElement& parentElement,
                                    const std::string& attributeName,
                                    const std::string& attributeNameForSchema,
                                    const Reflex::Type& attributeType,
                                    bool arraysInBlobs )
{
  Reflex::Type resType = ClassUtils::resolvedType( attributeType );
  
  // primitive and string
  if ( ora::ClassUtils::isTypePrimitive(resType) ) {
    // Correct the string type name
    return this->processPrimitive( parentElement,
                                   attributeName,
                                   attributeNameForSchema,
                                   resType );
  }
  else if ( resType.IsArray() ){
    
    if ( arraysInBlobs ){
      return this->processBlob( parentElement,
                                attributeName,
                                attributeNameForSchema,
                                resType );
    }
    return this->processCArray( parentElement,
                                attributeName,
                                attributeNameForSchema,
                                resType);
  }
  else if ( ora::ClassUtils::isTypeContainer( resType ) ) {

    if ( arraysInBlobs ){
      return this->processBlob( parentElement,
                                attributeName,
                                attributeNameForSchema,
                                resType );
    }
    return this->processArray( parentElement,
                               attributeName,
                               attributeNameForSchema,
                               resType);
  }
  else if ( resType.IsPointer() || resType.IsReference() ){
    return false;
  }
  else if ( ora::ClassUtils::isTypeOraPointer( resType )){
    return this->processOraPtr( parentElement,
                                attributeName,
                                attributeNameForSchema,
                                resType,
                                arraysInBlobs );
  }
  else if ( ora::ClassUtils::isTypeUniqueReference( resType )){
    if( parentElement.isDependentTree() ) {
      return false;
    }
    return this->processUniqueReference( parentElement,
                                         attributeName,
                                         attributeNameForSchema,
                                         resType );
  }
  else if ( resType.TypeInfo() == typeid(ora::Reference) ||
            resType.HasBase( Reflex::Type::ByTypeInfo( typeid(ora::Reference) ) ) ){
    return this->processOraReference( parentElement,
                                      attributeName,
                                      attributeNameForSchema,
                                      attributeType );
  }
  else { // embeddedobject
    return this->processEmbeddedClass( parentElement,
                                       attributeName,
                                       attributeNameForSchema,
                                       attributeType );
  }
}

bool
ora::MappingGenerator::processArray( ora::MappingElement& parentelement,
                                     const std::string& attributeName,
                                     const std::string& attributeNameForSchema,
                                     const Reflex::Type& attributeType ){
  std::string mainTableName = parentelement.mainTableName();
  std::string initialTable(mainTableName);

  std::string arrayTable(initialTable);
  unsigned int i=0;
  while(m_tableRegister.checkTable(arrayTable)){
    arrayTable = ora::MappingRules::newNameForSchemaObject( initialTable, i, ora::MappingRules::MaxTableNameLength );
    i++;
  }
  m_tableRegister.insertTable(arrayTable);

  std::string className = attributeType.Name(Reflex::SCOPED);

  std::string elementType = ora::MappingElement::arrayMappingElementType();
  if(ora::ClassUtils::isTypePVector(attributeType) || ora::ClassUtils::isTypeQueryableVector(attributeType)){
    elementType = ora::MappingElement::OraArrayMappingElementType();
  }
  ora::MappingElement& me = parentelement.appendSubElement( elementType,attributeName,className,arrayTable );
  std::vector<std::string> parentColumns = parentelement.columnNames();
  std::vector<std::string> columns;
  // always comes the oid first
  columns.push_back( ora::MappingRules::columnNameForId() );
  std::vector<std::string>::const_iterator iColumn = parentColumns.begin();
  // then copy the columns except the id...
  iColumn++;
  for ( ;iColumn != parentColumns.end(); ++iColumn ) {
    columns.push_back( ora::MappingRules::columnNameForId() + "_" + *iColumn );
  }
  // and finally add the position!
  columns.push_back( ora::MappingRules::columnNameForPosition() );
  
  me.setColumnNames( columns );
  me.setVariableNameForSchema(parentelement.variableNameForSchema(),attributeNameForSchema);
  
  bool singleItemContainer =  ora::ClassUtils::isTypeNonAssociativeContainer(attributeType);
  bool associativeContainer =  ora::ClassUtils::isTypeAssociativeContainer(attributeType);

  Reflex::Type contentType;
  Reflex::Type keyType;
  
  if( singleItemContainer ){
    contentType = ClassUtils::containerValueType(attributeType);
  }
  else if ( associativeContainer ) { // This is an associative container type
    contentType = ClassUtils::containerDataType( attributeType );
    keyType = ClassUtils::containerKeyType( attributeType );
    if( !keyType || !ClassUtils::resolvedType(keyType) ){
      throwException( "Cannot not resolve the type of the key item of container \""+attributeType.Name(Reflex::SCOPED)+"\".",
                      "MappingGenerator::processArray");
    }
  }
  else {
    // Not supported container
      throwException( "Container type=\""+attributeType.Name(Reflex::SCOPED)+"\".is not supported.",
                      "MappingGenerator::processArray");    
  }

  if( !contentType || !ClassUtils::resolvedType(contentType) ){
      throwException( "Cannot not resolve the type of the content item of container \""+attributeType.Name(Reflex::SCOPED)+"\".",
                      "MappingGenerator::processArray");
  }
  if ( keyType ) {
    std::string keyTypeName = keyType.Name();
    std::string keyTypeNameForSchema = MappingRules::variableNameForContainerKey();
    if ( !( this->processItem( me, keyTypeName, keyTypeNameForSchema, keyType, false ) ) )
      return false;
  }
  std::string contentTypeName = contentType.Name();
  std::string contentTypeNameForSchema = MappingRules::variableNameForContainerValue();
  return this->processItem( me, contentTypeName, contentTypeNameForSchema, contentType, false );
}

bool
ora::MappingGenerator::processCArray( MappingElement& parentelement,
                                      const std::string& attributeName,
                                      const std::string& attributeNameForSchema,
                                      const Reflex::Type& attributeType ){
  
  Reflex::Type arrayElementType = attributeType.ToType();
  if( !arrayElementType || !ClassUtils::resolvedType( arrayElementType ) ){
    throwException("Cannot resolve the type of the content of the array \""+attributeType.Name(Reflex::SCOPED)+"\".",
                   "MappingGenerator::processCArray");
  }

  if(!m_tableRegister.checkTable(parentelement.tableName())){
    throwException("Table \""+parentelement.tableName()+"\" has not been allocated.",
                   "MappingGenerator::processCArray");
  }
  //size_t arraySizeInColumns = sizeInColumns( attributeType, true );
  std::pair<bool,size_t> arraySizeInColumns = sizeInColumnsForCArray( attributeType );
  //size_t arraySizeInColumns = sizeInColumnsForCArray( attributeType );
  if( !arraySizeInColumns.first && arraySizeInColumns.second < MappingRules::MaxColumnsForInlineCArray ) {
    //if( arraySizeInColumns < MappingRules::MaxColumnsForInlineCArray ) {
    //if( arraySizeInColumns < MappingRules::MaxColumnsPerTable ) {
    size_t columnsInTable = m_tableRegister.numberOfColumns(parentelement.tableName()) + arraySizeInColumns.second;
    if( columnsInTable < MappingRules::MaxColumnsPerTable ){
      return processInlineCArrayItem(parentelement,attributeName,attributeNameForSchema,attributeType, arrayElementType);    
    }
  }
  return processCArrayItem(parentelement,attributeName,attributeNameForSchema,attributeType,arrayElementType );
}

bool
ora::MappingGenerator::processCArrayItem( MappingElement& parentelement,
                                          const std::string& attributeName,
                                          const std::string& attributeNameForSchema,
                                          const Reflex::Type& attributeType,
                                          const Reflex::Type& arrayElementType ){
  std::string mainTableName = parentelement.mainTableName();
  std::string initialTable(mainTableName);

  std::string arrayTable(initialTable);
  unsigned int i=0;
  while(m_tableRegister.checkTable(arrayTable)){
    arrayTable = ora::MappingRules::newNameForSchemaObject( initialTable, i, ora::MappingRules::MaxTableNameLength );
    i++;
  }
  m_tableRegister.insertTable(arrayTable);
  
  std::string className = attributeType.Name(Reflex::SCOPED);
  ora::MappingElement& me = parentelement.appendSubElement( ora::MappingElement::CArrayMappingElementType(),
                                                            attributeName, attributeName, arrayTable );
  //attributeName, className, arrayTable );
  std::vector<std::string> parentColumns = parentelement.columnNames();
  //if ( parentelement.elementType() == ora::MappingElement::OraReference ) {
  //  parentColumns.pop_back();
  //  parentColumns.pop_back();
  //}
  std::vector<std::string> columns;
  // always comes the oid first
  columns.push_back( ora::MappingRules::columnNameForId() );
  std::vector<std::string>::const_iterator iColumn = parentColumns.begin();
  // then copy the columns except the id...
  iColumn++;
  for ( ;iColumn != parentColumns.end(); ++iColumn ) {
    columns.push_back( ora::MappingRules::columnNameForId() + "_" + *iColumn );
  }
  // and finally add the position!
  columns.push_back( ora::MappingRules::columnNameForPosition() );

  me.setColumnNames( columns );
  me.setVariableNameForSchema(parentelement.variableNameForSchema(),attributeNameForSchema);

  std::string contentTypeName = arrayElementType.Name();
  std::string variableNameForSchema = attributeName;
  //std::string variableNameForSchema = ora::MappingRules::variableNameForArrayColumn( attributeType );
  return this->processItem( me, contentTypeName, variableNameForSchema, arrayElementType, false );  
}

bool
ora::MappingGenerator::processInlineCArrayItem( MappingElement& parentelement,
                                                const std::string& attributeName,
                                                const std::string& attributeNameForSchema,
                                                const Reflex::Type& attributeType,
                                                const Reflex::Type& arrayElementType ){
  std::string className = attributeType.Name(Reflex::SCOPED);
  std::string mappingElementType = ora::MappingElement::inlineCArrayMappingElementType();
  ora::MappingElement& me = parentelement.appendSubElement( mappingElementType, attributeName, className, parentelement.tableName() );
  std::vector< std::string > idColumns = parentelement.columnNames();
  //if ( parentelement.elementType() == ora::MappingElement::OraReference ) {
  //  idColumns.pop_back();
  //  idColumns.pop_back();
  //}
  me.setColumnNames( idColumns );
  me.setVariableNameForSchema(parentelement.variableNameForSchema(),attributeNameForSchema);
  
  for(size_t i=0;i<attributeType.ArrayLength();i++){
    if ( ! this->processItem( me,
                              MappingRules::variableNameForArrayIndex(attributeName,i),
//                              MappingRules::variableNameForArrayColumn(attributeName,i),
//                              MappingRules::variableNameForArrayIndex(attributeName,i),
                              MappingRules::variableNameForArrayIndex(attributeNameForSchema,i),
                              arrayElementType, false ) ) return false;
  }
  return true;  
}

bool
ora::MappingGenerator::processOraPtr( MappingElement& parentelement,
                                      const std::string& attributeName,
                                      const std::string& attributeNameForSchema,
                                      const Reflex::Type& attributeType,
                                      bool arraysInBlobs){
  if(parentelement.find( attributeName )!=parentelement.end()){
    throwException( "Attribute name \""+attributeName+"\" is already defined in the mapping element of variable \""+
                    parentelement.variableName()+"\".",
                    "MappingGenerator::processOraPtr");
  }

  std::string typeName = attributeType.Name(Reflex::SCOPED);
  ora::MappingElement& me = parentelement.appendSubElement( ora::MappingElement::OraPointerMappingElementType(), attributeName, typeName, parentelement.tableName() );
  me.setColumnNames( parentelement.columnNames() );

  Reflex::Type ptrType = attributeType.TemplateArgumentAt(0);
  std::string ptrTypeName = ptrType.Name();

  return this->processItem( me, ptrTypeName, attributeNameForSchema, ptrType, arraysInBlobs );  
}

bool
ora::MappingGenerator::processUniqueReference( MappingElement& parentelement,
                                               const std::string& attributeName,
                                               const std::string& attributeNameForSchema,
                                               const Reflex::Type& attributeType ){
  
  if(parentelement.find( attributeName )!=parentelement.end()){
    throwException( "Attribute name \""+attributeName+"\" is already defined in the mapping element of variable \""+
                    parentelement.variableName()+"\"",
                    "MappingGenerator::processUniqueReference");
  }
  
  if(!m_tableRegister.checkTable( parentelement.tableName())){
    throwException("Table \""+parentelement.tableName()+"\" has not been allocated.",
                   "MappingGenerator::processUniqueReference");
  }
  std::string typeName = attributeType.Name(Reflex::SCOPED);
  ora::MappingElement& me = parentelement.appendSubElement( ora::MappingElement::uniqueReferenceMappingElementType(), attributeName, typeName, parentelement.tableName() );

  std::vector< std::string > cols;
  std::string inputCol = ora::MappingRules::columnNameForRefMetadata( attributeNameForSchema, parentelement.variableNameForSchema());
  std::string columnName(inputCol);
  unsigned int i=0;
  while(m_tableRegister.checkColumn(parentelement.tableName(),columnName)){
    columnName = ora::MappingRules::newNameForSchemaObject( inputCol, i, ora::MappingRules::MaxColumnNameLength );
    i++;
  }
  m_tableRegister.insertColumn(parentelement.tableName(),columnName);
  cols.push_back(columnName);

  std::string idCol = ora::MappingRules::columnNameForRefId( attributeNameForSchema, parentelement.variableNameForSchema());
  columnName = idCol;
  i=0;
  while(m_tableRegister.checkColumn(parentelement.tableName(),columnName)){
    columnName = ora::MappingRules::newNameForSchemaObject( inputCol, i, ora::MappingRules::MaxColumnNameLength );
    i++;
  }
  m_tableRegister.insertColumn(parentelement.tableName(),columnName);
  cols.push_back(columnName);

  me.setColumnNames( cols );
  return true;
}

ora::TableRegister&
ora::MappingGenerator::tableRegister(){
  return m_tableRegister;
}

size_t
ora::MappingGenerator::sizeInColumns(const Reflex::Type& topLevelClassType ){
  size_t sz=0;
  bool hasDependencies = false;
  _sizeInColumns(topLevelClassType, sz, hasDependencies );
  return sz;
}

std::pair<bool,size_t>
ora::MappingGenerator::sizeInColumnsForCArray(const Reflex::Type& topLevelClassType ){
  size_t sz=0;
  bool hasDependencies = false;
  _sizeInColumnsForCArray(topLevelClassType, sz, hasDependencies );
  return std::make_pair(hasDependencies,sz);
}


void
ora::MappingGenerator::_sizeInColumns(const Reflex::Type& topLevelClassType,
                                      size_t& sz,
                                      bool& hasDependencies ){
  // resolve possible typedef chains
  Reflex::Type typ = ClassUtils::resolvedType( topLevelClassType );
  bool isOraPolyPointer = ora::ClassUtils::isTypeUniqueReference(typ);
  bool isPrimitive = ora::ClassUtils::isTypePrimitive( typ );

  // primitive and string
  if( isPrimitive || isOraPolyPointer ) {
    ++sz;
  } else if (typ.IsArray()){
    size_t arraySize = 0;
    _sizeInColumnsForCArray( typ,arraySize, hasDependencies );
    if( arraySize < MappingRules::MaxColumnsForInlineCArray ) sz += arraySize;
    else hasDependencies = true;
    /**
    size_t arraySize = typ.ArrayLength();
    Reflex::Type arrayType = typ.ToType();
    size_t arrayElementSize = 0;
    _sizeInColumns(arrayType, arrayElementSize);
    size_t totSize = arraySize*arrayElementSize;
    size_t currSize = sz + totSize;
    if(currSize < MaxColumnsPerTable) sz += totSize;
    **/
  } else if (typ.TypeInfo() == typeid(ora::Reference) ||
             typ.HasBase( Reflex::Type::ByTypeInfo( typeid(ora::Reference) ) )){
    sz += 2;
  } else {
  
    bool isContainer =  ora::ClassUtils::isTypeNonAssociativeContainer(typ) ||
      ora::ClassUtils::isTypeAssociativeContainer(typ);
    bool isOraPointer = ora::ClassUtils::isTypeOraPointer(typ);
    if( !isContainer && !isOraPointer ){
      
      // loop over the data members
      typ.UpdateMembers();
      //std::vector<Reflex::Type> carrays;
      for ( size_t i=0; i< typ.DataMemberSize(); i++){
        Reflex::Member objMember = typ.DataMemberAt(i);

        // Skip the transient ones
        if ( objMember.IsTransient() ) continue;

        // Retrieve the field type
        Reflex::Type objMemberType = objMember.TypeOf();

        //if(objMemberType.IsArray()){
        //  carrays.push_back( objMemberType );
        //} else {
        _sizeInColumns(objMemberType,sz, hasDependencies );
          //}
      }
      //for( size_t i=0; i< carrays.size(); i++){
      //  _sizeInColumns( carrays[i], sz );
      //}
      
    } else {
      hasDependencies = true;
    }
    
  }
}

void
ora::MappingGenerator::_sizeInColumnsForCArray(const Reflex::Type& topLevelClassType,
                                               size_t& sz,
                                               bool& hasDependencies){
  // resolve possible typedef chains
  Reflex::Type typ = ClassUtils::resolvedType( topLevelClassType );
  if( !typ.IsArray()){
    return;
  }

  size_t arraySize = typ.ArrayLength();
  Reflex::Type arrayType = typ.ToType();
  size_t arrayElementSize = 0;
  _sizeInColumns(arrayType, arrayElementSize, hasDependencies);
  size_t totSize = arraySize*arrayElementSize;
  sz += totSize;
}
