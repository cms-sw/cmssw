#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Reference.h"
#include "CondCore/ORA/interface/NamedRef.h"
#include "RelationalMapping.h"
#include "TableRegister.h"
#include "MappingElement.h"
#include "MappingRules.h"
#include "ClassUtils.h"
// externals 
#include "Reflex/Reflex.h"
#include "CoralBase/AttributeSpecification.h"

size_t
ora::RelationalMapping::sizeInColumns(const Reflex::Type& topLevelClassType ){
  size_t sz=0;
  bool hasDependencies = false;
  _sizeInColumns(topLevelClassType, sz, hasDependencies );
  return sz;
}

std::pair<bool,size_t>
ora::RelationalMapping::sizeInColumnsForCArray(const Reflex::Type& topLevelClassType ){
  size_t sz=0;
  bool hasDependencies = false;
  _sizeInColumnsForCArray(topLevelClassType, sz, hasDependencies );
  return std::make_pair(hasDependencies,sz);
}


void
ora::RelationalMapping::_sizeInColumns(const Reflex::Type& topLevelClassType,
                                       size_t& sz,
                                       bool& hasDependencies ){
  // resolve possible typedef chains
  Reflex::Type typ = ClassUtils::resolvedType( topLevelClassType );
  bool isOraPolyPointer = ora::ClassUtils::isTypeUniqueReference(typ);
  bool isPrimitive = ora::ClassUtils::isTypePrimitive( typ );

  // primitive and string
  if( isPrimitive || isOraPolyPointer || ora::ClassUtils::isTypeNamedReference( typ)) {
    ++sz;
  } else if (typ.IsArray()){
    size_t arraySize = 0;
    _sizeInColumnsForCArray( typ,arraySize, hasDependencies );
    if( arraySize < MappingRules::MaxColumnsForInlineCArray ) sz += arraySize;
    else hasDependencies = true;
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

        _sizeInColumns(objMemberType,sz, hasDependencies );
      }
    } else {
      hasDependencies = true;
    }    
  }
}

void
ora::RelationalMapping::_sizeInColumnsForCArray(const Reflex::Type& topLevelClassType,
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


ora::RelationalMappingFactory::RelationalMappingFactory( TableRegister& tableRegister ):
  m_tableRegister( tableRegister ){
}

ora::RelationalMappingFactory::~RelationalMappingFactory(){
}

ora::IRelationalMapping* ora::RelationalMappingFactory::newProcessor( const Reflex::Type& attributeType,
                                                                      bool blobStreaming ){
  if( blobStreaming ){
    return new BlobMapping( attributeType, m_tableRegister );
  }
  Reflex::Type resType = ClassUtils::resolvedType( attributeType );
  if ( ora::ClassUtils::isTypePrimitive(resType) ) {
    return new PrimitiveMapping( attributeType, m_tableRegister );
  }
  else if ( resType.IsArray() ){
    return new CArrayMapping( attributeType, m_tableRegister );
  }
  else if ( ora::ClassUtils::isTypeContainer( resType ) ) {
    return new ArrayMapping( attributeType, m_tableRegister );
  }
  else if ( resType.IsPointer() || resType.IsReference() ){
    return new EmptyMapping();
  }
  else if ( ora::ClassUtils::isTypeOraPointer( resType )){
    return new OraPtrMapping( attributeType, m_tableRegister );
  }
  else if ( ora::ClassUtils::isTypeUniqueReference( resType )){
    return new UniqueReferenceMapping( attributeType, m_tableRegister );
  }
  else if ( resType.TypeInfo() == typeid(ora::Reference) ||
            resType.HasBase( Reflex::Type::ByTypeInfo( typeid(ora::Reference) ) ) ){
    return new OraReferenceMapping( attributeType, m_tableRegister );
  }
  else if ( resType.TypeInfo() == typeid(ora::NamedReference) ||
            resType.HasBase( Reflex::Type::ByTypeInfo( typeid(ora::NamedReference) ) ) ){
    return new NamedRefMapping( attributeType, m_tableRegister );
  }
  else { // embeddedobject
    return new ObjectMapping( attributeType, m_tableRegister );
  } 
}

namespace ora {
  void processLeafElement( MappingElement& parentElement,
                           const std::string& elementType,
                           const std::string& typeName,
                           const std::string& attributeName,
                           const std::string& attributeNameForSchema,
                           const std::string& scopeNameForSchema,
                           TableRegister& tableRegister){
    if(!tableRegister.checkTable( parentElement.tableName())){
      throwException("Table \""+parentElement.tableName()+"\" has not been allocated.",
                     "processLeafElement");
    }
    ora::MappingElement& me = parentElement.appendSubElement( elementType, attributeName, typeName, parentElement.tableName() );
    std::string inputCol = ora::MappingRules::columnNameForVariable( attributeNameForSchema, scopeNameForSchema );
    std::string columnName(inputCol);
    unsigned int i=0;
    while(tableRegister.checkColumn(parentElement.tableName(),columnName)){
      columnName = ora::MappingRules::newNameForSchemaObject( inputCol, i, ora::MappingRules::MaxColumnNameLength );
      i++;
    }
    tableRegister.insertColumn(parentElement.tableName(),columnName);
    me.setColumnNames( std::vector< std::string >( 1, columnName ) );
  }
}

ora::PrimitiveMapping::PrimitiveMapping( const Reflex::Type& attributeType, TableRegister& tableRegister ):
  m_type(attributeType),m_tableRegister( tableRegister ){
}

ora::PrimitiveMapping::~PrimitiveMapping(){
}

void ora::PrimitiveMapping::process( MappingElement& parentElement,
                                     const std::string& attributeName,
                                     const std::string& attributeNameForSchema,
                                     const std::string& scopeNameForSchema ){
  Reflex::Type t = ClassUtils::resolvedType( m_type );
  const std::type_info* attrType = &t.TypeInfo();
  if(t.IsEnum()) attrType = &typeid(int);
  //std::string tn = ClassUtils::demangledName(*attrType);
  if(ClassUtils::isTypeString( t )) attrType = &typeid(std::string);
  std::string typeName = coral::AttributeSpecification::typeNameForId(*attrType);
  
  processLeafElement(parentElement,
                     ora::MappingElement::primitiveMappingElementType(),
                     typeName,
                     attributeName,
                     attributeNameForSchema,
                     scopeNameForSchema,
                     m_tableRegister);  
}

ora::BlobMapping::BlobMapping( const Reflex::Type& attributeType, TableRegister& tableRegister ):
  m_type(attributeType),m_tableRegister( tableRegister ){
}

ora::BlobMapping::~BlobMapping(){
}
void ora::BlobMapping::process( MappingElement& parentElement,
                                const std::string& attributeName,
                                const std::string& attributeNameForSchema,
                                const std::string& scopeNameForSchema ){
  std::string className = m_type.Name(Reflex::SCOPED);
  processLeafElement(parentElement,
                     ora::MappingElement::blobMappingElementType(),
                     className,
                     attributeName,
                     attributeNameForSchema,
                     scopeNameForSchema,
                     m_tableRegister);
}

ora::OraReferenceMapping::OraReferenceMapping( const Reflex::Type& attributeType, TableRegister& tableRegister ):
  m_type(attributeType),m_tableRegister( tableRegister ){
}

ora::OraReferenceMapping::~OraReferenceMapping(){
}

void ora::OraReferenceMapping::process( MappingElement& parentElement,
                                        const std::string& attributeName,
                                        const std::string& attributeNameForSchema,
                                        const std::string& scopeNameForSchema ){
  std::string className = m_type.Name(Reflex::SCOPED);
  std::string elementType = ora::MappingElement::OraReferenceMappingElementType();
  if(!m_tableRegister.checkTable( parentElement.tableName())){
    throwException("Table \""+parentElement.tableName()+"\" has not been allocated.",
                   "OraReferenceMapping::process");
  }
  ora::MappingElement& me = parentElement.appendSubElement( elementType, attributeName, className, parentElement.tableName() );

  std::vector<std::string> cols;
  for(unsigned int j=0;j<2;j++){
    std::string inputCol = ora::MappingRules::columnNameForOID( attributeNameForSchema, scopeNameForSchema, j );
    std::string columnName(inputCol);
    unsigned int i=0;
    while(m_tableRegister.checkColumn(parentElement.tableName(),columnName)){
      columnName = ora::MappingRules::newNameForSchemaObject( inputCol, i, ora::MappingRules::MaxColumnNameLength );
      i++;
    }
    m_tableRegister.insertColumn(parentElement.tableName(),columnName);
    cols.push_back( columnName );
  }
  me.setColumnNames( cols );
}

ora::UniqueReferenceMapping::UniqueReferenceMapping( const Reflex::Type& attributeType, TableRegister& tableRegister ):
  m_type(attributeType),m_tableRegister( tableRegister ){
}

ora::UniqueReferenceMapping::~UniqueReferenceMapping(){
}

void ora::UniqueReferenceMapping::process( MappingElement& parentElement,
                                           const std::string& attributeName,
                                           const std::string& attributeNameForSchema,
                                           const std::string& scopeNameForSchema ){
    
  std::string typeName = m_type.Name(Reflex::SCOPED);
  if(!m_tableRegister.checkTable( parentElement.tableName())){
    throwException("Table \""+parentElement.tableName()+"\" has not been allocated.",
                   "UniqueReferenceMapping::process");
  }
  ora::MappingElement& me = parentElement.appendSubElement( ora::MappingElement::uniqueReferenceMappingElementType(), attributeName, typeName, parentElement.tableName() );

  std::vector< std::string > cols;
  std::string inputCol = ora::MappingRules::columnNameForRefMetadata( attributeNameForSchema, scopeNameForSchema );
  std::string columnName(inputCol);
  unsigned int i=0;
  while(m_tableRegister.checkColumn(parentElement.tableName(),columnName)){
    columnName = ora::MappingRules::newNameForSchemaObject( inputCol, i, ora::MappingRules::MaxColumnNameLength );
    i++;
  }
  m_tableRegister.insertColumn(parentElement.tableName(),columnName);
  cols.push_back(columnName);

  std::string idCol = ora::MappingRules::columnNameForRefId( attributeNameForSchema, scopeNameForSchema);
  columnName = idCol;
  i=0;
  while(m_tableRegister.checkColumn(parentElement.tableName(),columnName)){
    columnName = ora::MappingRules::newNameForSchemaObject( inputCol, i, ora::MappingRules::MaxColumnNameLength );
    i++;
  }
  m_tableRegister.insertColumn(parentElement.tableName(),columnName);
  cols.push_back(columnName);

  me.setColumnNames( cols );
}

ora::OraPtrMapping::OraPtrMapping( const Reflex::Type& attributeType, TableRegister& tableRegister ):
  m_type(attributeType), m_tableRegister( tableRegister ){
}

ora::OraPtrMapping::~OraPtrMapping(){
}

void ora::OraPtrMapping::process( MappingElement& parentElement,
                                  const std::string& attributeName,
                                  const std::string& attributeNameForSchema,
                                  const std::string& scopeNameForSchema ){
  
  std::string typeName = m_type.Name(Reflex::SCOPED);
  ora::MappingElement& me = parentElement.appendSubElement( ora::MappingElement::OraPointerMappingElementType(), attributeName, typeName, parentElement.tableName() );
  me.setColumnNames( parentElement.columnNames() );

  Reflex::Type ptrType = m_type.TemplateArgumentAt(0);
  std::string ptrTypeName = ptrType.Name();

  RelationalMappingFactory factory( m_tableRegister );
  std::auto_ptr<IRelationalMapping> processor( factory.newProcessor( ptrType ) );
  processor->process( me, ptrTypeName, attributeNameForSchema, scopeNameForSchema );
}

ora::NamedRefMapping::NamedRefMapping( const Reflex::Type& attributeType, TableRegister& tableRegister ):
  m_type( attributeType ),
  m_tableRegister( tableRegister ){
}

ora::NamedRefMapping::~NamedRefMapping(){
}

void ora::NamedRefMapping::process( MappingElement& parentElement, 
                                    const std::string& attributeName,
                                    const std::string& attributeNameForSchema, 
                                    const std::string& scopeNameForSchema ){
  std::string typeName = m_type.Name(Reflex::SCOPED);
  ora::MappingElement& me = parentElement.appendSubElement( ora::MappingElement::namedReferenceMappingElementType(), attributeName, typeName, parentElement.tableName() );

  std::vector< std::string > cols;
  std::string inputCol = ora::MappingRules::columnNameForNamedReference( attributeNameForSchema, scopeNameForSchema );
  std::string columnName(inputCol);
  unsigned int i=0;
  while(m_tableRegister.checkColumn(parentElement.tableName(),columnName)){
    columnName = ora::MappingRules::newNameForSchemaObject( inputCol, i, ora::MappingRules::MaxColumnNameLength );
    i++;
  }
  m_tableRegister.insertColumn(parentElement.tableName(),columnName);
  cols.push_back(columnName);

  me.setColumnNames( cols );
  
}

ora::ArrayMapping::ArrayMapping( const Reflex::Type& attributeType, TableRegister& tableRegister ):
  m_type(attributeType), m_tableRegister( tableRegister ){
}

ora::ArrayMapping::~ArrayMapping(){
}

void ora::ArrayMapping::process( MappingElement& parentElement,
                                 const std::string& attributeName,
                                 const std::string& attributeNameForSchema,
                                 const std::string& scopeNameForSchema ){
  std::string tableName = parentElement.tableName();
  std::string initialTable(tableName);

  std::string arrayTable(initialTable);
  unsigned int i=0;
  while(m_tableRegister.checkTable(arrayTable)){
    arrayTable = ora::MappingRules::newNameForArraySchemaObject( initialTable, i, ora::MappingRules::MaxTableNameLength );
    i++;
  }
  m_tableRegister.insertTable(arrayTable);

  std::string className = m_type.Name(Reflex::SCOPED);

  std::string elementType = ora::MappingElement::arrayMappingElementType();
  if(ora::ClassUtils::isTypePVector(m_type) || ora::ClassUtils::isTypeQueryableVector(m_type)){
    elementType = ora::MappingElement::OraArrayMappingElementType();
  }
  ora::MappingElement& me = parentElement.appendSubElement( elementType,attributeName,className,arrayTable );
  const std::vector<std::string>& parentColumns = parentElement.columnNames();
  if( parentColumns.empty()){
    throwException( "No column name found in the parent mapping element.","ArrayMapping::process");
  }
  
  std::vector<std::string> columns;
  // always comes the oid first
  columns.push_back( ora::MappingRules::columnNameForId() );
  std::vector<std::string>::const_iterator iColumn = parentColumns.begin();
  // then copy the columns except the id...
  iColumn++;
  for ( ;iColumn != parentColumns.end(); iColumn++ ) {
    columns.push_back( ora::MappingRules::columnNameForId() + "_" + *iColumn );
  }
  // and finally add the position!
  columns.push_back( ora::MappingRules::columnNameForPosition() );
  
  me.setColumnNames( columns );
  m_tableRegister.insertColumns(arrayTable, columns );

  std::string arrayScopeNameForSchema = scopeNameForSchema;
  if( !arrayScopeNameForSchema.empty() ) arrayScopeNameForSchema +="_";
  arrayScopeNameForSchema += attributeNameForSchema;

  
  bool singleItemContainer =  ora::ClassUtils::isTypeNonAssociativeContainer(m_type);
  bool associativeContainer =  ora::ClassUtils::isTypeAssociativeContainer(m_type);

  Reflex::Type contentType;
  Reflex::Type keyType;
  
  if( singleItemContainer ){
    contentType = ClassUtils::containerValueType(m_type);
  }
  else if ( associativeContainer ) { // This is an associative container type
    contentType = ClassUtils::containerDataType( m_type );
    keyType = ClassUtils::containerKeyType( m_type );
    if( !keyType || !ClassUtils::resolvedType(keyType) ){
      throwException( "Cannot not resolve the type of the key item of container \""+m_type.Name(Reflex::SCOPED)+"\".",
                      "ArrayMapping::process");
    }
  }
  else {
    // Not supported container
      throwException( "Container type=\""+m_type.Name(Reflex::SCOPED)+"\".is not supported.",
                      "ArrayMapping::process");    
  }

  if( !contentType || !ClassUtils::resolvedType(contentType) ){
      throwException( "Cannot not resolve the type of the content item of container \""+m_type.Name(Reflex::SCOPED)+"\".",
                      "ArrayMapping::process");
  }
  RelationalMappingFactory mappingFactory( m_tableRegister );
  if ( keyType ) {
    std::string keyTypeName = keyType.Name();
    std::string keyTypeNameForSchema = MappingRules::variableNameForContainerKey();
    std::auto_ptr<IRelationalMapping> keyProcessor( mappingFactory.newProcessor( keyType ) );
    keyProcessor->process( me, keyTypeName, keyTypeNameForSchema, arrayScopeNameForSchema  );
  }
  std::string contentTypeName = contentType.Name();
  std::string contentTypeNameForSchema = MappingRules::variableNameForContainerValue();
  std::auto_ptr<IRelationalMapping> contentProcessor( mappingFactory.newProcessor( contentType ) );
  contentProcessor->process( me, contentTypeName, contentTypeNameForSchema, arrayScopeNameForSchema );
}

ora::CArrayMapping::CArrayMapping( const Reflex::Type& attributeType, TableRegister& tableRegister ):
  m_type(attributeType), m_tableRegister( tableRegister ){
}

ora::CArrayMapping::~CArrayMapping(){
}

void ora::CArrayMapping::process( MappingElement& parentElement,
                                  const std::string& attributeName,
                                  const std::string& attributeNameForSchema,
                                  const std::string& scopeNameForSchema ){
  Reflex::Type arrayElementType = m_type.ToType();
  if( !arrayElementType || !ClassUtils::resolvedType( arrayElementType ) ){
    throwException("Cannot resolve the type of the content of the array \""+m_type.Name(Reflex::SCOPED)+"\".",
                   "CArrayMapping::process");
  }

  if(!m_tableRegister.checkTable(parentElement.tableName())){
    throwException("Table \""+parentElement.tableName()+"\" has not been allocated.",
                   "CArrayMapping::process");
  }
  std::string className = m_type.Name(Reflex::SCOPED);
  RelationalMappingFactory mappingFactory( m_tableRegister );

  std::string arrayScopeNameForSchema = scopeNameForSchema;
  if( !arrayScopeNameForSchema.empty() ) arrayScopeNameForSchema +="_";
  arrayScopeNameForSchema += attributeNameForSchema;

  std::pair<bool,size_t> arraySizeInColumns = RelationalMapping::sizeInColumnsForCArray( m_type );
  if( !arraySizeInColumns.first && arraySizeInColumns.second < MappingRules::MaxColumnsForInlineCArray ) {
    size_t columnsInTable = m_tableRegister.numberOfColumns(parentElement.tableName()) + arraySizeInColumns.second;
    if( columnsInTable < MappingRules::MaxColumnsPerTable ){
      // Inline C-Array
      std::string mappingElementType = ora::MappingElement::inlineCArrayMappingElementType();
      ora::MappingElement& me = parentElement.appendSubElement( mappingElementType, attributeName, className, parentElement.tableName() );
      me.setColumnNames( parentElement.columnNames() );
      std::auto_ptr<IRelationalMapping> processor( mappingFactory.newProcessor( arrayElementType ) );
      for(size_t i=0;i<m_type.ArrayLength();i++){
        processor->process( me, MappingRules::variableNameForArrayIndex(attributeName,i), 
                            MappingRules::variableNameForArrayColumn(i), arrayScopeNameForSchema );
      }
      return;
    }
  }
  /// otherwise, process as standard CArrays in separate tables
  std::string tableName = parentElement.tableName();
  std::string initialTable(tableName);

  std::string arrayTable(initialTable);
  unsigned int i=0;
  while(m_tableRegister.checkTable(arrayTable)){
    arrayTable = ora::MappingRules::newNameForArraySchemaObject( initialTable, i, ora::MappingRules::MaxTableNameLength );
    i++;
  }
  m_tableRegister.insertTable(arrayTable);
  ora::MappingElement& me = parentElement.appendSubElement( ora::MappingElement::CArrayMappingElementType(),
                                                            attributeName, attributeName, arrayTable );
  const std::vector<std::string>& parentColumns = parentElement.columnNames();
  if( parentColumns.empty()){
    throwException( "No column name found in the parent mapping element.","CArrayMapping::process");
  }
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
  m_tableRegister.insertColumns(arrayTable, columns );

  std::string contentTypeName = arrayElementType.Name();
  std::string variableNameForSchema = MappingRules::variableNameForArrayColumn( m_type  );
  std::auto_ptr<IRelationalMapping> processor( mappingFactory.newProcessor( arrayElementType ) );
  processor->process( me, contentTypeName, variableNameForSchema, arrayScopeNameForSchema );
}

ora::ObjectMapping::ObjectMapping( const Reflex::Type& attributeType, TableRegister& tableRegister ):
  m_type(attributeType), m_tableRegister( tableRegister ){
}

ora::ObjectMapping::~ObjectMapping(){
}

namespace ora {

  bool isLoosePersistencyOnWriting( const Reflex::Member& dataMember ){
    std::string persistencyType("");
    Reflex::PropertyList memberProps = dataMember.Properties();
    if( memberProps.HasProperty(ora::MappingRules::persistencyPropertyNameInDictionary())){
       persistencyType = memberProps.PropertyAsString(ora::MappingRules::persistencyPropertyNameInDictionary());
    }
    return ora::MappingRules::isLooseOnWriting( persistencyType );
  }
  
  void processBaseClasses( MappingElement& mappingElement,
                           const Reflex::Type& objType,
                           const std::string& scopeNameForSchema,
                           TableRegister& tableRegister ){
    std::string className = objType.Name(Reflex::SCOPED);
    for ( size_t i=0; i< objType.BaseSize(); i++){
      Reflex::Base base = objType.BaseAt(i);
      Reflex::Type baseType = ClassUtils::resolvedType( base.ToType() );
      if(!baseType){
        throwException( "Class for base \""+base.Name()+"\" is not in the dictionary.","ObjectMapping::process");
      }

      // TO BE FIXED:: here there is still to fix the right scopeName to pass 
      processBaseClasses( mappingElement, baseType, scopeNameForSchema, tableRegister );
      for ( size_t j=0; j< baseType.DataMemberSize(); j++){
        Reflex::Member baseMember = baseType.DataMemberAt(j);
        // Skip the transient and the static ones
        if ( baseMember.IsTransient() || baseMember.IsStatic() || isLoosePersistencyOnWriting( baseMember ) ) continue;
        // Retrieve the data member type
        Reflex::Type type = ClassUtils::resolvedType( baseMember.TypeOf() );
        Reflex::Type declaringType = ClassUtils::resolvedType( baseMember.DeclaringType());
        std::string scope = declaringType.Name(Reflex::SCOPED);
        // Retrieve the field name
        std::string objectMemberName = ora::MappingRules::scopedVariableName( baseMember.Name(), scope );
        std::string objectMemberNameForSchema = ora::MappingRules::scopedVariableForSchemaObjects( baseMember.Name(), scope );

        std::string mappingType("");
        Reflex::PropertyList memberProps = baseMember.Properties();
        if( memberProps.HasProperty(ora::MappingRules::mappingPropertyNameInDictionary())){
          mappingType = memberProps.PropertyAsString(ora::MappingRules::mappingPropertyNameInDictionary());
        }
        bool blobStreaming = ora::MappingRules::isMappedToBlob( mappingType );
        
        RelationalMappingFactory mappingFactory( tableRegister );
        std::auto_ptr<IRelationalMapping> processor( mappingFactory.newProcessor( type, blobStreaming ) );
        processor->process( mappingElement, objectMemberName, objectMemberNameForSchema, scopeNameForSchema );
      }
    }
  }
}

void ora::ObjectMapping::process( MappingElement& parentElement,
                                  const std::string& attributeName,
                                  const std::string& attributeNameForSchema,
                                  const std::string& scopeNameForSchema ){
  std::string className = m_type.Name(Reflex::SCOPED);
  std::string elementType = ora::MappingElement::objectMappingElementType();
  ora::MappingElement& me = parentElement.appendSubElement( elementType, attributeName, className, parentElement.tableName() );
  me.setColumnNames( parentElement.columnNames() );

  // resolve possible typedef chains
  Reflex::Type objectType = ClassUtils::resolvedType(m_type);
  // process base class data members
  processBaseClasses( me, m_type, scopeNameForSchema, m_tableRegister );
  RelationalMappingFactory mappingFactory( m_tableRegister );
  std::string scope = attributeName;
  std::string objectScopeNameForSchema = scopeNameForSchema;
  if( !objectScopeNameForSchema.empty() ) objectScopeNameForSchema +="_";
  objectScopeNameForSchema += attributeNameForSchema;

  // loop over the data members 
  for ( size_t i=0; i< objectType.DataMemberSize(); i++){

    Reflex::Member objectMember = m_type.DataMemberAt(i);
    // Skip the transient and the static ones
    if ( objectMember.IsTransient() || objectMember.IsStatic() || isLoosePersistencyOnWriting( objectMember )) continue;

    // Retrieve the field type
    Reflex::Type type = ClassUtils::resolvedType( objectMember.TypeOf() );
    // Check for the existence of the dictionary information
    if ( !type ){
      throwException( "Type for data member \""+objectMember.Name()+"\" of class \""+className+
                      "\" has not been found in the dictionary.",
                      "ObjectMapping::process");
    }
    
    // check if the member is from a class in the inheritance tree
    Reflex::Type declaringType = ClassUtils::resolvedType( objectMember.DeclaringType());
    if( declaringType != objectType ){
      continue;
    }
    // Retrieve the field name
    std::string objectMemberName = objectMember.Name();
    std::string objectNameForSchema = objectMember.Name();
    
    std::string mappingType("");
    Reflex::PropertyList memberProps = objectMember.Properties();
    if( memberProps.HasProperty(ora::MappingRules::mappingPropertyNameInDictionary())){
      mappingType = memberProps.PropertyAsString(ora::MappingRules::mappingPropertyNameInDictionary());
    }
    bool blobStreaming = ora::MappingRules::isMappedToBlob( mappingType );

    std::auto_ptr<IRelationalMapping> processor( mappingFactory.newProcessor( type, blobStreaming ) );
    processor->process( me, objectMemberName, objectNameForSchema, objectScopeNameForSchema  );
  }
  
}

ora::EmptyMapping::EmptyMapping(){
}

ora::EmptyMapping::~EmptyMapping(){
}

void ora::EmptyMapping::process( MappingElement&,
                                 const std::string&,
                                 const std::string&,
                                 const std::string& ){
}
