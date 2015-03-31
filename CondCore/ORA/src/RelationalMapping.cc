#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Reference.h"
#include "CondCore/ORA/interface/NamedRef.h"
#include "RelationalMapping.h"
#include "TableRegister.h"
#include "MappingElement.h"
#include "MappingRules.h"
#include "ClassUtils.h"
// externals 
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "CoralBase/AttributeSpecification.h"

size_t
ora::RelationalMapping::sizeInColumns(const edm::TypeWithDict& topLevelClassType ){
  size_t sz=0;
  bool hasDependencies = false;
  _sizeInColumns(topLevelClassType, sz, hasDependencies );
  return sz;
}

std::pair<bool,size_t>
ora::RelationalMapping::sizeInColumnsForCArray(const edm::TypeWithDict& topLevelClassType ){
  size_t sz=0;
  bool hasDependencies = false;
  _sizeInColumnsForCArray(topLevelClassType, sz, hasDependencies );
  return std::make_pair(hasDependencies,sz);
}


void
ora::RelationalMapping::_sizeInColumns(const edm::TypeWithDict& topLevelClassType,
                                       size_t& sz,
                                       bool& hasDependencies ){
  // resolve possible typedef chains
  edm::TypeWithDict typ = ClassUtils::resolvedType( topLevelClassType );
  bool isOraPolyPointer = ora::ClassUtils::isTypeUniqueReference(typ);
  bool isPrimitive = ora::ClassUtils::isTypePrimitive( typ );

  // primitive and string
  if( isPrimitive || isOraPolyPointer || ora::ClassUtils::isTypeNamedReference( typ)) {
    ++sz;
  } else if (typ.isArray()){
    size_t arraySize = 0;
    _sizeInColumnsForCArray( typ,arraySize, hasDependencies );
    if( arraySize < MappingRules::MaxColumnsForInlineCArray ) sz += arraySize;
    else hasDependencies = true;
  } else if (typ == typeid(ora::Reference) ||
             typ.hasBase( edm::TypeWithDict( typeid(ora::Reference) ) )){
    sz += 2;
  } else {
  
    bool isContainer =  ora::ClassUtils::isTypeNonAssociativeContainer(typ) ||
      ora::ClassUtils::isTypeAssociativeContainer(typ);
    bool isOraPointer = ora::ClassUtils::isTypeOraPointer(typ);
    if( !isContainer && !isOraPointer ){
      
      // loop over the data members
      //-ap ignore for now:  typ.UpdateMembers();
      //std::vector<edm::TypeWithDict> carrays;
      edm::TypeDataMembers members(typ);
      for (auto const & member : members) {
        edm::MemberWithDict objMember(member);

        // Skip the transient ones
        if ( objMember.isTransient() ) continue;

        // Retrieve the field type
        edm::TypeWithDict objMemberType = objMember.typeOf();

        _sizeInColumns(objMemberType,sz, hasDependencies );
      }
    } else {
      hasDependencies = true;
    }    
  }
}

void
ora::RelationalMapping::_sizeInColumnsForCArray(const edm::TypeWithDict& topLevelClassType,
                                                size_t& sz,
                                                bool& hasDependencies){
  // resolve possible typedef chains
  edm::TypeWithDict typ = ClassUtils::resolvedType( topLevelClassType );
  if( !typ.isArray()){
    return;
  }

  size_t arraySize = ClassUtils::arrayLength( typ );
  edm::TypeWithDict arrayType = typ.toType();
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

ora::IRelationalMapping* ora::RelationalMappingFactory::newProcessor( const edm::TypeWithDict& attributeType,
                                                                      bool blobStreaming ){
  if( blobStreaming ){
    return new BlobMapping( attributeType, m_tableRegister );
  }
  edm::TypeWithDict resType = ClassUtils::resolvedType( attributeType );
  if ( ora::ClassUtils::isTypePrimitive(resType) ) {
    return new PrimitiveMapping( attributeType, m_tableRegister );
  }
  else if ( resType.isArray() ){
    return new CArrayMapping( attributeType, m_tableRegister );
  }
  else if ( ora::ClassUtils::isTypeContainer( resType ) ) {
    return new ArrayMapping( attributeType, m_tableRegister );
  }
  else if ( resType.isPointer() || resType.isReference() ){
    return new EmptyMapping();
  }
  else if ( ora::ClassUtils::isTypeOraPointer( resType )){
    return new OraPtrMapping( attributeType, m_tableRegister );
  }
  else if ( ora::ClassUtils::isTypeUniqueReference( resType )){
    return new UniqueReferenceMapping( attributeType, m_tableRegister );
  }
  else if ( resType == typeid(ora::Reference) ||
            resType.hasBase( edm::TypeWithDict( typeid(ora::Reference) ) ) ){
    return new OraReferenceMapping( attributeType, m_tableRegister );
  }
  else if ( resType == typeid(ora::NamedReference) ||
            resType.hasBase( edm::TypeWithDict( typeid(ora::NamedReference) ) ) ){
    return new NamedRefMapping( attributeType, m_tableRegister );
  }
  else { // embeddedobject
    return new ObjectMapping( attributeType, m_tableRegister );
  } 
  return 0; // make the compiler happy -- we should never come here !! 
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
    std::vector<std::string> cols;
    cols.push_back( columnName );
    // add metadata column for blobs 
    if( elementType == ora::MappingElement::blobMappingElementType() ){
      std::string metaDataColumnName = ora::MappingRules::columnNameForBlobMetadata( columnName );
      tableRegister.insertColumn(parentElement.tableName(),metaDataColumnName );
      cols.push_back( metaDataColumnName );      
    }
    me.setColumnNames( cols );
  }
}

ora::PrimitiveMapping::PrimitiveMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister ):
  m_type(attributeType),m_tableRegister( tableRegister ){
}

ora::PrimitiveMapping::~PrimitiveMapping(){
}

void ora::PrimitiveMapping::process( MappingElement& parentElement,
                                     const std::string& attributeName,
                                     const std::string& attributeNameForSchema,
                                     const std::string& scopeNameForSchema ){
  edm::TypeWithDict t = ClassUtils::resolvedType( m_type );
  const std::type_info* attrType = t.isEnum() ? &typeid(int) : &t.typeInfo();
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

ora::BlobMapping::BlobMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister ):
  m_type(attributeType),m_tableRegister( tableRegister ){
}

ora::BlobMapping::~BlobMapping(){
}
void ora::BlobMapping::process( MappingElement& parentElement,
                                const std::string& attributeName,
                                const std::string& attributeNameForSchema,
                                const std::string& scopeNameForSchema ){
  std::string className = m_type.cppName();
  processLeafElement(parentElement,
                     ora::MappingElement::blobMappingElementType(),
                     className,
                     attributeName,
                     attributeNameForSchema,
                     scopeNameForSchema,
                     m_tableRegister);
}

ora::OraReferenceMapping::OraReferenceMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister ):
  m_type(attributeType),m_tableRegister( tableRegister ){
}

ora::OraReferenceMapping::~OraReferenceMapping(){
}

void ora::OraReferenceMapping::process( MappingElement& parentElement,
                                        const std::string& attributeName,
                                        const std::string& attributeNameForSchema,
                                        const std::string& scopeNameForSchema ){
  std::string className = m_type.cppName();
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

ora::UniqueReferenceMapping::UniqueReferenceMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister ):
  m_type(attributeType),m_tableRegister( tableRegister ){
}

ora::UniqueReferenceMapping::~UniqueReferenceMapping(){
}

void ora::UniqueReferenceMapping::process( MappingElement& parentElement,
                                           const std::string& attributeName,
                                           const std::string& attributeNameForSchema,
                                           const std::string& scopeNameForSchema ){
    
  std::string typeName = m_type.cppName();
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

ora::OraPtrMapping::OraPtrMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister ):
  m_type(attributeType), m_tableRegister( tableRegister ){
}

ora::OraPtrMapping::~OraPtrMapping(){
}

void ora::OraPtrMapping::process( MappingElement& parentElement,
                                  const std::string& attributeName,
                                  const std::string& attributeNameForSchema,
                                  const std::string& scopeNameForSchema ){
  
  std::string typeName = m_type.cppName();
  ora::MappingElement& me = parentElement.appendSubElement( ora::MappingElement::OraPointerMappingElementType(), attributeName, typeName, parentElement.tableName() );
  me.setColumnNames( parentElement.columnNames() );

  edm::TypeWithDict ptrType = m_type.templateArgumentAt(0);
  std::string ptrTypeName = ptrType.name();

  RelationalMappingFactory factory( m_tableRegister );
  std::auto_ptr<IRelationalMapping> processor( factory.newProcessor( ptrType ) );
  processor->process( me, ptrTypeName, attributeNameForSchema, scopeNameForSchema );
}

ora::NamedRefMapping::NamedRefMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister ):
  m_type( attributeType ),
  m_tableRegister( tableRegister ){
}

ora::NamedRefMapping::~NamedRefMapping(){
}

void ora::NamedRefMapping::process( MappingElement& parentElement, 
                                    const std::string& attributeName,
                                    const std::string& attributeNameForSchema, 
                                    const std::string& scopeNameForSchema ){
  std::string typeName = m_type.cppName();
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

ora::ArrayMapping::ArrayMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister ):
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

  std::string className = m_type.cppName();

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

  edm::TypeWithDict contentType;
  edm::TypeWithDict keyType;
  std::string contentTypeName;
  
  if( singleItemContainer ){
    contentType = ClassUtils::containerValueType(m_type);
    contentTypeName = "value_type";
  }
  else if ( associativeContainer ) { // This is an associative container type
    contentType = ClassUtils::containerDataType( m_type );
    contentTypeName = "mapped_type";
    keyType = ClassUtils::containerKeyType( m_type );
    if( !keyType || !ClassUtils::resolvedType(keyType) ){
      throwException( "Cannot not resolve the type of the key item of container \""+m_type.cppName()+"\".",
                      "ArrayMapping::process");
    }
  }
  else {
    // Not supported container
      throwException( "Container type=\""+m_type.cppName()+"\".is not supported.",
                      "ArrayMapping::process");    
  }

  if( !contentType || !ClassUtils::resolvedType(contentType) ){
      throwException( "Cannot not resolve the type of the content item of container \""+m_type.cppName()+"\".",
                      "ArrayMapping::process");
  }
  RelationalMappingFactory mappingFactory( m_tableRegister );
  if ( keyType ) {
    std::string keyTypeName = "key_type";
    std::string keyTypeNameForSchema = MappingRules::variableNameForContainerKey();
    std::auto_ptr<IRelationalMapping> keyProcessor( mappingFactory.newProcessor( keyType ) );
    keyProcessor->process( me, keyTypeName, keyTypeNameForSchema, arrayScopeNameForSchema  );
  }
  std::string contentTypeNameForSchema = MappingRules::variableNameForContainerValue();
  std::auto_ptr<IRelationalMapping> contentProcessor( mappingFactory.newProcessor( contentType ) );
  contentProcessor->process( me, contentTypeName, contentTypeNameForSchema, arrayScopeNameForSchema );
}

ora::CArrayMapping::CArrayMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister ):
  m_type(attributeType), m_tableRegister( tableRegister ){
}

ora::CArrayMapping::~CArrayMapping(){
}

void ora::CArrayMapping::process( MappingElement& parentElement,
                                  const std::string& attributeName,
                                  const std::string& attributeNameForSchema,
                                  const std::string& scopeNameForSchema ){
  edm::TypeWithDict arrayElementType = m_type.toType();
  if( !arrayElementType || !ClassUtils::resolvedType( arrayElementType ) ){
    throwException("Cannot resolve the type of the content of the array \""+m_type.cppName()+"\".",
                   "CArrayMapping::process");
  }

  if(!m_tableRegister.checkTable(parentElement.tableName())){
    throwException("Table \""+parentElement.tableName()+"\" has not been allocated.",
                   "CArrayMapping::process");
  }
  std::string className = m_type.cppName();
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
      size_t arraySize = ClassUtils::arrayLength( m_type );
      for(size_t i=0;i<arraySize;i++){
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
                                                            attributeName, className, arrayTable );
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

  std::string contentTypeName = arrayElementType.name();
  std::string variableNameForSchema = MappingRules::variableNameForArrayColumn( m_type  );
  std::auto_ptr<IRelationalMapping> processor( mappingFactory.newProcessor( arrayElementType ) );
  processor->process( me, contentTypeName, variableNameForSchema, arrayScopeNameForSchema );
}

ora::ObjectMapping::ObjectMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister ):
  m_type(attributeType), m_tableRegister( tableRegister ){
}

ora::ObjectMapping::~ObjectMapping(){
}

namespace ora {

 bool isLoosePersistencyOnWriting( const edm::MemberWithDict& dataMember ){
   return ora::MappingRules::isLooseOnWriting( ClassUtils::getDataMemberProperty(ora::MappingRules::persistencyPropertyNameInDictionary(), dataMember  ) );
  }

  bool isMappedToBlob( const edm::MemberWithDict& dataMember ){
    return ora::MappingRules::isMappedToBlob( ClassUtils::getDataMemberProperty( ora::MappingRules::mappingPropertyNameInDictionary(), dataMember ) );
  }
  
  void processBaseClasses( MappingElement& mappingElement,
                           const edm::TypeWithDict& objType,
                           const std::string& scopeNameForSchema,
                           TableRegister& tableRegister ){
    std::string className = objType.cppName();
    edm::TypeBases bases(objType);
    for (auto const & b : bases) {
      edm::BaseWithDict base(b);
      edm::TypeWithDict baseType = ClassUtils::resolvedType( base.typeOf().toType() );
      if(!baseType){
        throwException( "Class for base \""+base.name()+"\" is not in the dictionary.","ObjectMapping::process");
      }

      // TO BE FIXED:: here there is still to fix the right scopeName to pass 
      processBaseClasses( mappingElement, baseType, scopeNameForSchema, tableRegister );
      edm::TypeDataMembers members(baseType);
      for (auto const & member : members) {
        edm::MemberWithDict baseMember(member);
        // Skip the transient and the static ones
        if ( baseMember.isTransient() || baseMember.isStatic() || isLoosePersistencyOnWriting( baseMember ) ) continue;
        // Retrieve the data member type
        edm::TypeWithDict type = ClassUtils::resolvedType( baseMember.typeOf() );
        edm::TypeWithDict declaringType = ClassUtils::resolvedType( baseMember.declaringType());
        std::string scope = declaringType.cppName();
        // Retrieve the field name
        std::string objectMemberName = ora::MappingRules::scopedVariableName( baseMember.name(), scope );
        std::string objectMemberNameForSchema = ora::MappingRules::scopedVariableForSchemaObjects( baseMember.name(), scope );

        bool blobStreaming = isMappedToBlob( baseMember ); 
        
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
  std::string className = m_type.cppName();
  std::string elementType = ora::MappingElement::objectMappingElementType();
  ora::MappingElement& me = parentElement.appendSubElement( elementType, attributeName, className, parentElement.tableName() );
  me.setColumnNames( parentElement.columnNames() );

  // resolve possible typedef chains
  edm::TypeWithDict objectType = ClassUtils::resolvedType(m_type);
  // process base class data members
  processBaseClasses( me, m_type, scopeNameForSchema, m_tableRegister );
  RelationalMappingFactory mappingFactory( m_tableRegister );
  std::string scope = attributeName;
  std::string objectScopeNameForSchema = scopeNameForSchema;
  if( !objectScopeNameForSchema.empty() ) objectScopeNameForSchema +="_";
  objectScopeNameForSchema += attributeNameForSchema;

  // loop over the data members 
  edm::TypeDataMembers members(objectType);
  for (auto const & member : members) {
    edm::MemberWithDict objectMember(member);
    // Skip the transient and the static ones
    if ( objectMember.isTransient() || objectMember.isStatic() || isLoosePersistencyOnWriting( objectMember )) continue;

    // Retrieve the field type
    edm::TypeWithDict type = ClassUtils::resolvedType( objectMember.typeOf() );
    // Check for the existence of the dictionary information
    if ( !type ){
      throwException( "Type for data member \""+objectMember.name()+"\" of class \""+className+
                      "\" has not been found in the dictionary.",
                      "ObjectMapping::process");
    }
    
    // check if the member is from a class in the inheritance tree
    edm::TypeWithDict declaringType = ClassUtils::resolvedType( objectMember.declaringType());
    if( declaringType != objectType ){
      continue;
    }
    // Retrieve the field name
    std::string objectMemberName = objectMember.name();
    std::string objectNameForSchema = objectMember.name();
    
   bool blobStreaming = isMappedToBlob( objectMember ); 

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
