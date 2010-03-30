#include "MappingRules.h"
//
#include <sstream>
#include <vector>
#include <cctype>
// externals
#include "Reflex/Type.h"

std::string ora::MappingRules::containerIdSequenceName(){
  static std::string s_sequenceName("CONTAINER_ID");
  return s_sequenceName;
}

std::string ora::MappingRules::mappingSequenceName(){
  static std::string s_mappingSequenceName("MAPPING_ELEMENT_ID");
  return s_mappingSequenceName;
}

std::string ora::MappingRules::sequenceNameForContainer( const std::string& containerName ){
  std::string ret("C_");
  return ret+containerName;
}

std::string
ora::MappingRules::shortName( const std::string& nameCut,
                              size_t maxLength){
  size_t siz = nameCut.size();
  if( siz < maxLength ) return nameCut;
  size_t ind = nameCut.find('<',0);
  size_t maxPrefixSize = (maxLength/3);
  if(maxPrefixSize>5) {
    maxPrefixSize = 5;
  } else if( maxPrefixSize == 0 ) maxPrefixSize = 1;
  if( ind!=std::string::npos ){
    std::stringstream shName;
    size_t lastInd = nameCut.rfind('>'); 
    std::vector< size_t>  uppers;
    for( size_t i=1;i<ind;i++) {
      if(::isupper(nameCut[i])) uppers.push_back(i);
    }
    size_t usize = uppers.size();
    if( usize < (maxPrefixSize-1) ){
      size_t start = 0;
      size_t cut = maxPrefixSize-usize;
      if( usize && (cut > uppers[0]) ) cut = uppers[0];
      shName << nameCut.substr( start, cut );
      size_t left = maxPrefixSize-cut-usize;
      if( usize > 1) left = 0;
      for( size_t i=0; i<usize;i++){
        size_t st = uppers[i];
        shName << nameCut.substr(st,left+1);
        left = 0;
      }
    } else {
      shName << nameCut[0];
      for( size_t i=0;i<maxPrefixSize-1;i++) shName << nameCut[uppers[i]];
    }
    size_t preSz = shName.str().size();
    shName << "_" << shortName( nameCut.substr(ind+1,lastInd-ind-1), maxLength-preSz-1 );
    return shName.str();
  } else {
    return formatForSchema( nameCut, maxLength );
  }
}

std::string
ora::MappingRules::nameForSchema( const std::string& variableName ){
  std::string varName( variableName );
  //first turn className into uppercase
  MappingRules::ToUpper up(std::locale::classic());
  std::transform(varName.begin(), varName.end(),varName.begin(),up);
  std::ostringstream result;
  char p = 0;
  // clean up string...
  for ( std::string::size_type i = 0; i < varName.size(); ++i ) {
    char c = varName[i];
    if ( c == ':' || c == '<' || c == '>' || c == ',' || c == ' ' ) c = '_';
    if(c!='_' || p!='_'){ 
      if(c!='_' || (i!=0 && i!=(varName.size()-1))) {
        result << c;
        p = c;
      }
    }
  }
  size_t cutPoint = result.str().size();
  if(p == '_' && cutPoint) cutPoint -= 1;
  return result.str().substr(0,cutPoint);  
}

std::string
ora::MappingRules::formatForSchema( const std::string& variableName, size_t maxLength){
  if( variableName.empty() ) return variableName;
  if(variableName.size()<maxLength) maxLength = variableName.size();
  if(variableName[maxLength-1]=='_') maxLength -= 1;
  return variableName.substr(0, maxLength);
}

std::string
ora::MappingRules::indexNameForIdentity( const std::string& tableName )
{
  return tableName + "_ID_IDX";
}

std::string
ora::MappingRules::fkNameForIdentity( const std::string& tableName )
{
  return tableName + "_ID_FK";
}

std::string
ora::MappingRules::newMappingVersionForContainer( const std::string& containerName,
                                                  int iteration )
{
  std::ostringstream os;
  os << containerName;
  os << "_C";
  if ( iteration > 0 ) {
    if ( iteration < 10 ) os << "0";
    if ( iteration < 100 ) os << "0";
    os << iteration;
  } else {
    os << "000";
  }
  return os.str();
}

std::string
ora::MappingRules::newMappingVersionForDependentClass( const std::string& containerName,
                                                       const std::string& className,
                                                       int iteration )
{
  std::string contDependencyName = containerName+"_"+className;
  std::ostringstream os;
  os << contDependencyName;
  os << "_D";
  if ( iteration > 0 ) {
    if ( iteration < 10 ) os << "0";
    if ( iteration < 100 ) os << "0";
    os << iteration;
  } else {
    os << "000";
  }
  return os.str();
}

std::string
ora::MappingRules::classVersionPropertyNameInDictionary()
{
  static std::string s_propertyName("class_version");
  return s_propertyName;

}

std::string
ora::MappingRules::mappingPropertyNameInDictionary()
{
  static std::string s_propertyName("mapping");
  return s_propertyName;
}

bool
ora::MappingRules::isMappedToBlob(const std::string& mappingProperty){
  return (mappingProperty == "Blob" || mappingProperty == "blob" || mappingProperty == "BLOB" );
}

std::string
ora::MappingRules::baseClassVersion()
{
  static std::string classVersion("BASE");
  return classVersion;
}

std::string
ora::MappingRules::defaultClassVersion(const std::string& className)
{
  std::string classVersion(className);
  classVersion.append("_default");
  return classVersion;
}

std::string
ora::MappingRules::classId( const std::string& className,
                            const std::string& classVersion ){
  return className+".V"+classVersion;
}

std::string ora::MappingRules::classVersionFromId( const std::string& classId ){
  std::string ret("");
  size_t idx = classId.find('.');
  if( idx != std::string::npos ){
    ret = classId.substr( idx+2 );
  }
  return ret;
}

std::string
ora::MappingRules::baseIdForClass( const std::string& className ){
  return className+"."+baseClassVersion();
}

std::string
ora::MappingRules::fullNameForSchema( const std::string& parentVariableName,
                                      const std::string& variableName ){
  std::ostringstream fullNameForSchema;
  std::string varName = formatForSchema( variableName, MaxTableNameLength-ClassNameLengthForSchema );
  int parentVarNameSize = (MaxTableNameLength-ClassNameLengthForSchema)/3;
  int extraSize = MaxTableNameLength-ClassNameLengthForSchema-varName.size(); 
  if( extraSize>0 && extraSize>parentVarNameSize ) parentVarNameSize = extraSize;
  std::string parVarName = formatForSchema( parentVariableName, parentVarNameSize-2);
  fullNameForSchema << parVarName;
  if(!fullNameForSchema.str().empty()) fullNameForSchema << "_";
  fullNameForSchema << varName;
  return fullNameForSchema.str();
}

std::string
ora::MappingRules::tableNameForClass( const std::string& className,
                                      size_t maxLength )
{
  //std::string tableName = nameForSchema( className );
  //return formatForSchema( tableName, MaxTableNameLength );
  return nameForSchema( shortName( className, maxLength ));
}

std::string
ora::MappingRules::tableNameForVariable( const std::string& variableName,
                                         const std::string& mainTableName )
{
  std::ostringstream result;
  if ( ! mainTableName.empty() ) {
    result << formatForSchema(mainTableName, ClassNameLengthForSchema);
  }
  if(result.str()[result.str().size()-1]!='_') result << "_";
  result << tableNameForClass( variableName,MaxTableNameLength-ClassNameLengthForSchema );

  return formatForSchema( result.str(),MaxTableNameLength);
}

std::string
ora::MappingRules::tableNameForDependency( const std::string& mainTableName,
                                           unsigned int index )
{
  std::ostringstream postfix;
  postfix << "D"<< index;
  size_t maxSize = MaxTableNameLength-postfix.str().size()+1; // add 1 to allow the "_"
  std::ostringstream result;
  if ( ! mainTableName.empty() ) {
    result << formatForSchema(mainTableName, maxSize);
  }
  if(result.str()[result.str().size()-1]!='_') result << "_";
  result << postfix.str();
  //return formatForSchema( result.str(),MaxTableNameLength);
  return result.str();
}


std::string
ora::MappingRules::columnNameForId()
{
  return std::string("ID");
}

std::string
ora::MappingRules::columnNameForRefColumn()
{
  return std::string("REF_ID");
}

std::string
ora::MappingRules::columnNameForPosition()
{
  return std::string("POS");
}

std::string
ora::MappingRules::columnNameForOID( const std::string& variableName,
                                     const std::string& scope,
                                     unsigned int index )
{
  std::stringstream ret;
  ret << "R" << columnNameForVariable( variableName, scope, false )<<"_OID"<<index;
  return ret.str();
}

std::string
ora::MappingRules::columnNameForRefMetadata( const std::string& variableName,
                                             const std::string& scope )
{
  std::stringstream ret;
  ret << "M" << columnNameForVariable( variableName, scope, false );
  return ret.str();
}

std::string
ora::MappingRules::columnNameForRefId( const std::string& variableName,
                                       const std::string& scope )
{
  std::stringstream ret;
  ret << "RID_" << columnNameForVariable( variableName, scope, false );
  return ret.str();
}

std::string
ora::MappingRules::sequenceNameForDependentClass( const std::string& containerName,
                                                  const std::string& className ){
  std::string ret(containerName);
  ret+="_";
  ret+=className;
  return ret;
}

std::string
ora::MappingRules::columnNameForVariable( const std::string& variableName,
                                          const std::string& scopeName,
                                          bool forData )
{
  std::ostringstream totalString;
  int scopeMaxSize = MaxColumnNameLength/4-1;
  int extraSize = MaxColumnNameLength-variableName.size();
  if( extraSize>0 && extraSize>scopeMaxSize ) scopeMaxSize = extraSize;
  if(forData) totalString << "D";
  if( !scopeName.empty() ) {
    size_t scopeCut = scopeName.size();
    if( scopeCut> (size_t)scopeMaxSize ) scopeCut = scopeMaxSize;
    totalString << scopeName.substr(0,scopeCut);
    totalString << "_";
  }
  size_t varMaxSize = MaxColumnNameLength-totalString.str().size();

  size_t fp = variableName.find('[');
  if( fp != std::string::npos ){
    // process for c-arrays
    std::string arrayVar = variableName.substr(0,fp);
    std::string indexVar = variableName.substr(fp);
    for( size_t ind = 0; ind!=std::string::npos; ind=indexVar.find('[',ind+1 ) ){
      indexVar = indexVar.replace(ind,1,"I");
    }
    for( size_t ind = indexVar.find(']'); ind!=std::string::npos; ind=indexVar.find(']',ind+1 ) ){
      indexVar = indexVar.replace(ind,1,"_");
    }
    size_t arrayVarCut = 0;
    size_t varCut = variableName.size()+1;
    if( varCut>varMaxSize ) varCut = varMaxSize;
    if( varCut>(indexVar.size()+1) ) arrayVarCut = varCut-indexVar.size()-1;
    totalString << arrayVar.substr(0,arrayVarCut);
    totalString << "_" << indexVar;
  } else {
    size_t varCut = variableName.size();
    if( varCut>varMaxSize ) varCut = varMaxSize;
    // other types
    totalString << variableName.substr(0,varCut);
  }

  std::string ret(nameForSchema(totalString.str()));
  return formatForSchema(ret,MaxColumnNameLength);
}

std::string
ora::MappingRules::newNameForSchemaObject( const std::string& initialName,
                                           unsigned int index,
                                           size_t maxLength)
{
  unsigned int digitsForPostfix = 3;
  if(index<10) digitsForPostfix = 2;
  size_t newSize = initialName.size()+digitsForPostfix;
  if(newSize > maxLength) newSize = maxLength;
  unsigned int cutSize = newSize - digitsForPostfix;
  std::stringstream newStr("");
  std::string cutString = initialName.substr(0, cutSize );
  newStr << cutString;
  if(cutString[cutString.size()-1]!='_') newStr << "_";
  //newStr << "D"<<index;
  newStr <<index;
  return newStr.str();
}

std::string
ora::MappingRules::newNameForDepSchemaObject( const std::string& initialName,
                                           unsigned int index,
                                           size_t maxLength)
{
  unsigned int digitsForPostfix = 4;
  if(index<10) digitsForPostfix = 3;
  size_t newSize = initialName.size()+digitsForPostfix;
  if(newSize > maxLength) newSize = maxLength;
  unsigned int cutSize = newSize - digitsForPostfix;
  std::stringstream newStr("");
  std::string cutString = initialName.substr(0, cutSize );
  newStr << cutString;
  if(cutString[cutString.size()-1]!='_') newStr << "_";
  newStr << "D"<<index;
  return newStr.str();
}


std::string
ora::MappingRules::variableNameForArrayIndex( const std::string& arrayVariable,
                                              unsigned int index ){
  std::ostringstream arrayElementLabel;
  arrayElementLabel << arrayVariable << "[" << index << "]";
  return arrayElementLabel.str();
}

std::string
ora::MappingRules::variableNameForArrayColumn(const std::string& arrayVariable,
                                              unsigned int arrayIndex ){
  std::ostringstream arrayElementLabel;
  arrayElementLabel << arrayVariable << "_I" << arrayIndex;
  return arrayElementLabel.str();
}

std::string
ora::MappingRules::variableNameForArrayColumn( unsigned int arrayIndex ){
  std::ostringstream arrayElementLabel;
  arrayElementLabel << "I" << arrayIndex;
  return arrayElementLabel.str();
}

std::string
ora::MappingRules::scopedVariableName( const std::string& variableName,
                                       const std::string& scope ){
  std::stringstream scopedName;
  if(!scope.empty()) scopedName << scope << "::";
  scopedName << variableName;
  return scopedName.str();
}

std::string
ora::MappingRules::scopedVariableForSchemaObjects( const std::string& variableName,
                                                   const std::string& scope ){
  std::stringstream scopedName;
  if(!scope.empty()) scopedName << formatForSchema(scope, ClassNameLengthForSchema) << "_";
  scopedName << variableName;
  return scopedName.str();  
}

std::string
ora::MappingRules::variableNameForArrayColumn( const Reflex::Type& array ){
  std::stringstream contentTypeName;
  contentTypeName << "A" << array.ArrayLength();
  return contentTypeName.str();
}

std::string
ora::MappingRules::referenceColumnKey( const std::string& tableName,
                                       const std::string& columnName )
{
  std::stringstream referenceColumn;
  referenceColumn << tableName << "." << columnName;
  return referenceColumn.str();
}

std::string ora::MappingRules::variableNameForContainerValue(){
  static std::string s_cv("CV");
  return s_cv;
}

std::string ora::MappingRules::variableNameForContainerKey(){
  static std::string s_ck("CK");
  return s_ck;
}

