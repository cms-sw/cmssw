#include "MappingRules.h"
//
#include <sstream>
#include <vector>
#include <cctype>
// externals
#include "Reflex/Type.h"

std::string ora::MappingRules::sequenceNameForContainerId(){
  static std::string s_sequenceName("CONTAINER_ID");
  return s_sequenceName;
}

std::string ora::MappingRules::sequenceNameForContainer( const std::string& containerName ){
  std::string ret("C_");
  return ret+containerName;
}

std::string
ora::MappingRules::sequenceNameForDependentClass( const std::string& containerName,
                                                  const std::string& className ){
  std::string ret(containerName);
  ret+="_";
  ret+=className;
  return ret;
}

std::string ora::MappingRules::sequenceNameForMapping(){
  static std::string s_mappingSequenceName("MAPPING_ELEMENT_ID");
  return s_mappingSequenceName;
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
ora::MappingRules::persistencyPropertyNameInDictionary(){
  static std::string s_propertyName("persistency");
  return s_propertyName;
}

bool
ora::MappingRules::isLooseOnReading(const std::string& persistencyProperty){
  return (persistencyProperty == "loose_on_reading" || persistencyProperty == "LOOSE_ON_READING" ); 
}

bool
ora::MappingRules::isLooseOnWriting(const std::string& persistencyProperty){
  return (persistencyProperty == "loose_on_writing" || persistencyProperty == "LOOSE_ON_WRITING" ); 
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
ora::MappingRules::baseClassVersion()
{
  static std::string classVersion("BASE");
  return classVersion;
}

std::pair<bool,std::string> ora::MappingRules::classNameFromBaseId( const std::string& classId ){
  std::pair<bool,std::string> ret(false,"");
  size_t cut = classId.find("."+baseClassVersion() );
  if( cut != std::string::npos ){
    ret.first = true;
    ret.second = classId.substr(0,cut);
  }
  return ret;
}

std::string
ora::MappingRules::defaultClassVersion(const std::string& className)
{
  std::string classVersion(className);
  classVersion.append("_default");
  return classVersion;
}

std::string
ora::MappingRules::classVersionPropertyNameInDictionary()
{
  static std::string s_propertyName("class_version");
  return s_propertyName;

}

std::string
ora::MappingRules::newMappingVersion( const std::string& itemName,
                                      int iteration,
                                      char versionTrailer ){
  std::ostringstream os;
  os << itemName;
  os << "_" << versionTrailer;
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
ora::MappingRules::newMappingVersionForContainer( const std::string& containerName,
                                                  int iteration ){
  return newMappingVersion( containerName, iteration, 'M' );
}

std::string
ora::MappingRules::newMappingVersionForDependentClass( const std::string& containerName,
                                                       const std::string& className,
                                                       int iteration )
{
  std::string contDependencyName = containerName+"_"+className;
  return newMappingVersion( contDependencyName, iteration, 'D' );
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
ora::MappingRules::variableNameForArrayIndex( const std::string& arrayVariable,
                                              unsigned int index ){
  std::ostringstream arrayElementLabel;
  arrayElementLabel << arrayVariable << "[" << index << "]";
  return arrayElementLabel.str();
}

std::string
ora::MappingRules::variableNameForArrayColumn( unsigned int arrayIndex ){
  std::ostringstream arrayElementLabel;
  arrayElementLabel << "I" << arrayIndex;
  return arrayElementLabel.str();
}

std::string
ora::MappingRules::variableNameForArrayColumn( const Reflex::Type& array ){
  std::stringstream contentTypeName;
  contentTypeName << "A" << array.ArrayLength();
  return contentTypeName.str();
}

std::string ora::MappingRules::variableNameForContainerValue(){
  static std::string s_cv("CV");
  return s_cv;
}

std::string ora::MappingRules::variableNameForContainerKey(){
  static std::string s_ck("CK");
  return s_ck;
}

std::string
ora::MappingRules::scopedVariableForSchemaObjects( const std::string& variableName,
                                                   const std::string& scope ){
  std::stringstream scopedName;
  if(!scope.empty()) scopedName << formatName(scope, ClassNameLengthForSchema) << "_";
  scopedName << variableName;
  return scopedName.str();  
}

/// TO BE REMOVED
#include "CondCore/ORA/interface/Exception.h"
namespace ora {
  static std::string validChars("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-0123456789");
  void checkString( const std::string& s, int code, bool thro=true){
    for( size_t i=0;i<s.size();i++){
      if( validChars.find( s[i])==std::string::npos ) {
        std::stringstream mess;
        mess <<" code="<<code<<" in string ["<<s<<"] found a wrong char=["<<s[i]<<"].";
        if( thro ) throwException( mess.str(),"validChars");
      }
    }    
  } 
  
}

std::string
ora::MappingRules::newNameForSchemaObject( const std::string& initialName,
                                           unsigned int index,
                                           size_t maxLength,
                                           char indexTrailer){
  unsigned int digitsForPostfix = 3;
  if(index<10) digitsForPostfix = 2;
  size_t newSize = initialName.size()+digitsForPostfix;
  if(newSize > maxLength) newSize = maxLength;
  unsigned int cutSize = newSize - digitsForPostfix;
  std::stringstream newStr("");
  if( initialName[cutSize-1]=='_' ) cutSize -= 1;
  std::string cutString = initialName.substr(0, cutSize );
  newStr << cutString << "_";
  if( indexTrailer !=0 ) newStr<< indexTrailer;
  newStr<< index;
  //checkString( newStr.str(), 7 );
  return newStr.str();
}

std::string
ora::MappingRules::newNameForDepSchemaObject( const std::string& initialName,
                                           unsigned int index,
                                           size_t maxLength)
{
  return newNameForSchemaObject( initialName, index, maxLength, 'D' );
}

std::string
ora::MappingRules::newNameForArraySchemaObject( const std::string& initialName,
                                                unsigned int index,
                                                size_t maxL)
{
  return newNameForSchemaObject( initialName, index, maxL, 'A' );
}

std::string
ora::MappingRules::nameForSchema( const std::string& variableName ){
  std::string varName( variableName );
  //first turn className into uppercase
  MappingRules::ToUpper up(std::locale::classic());
  std::transform(varName.begin(), varName.end(),varName.begin(),up);
  size_t cutPoint = varName.size();
  if(cutPoint && varName[varName.size()-1] == '_' ) cutPoint -= 1;
  if(!cutPoint) return "";
  size_t start = 0;
  if(varName.size() && varName[0]== '_' ) start = 1;
  return varName.substr(start,cutPoint);
}

std::string ora::MappingRules::shortNameByUpperCase( const std::string& className,
                                                     size_t maxL ){
  if( !maxL ) return "";
  if( className.size() < maxL ) return className;
  std::vector< size_t>  uppers;
  for( size_t i=0;i<className.size();i++) {
    if(::isupper(className[i]) || ::isdigit(className[i]) ) uppers.push_back(i);
  }
  std::stringstream shName;
  size_t usize = uppers.size();
  if( usize < maxL ){
    size_t start = 0;
    size_t cut = maxL-usize;
    if( usize && (cut > uppers[0]) ) cut = uppers[0];
    shName << className.substr( start, cut );
    size_t left = maxL-cut-usize;
    if( usize > 1) left = 0;
    size_t curs = 0;
    for( size_t i=0; i<usize;i++){
      size_t st = uppers[i];
      curs = st+left+1;
      shName << className.substr(st,left+1);
      left = 0;
    }
    size_t maxIndex = className.size();
    if( shName.str().size()<maxL && curs < maxIndex ){
      size_t more = maxL - shName.str().size();
      size_t max = curs+more;
      if(max > className.size()) max = className.size();
      for( size_t j=curs;j<max-1;j++ ){
        shName << className[j];
      }
    }
    
    //checkString( shName.str(), 0 );
  } else {
    shName << className[0];
    for(size_t i=0 ;i<maxL-1;i++) {
      if( uppers[i] != 0 ) shName << className[uppers[i]];
    }
    //checkString( shName.str(), 1 );
  }
  /// 
  return shName.str();
}

std::string ora::MappingRules::shortScopedName( const std::string& scopedClassName,
                                                size_t maxLength ){
  if( !maxLength ) return "";
  std::string cn = scopedClassName;
  std::string sn("");
  size_t ns = cn.rfind("::");
  if( ns!= std::string::npos ){
    cn = scopedClassName.substr( ns+2 );
    sn = scopedClassName.substr( 0, ns );
  }
  //
  ns = cn.find(" ");
  while( ns != std::string::npos ){
    cn = cn.replace( ns, 1, "_");
    ns = cn.find(" ");
  }
  //
  ns = sn.find("::");
  if( ns == 0 ){
    sn = sn.substr( 2 );
    ns = sn.find("::");
  } 
  while( ns != std::string::npos ){
    sn = sn.replace( ns, 2, "_");
    ns = sn.find("::");
  }
  //
  if( sn[sn.size()-1]=='_' ) sn = sn.substr(0,sn.size()-1);
  // ignore if namespace==std
  if( sn == "std" ) sn = "";
  
  size_t currSize = sn.size()+cn.size()+1;
  if( currSize > maxLength+1 ){
    // a cut is required...
    size_t maxScopeLen = maxLength/3;
    if( maxScopeLen ==0 ) maxScopeLen = 1;
    if( maxScopeLen > 1 ) maxScopeLen -= 1;
    if( sn.size() > maxScopeLen ){
      sn = sn.substr( 0,maxScopeLen );
    }
    size_t availableSize = maxLength-sn.size();
    if( sn.size() ) availableSize -= 1;
    cn =shortNameByUpperCase( cn, availableSize );
  }
  std::string ret = sn;
  if(!ret.empty()) ret += "_";
  ret += cn;
  //checkString( ret, 2 );
  return ret;
}

std::string ora::MappingRules::nameFromTemplate( const std::string templateClassName,
                                                 size_t maxLength ){
  if( !maxLength ) return "";
  std::string newName("");
  size_t ind0 = templateClassName.find('<',0);
  if( ind0 != std::string::npos ){
    size_t ind1 = templateClassName.rfind('>');
    std::string templArg = templateClassName.substr( ind0+1, ind1-ind0-1 );
    std::string prefix = shortScopedName( templateClassName.substr( 0, ind0 ), maxLength );
    size_t currSize = templArg.size()+prefix.size()+1;
    if( currSize > maxLength+1 ){
      // a cut is required...
      size_t prefixL = maxLength/3;
      if( prefixL == 0 ) prefixL = 1;
      if( prefixL >1 ) prefixL -=1;
      prefix = shortScopedName( prefix, prefixL );
    }
    size_t templMaxSize = maxLength-prefix.size()-1;
    templArg = nameFromTemplate( templArg,  templMaxSize );
    newName = prefix+"_"+ templArg;
  } else {
    newName = shortScopedName( templateClassName, maxLength );
  }
  //checkString( newName, 3 );
  return newName;
}

std::string
ora::MappingRules::tableNameForItem( const std::string& itemName )
{
  return "ORA_C_"+nameForSchema(formatName( itemName, MaxTableNameLength-5 ));
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
ora::MappingRules::columnNameForVariable( const std::string& variableName,
                                          const std::string& scopeName,
                                          bool forData )
{
  std::ostringstream totalString;
  int scopeMaxSize = MaxColumnNameLength/4-1;
  int extraSize = MaxColumnNameLength-variableName.size()-2;
  if( extraSize>0 && extraSize>scopeMaxSize ) scopeMaxSize = extraSize;
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

  std::stringstream ret;
  if(forData) ret << "D";
  ret << nameForSchema(totalString.str());
  //return formatForSchema(ret.str(),MaxColumnNameLength);
  return formatName(ret.str(),MaxColumnNameLength);
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
ora::MappingRules::columnNameForNamedReference( const std::string& variableName,
                                                const std::string& scope )
{
  std::stringstream ret;
  ret << "R" << columnNameForVariable( variableName, scope, false )<<"_NAME";
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
ora::MappingRules::columnNameForPosition()
{
  return std::string("POS");
}

std::string
ora::MappingRules::fkNameForIdentity( const std::string& tableName, int index )
{
  std::stringstream ret;
  ret << tableName <<"_ID_FK";
  if( index ) ret << "_"<<index;
  return ret.str();
}

std::string ora::MappingRules::formatName( const std::string& variableName,
                                           size_t maxLength ){
  return nameFromTemplate( variableName, maxLength );
}

/**
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
    std::string templArg = nameCut.substr( ind+1, lastInd-ind-1 );
    size_t ns = templArg.rfind("::");
    if( ns!= std::string::npos ){
      templArg = templArg.substr( ns+2 );
    }
    
    std::vector< size_t>  uppers;
    for( size_t i=0;i<templArg.size();i++) {
      if(::isupper(templArg[i])) uppers.push_back(i);
    }
    size_t usize = uppers.size();
    if( usize < (maxPrefixSize-1) ){
      size_t start = 0;
      size_t cut = maxPrefixSize-usize;
      if( usize && (cut > uppers[0]) ) cut = uppers[0];
      shName << templArg.substr( start, cut );
      size_t left = maxPrefixSize-cut-usize;
      if( usize > 1) left = 0;
      for( size_t i=0; i<usize;i++){
        size_t st = uppers[i];
        shName << templArg.substr(st,left+1);
        left = 0;
      }
    } else {
      shName << templArg[0];
      for( size_t i=0;i<maxPrefixSize-1;i++) shName << templArg[uppers[i]];
    }
    size_t preSz = shName.str().size();
    shName << "_" << shortName( templArg, maxLength-preSz-1 );
    return shName.str();
  } else {
    return formatForSchema( nameCut, maxLength );
  }
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

//std::string
//ora::MappingRules::tableNameForVariable( const std::string& variableName,
//                                         const std::string& mainTableName )
//{
//  std::ostringstream result;
//  if ( ! mainTableName.empty() ) {
//    result << formatForSchema(mainTableName, ClassNameLengthForSchema);
//  }
//  if(result.str()[result.str().size()-1]!='_') result << "_";
//  result << tableNameForClass( variableName,MaxTableNameLength-ClassNameLengthForSchema );
//
//  return formatForSchema( result.str(),MaxTableNameLength);
//  }

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
ora::MappingRules::variableNameForArrayColumn(const std::string& arrayVariable,
                                              unsigned int arrayIndex ){
  std::ostringstream arrayElementLabel;
  arrayElementLabel << arrayVariable << "_I" << arrayIndex;
  return arrayElementLabel.str();
}

std::string
ora::MappingRules::referenceColumnKey( const std::string& tableName,
                                       const std::string& columnName )
{
  std::stringstream referenceColumn;
  referenceColumn << tableName << "." << columnName;
  return referenceColumn.str();
}

std::string ora::MappingRules::nameFromCArray( const std::string variableName,
                                               size_t maxLength ){
  if( !maxLength ) return "";
  std::stringstream totalString;
  size_t fp = variableName.find('[');
  if( fp == std::string::npos ){
    // other types
    totalString << shortNameByUpperCase( variableName, maxLength );
  }else{
    // process for c-arrays
    std::string arrayVar = variableName.substr(0,fp);
    std::string indexVar = variableName.substr(fp);
    for( size_t ind = 0; ind!=std::string::npos; ind=indexVar.find('[',ind+1 ) ){
      indexVar = indexVar.replace(ind,1,"A");
    }
    for( size_t ind = indexVar.find(']'); ind!=std::string::npos; ind=indexVar.find(']',ind+1 ) ){
      indexVar = indexVar.replace(ind,1,"_");
    }
    size_t arrayVarCut = 0;
    size_t varCut = variableName.size()+1;
    if( varCut>maxLength ) varCut = maxLength;
    if( varCut>(indexVar.size()+1) ) arrayVarCut = varCut-indexVar.size()-1;
    totalString << shortNameByUpperCase( arrayVar, arrayVarCut);
    totalString << "_" << indexVar;
  }
  return totalString.str();
}


**/
