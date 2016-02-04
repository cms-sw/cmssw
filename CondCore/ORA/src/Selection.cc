#include "CondCore/ORA/interface/Selection.h"
#include "CondCore/ORA/interface/Exception.h"
//
#include <sstream>
//
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"

std::vector<std::string>& ora::Selection::selectionTypes(){
  static std::vector<std::string> types;
  types.push_back("=");
  types.push_back("!=");
  types.push_back(">");
  types.push_back(">=");
  types.push_back("<");
  types.push_back("<=");
  return types;
}

std::string  ora::Selection::variableNameFromUniqueString(const std::string& uniqueString)
{
  size_t ind = uniqueString.rfind("_");
  return uniqueString.substr(0,ind);
}

std::string ora::Selection::indexVariable(){
  static std::string s_var("ora::ContainerIndex");
  return s_var;
}

ora::Selection::Selection():m_items(),m_data( new coral::AttributeList ){
}

ora::Selection::~Selection(){
}

ora::Selection::Selection( const ora::Selection& rhs ):
  m_items( rhs.m_items ),
  m_data( new coral::AttributeList( *rhs.m_data )){
}

ora::Selection& ora::Selection::operator=( const ora::Selection& rhs ){
  m_items = rhs.m_items;
  m_data.reset( new coral::AttributeList( *rhs.m_data ) );
  return *this;
}

std::string
ora::Selection::uniqueVariableName(const std::string& varName) const {
  std::stringstream uniqueVarName;
  unsigned int i = 0;
  bool notUnique = true;
  while(notUnique){
    bool found = false;
    uniqueVarName.str("");
    uniqueVarName << varName;
    uniqueVarName << "_" << i;
    for(coral::AttributeList::const_iterator iAttr = m_data->begin();
        iAttr!=m_data->end() && !found; ++iAttr){
      if( iAttr->specification().name() == uniqueVarName.str() ) found = true;
    }
    notUnique = found;
    i++;
  }
  return uniqueVarName.str();
}

void ora::Selection::addIndexItem( int startIndex, 
                                   int endIndex ){
  if(endIndex<startIndex && endIndex>=0) {
    throwException("Cannot select with endIndex<startIndex.",
                   "Selection::addIndexItem");
  } else if( startIndex==endIndex && endIndex>=0){
    std::string varName = uniqueVariableName( indexVariable() );
    SelectionItemType selType = ora::EQ;
    m_items.push_back(std::make_pair(varName,selectionTypes()[selType]));
    m_data->extend<int>(varName);
    (*m_data)[varName].data<int>() = startIndex;    
  } else {
    if(startIndex>0){
      std::string varName0 = uniqueVariableName( indexVariable() );
      SelectionItemType firstType = ora::GE;
      m_items.push_back(std::make_pair(varName0,selectionTypes()[firstType]));
      m_data->extend<int>(varName0);
      (*m_data)[varName0].data<int>() = startIndex;
    }
    if(endIndex>0){
      std::string varName1 = uniqueVariableName( indexVariable() );
      SelectionItemType secondType = ora::LE;
      m_items.push_back(std::make_pair(varName1,selectionTypes()[secondType]));
      m_data->extend<int>(varName1);
      (*m_data)[varName1].data<int>() = endIndex;
    }
  }
}

void ora::Selection::addUntypedDataItem( const std::string& dataMemberName, 
                                          SelectionItemType stype, 
                                          const std::type_info& primitiveType, 
                                          void* data ){
  std::string varName = uniqueVariableName( dataMemberName );
  m_items.push_back(std::make_pair(varName,selectionTypes()[stype]));
  m_data->extend( varName, primitiveType );
  (*m_data)[varName].setValueFromAddress( data );
}

bool
ora::Selection::isEmpty() const {
  return m_items.empty();
}

const std::vector<std::pair<std::string,std::string> >&
ora::Selection::items() const {
  return m_items;
}

const coral::AttributeList&
ora::Selection::data() const {
  return *m_data;
}
