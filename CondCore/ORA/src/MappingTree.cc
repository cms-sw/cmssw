#include "MappingTree.h"

ora::MappingTree::MappingTree():
  m_version( "" ),
  m_element(){
}

ora::MappingTree::MappingTree( const std::string& version ):
  m_version( version ),
  m_element(){
}

ora::MappingElement&
ora::MappingTree::setTopElement( const std::string& className,
                                 const std::string& tableName,
                                 bool isDependency ){
  std::string elementType = ora::MappingElement::classMappingElementType();
  if( isDependency ) elementType = ora::MappingElement::dependencyMappingElementType();
  m_element = ora::MappingElement( elementType,
                                   className,
                                   className,
                                   tableName );
  return m_element;
}

void ora::MappingTree::override(const MappingTree& source)
{
  if( className() == source.className() ) m_element.override( source.m_element );
}

