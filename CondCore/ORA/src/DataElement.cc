#include "DataElement.h"
#include <stddef.h>     /* offsetof */

ora::DataElement::DataElement():
  m_parent(0),
  m_children(),
  m_declaringScopeOffset(0),
  /*m_offsetFunction(0)*/ m_offset(0){
}

ora::DataElement::DataElement( size_t declaringScopeOffset,
                               size_t /*offsetFunc*/ offset ):
  m_parent(0),
  m_children(),
  m_declaringScopeOffset(declaringScopeOffset),
  /*m_offsetFunction(offsetFunc)*/ m_offset(offset){
}

ora::DataElement::~DataElement(){
  for(std::vector<DataElement*>::const_iterator iEl = m_children.begin();
      iEl != m_children.end(); ++iEl ){
    delete *iEl;
  }
}

ora::DataElement&
ora::DataElement::addChild( size_t declaringScopeOffset,
                            size_t /*offsetFunction*/ offset ){
  DataElement* child = new DataElement( declaringScopeOffset, /*offsetFunction*/ offset );
  child->m_parent = this;
  m_children.push_back(child);
  return *child;
}

size_t ora::DataElement::offset( const void* topLevelAddress ) const {
  //const void* address = topLevelAddress;
  size_t offset = m_declaringScopeOffset;
  if(m_parent){
    size_t parentOffset = m_parent->offset( topLevelAddress );
    offset += parentOffset;
    //address = static_cast<char*>(const_cast<void*>(topLevelAddress))+parentOffset;
  }
  if(/*m_offsetFunction*/ m_offset){
    offset += /*m_offsetFunction( const_cast<void*>(address))*/ m_offset;
  }
  return offset;  
}

void* ora::DataElement::address( const void* topLevelAddress ) const {
  void* elementAddress = static_cast< char* >( const_cast<void*>(topLevelAddress)) + offset( topLevelAddress );
  return elementAddress;
}

size_t ora::DataElement::declaringScopeOffset() const 
{
  return m_declaringScopeOffset;
}

void ora::DataElement::clear(){
  for(std::vector<DataElement*>::const_iterator iEl = m_children.begin();
      iEl != m_children.end(); ++iEl ){
    delete *iEl;
  }
  m_children.clear();
}

