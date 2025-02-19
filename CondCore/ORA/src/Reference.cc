#include "CondCore/ORA/interface/Reference.h"

ora::Reference::Reference():
  m_containerId(0),
  m_itemId(0){
}

ora::Reference::Reference( const OId& oid ):
  m_containerId(oid.containerId()),
  m_itemId(oid.itemId()){
}

ora::Reference::Reference( const ora::Reference& rhs ):
  m_containerId(rhs.m_containerId),
  m_itemId(rhs.m_itemId){
}

ora::Reference::~Reference(){
}

ora::Reference& ora::Reference::operator=( const ora::Reference& rhs ){
  m_containerId = rhs.m_containerId;
  m_itemId = rhs.m_itemId;
  return *this;
}

void ora::Reference::set( const ora::OId& oid ){
  m_containerId = oid.containerId();
  m_itemId = oid.itemId();  
}

ora::OId ora::Reference::oid() const {
  return OId( m_containerId, m_itemId );
}

  
    
    
