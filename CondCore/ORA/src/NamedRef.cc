#include "CondCore/ORA/interface/NamedRef.h"

ora::NamedReference::NamedReference():
  m_name(""),m_isPersistent(false),m_ptr(){
}
    
ora::NamedReference::NamedReference( const std::string& name ):
  m_name(name),m_isPersistent(false),m_ptr(){
}

ora::NamedReference::NamedReference( const std::string& name, boost::shared_ptr<void> ptr ):  
  m_name(name),m_isPersistent(false),m_ptr( ptr ){
}

ora::NamedReference::NamedReference( const NamedReference& rhs ):
  m_name(rhs.m_name),m_isPersistent(rhs.m_isPersistent),m_ptr(rhs.m_ptr){
}

ora::NamedReference::~NamedReference(){
}

ora::NamedReference::NamedReference& ora::NamedReference::operator=( const NamedReference& rhs ){
  if( this != &rhs ){
    m_name = rhs.m_name;
    m_isPersistent = rhs.m_isPersistent;
    m_ptr = rhs.m_ptr;
  }
  return *this;
}

void ora::NamedReference::set( const std::string& name ){
  m_name = name;
  m_isPersistent = false;
}
 
const std::string& ora::NamedReference::name() const {
  return m_name;
}

bool ora::NamedReference::isPersistent() const {
  return m_isPersistent;
}

boost::shared_ptr<void>& ora::NamedReference::ptr() const {
  return m_ptr;
}
