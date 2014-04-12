#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/Singleton.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <sstream>

std::ostream & operator<<(std::ostream & os, const DDName & n)
{ 
  os << n.ns() << ':' << n.name();
  static const char * ev = getenv("DDNAMEID");
  if (ev) os << '[' << n.id() << ']';
  return os;
}  

// Implementation of DDName

DDName::DDName( const std::string & name, const std::string & ns)
 : id_(registerName(std::make_pair(name,ns))->second)
{ }


DDName::DDName( const std::string & name )
 : id_(0)
{ 
  std::pair<std::string,std::string> result = DDSplit(name);
  if (result.second == "") {
    id_ = registerName(std::make_pair(result.first,DDCurrentNamespace::ns()))->second;
  }  
  else {
    id_ = registerName(result)->second;
  }
}


DDName::DDName( const char* name )
 : id_(0)
{ 
  std::pair<std::string,std::string> result = DDSplit(name);
  if (result.second == "") {
    id_ = registerName(std::make_pair(result.first,DDCurrentNamespace::ns()))->second;
  }  
  else {
    id_ = registerName(result)->second;
  }
}


DDName::DDName( const char* name, const char* ns)
 : id_(registerName(std::make_pair(std::string(name),std::string(ns)))->second)
{ }


DDName::DDName()
 : id_(0)
{ } 

DDName::DDName(DDName::id_type id)
 : id_(id)
{ }

void DDName::defineId(const std::pair<std::string,std::string> & nm, DDName::id_type id)
{ 
  IdToName & id2n = DDI::Singleton<IdToName>::instance();
  
  /* 
    Semantics:
    If id exists && the registered value matches the given one, do nothing
    If id exists && the registered value DOES NOT match, throw
    If id DOES NOT exists, register with the given value
  */
  if ( id < id_type(id2n.size()) ) {
    if(id2n[id]->first != nm) {
      std::stringstream s;
      s << id;
      throw cms::Exception("DDException") << "DDName::DDName(std::pair<std::string,std::string>,id_type): id=" + s.str() + " reg-name=?";
    }
  }
  else {
    id2n.resize(id+1);
    DDI::Singleton<Registry>::instance()[nm]=id;
    id2n[id] = DDI::Singleton<Registry>::instance().find(nm);
  }
}

const std::string & DDName::name() const 
{
  const static std::string ano_("anonymous");
  const std::string * result;
  if (id_ < 0) {
      result = &ano_;
  }
  else {
    result = &DDI::Singleton<IdToName>::instance()[id_]->first.first;
  }
  return *result; 
}


const std::string & DDName::ns() const
{
  const static std::string ano_("anonymous");
  const std::string * result;
  if (id_ < 0) {
      result = &ano_;
  }
  else {
    result = &DDI::Singleton<IdToName>::instance()[id_]->first.second;
  }
  return *result;
}



bool DDName::exists(const std::string & name, const std::string & ns)
{
   const std::pair<std::string,std::string> p(name,ns);
   Registry::const_iterator it = DDI::Singleton<Registry>::instance().find(p);
   return it != DDI::Singleton<Registry>::instance().end() ? true : false;
}


DDName::Registry::iterator DDName::registerName(const std::pair<std::string,std::string> & nm) {
    Registry& reg_ = DDI::Singleton<Registry>::instance();
    IdToName & idToName = DDI::Singleton<IdToName>::instance();  
    Registry::size_type sz = reg_.size();
    if (!sz) {
      reg_[std::make_pair(std::string(""),std::string(""))] = 0;
      idToName.push_back(reg_.begin());
      ++sz;
    }
    Registry::value_type val(nm, sz);
    std::pair<Registry::iterator,bool> result = reg_.insert(val);
    if (result.second) {
      idToName.push_back(result.first);
    }
    return result.first;
}

