#ifndef DDD_DDOBJECT_H
#define DDD_DDOBJECT_H

#include <string>
#include <map>
#include "Utilities/Loki/interface/TypeManip.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Base/interface/Singleton.h"
#include "DetectorDescription/Base/interface/Ptr.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDRegistry.h"


template<class T>
class DDObject
{
public:
  typedef Ptr<T> pointer;
  //! std::maps a DDName to the pointer to be wrapped
  typedef DDRegistry<pointer> RegistryType; 
  //! iterator
  typedef typename RegistryType::iterator iterator;
  
  typedef typename RegistryType::const_iterator const_iterator;
  
public:
  //! unitialized, anonymous object
  DDObject();
  
  //! initialized (named) object
  explicit DDObject(const DDName & name);
  
  //! initialized (named) object, implementation provided by T*
  DDObject(const DDName &, T*);
  
  //! named object, implemenation derived from DDName & thus provides the name
  explicit DDObject(DDName*);
  
  //! anonymous (un-named) object
  //explicit DDObject(T*);
  
  //! the name of the wrapped object
  const DDName & name() const;
  
  const T* operator->() const;
  
  T* operator->();
  
  const T& operator*() const;
  
  T& operator*();
  
  operator bool() const 
  {
    if (rep_ != registry().end()) {
      return bool(rep_->second);
    }
    throw DDException("unbound DDObject, typeid.name="+std::string(typeid(*rep_->second).name()));
  }; 
  
  static iterator begin() { return registry().begin(); }
  //static const_iterator begin() const { return registry().begin();}
  static iterator end() { return registry().end();}
  //static const_iterator end() const { return registry().end();}
  
private:
 
  void init();
  
  void registerObject(const DDName & name, T* object);
  void registerAnoObject(const DDName & name, T* object);
  
  //! the static registry for the wrapped pointer
  static RegistryType & registry()
  {
    static RegistryType reg_;
    return reg_;
  };
  
  //! static registry for anonymous pointers (un-named pointers)
  static RegistryType & anoRegistry()
  {
    static RegistryType reg_;
    return reg_;
  }
  
  //! representation of the wrapped object is a pointer into a registry-std::map
  typename RegistryType::iterator rep_;
  
};


template<class T>
DDObject<T>::DDObject()
: rep_(registry().end()) 
 { }


template<class T>
DDObject<T>::DDObject(const DDName & name)  
{ 
  registerObject(name,(T*)0);
}


template<class T>
DDObject<T>::DDObject(const DDName & name, T* object)  
{ 
   registerObject(name,object);
}

/*
template<class T>
DDObject<T>::DDObject(T* object)  
{ 
  static int id_(-1);
  DDName name(id_);
  --id_;
  registerAnoObject(name,object);
}
*/
 
template<class T>
DDObject<T>::DDObject(DDName* object)  
{ 
  // Loki compile time check whether T is sub-classed from DDName
  //SUPERSUBCLASS(DDName,T);
  registerObject(*object,static_cast<T*>(object));
}


template<class T>
void DDObject<T>::registerObject(const DDName & name, T* object)
{
  std::pair<typename RegistryType::iterator,bool> result 
    = registry().insert(typename RegistryType::value_type(name,pointer(object)));
  if (!result.second) {
    result.first->second = pointer(object);
  }
  rep_ = result.first; 
}


template<class T>
void DDObject<T>::registerAnoObject(const DDName & name, T* object)
{
  std::pair<typename RegistryType::iterator,bool> result 
    = anoRegistry().insert(typename RegistryType::value_type(name,pointer(object)));
  if (!result.second) {
    result.first->second = pointer(object);
  }
  rep_ = result.first; 
}

template<class T>
const DDName & DDObject<T>::name() const
{
  //static DDName name_;
  return rep_->first;	
}


template<class T>
const T& DDObject<T>::operator*() const
{
   return *(rep_->second);	
}


template<class T>
T& DDObject<T>::operator*()
{
  return *(rep_->second);
}


template<class T>
const T* DDObject<T>::operator->() const
{
	return &(*rep_->second);
}


template<class T>
T* DDObject<T>::operator->()
{
	return &(*rep_->second);
}


#endif // DDD_DDOBJECT_H
