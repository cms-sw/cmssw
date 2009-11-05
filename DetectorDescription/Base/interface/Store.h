#ifndef DDI_Store_h
#define DDI_Store_h

#include <map>
/* #include <iostream> */
#include <DetectorDescription/Base/interface/rep_type.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
//;
//FIXME: Store : implement readOnly-behaviour ..
namespace DDI {

 /** 
   A Store provides a place for objects of type I which are uniquely identified
   by there name of type N. The objects themselves can be accessed indirectly
   by the prep_type of the Store like
   
   typedef Store<std::string,double> NamedDouble;
   NamedDouble::prep_type d = NamedDouble("Four", new double(4.));
   double val  = *(d->second);
   std::string name = d->first;

   K is the key_type which is used as an index in the storage.
   It must fulfill all requirements for a key in a sorted associative container.
   N is the user-friendly name_type, which must be mapped uniquely
   to the key_type and vice versa. N itself must also fulfill all requirements
   of a key in a sorted associative container.
   The reason to provide K is that one might save some memory by compacting
   common parts of information contained in different instances of N, e.g.
   if N is a pair<string,string>, the first string being a 'namespace' the 
   second a 'name' then K could be a pair<int,string> thus compacting the
   namespace-string to a simple int.
   K and N must support following unique conversions:
   - from N to K, thus N(const K &)
   - from K to N, thus K(const N &)
 */  
 template <class N, class I, class K=I>
 class Store 
 {
 public:
   typedef N name_type;
   typedef I  pimpl_type;
   //typedef I* pimpl_type;
   typedef K key_type;
   typedef rep_type<name_type, pimpl_type> Rep_type;  
   // direct way - no comfortable dereferencing of a pointer to rep_type
   //typedef typename std::pair<name_type,pimpl_type> rep_type;
  /*
   struct rep_type {
     rep_type(const name_type & n, pimpl_type p) : first(n), second(p) { }
     key_type first;
     pimpl_type second;
   };
  */ 
   //typedef typename Rep_type<N,I> rep_type;
   
   typedef Rep_type* prep_type;
   
   typedef std::map<name_type,prep_type> registry_type;
   typedef typename registry_type::iterator iterator;
   typedef typename registry_type::const_iterator const_iterator;
   typedef typename registry_type::size_type size_type;
   
   iterator begin() { return reg_.begin(); }
   const_iterator begin() const { return reg_.begin(); }
   iterator end() { return reg_.end(); }
   const_iterator end() const { return reg_.end(); }
   size_type size() const { return reg_.size(); } 
   
   // empty shell or fetch from registry
   prep_type create(const name_type &);
   
   // full new object or replace existing with new one
   prep_type create(const name_type &, pimpl_type);
   
   // anonymous object, not stored
   prep_type create(pimpl_type);
   
   // clear all objects
   void clear();

   // swap moves the registry from this guy to another of the same 
   // type
   void swap ( Store& ) ; 
   
   bool isDefined(const name_type & n ) const;
   void setReadOnly(bool b) { readOnly_ = b; }
   bool readOnly() const { return readOnly_; }
   
   Store() : readOnly_(false) { }
   ~Store();

 protected:
   registry_type reg_;
   Store(const Store &);
   Store & operator=(const Store &);
   bool readOnly_;
 };

 
//   rep_type as nested type inside Store, currently gives
//   internal compiler error on gcc-2.95.2, but works on SUNWspro62Apr02Sun
//    
// template<class N, class I, class K>
// Store<N,I,K>::rep_type::rep_type(const name_type & n, pimpl_type p)
//  : std::pair(n,p)
// { }

template<class N, class I, class K>
void
Store<N,I,K>::clear()
{
  typename registry_type::iterator it = reg_.begin();
  for (; it != reg_.end(); ++it) {
    delete it->second->second;
    delete it->second;
  }  
  reg_.clear();
}

 template<class N, class I, class K>
 typename Store<N,I,K>::prep_type 
 Store<N,I,K>::create(const name_type & n)
 {
   prep_type tmp = 0;
   std::pair<typename registry_type::iterator,bool> result 
     = reg_.insert(std::make_pair(n,tmp));
   if (result.second) {
     if (readOnly_) throw cms::Exception("DetectorDescriptionStore")<<" Store has been locked.  Illegal attempt to add " << n << " to a global store."; 
     // ELSE     
     result.first->second = new Rep_type(n,(I)0);
   }
   return result.first->second;    
 }
 
 
 template<class N, class I, class K>
 typename Store<N,I,K>::prep_type 
 Store<N,I,K>::create(const name_type & n, 
                      pimpl_type p)
 {			
   if (readOnly_) throw cms::Exception("DetectorDescriptionStore")<<" Store has been locked.  Illegal attempt to add " << n << " to a global store."; 
   // ELSE     
   prep_type tmp = 0;
   std::pair<typename registry_type::iterator,bool> result 
     = reg_.insert(std::make_pair(n,tmp));
   if (!result.second) {
     delete result.first->second->second;
     result.first->second->second = p;
     //delete result.first->second->swap(p);
   }
   else {
     result.first->second = new Rep_type(n,p);
   }
   return result.first->second;
 }

 
 template<class N, class I, class K>
 typename Store<N,I,K>::prep_type 
 Store<N,I,K>::create(typename Store<N,I,K>::pimpl_type p)
 {					  
   if (readOnly_) throw cms::Exception("DetectorDescriptionStore")<<" Store has been locked.  Illegal attempt to add " << name_type() << " to a global store."; 
    return new Rep_type(name_type(),p);
 }


 template<class N, class I, class K>
 Store<N,I,K>::~Store()
 {
   typename registry_type::iterator it = reg_.begin();
   for(; it != reg_.end(); ++it) {
//     std::cout << "deleting " << it->first << std::endl;
     delete it->second->second; it->second->second = 0;
     delete it->second; it->second = 0;
   }
 } 

template<class N, class I, class K>
bool Store<N,I,K>::isDefined(const name_type & n ) const
{
  if (readOnly_) edm::LogWarning("DetectorDescriptionStore") << " Store is locked and most likely empty.  isDefined will be false.";
  typename registry_type::const_iterator it = reg_.find(n);
  bool result(false);
  if (it != reg_.end()) {
    if (it->second->second) {
      result=true;
    }  
  }
  return result;
}

template<class N, class I, class K>
void Store<N, I, K>::swap ( Store<N, I, K>& storeToSwap ) {
  reg_.swap(storeToSwap.reg_);
  storeToSwap.readOnly_ = readOnly_;
}
}

#endif
