#ifndef DDI_Store_h
#define DDI_Store_h

#include <map>
#include "DetectorDescription/Base/interface/rep_type.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
      typedef K key_type;
      typedef rep_type<name_type, pimpl_type> Rep_type;  
      typedef Rep_type* prep_type;
      typedef std::map<name_type,prep_type> registry_type;
      typedef typename registry_type::iterator iterator;
      
      auto begin() { return reg_.begin(); }
      auto end() { return reg_.end(); }
      auto size() const { return reg_.size(); } 
      
      // empty shell or fetch from registry
      prep_type create(const name_type &);
      
      // full new object or replace existing with new one
      prep_type create(const name_type &, pimpl_type);
      
      // clear all objects
      void clear();
      
      // swap moves the registry from this guy to another of the same type
      void swap ( Store& );
      
      bool isDefined(const name_type & n ) const;
      void setReadOnly(bool b) { readOnly_ = b; }
      bool readOnly() const { return readOnly_; }
      
      Store() : readOnly_(false) { }
      ~Store();
      
   protected:
      std::map<name_type,prep_type> reg_;
      Store(const Store &);
      Store & operator=(const Store &);
      bool readOnly_;
   };
   
   template<class N, class I, class K>
   typename Store<N,I,K>::prep_type 
   Store<N,I,K>::create(const name_type & n)
   {
      prep_type tmp = 0;
      auto result = reg_.insert(std::make_pair(n,tmp));
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
      auto result = reg_.insert(std::make_pair(n,tmp));
      if (!result.second) {
         delete result.first->second->second;
         result.first->second->second = p;
      }
      else {
         result.first->second = new Rep_type(n,p);
      }
      return result.first->second;
   }
   
   template<class N, class I, class K>
   Store<N,I,K>::~Store()
   {
      for( auto it : reg_ ) {
         delete it.second->second;
	 it.second->second = 0;
         delete it.second;
	 it.second = 0;
      }
   } 
   
   template<class N, class I, class K>
   bool Store<N,I,K>::isDefined(const name_type & n ) const
   {
      if (readOnly_) edm::LogWarning("DetectorDescriptionStore") << " Store is locked and most likely empty.  isDefined will be false.";
      auto it = reg_.find(n);
      bool result(false);
      if (it != reg_.end()) {
         if (it.second->second) {
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
   
} // namespace DDI

#endif
