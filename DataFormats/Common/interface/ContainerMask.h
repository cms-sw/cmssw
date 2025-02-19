#ifndef DataFormats_Common_ContainerMask_h
#define DataFormats_Common_ContainerMask_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     ContainerMask
// 
/**\class ContainerMask ContainerMask.h DataFormats/Common/interface/ContainerMask.h

 Description: Provides a 'mask' associated with one container

 Usage:
    This class is used in conjunction with a container in the event. The container 
    must be 'indexable', i.e. the elements must be addressable via an unsigned int.

*/
//
// Original Author:  
//         Created:  Fri Sep 23 17:05:43 CDT 2011
// $Id: ContainerMask.h,v 1.1 2011/12/01 13:02:17 vlimant Exp $
//

// system include files
#include <vector>
#include <cassert>
#include <algorithm>

// user include files
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/ContainerMaskTraits.h"
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

// forward declarations
namespace edm {
   template< typename T>
   class ContainerMask {

   public:
      ContainerMask() {}
      ContainerMask(const edm::RefProd<T>& iProd, const std::vector<bool>& iMask);
      //virtual ~ContainerMask();

      // ---------- const member functions ---------------------
     	bool mask(unsigned int iIndex) const {
         assert(iIndex<m_mask.size());
         return m_mask[iIndex];
     	}
     	bool mask(const typename ContainerMaskTraits<T>::value_type *);
     	void applyOrTo( std::vector<bool>&) const;
     	void copyMaskTo( std::vector<bool>&) const;


      size_t size() const { return m_mask.size();}

      const edm::RefProd<T>& refProd() const {return m_prod;}
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void swap(ContainerMask<T>& iOther);

      //Used by ROOT storage
      CMS_CLASS_VERSION(10)
      
   private:
      //ContainerMask(const ContainerMask&); // stop default

      //const ContainerMask& operator=(const ContainerMask&); // stop default

      // ---------- member data --------------------------------
      edm::RefProd<T> m_prod;
      std::vector<bool> m_mask;
   };

   template<typename T>
   ContainerMask<T>::ContainerMask(const edm::RefProd<T>& iProd, const std::vector<bool>& iMask):
   m_prod(iProd), m_mask(iMask) {
      assert(iMask.size() == ContainerMaskTraits<T>::size(m_prod.product()));
   }
   
   
   template<typename T>
   bool ContainerMask<T>::mask(const typename ContainerMaskTraits<T>::value_type * iElement )
   {
      unsigned int index = ContainerMaskTraits<T>::indexFor(iElement,m_prod.product());
      return this->mask(index);
   }
   
   template<typename T>
   void ContainerMask<T>::copyMaskTo(std::vector<bool>& iTo) const {
      iTo.assign(m_mask.begin(),m_mask.end());
   }
   
   template<typename T>
   void ContainerMask<T>::applyOrTo(std::vector<bool>& iTo) const {
      assert(iTo.size()==m_mask.size());
      std::transform(m_mask.begin(),m_mask.end(),iTo.begin(), iTo.begin(),std::logical_or<bool>());
   }
   
   template<typename T>
   void ContainerMask<T>::swap(ContainerMask<T>& iOther) {
      m_prod.swap(iOther.m_prod);
      std::swap(m_mask,iOther.m_mask);
   }
}

#endif
