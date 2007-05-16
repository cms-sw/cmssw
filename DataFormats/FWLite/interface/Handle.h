#ifndef DataFormats_FWLite_Handle_h
#define DataFormats_FWLite_Handle_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     Handle
// 
/**\class Handle Handle.h DataFormats/FWLite/interface/Handle.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue May  8 15:01:26 EDT 2007
// $Id: Handle.h,v 1.3 2007/05/12 20:27:06 chrjones Exp $
//

// system include files

// user include files
#ifndef __CINT__
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/FWLite/interface/Event.h"
#endif

// forward declarations
namespace fwlite {
template <class T>
class Handle
{

   public:
      Handle() : data_(0){}
      //virtual ~Handle();

      // ---------- const member functions ---------------------
      const T* ptr() const { return data_;}
      const T& ref() const { return *data_;}
  
      const T* operator->() const {
        return data_;
      }
  
      const T& operator*() const {
        return *data_;
      }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
  /*
      void getByBranchName(const fwlite::Event& iEvent, const char* iBranchName) {
        iEvent.getByBranchName(edm::Wrapper<T>::typeInfo(), iBranchName, data_);
      }
   */
  
      void getByLabel(const fwlite::Event& iEvent, 
                      const char* iModuleLabel,
                      const char* iProductInstanceLabel = 0,
                      const char* iProcessLabel=0) {
        edm::Wrapper<T>* temp;
        void* pTemp = &temp;
        iEvent.getByLabel(edm::Wrapper<T>::typeInfo(),
                          iModuleLabel,
                          iProductInstanceLabel,
                          iProcessLabel,
                          pTemp);
data_ = temp->product();
        if(data_==0) {
          iEvent.throwProductNotFoundException(edm::Wrapper<T>::typeInfo(),
                                               iModuleLabel,
                                               iProductInstanceLabel,
                                               iProcessLabel);
        }
      }
  
   private:
      //Handle(const Handle&); // stop default

      //const Handle& operator=(const Handle&); // stop default

      // ---------- member data --------------------------------
      const T* data_;
};

}
#endif
