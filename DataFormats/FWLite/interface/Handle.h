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
// $Id: Handle.h,v 1.6 2007/08/06 15:01:55 chrjones Exp $
//

// system include files

// user include files
#if !defined(__CINT__) && !defined(__MAKECINT__)
//CINT can't handle parsing these files
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#endif

// forward declarations
namespace fwlite {
template <class T>
class Handle
{

   public:
      // this typedef is to avoid a Cint bug where calling
      // edm::Wrapper<T>::typeInfo() fails for some types with
      // very long expansions.  First noticed for the expansion
      // of reco::JetTracksAssociation::Container
      typedef edm::Wrapper<T> TempWrapT;

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
        TempWrapT* temp;
        void* pTemp = &temp;
        iEvent.getByLabel(TempWrapT::typeInfo(),
                          iModuleLabel,
                          iProductInstanceLabel,
                          iProcessLabel,
                          pTemp);
data_ = temp->product();
        if(data_==0) {
          iEvent.throwProductNotFoundException(TempWrapT::typeInfo(),
                                               iModuleLabel,
                                               iProductInstanceLabel,
                                               iProcessLabel);
        }
      }

  void getByLabel(const fwlite::ChainEvent& iEvent, 
                  const char* iModuleLabel,
                  const char* iProductInstanceLabel = 0,
                  const char* iProcessLabel=0) {
    TempWrapT* temp;
    void* pTemp = &temp;
    iEvent.getByLabel(TempWrapT::typeInfo(),
                      iModuleLabel,
                      iProductInstanceLabel,
                      iProcessLabel,
                      pTemp);
    data_ = temp->product();
    if(data_==0) {
      iEvent.throwProductNotFoundException(TempWrapT::typeInfo(),
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
