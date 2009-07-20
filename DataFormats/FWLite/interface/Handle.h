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
// $Id: Handle.h,v 1.12 2009/07/12 05:09:08 srappocc Exp $
//

// system include files

// user include files
#if !defined(__CINT__) && !defined(__MAKECINT__)
//CINT can't handle parsing these files
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/FWLite/interface/EventBase.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/FWLite/interface/MultiChainEvent.h"
#include "DataFormats/FWLite/interface/ErrorThrower.h"
#endif

// forward declarations
namespace fwlite {
   class ErrorThrower;
   
template <class T>
class Handle
{

   public:
      // this typedef is to avoid a Cint bug where calling
      // edm::Wrapper<T>::typeInfo() fails for some types with
      // very long expansions.  First noticed for the expansion
      // of reco::JetTracksAssociation::Container
      typedef edm::Wrapper<T> TempWrapT;
 
      Handle() : data_(0), errorThrower_(ErrorThrower::unsetErrorThrower()) {}
      ~Handle() { delete errorThrower_;}

      Handle(const Handle<T>& iOther) : data_(iOther.data_),
      errorThrower_( iOther.errorThrower_? iOther.errorThrower_->clone(): 0) {}
   
      const Handle<T>& operator=(const Handle<T>& iOther) {
         Handle<T> temp(iOther);
         swap(iOther);
      }
   
      // ---------- const member functions ---------------------
      bool isValid() const { return data_ != 0; }
   
      ///Returns true only if Handle was used in a 'get' call and the data could not be found
      bool failedToGet() const {return errorThrower_ != 0; }
   
      T const* product() const { check(); return data_;}

      const T* ptr() const { check(); return data_;}
      const T& ref() const { check(); return *data_;}
  
      const T* operator->() const {
        check();
        return data_;
      }
  
      const T& operator*() const {
        check();
        return *data_;
      }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
  /*
      void getByBranchName(const fwlite::Event& iEvent, const char* iBranchName) {
        iEvent.getByBranchName(edm::Wrapper<T>::typeInfo(), iBranchName, data_);
      }
   */
  
      void getByLabel(const fwlite::EventBase& iEvent, 
                      const char* iModuleLabel,
                      const char* iProductInstanceLabel = 0,
                      const char* iProcessLabel = 0) {
        TempWrapT* temp;
        void* pTemp = &temp;
        iEvent.getByLabel(TempWrapT::typeInfo(),
                          iModuleLabel,
                          iProductInstanceLabel,
                          iProcessLabel,
                          pTemp);
        delete errorThrower_;
        errorThrower_ = 0;
        if(0==temp) {
           errorThrower_=ErrorThrower::errorThrowerBranchNotFoundException(TempWrapT::typeInfo(),
                                                                           iModuleLabel,
                                                                           iProductInstanceLabel,
                                                                           iProcessLabel);
	   return;
        }
        data_ = temp->product();
        if(data_==0) {
           errorThrower_=ErrorThrower::errorThrowerProductNotFoundException(TempWrapT::typeInfo(),
                                                                            iModuleLabel,
                                                                            iProductInstanceLabel,
                                                                            iProcessLabel);
        }
      }

      // void getByLabel(const fwlite::Event& iEvent, 
      //                 const char* iModuleLabel,
      //                 const char* iProductInstanceLabel = 0,
      //                 const char* iProcessLabel = 0) {
      //   TempWrapT* temp;
      //   void* pTemp = &temp;
      //   iEvent.getByLabel(TempWrapT::typeInfo(),
      //                     iModuleLabel,
      //                     iProductInstanceLabel,
      //                     iProcessLabel,
      //                     pTemp);
      //   delete errorThrower_;
      //   errorThrower_ = 0;
      //   if(0==temp) {
      //      errorThrower_=ErrorThrower::errorThrowerBranchNotFoundException(TempWrapT::typeInfo(),
      //                                                                      iModuleLabel,
      //                                                                      iProductInstanceLabel,
      //                                                                      iProcessLabel);
	  //  return;
      //   }
      //   data_ = temp->product();
      //   if(data_==0) {
      //      errorThrower_=ErrorThrower::errorThrowerProductNotFoundException(TempWrapT::typeInfo(),
      //                                                                       iModuleLabel,
      //                                                                       iProductInstanceLabel,
      //                                                                       iProcessLabel);
      //   }
      // }
      // 
      // void getByLabel(const fwlite::ChainEvent& iEvent, 
      //                 const char* iModuleLabel,
      //                 const char* iProductInstanceLabel = 0,
      //                 const char* iProcessLabel = 0) {
      // TempWrapT* temp;
      // void* pTemp = &temp;
      // iEvent.getByLabel(TempWrapT::typeInfo(),
      //                   iModuleLabel,
      //                   iProductInstanceLabel,
      //                   iProcessLabel,
      //                   pTemp);
      //    delete errorThrower_;
      //    errorThrower_ = 0;
      //    if(0==temp) {
      //       errorThrower_=ErrorThrower::errorThrowerBranchNotFoundException(TempWrapT::typeInfo(),
      //                                                                       iModuleLabel,
      //                                                                       iProductInstanceLabel,
      //                                                                       iProcessLabel);
	  //   return;
      //    }
      //    data_ = temp->product();
      //    if(data_==0) {
      //       errorThrower_=ErrorThrower::errorThrowerProductNotFoundException(TempWrapT::typeInfo(),
      //                                                                        iModuleLabel,
      //                                                                        iProductInstanceLabel,
      //                                                                        iProcessLabel);
      //    }
      // }
      // 
      // 
      // void getByLabel(const fwlite::MultiChainEvent& iEvent, 
      //                 const char* iModuleLabel,
      //                 const char* iProductInstanceLabel = 0,
      //                 const char* iProcessLabel = 0) {
      // TempWrapT* temp;
      // void* pTemp = &temp;
      // iEvent.getByLabel(TempWrapT::typeInfo(),
      //                   iModuleLabel,
      //                   iProductInstanceLabel,
      //                   iProcessLabel,
      //                   pTemp);
      //    if ( 0 != errorThrower_ ) delete errorThrower_;
      //    errorThrower_ = 0;
      //    if(0==temp) {
      //       errorThrower_=ErrorThrower::errorThrowerBranchNotFoundException(TempWrapT::typeInfo(),
      //                                                                       iModuleLabel,
      //                                                                       iProductInstanceLabel,
      //                                                                       iProcessLabel);
	  //   return;
      //    }
      //    data_ = temp->product();
      //    if(data_==0) {
      //       errorThrower_=ErrorThrower::errorThrowerProductNotFoundException(TempWrapT::typeInfo(),
      //                                                                        iModuleLabel,
      //                                                                        iProductInstanceLabel,
      //                                                                        iProcessLabel);
      //    }
      // }

      const std::string getBranchNameFor(const fwlite::EventBase& iEvent, 
                                         const char* iModuleLabel,
                                         const char* iProductInstanceLabel = 0,
                                         const char* iProcessLabel = 0) 
      {
         return iEvent.getBranchNameFor(TempWrapT::typeInfo(),
                                        iModuleLabel,
                                        iProductInstanceLabel,
                                        iProcessLabel);
      }
      
      // const std::string getBranchNameFor(const fwlite::Event& iEvent, 
      //                                    const char* iModuleLabel,
      //                                    const char* iProductInstanceLabel = 0,
      //                                    const char* iProcessLabel = 0) {
      //    return iEvent.getBranchNameFor(TempWrapT::typeInfo(),
      //                                   iModuleLabel,
      //                                   iProductInstanceLabel,
      //                                   iProcessLabel);
      // }
      // 
      // const std::string getBranchNameFor(const fwlite::ChainEvent& iEvent, 
      //                                    const char* iModuleLabel,
      //                                    const char* iProductInstanceLabel = 0,
      //                                    const char* iProcessLabel = 0) {
      //    return iEvent.getBranchNameFor(TempWrapT::typeInfo(),
      //                                   iModuleLabel,
      //                                   iProductInstanceLabel,
      //                                   iProcessLabel);
      // }
      // 
      // 
      // 
      // const std::string getBranchNameFor(const fwlite::MultiChainEvent& iEvent, 
      //                                    const char* iModuleLabel,
      //                                    const char* iProductInstanceLabel = 0,
      //                                    const char* iProcessLabel = 0) {
      //    return iEvent.getBranchNameFor(TempWrapT::typeInfo(),
      //                                   iModuleLabel,
      //                                   iProductInstanceLabel,
      //                                   iProcessLabel);
      // }

   void swap( Handle<T>& iOther) {
      const T* temp = data_;
      data_ = iOther.data_;
      iOther.data_ = temp;
      ErrorThrower* tempE = errorThrower_;
      errorThrower_ = iOther.errorThrower_;
      iOther.errorThrower = tempE;
   }
   private:
      void check() const { if(errorThrower_) { errorThrower_->throwIt();} }


      // ---------- member data --------------------------------
      const T* data_;
      ErrorThrower*  errorThrower_;
};

}
#endif
