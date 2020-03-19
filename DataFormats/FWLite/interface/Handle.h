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
//

// system include files

// user include files
namespace edm {
  template <typename T>
  class Wrapper;
}

#include "DataFormats/FWLite/interface/EventBase.h"
#include "DataFormats/FWLite/interface/ErrorThrower.h"
#include "DataFormats/Common/interface/Wrapper.h"

// forward declarations
namespace fwlite {
  class ErrorThrower;
  class EventBase;

  template <class T>
  class Handle {
  public:
    // this typedef is to avoid a Cint bug where calling
    // edm::Wrapper<T>::typeInfo() fails for some types with
    // very long expansions.  First noticed for the expansion
    // of reco::JetTracksAssociation::Container
    typedef edm::Wrapper<T> TempWrapT;

    Handle() : data_(nullptr), errorThrower_(ErrorThrower::unsetErrorThrower()) {}
    ~Handle() { delete errorThrower_; }

    Handle(const Handle<T>& iOther)
        : data_(iOther.data_), errorThrower_(iOther.errorThrower_ ? iOther.errorThrower_->clone() : nullptr) {}

    const Handle<T>& operator=(const Handle<T>& iOther) {
      Handle<T> temp(iOther);
      swap(temp);
      return *this;
    }

    // ---------- const member functions ---------------------
    bool isValid() const { return data_ != nullptr; }

    ///Returns true only if Handle was used in a 'get' call and the data could not be found
    bool failedToGet() const { return errorThrower_ != nullptr; }

    T const* product() const {
      check();
      return data_;
    }

    const T* ptr() const {
      check();
      return data_;
    }
    const T& ref() const {
      check();
      return *data_;
    }

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

    // P can be Run, Event, or LuminosityBlock

    template <class P>
    void getByLabel(const P& iP,
                    const char* iModuleLabel,
                    const char* iProductInstanceLabel = nullptr,
                    const char* iProcessLabel = nullptr) {
      TempWrapT* temp;
      void* pTemp = &temp;
      iP.getByLabel(TempWrapT::typeInfo(), iModuleLabel, iProductInstanceLabel, iProcessLabel, pTemp);
      delete errorThrower_;
      errorThrower_ = nullptr;
      if (nullptr == temp) {
        errorThrower_ = ErrorThrower::errorThrowerBranchNotFoundException(
            TempWrapT::typeInfo(), iModuleLabel, iProductInstanceLabel, iProcessLabel);
        return;
      }
      data_ = temp->product();
      if (data_ == nullptr) {
        errorThrower_ = ErrorThrower::errorThrowerProductNotFoundException(
            TempWrapT::typeInfo(), iModuleLabel, iProductInstanceLabel, iProcessLabel);
      }
    }

    // void getByLabel(const fwlite::Event& iEvent,
    //                 const char* iModuleLabel,
    //                 const char* iProductInstanceLabel = nullptr,
    //                 const char* iProcessLabel = nullptr) {
    //   TempWrapT* temp;
    //   void* pTemp = &temp;
    //   iEvent.getByLabel(TempWrapT::typeInfo(),
    //                     iModuleLabel,
    //                     iProductInstanceLabel,
    //                     iProcessLabel,
    //                     pTemp);
    //   delete errorThrower_;
    //   errorThrower_ = nullptr;
    //   if(nullptr == temp) {
    //      errorThrower_=ErrorThrower::errorThrowerBranchNotFoundException(TempWrapT::typeInfo(),
    //                                                                      iModuleLabel,
    //                                                                      iProductInstanceLabel,
    //                                                                      iProcessLabel);
    //  return;
    //   }
    //   data_ = temp->product();
    //   if(data_ == nullptr) {
    //      errorThrower_=ErrorThrower::errorThrowerProductNotFoundException(TempWrapT::typeInfo(),
    //                                                                       iModuleLabel,
    //                                                                       iProductInstanceLabel,
    //                                                                       iProcessLabel);
    //   }
    // }
    //
    // void getByLabel(const fwlite::ChainEvent& iEvent,
    //                 const char* iModuleLabel,
    //                 const char* iProductInstanceLabel = nullptr,
    //                 const char* iProcessLabel = nullptr) {
    // TempWrapT* temp;
    // void* pTemp = &temp;
    // iEvent.getByLabel(TempWrapT::typeInfo(),
    //                   iModuleLabel,
    //                   iProductInstanceLabel,
    //                   iProcessLabel,
    //                   pTemp);
    //    delete errorThrower_;
    //    errorThrower_ = nullptr;
    //    if(nullptr == temp) {
    //       errorThrower_=ErrorThrower::errorThrowerBranchNotFoundException(TempWrapT::typeInfo(),
    //                                                                       iModuleLabel,
    //                                                                       iProductInstanceLabel,
    //                                                                       iProcessLabel);
    //   return;
    //    }
    //    data_ = temp->product();
    //    if(data_ == nullptr) {
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
    //                 const char* iProductInstanceLabel = nullptr,
    //                 const char* iProcessLabel = nullptr) {
    // TempWrapT* temp;
    // void* pTemp = &temp;
    // iEvent.getByLabel(TempWrapT::typeInfo(),
    //                   iModuleLabel,
    //                   iProductInstanceLabel,
    //                   iProcessLabel,
    //                   pTemp);
    //    if ( nullptr != errorThrower_ ) delete errorThrower_;
    //    errorThrower_ = nullptr;
    //    if(nullptr == temp) {
    //       errorThrower_=ErrorThrower::errorThrowerBranchNotFoundException(TempWrapT::typeInfo(),
    //                                                                       iModuleLabel,
    //                                                                       iProductInstanceLabel,
    //                                                                       iProcessLabel);
    //   return;
    //    }
    //    data_ = temp->product();
    //    if(data_ == nullptr) {
    //       errorThrower_=ErrorThrower::errorThrowerProductNotFoundException(TempWrapT::typeInfo(),
    //                                                                        iModuleLabel,
    //                                                                        iProductInstanceLabel,
    //                                                                        iProcessLabel);
    //    }
    // }

    const std::string getBranchNameFor(const fwlite::EventBase& iEvent,
                                       const char* iModuleLabel,
                                       const char* iProductInstanceLabel = nullptr,
                                       const char* iProcessLabel = nullptr) {
      return iEvent.getBranchNameFor(TempWrapT::typeInfo(), iModuleLabel, iProductInstanceLabel, iProcessLabel);
    }

    // const std::string getBranchNameFor(const fwlite::Event& iEvent,
    //                                    const char* iModuleLabel,
    //                                    const char* iProductInstanceLabel = nullptr,
    //                                    const char* iProcessLabel = nullptr) {
    //    return iEvent.getBranchNameFor(TempWrapT::typeInfo(),
    //                                   iModuleLabel,
    //                                   iProductInstanceLabel,
    //                                   iProcessLabel);
    // }
    //
    // const std::string getBranchNameFor(const fwlite::ChainEvent& iEvent,
    //                                    const char* iModuleLabel,
    //                                    const char* iProductInstanceLabel = nullptr,
    //                                    const char* iProcessLabel = nullptr) {
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
    //                                    const char* iProductInstanceLabel = nullptr,
    //                                    const char* iProcessLabel = nullptr) {
    //    return iEvent.getBranchNameFor(TempWrapT::typeInfo(),
    //                                   iModuleLabel,
    //                                   iProductInstanceLabel,
    //                                   iProcessLabel);
    // }

    void swap(Handle<T>& iOther) {
      const T* temp = data_;
      data_ = iOther.data_;
      iOther.data_ = temp;
      ErrorThrower const* tempE = errorThrower_;
      errorThrower_ = iOther.errorThrower_;
      iOther.errorThrower_ = tempE;
    }

  private:
    void check() const {
      if (errorThrower_) {
        errorThrower_->throwIt();
      }
    }

    // ---------- member data --------------------------------
    const T* data_;
    ErrorThrower const* errorThrower_;
  };

}  // namespace fwlite

#endif
