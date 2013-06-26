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
  template<typename T> class Wrapper;
}

#if !defined(__CINT__) && !defined(__MAKECINT__)
//CINT can't handle parsing these files
#include "DataFormats/FWLite/interface/EventBase.h"
#include "DataFormats/FWLite/interface/ErrorThrower.h"
#include "DataFormats/Common/interface/Wrapper.h"
#endif

// forward declarations
namespace fwlite {
   class ErrorThrower;
   class EventBase;

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
         swap(temp);
         return *this;
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

      // P can be Run, Event, or LuminosityBlock

      template <class P> void getByLabel(const P& iP,
                      const char* iModuleLabel,
                      const char* iProductInstanceLabel = 0,
                      const char* iProcessLabel = 0) {
        TempWrapT* temp;
        void* pTemp = &temp;
        iP.getByLabel(TempWrapT::typeInfo(),
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
      iOther.errorThrower_ = tempE;
   }
   private:
      void check() const { if(errorThrower_) { errorThrower_->throwIt();} }


      // ---------- member data --------------------------------
      const T* data_;
      ErrorThrower*  errorThrower_;
};

}

#if defined(__CINT__) || defined(__MAKECINT__)
#include <RVersion.h>
#if ROOT_VERSION_CODE >= 336384 // ROOT_VERSION(5,34,0), doesn't work

// "magic" typedefs for CINT
#define DECL_HANDLE_VECTOR(T) \
typedef fwlite::Handle<vector<T> > Handle<vector<T,allocator<T> > >

// derived from RootAutoLibraryLoader specials list
namespace edm {
  DECL_HANDLE_VECTOR(bool);

  DECL_HANDLE_VECTOR(char);
  DECL_HANDLE_VECTOR(unsigned char);
  // vector<signed char> gives CINT conniptions
  //  DECL_HANDLE_VECTOR(signed char);
  DECL_HANDLE_VECTOR(short);
  DECL_HANDLE_VECTOR(unsigned short);
  DECL_HANDLE_VECTOR(int);
  DECL_HANDLE_VECTOR(unsigned int);
  DECL_HANDLE_VECTOR(long);
  DECL_HANDLE_VECTOR(unsigned long);
  DECL_HANDLE_VECTOR(long long);
  DECL_HANDLE_VECTOR(unsigned long long);

  DECL_HANDLE_VECTOR(float);
  DECL_HANDLE_VECTOR(double);
}
#endif
#endif
#endif
