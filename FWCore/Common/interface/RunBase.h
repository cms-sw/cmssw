#ifndef FWCore_Common_RunBase_h
#define FWCore_Common_RunBase_h

// -*- C++ -*-
//
// Package:     FWCore/Common
// Class  :     RunBase
//
/**\class RunBase RunBase.h FWCore/Common/interface/RunBase.h

 Description: Base class for Runs in both the full and light framework

 Usage:
    One can use this class for code which needs to work in both the full and the
 light (i.e. FWLite) frameworks.  Data can be accessed using the same getByLabel
 interface which is available in the full framework.

*/
//
// Original Author:  Eric Vaandering
//         Created:  Tue Jan 12 15:31:00 CDT 2010
//

#if !defined(__CINT__) && !defined(__MAKECINT__)

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"

namespace edm {

  class RunBase {
  public:
    RunBase();
    virtual ~RunBase();

    // AUX functions.
    RunID const& id() const {return runAuxiliary().id();}
    RunNumber_t run() const {return runAuxiliary().run();}
    Timestamp const& beginTime() const {return runAuxiliary().beginTime();}
    Timestamp const& endTime() const {return runAuxiliary().endTime();}


    virtual edm::RunAuxiliary const& runAuxiliary() const = 0;

    /// same as above, but using the InputTag class
    template <typename PROD>
    bool
    getByLabel(InputTag const& tag, Handle<PROD>& result) const;

  private:

    virtual BasicHandle getByLabelImpl(const std::type_info& iWrapperType, const std::type_info& iProductType, const InputTag& iTag) const = 0;

  };

   template <class T>
   bool
   RunBase::getByLabel(const InputTag& tag, Handle<T>& result) const {
      result.clear();
      BasicHandle bh = this->getByLabelImpl(typeid(edm::Wrapper<T>), typeid(T), tag);
      convert_handle(bh, result);  // throws on conversion error
      if (bh.failedToGet()) {
         return false;
      }
      return true;
   }


}
#endif /*!defined(__CINT__) && !defined(__MAKECINT__)*/
#endif
