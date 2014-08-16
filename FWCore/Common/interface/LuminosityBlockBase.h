#ifndef FWCore_Common_LuminosityBlockBase_h
#define FWCore_Common_LuminosityBlockBase_h

// -*- C++ -*-
//
// Package:     FWCore/Common
// Class  :     LuminosityBlockBase
//
/**\class LuminosityBlockBase LuminosityBlockBase.h FWCore/Common/interface/LuminosityBlockBase.h

 Description: Base class for LuminosityBlocks in both the full and light framework

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

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {

  class LuminosityBlockBase {
  public:
    LuminosityBlockBase();
    virtual ~LuminosityBlockBase();

    // AUX functions.
    LuminosityBlockNumber_t luminosityBlock() const {
      return luminosityBlockAuxiliary().luminosityBlock();
    }

    RunNumber_t run() const {
      return luminosityBlockAuxiliary().run();
    }

    LuminosityBlockID id() const {
      return luminosityBlockAuxiliary().id();
    }

    Timestamp const& beginTime() const {
      return luminosityBlockAuxiliary().beginTime();
    }
    Timestamp const& endTime() const {
      return luminosityBlockAuxiliary().endTime();
    }

    virtual edm::LuminosityBlockAuxiliary const& luminosityBlockAuxiliary() const = 0;

    /// same as above, but using the InputTag class
    template<typename PROD>
    bool
    getByLabel(InputTag const& tag, Handle<PROD>& result) const;

  private:
    virtual BasicHandle getByLabelImpl(std::type_info const& iWrapperType, std::type_info const& iProductType, const InputTag& iTag) const = 0;

  };

#if !defined(__REFLEX__)
   template<class T>
   bool
   LuminosityBlockBase::getByLabel(const InputTag& tag, Handle<T>& result) const {
      result.clear();
      BasicHandle bh = this->getByLabelImpl(typeid(Wrapper<T>), typeid(T), tag);
      convert_handle(std::move(bh), result);  // throws on conversion error
      if (result.failedToGet()) {
         return false;
      }
      return true;
   }
#endif

}
#endif /*!defined(__CINT__) && !defined(__MAKECINT__)*/
#endif
