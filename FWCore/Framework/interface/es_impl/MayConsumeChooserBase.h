#ifndef FWCore_Framework_MayConsumeChooserBase_h
#define FWCore_Framework_MayConsumeChooserBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     MayConsumeChooserBase
//
/**\class MayConsumeChooserBase MayConsumeChooserBase.h "MayConsumeChooserBase.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 19 Sep 2019 15:29:33 GMT
//

// system include files
#include "FWCore/Framework/interface/ESTagGetter.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Utilities/interface/ESIndices.h"

// user include files

// forward declarations

namespace edm::eventsetup::impl {

  class MayConsumeChooserCore {
  public:
    virtual ~MayConsumeChooserCore() = default;

    virtual EventSetupRecordKey recordKey() const noexcept = 0;
    virtual TypeTag productType() const noexcept = 0;

    void setTagGetter(ESTagGetter iGetter) { getter_ = std::move(iGetter); }

  protected:
    ESTagGetter const& tagGetter() const { return getter_; }

  private:
    ESTagGetter getter_;
  };

  template <typename RCD>
  class MayConsumeChooserBase : public MayConsumeChooserCore {
  public:
    MayConsumeChooserBase() = default;
    ~MayConsumeChooserBase() override = default;

    // ---------- const member functions ---------------------
    virtual ESProxyIndex makeChoice(RCD const& iRecord) const = 0;
  };
}  // namespace edm::eventsetup::impl

#endif
