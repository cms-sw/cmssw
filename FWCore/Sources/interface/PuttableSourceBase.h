#ifndef FWCore_Sources_PuttableSourceBase_h
#define FWCore_Sources_PuttableSourceBase_h
// -*- C++ -*-
//
// Package:     FWCore/Sources
// Class  :     PuttableSourceBase
//
/**\class PuttableSourceBase PuttableSourceBase.h "PuttableSourceBase.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Tue, 26 Sep 2017 20:51:50 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProducerBase.h"

// forward declarations
namespace edm {
  class PuttableSourceBase : public InputSource, public ProducerBase {
  public:
    PuttableSourceBase(ParameterSet const&, InputSourceDescription const&);

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    using ProducerBase::registerProducts;
    using ProducerBase::resolvePutIndicies;
    void registerProducts() final;

    bool hasAbilityToProduceInBeginRuns() const final { return true; }

    bool hasAbilityToProduceInBeginLumis() const final { return true; }

  protected:
    //If inheriting class overrides, they need to call this function as well
    void beginJob() override;

  private:
    void doBeginLumi(LuminosityBlockPrincipal& lbp, ProcessContext const*) override;
    void doBeginRun(RunPrincipal& rp, ProcessContext const*) override;

    virtual void beginRun(Run&);
    virtual void beginLuminosityBlock(LuminosityBlock&);

    PuttableSourceBase(const PuttableSourceBase&) = delete;

    PuttableSourceBase& operator=(const PuttableSourceBase&) = delete;

    // ---------- member data --------------------------------
  };
}  // namespace edm

#endif
