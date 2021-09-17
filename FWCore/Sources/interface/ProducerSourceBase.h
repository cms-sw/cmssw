#ifndef Framework_Sources_ProducerSourceBase_h
#define Framework_Sources_ProducerSourceBase_h

/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/PuttableSourceBase.h"
#include "FWCore/Sources/interface/IDGeneratorSourceBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

#include <memory>
#include <vector>

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
  class ProducerSourceBase : public IDGeneratorSourceBase<PuttableSourceBase> {
  public:
    explicit ProducerSourceBase(ParameterSet const& pset, InputSourceDescription const& desc, bool realData);
    ~ProducerSourceBase() noexcept(false) override;

  protected:
  private:
    virtual void produce(Event& e) = 0;

    void readEvent_(EventPrincipal& eventPrincipal) override;
  };
}  // namespace edm
#endif
