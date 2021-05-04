/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <cerrno>

#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ExceptionHelpers.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"

namespace edm {
  ProducerSourceBase::ProducerSourceBase(ParameterSet const& pset, InputSourceDescription const& desc, bool realData)
      : IDGeneratorSourceBase<PuttableSourceBase>(pset, desc, realData) {}

  ProducerSourceBase::~ProducerSourceBase() noexcept(false) {}

  void ProducerSourceBase::readEvent_(EventPrincipal& eventPrincipal) {
    doReadEvent(eventPrincipal, [this](auto& eventPrincipal) {
      Event e(eventPrincipal, moduleDescription(), nullptr);
      e.setProducer(this, nullptr);
      produce(e);
      e.commit_(std::vector<ProductResolverIndex>());
    });
  }
}  // namespace edm
