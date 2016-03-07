#include "FWCore/Framework/interface/EventForOutput.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {

  EventForOutput::EventForOutput(EventPrincipal const& ep, ModuleDescription const& md, ModuleCallingContext const* moduleCallingContext) :
      provRecorder_(ep, md),
      aux_(ep.aux()),
      luminosityBlock_(ep.luminosityBlockPrincipalPtrValid() ? new LuminosityBlockForOutput(ep.luminosityBlockPrincipal(), md, moduleCallingContext) : nullptr),
      streamID_(ep.streamID()),
      moduleCallingContext_(moduleCallingContext)
  {
  }

  EventForOutput::~EventForOutput() {
  }

  void
  EventForOutput::setConsumer(EDConsumerBase const* iConsumer) {
    provRecorder_.setConsumer(iConsumer);
    const_cast<LuminosityBlockForOutput*>(luminosityBlock_.get())->setConsumer(iConsumer);
  }
  
  EventPrincipal const&
  EventForOutput::eventPrincipal() const {
    return dynamic_cast<EventPrincipal const&>(provRecorder_.principal());
  }

  RunForOutput const&
  EventForOutput::getRun() const {
    return getLuminosityBlock().getRun();
  }

  EventSelectionIDVector const&
  EventForOutput::eventSelectionIDs() const {
    return eventPrincipal().eventSelectionIDs();
  }

  ProcessHistoryID const&
  EventForOutput::processHistoryID() const {
    return eventPrincipal().processHistoryID();
  }

  Provenance
  EventForOutput::getProvenance(BranchID const& bid) const {
    return provRecorder_.principal().getProvenance(bid, moduleCallingContext_);
  }

  void
  EventForOutput::getAllProvenance(std::vector<Provenance const*>& provenances) const {
    provRecorder_.principal().getAllProvenance(provenances);
  }

  ProductProvenanceRetriever const*
  EventForOutput::productProvenanceRetrieverPtr() const {
   return eventPrincipal().productProvenanceRetrieverPtr();;
  }

  bool
  EventForOutput::getProcessParameterSet(std::string const& processName,
                                ParameterSet& ps) const {
    ProcessConfiguration config;
    bool process_found = processHistory().getConfigurationForProcess(processName, config);
    if(process_found) {
      pset::Registry::instance()->getMapped(config.parameterSetID(), ps);
      assert(!ps.empty());
    }
    return process_found;
  }

  ProcessHistory const&
  EventForOutput::processHistory() const {
    return provRecorder_.processHistory();
  }

  size_t
  EventForOutput::size() const {
    return provRecorder_.principal().size();
  }

  bool
  EventForOutput::getByToken(EDGetToken token, TypeID const& typeID, BasicHandle& result) const {
    result.clear();
    result = provRecorder_.getByToken_(typeID, PRODUCT_TYPE, token, moduleCallingContext_);
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  BranchListIndexes const&
  EventForOutput::branchListIndexes() const {
    return eventPrincipal().branchListIndexes();
  }

}
