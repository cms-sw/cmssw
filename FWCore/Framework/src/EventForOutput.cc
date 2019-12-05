#include "FWCore/Framework/interface/EventForOutput.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {

  EventForOutput::EventForOutput(EventPrincipal const& ep,
                                 ModuleDescription const& md,
                                 ModuleCallingContext const* moduleCallingContext)
      : OccurrenceForOutput(ep, md, moduleCallingContext, true /*always at end*/),
        aux_(ep.aux()),
        luminosityBlock_(ep.luminosityBlockPrincipalPtrValid()
                             ? new LuminosityBlockForOutput(
                                   ep.luminosityBlockPrincipal(), md, moduleCallingContext, false /*not at end*/)
                             : nullptr),
        streamID_(ep.streamID()) {}

  EventForOutput::~EventForOutput() {}

  EventPrincipal const& EventForOutput::eventPrincipal() const {
    return dynamic_cast<EventPrincipal const&>(principal());
  }

  RunForOutput const& EventForOutput::getRun() const { return getLuminosityBlock().getRun(); }

  EventSelectionIDVector const& EventForOutput::eventSelectionIDs() const {
    return eventPrincipal().eventSelectionIDs();
  }

  ProductProvenanceRetriever const* EventForOutput::productProvenanceRetrieverPtr() const {
    return eventPrincipal().productProvenanceRetrieverPtr();
  }

  BranchListIndexes const& EventForOutput::branchListIndexes() const { return eventPrincipal().branchListIndexes(); }

}  // namespace edm
