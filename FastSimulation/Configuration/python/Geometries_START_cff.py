import FWCore.ParameterSet.Config as cms

# Apply Tracker and Muon misalignment
def _fastSimGeometryCustomsStart(process):
    process.fastSimProducer.detectorDefinition.trackerAlignmentLabel = cms.untracked.string("")

from Configuration.Eras.Modifier_fastSim_cff import fastSim
modifyGeomStart_fastSim = fastSim.makeProcessModifier(_fastSimGeometryCustomsStart)

