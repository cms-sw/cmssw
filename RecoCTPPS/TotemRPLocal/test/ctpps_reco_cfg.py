import FWCore.ParameterSet.Config as cms
process = cms.Process("CTPPS")

process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
        #'/store/data/Run2016H/ZeroBias/RAW/v1/000/281/010/00000/20B9B8C4-6F7E-E611-8B60-02163E013864.root' # no diamond data
        '/store/data/Run2016H/ZeroBias/RAW/v1/000/284/036/00000/D2EE671D-D39E-E611-B272-FA163EA63BCC.root'
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

process.load('Geometry.VeryForwardGeometry.geometryRP_cfi')

process.load('RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff')
process.load('RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff')

# if a RP or projection needs adjustment of the above settings, you can use the following format
#process.totemRPUVPatternFinder.exceptionalSettings = cms.VPSet(
#    cms.PSet(
#        rpId = cms.uint32(1998061568), # RP id according to CTPPSDetId
#        minPlanesPerProjectionToFit_U = cms.uint32(2),
#        minPlanesPerProjectionToFit_V = cms.uint32(3),
#        threshold_U = cms.double(1.99),
#        threshold_V = cms.double(2.99)
#    )
#)

process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff')
process.ctppsLocalTrackLiteProducer.doNothing = cms.bool(False)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("file:miniAOD.root"),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_totemRP*_*_*',
        'keep *_ctpps*_*_*',
    ),
)

process.endjob_step = cms.EndPath(process.endOfProcess)
process.output_step = cms.EndPath(process.output)

# execution configuration
process.ctpps_reco_step = cms.Path(
    process.totemRPRawToDigi *
    process.totemRPLocalReconstruction *
    process.ctppsDiamondRawToDigi *
    process.ctppsDiamondLocalReconstruction
)
process.ctpps_miniaod_step = cms.Path(process.ctppsLocalTrackLiteProducer)
process.schedule = cms.Schedule(process.ctpps_reco_step, process.ctpps_miniaod_step, process.endjob_step, process.output_step)

from FWCore.ParameterSet.Utilities import convertToUnscheduled
process = convertToUnscheduled(process)
from FWCore.ParameterSet.Utilities import cleanUnscheduled
process = cleanUnscheduled(process)
