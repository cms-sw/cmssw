import FWCore.ParameterSet.Config as cms
process = cms.Process("CTPPS")

process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
#        'root://eoscms.cern.ch:1094//eos/totem/data/ctpps/run284036.root',
        '/store/data/Run2016H/ZeroBias/RAW/v1/000/281/010/00000/20B9B8C4-6F7E-E611-8B60-02163E013864.root'
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# diamonds mapping
#process.totemDAQMappingESSourceXML_TimingDiamond = cms.ESSource("TotemDAQMappingESSourceXML",
#    verbosity = cms.untracked.uint32(0),
#    subSystem = cms.untracked.string("TimingDiamond"),
#    configuration = cms.VPSet(
#        # before diamonds inserted in DAQ
#        cms.PSet(
#            validityRange = cms.EventRange("1:min - 283819:max"),
#            mappingFileNames = cms.vstring(),
#            maskFileNames = cms.vstring()
#        ),
#        # after diamonds inserted in DAQ
#        cms.PSet(
#            validityRange = cms.EventRange("283820:min - 999999999:max"),
#            mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_timing_diamond.xml"),
#            maskFileNames = cms.vstring()
#        )
#    )
#)
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

process.load('Geometry.VeryForwardGeometry.geometryRP_cfi')

process.load('RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff')
process.load('RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff')

process.load('RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cfi')
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
