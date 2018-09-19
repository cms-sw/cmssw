import FWCore.ParameterSet.Config as cms

process = cms.Process("Whatever")

#process.load('Configuration.Geometry.GeometryExtended2016Reco_cff')
process.load("Configuration.StandardSequences.GeometryDB_cff")  # load from DB
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '102X_dataRun2_Sep2018Rereco_v1')
print "Using GlobalTag: %s" % process.GlobalTag.globaltag.value()

# # Fake alignment is/should be ideal geometry
# # ==========================================
# process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
# process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource")

# process.source = cms.Source("EmptySource")
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(321988),
    lastValue = cms.uint64(321988),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.analyzer1 = cms.EDAnalyzer("MakeCoordLUT",
    # Verbosity level
    verbosity = cms.untracked.int32(1),

    # Output diectory
    outdir = cms.string("./pc_luts/firmware_data/"),

    # Produce "validate.root" to validate the LUTs
    please_validate = cms.bool(True),
)

process.path1 = cms.Path(process.analyzer1)
