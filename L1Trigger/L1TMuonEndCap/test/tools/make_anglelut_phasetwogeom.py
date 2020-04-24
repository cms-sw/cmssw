#
# This cfg calls MakeAngleLUT which is obsolete and completely unused.
#

import FWCore.ParameterSet.Config as cms

process = cms.Process("Whatever")

process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '90X_upgrade2023_realistic_v9', '')
print "Using GlobalTag: %s" % process.GlobalTag.globaltag.value()

# Fake alignment is/should be ideal geometry
# ==========================================
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.analyzer1 = cms.EDAnalyzer("MakeAngleLUT",
    # Verbosity level
    verbosity = cms.untracked.int32(1),

    # Output file
    outfile = cms.string("angle.root"),
)

process.path1 = cms.Path(process.analyzer1)
