import FWCore.ParameterSet.Config as cms

process = cms.Process("runCosMuoGen")
process.load("GeneratorInterface.CosmicMuonGenerator.CMSCGENsource_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.untracked.uint32(135799468)
)

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(500)
    input = cms.untracked.int32(10000)
)
process.CMSCGEN_out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('cosmic.root')
)

process.outpath = cms.EndPath(process.CMSCGEN_out)
process.CosMuoGenSource.MinP = 5.
process.CosMuoGenSource.MinTheta = 91.
process.CosMuoGenSource.MaxTheta = 180.

process.CosMuoGenSource.MaxEnu = 100000.

# Plug z-position [mm] (default=-14000.)
#process.CosMuoGenSource.PlugVz = -33000.;

