import FWCore.ParameterSet.Config as cms

process = cms.Process("runCosMuoGen")
process.load("GeneratorInterface.CosmicMuonGenerator.CMSCGENsource_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.untracked.uint32(135799468)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
#    input = cms.untracked.int32(10)
)
process.CMSCGEN_out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('cosmic.root')
)

process.outpath = cms.EndPath(process.CMSCGEN_out)

#process.CosMuoGenSource.MinP = 10.
process.CosMuoGenSource.MinP = 3.
#process.CosMuoGenSource.MinP = 300.

#process.CosMuoGenSource.MaxTheta = 80.
process.CosMuoGenSource.MaxTheta = 89.

# Plug z-position [mm] (default=-14000. = on Shaft) 
#process.CosMuoGenSource.PlugVz = -33000.;

# z-position of centre of target cylinder [mm] (default=0.)
#process.CosMuoGenSource.ZCentrOfTarget = 0.;
#process.CosMuoGenSource.ZCentrOfTarget = 15000.; 
