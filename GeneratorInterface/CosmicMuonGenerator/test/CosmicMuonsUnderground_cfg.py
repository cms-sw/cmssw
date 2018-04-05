import FWCore.ParameterSet.Config as cms

process = cms.Process("runCosMuoGen")
process.load("GeneratorInterface.CosmicMuonGenerator.CMSCGENproducer_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(500)
    input = cms.untracked.int32(100000)
)
process.CMSCGEN_out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('cosmic.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.CMSCGEN_out)
process.generator.MinP = 3.
process.generator.MaxTheta = 89.
#process.generator.MinTheta = 91.
#process.generator.MaxTheta = 180.
#process.generator.ElossScaleFactor = 0. #default=1.
##process.generator.MinEnu = 10.
##process.generator.MaxEnu = 100000.
#Neutrino production altitude (in [mm])
#process.generator.NuProdAlt = 7.5e6                       
##process.generator.NuProdAlt = 4.e7                       

# Plug z-position [mm] (default=-14000. = on Shaft)
#process.generator.PlugVz = 5000.;
#process.generator.PlugVz = -33000.;

# z-position of centre of target cylinder [mm] (default=0.)
#process.generator.ZCentrOfTarget = 0.;

#Read in Multi muon events or generate single muon events (MultiMuon=false = default)
#process.generator.MultiMuon = True
#process.generator.MultiMuonNmin = 2
#process.generator.MultiMuonFileName = "MultiEventsIn.root"
#process.generator.MultiMuonFileName = "CORSIKA6900_3_10TeV_100k.root"
###process.generator.MultiMuonFileFirstEvent = 1

# Accept all muons
#process.generator.AcptAllMu = True
