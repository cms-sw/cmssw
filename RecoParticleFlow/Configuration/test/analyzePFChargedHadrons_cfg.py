import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source (
    "PoolSource",    
    fileNames = cms.untracked.vstring(
        'rfio:/castor/cern.ch/user/p/pjanot/CMSSW390pre3/display_Matt_3.root'
      ),
    secondaryFileNames = cms.untracked.vstring(),
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
    )




process.pfChargedHadronAnalyzer = cms.EDAnalyzer(
    "PFChargedHadronAnalyzer",
    PFCandidates = cms.InputTag("particleFlow"),
    ptMin = cms.double(1.),                     # Minimum pt
    pMin = cms.double(3.),                      # Minimum p
    nPixMin = cms.int32(2),                     # Nb of pixel hits
    nHitMin = cms.vint32(14,17,20,17),          # Nb of track hits
    nEtaMin = cms.vdouble(1.4, 1.6, 2.0, 2.4),  # in these eta ranges
    hcalMin = cms.double(1.),                   # Minimum hcal energy
    ecalMax = cms.double(0.2),                  # Maximum ecal energy 
    verbose = cms.untracked.bool(True),         # not used.
)

process.load("FastSimulation.Configuration.EventContent_cff")
process.aod = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('aod.root')
)

#process.outpath = cms.EndPath(process.aod )


process.p = cms.Path(process.pfChargedHadronAnalyzer)


