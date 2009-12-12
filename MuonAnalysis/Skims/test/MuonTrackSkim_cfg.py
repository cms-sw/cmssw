import FWCore.ParameterSet.Config as cms

process = cms.Process("MUONSKIM")


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_31X_V2P::All"

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.source = cms.Source("PoolSource",
                            debugVerbosity = cms.untracked.uint32(0),
                            debugFlag = cms.untracked.bool(False),
                            fileNames = cms.untracked.vstring('file:/data/b/bellan/Run123592/RECO/E609699F-2BE2-DE11-A59D-003048D2C108.root',
                                                              'file:/data/b/bellan/Run123592/RECO/5C5983C5-2AE2-DE11-84A1-0019B9F72BAA.root'),
                            
                            secondaryFileNames = cms.untracked.vstring()
                            )

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/MuonAnalysis/Skims/test/MuonSkim_cfg.py,v $'),
    annotation = cms.untracked.string('BSC skim')
    )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))


########################## Muon tracks Filter ############################
process.load("MuonAnalysis.Skims.MuonSkim_cfi")
process.muonTracksSkim = cms.Path(process.muonSkim)
###########################################################################



process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('MuonSkim.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('Muon_skim')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("muonTracksSkim")
    )
)

process.e = cms.EndPath(process.out)

