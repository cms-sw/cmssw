import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/CRUZET3/CMSSW_2_1_2/src/DPGAnalysis/Skims/python/reco_50908_210_CRZT210_V1P.root')
                            )

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/DoubleMuon_cfg.py,v $'),
    annotation = cms.untracked.string('CRUZET4 DoubleMuon skim')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'auto:run3_data_prompt'
process.prefer("GlobalTag")

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")


process.doubleMuonFilter = cms.EDFilter("TrackCountFilter",
                                        src = cms.InputTag('cosmicMuonsBarrelOnly'),
                                        minNumber = cms.uint32(2) 
                                        )

process.doubleMuonPath = cms.Path(process.doubleMuonFilter)

process.out = cms.OutputModule("PoolOutputModule",
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('doubleMuonPath')),
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('RECO'),
                                         filterName = cms.untracked.string('doubleMuonPath')),
                               fileName = cms.untracked.string('doubleMuon.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)


