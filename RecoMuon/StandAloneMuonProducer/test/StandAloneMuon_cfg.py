import FWCore.ParameterSet.Config as cms

process = cms.Process("RecoSTAMuon")
process.load("RecoMuon.Configuration.RecoMuon_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023_cff')
#process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff') #!!!!!!!!!!!!!!!!!!!!!!!!!!
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'IDEAL_V9::All'
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_design', '')

# Fix DT and CSC Alignment #
############################
#from SLHCUpgradeSimulations.Configuration.combinedCustoms import fixDTAlignmentConditions
#process = fixDTAlignmentConditions(process)
#from SLHCUpgradeSimulations.Configuration.combinedCustoms import fixCSCAlignmentConditions
#process = fixCSCAlignmentConditions(process)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#        'root://cmsxrootd.fnal.gov///store/user/archie/PT200incleanarea/out_STA_reco_Pt200_withGems_clean.root'    
        'file:/tmp/archie/out_STA_reco_Pt200_withoutGems_new.root'
     )
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('cout')
)

#process.out = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string('RecoSTAMuons.root')
#)

minEtaTot = 1.64
maxEtaTot = 2.43

process.TFileService = cms.Service("TFileService", fileName = cms.string("NoGemRecHitTestPt200_newGT.root") )

## Analyzer to produce pT and 1/pT resolution plots
process.STAMuonAnalyzer = cms.EDAnalyzer("STAMuonAnalyzer",
                                         DataType = cms.untracked.string('SimData'),
                                         StandAloneTrackCollectionLabel = cms.untracked.InputTag('standAloneMuons','UpdatedAtVtx','STARECO'),
                                         MuonCollectionLabel = cms.untracked.InputTag('muons','','RECO'),
                                         NoGEMCase = cms.untracked.bool(True),
                                         isGlobalMuon = cms.untracked.bool(False),
                                         minEta = cms.untracked.double(minEtaTot),
                                         maxEta = cms.untracked.double(maxEtaTot),

#                                         MuonSeedCollectionLabel = cms.untracked.string('MuonSeed'),
#                                         rootFileName = cms.untracked.string('STAMuonAnalyzer.root')
                                         )

process.p = cms.Path(process.STAMuonAnalyzer)                             ## default path (no analyzer)
#process.p = cms.Path(process.MuonSeed * process.standAloneMuons * process.STAMuonAnalyzer)  ## path with analyzer
#process.this_is_the_end = cms.EndPath(process.out)
#from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023Muon

#call to customisation function cust_2019 imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
#process = cust_2023Muon(process)

# End of customisation functions

