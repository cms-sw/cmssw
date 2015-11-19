import FWCore.ParameterSet.Config as cms
#process = cms.Process("RecoSTAMuon")
process = cms.Process("STARECO")
process.load("RecoMuon.Configuration.RecoMuon_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
#process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
#process.load('Configuration.StandardSequences.MagneticField_38T_cff')
#process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff') #!!!!!!!!!!!!!!!!!!!!!!!!!!
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_design', '')

# Fix DT and CSC Alignment #
############################
#from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixDTAlignmentConditions
#process = fixDTAlignmentConditions(process)
#from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixCSCAlignmentConditions
#process = fixCSCAlignmentConditions(process)

process.maxEvents = cms.untracked.PSet(
input = cms.untracked.int32(100)
)

 # Seed generator
from RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cff import *

# Stand alone muon track producer
from RecoMuon.StandAloneMuonProducer.standAloneMuons_cff import *

# Beam Spot 
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *

process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(
#                  'file:/afs/cern.ch/work/a/archie/public/out_local_reco_SingleMuPt100_750pre1.root'
#                   'root://cmsxrootd.fnal.gov///store/user/archie/NewSamplesin75XforCSCGEM/out_local_reco_singleMuPt200_75X_neweta.root'
        #'file:/tmp/dnash/out_sim.root'
        #'file:/tmp/dnash/out_digi.root'
        #'file:out_digi_nocalo.root'
        'file:out_local_reco_test.root'
        #'file:out_digi_nocalo_experiment.root'
        #'file:out_digi.root'
        #'file:/tmp/dnash/out_local_reco.root'
                 )
           )


process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(
        'file:out_STA_reco_withGems_new.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('stareco_step')
    )

 )



process.MessageLogger = cms.Service("MessageLogger",
                    destinations       =  cms.untracked.vstring('debugmessages'),
                    #categories         = cms.untracked.vstring("TrackFitters"),
                    #categories         = cms.untracked.vstring('MuonDetLayerGeometryESProducer'),
                    #categories         = cms.untracked.vstring('MuonDetLayerMeasurements'),
                    #categories         = cms.untracked.vstring('RecoMuon','TrackFitters','ME0','ME0GeometryBuilderFromDDD','CSCGeometryBuilder'),
                    categories         = cms.untracked.vstring('RecoMuon','TrackFitters','ME0','CSCSegment','TrackProducer','MuonME0DetLayerGeometryBuilder'),
                                    debugModules  = cms.untracked.vstring('*'),
                                    
                                    debugmessages          = cms.untracked.PSet(
                                               threshold =  cms.untracked.string('DEBUG'),
                                               INFO       =  cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                               DEBUG   = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                               WARNING   = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                               ERROR  = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                               #WARNING = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                               #TrackFitters = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
                                               MuonME0DetLayerGeometryBuilder = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
                                               #MuonDetLayerGeometryESProducer = cms.untracked.PSet(limit = cms.untracked.int32(10000000))
                                               #MuonDetLayerMeasurements = cms.untracked.PSet(limit = cms.untracked.int32(10000000))
                                               TrackFitters = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
                                               RecoMuon = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
                                               ME0 = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
                                               CSCSegment = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
                                               TrackProducer = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
                                               #ME0GeometryBuilderFromDDD = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
                                               #CSCGeometryBuilder = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
                                                   )
)




process.stareco_step = cms.Path(offlineBeamSpot*standAloneMuonSeeds*process.standAloneMuons)
process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.stareco_step,
    process.endjob_step,
    process.out_step
)

