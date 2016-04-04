
import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMREC")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10))
#process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))

process.load('RecoLocalMuon.GEMRecHit.me0LocalReco_cff')
process.load('RecoLocalMuon.GEMRecHit.gemLocalReco_cff')
#process.load('RecoLocalMuon.GEMSegment.trackerGEM_cfi')

## process.MessageLogger.categories.append("GEMSegment")
## process.MessageLogger.categories.append("GEMSegmentBuilder")
## process.MessageLogger.categories.append("GEMSegAlgoPV")   
#process.MessageLogger.categories.append("GEMSegFit")      
#process.MessageLogger.categories.append("trackerGEM")      
## process.MessageLogger.categories.append("GEMSegFitMatrixDetails")      

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'/store/mc/TP2023HGCALDR/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM-DIGI-RAW/HGCALForMUO_PU140BX25_newsplit_PH2_1K_FB_V6-v2/20000/00F1E3B0-A231-E511-B092-008CFA1CB470.root'
        #'/store/mc/TP2023HGCALDR/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM-RECO/HGCALForMUO_PU140BX25_newsplit_PH2_1K_FB_V6-v2/20000/06E993BA-E531-E511-9803-008CFA0A57E4.root'
        #/GluGluToHToZZTo4m_M-125_14TeV-powheg-pythia6/TP2023HGCALDR-HGCALnewsplit_PU140BX25_newsplitPU140_PH2_1K_FB_V6-v1/GEN-SIM-RECO
        #'file:/afs/cern.ch/user/j/jlee/work/00E21A91-B9F9-E411-9107-00266CFADEC0.root'
	'file:out_digi.root'
    )
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(
        'file:out_reco.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
    )
)


#process.reco_step    = cms.Path(process.gemSegments*process.trackerGEM)
process.reco_step    = cms.Path(process.gemLocalReco)
process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.reco_step,
    process.endjob_step,
    process.out_step
)

