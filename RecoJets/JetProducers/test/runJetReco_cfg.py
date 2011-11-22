import FWCore.ParameterSet.Config as cms

process = cms.Process("JETRECO")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# input
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_4_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V12-v1/0002/0410E3D4-F5CB-DE11-A871-001D09F242EA.root',
'/store/relval/CMSSW_3_4_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V12-v1/0002/482F295C-F7CB-DE11-9C53-0030487A18A4.root',
'/store/relval/CMSSW_3_4_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V12-v1/0002/64775AFD-F7CB-DE11-9305-001D09F24DA8.root',
'/store/relval/CMSSW_3_4_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V12-v1/0002/64AC12F9-FFCB-DE11-AB45-0030487A322E.root',
'/store/relval/CMSSW_3_4_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V12-v1/0002/74E50396-F6CB-DE11-A6F4-0030487A322E.root',
'/store/relval/CMSSW_3_4_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V12-v1/0002/8AC096C6-F6CB-DE11-A494-0030487D0D3A.root',
'/store/relval/CMSSW_3_4_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V12-v1/0002/C0821C7D-F9CB-DE11-9209-001D09F253D4.root',
'/store/relval/CMSSW_3_4_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V12-v1/0002/C21534FF-F7CB-DE11-A43B-001D09F2423B.root'

        )
    )

# output
process.load('Configuration/EventContent/EventContent_cff')
process.output = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('testJetReco.root')
)
process.output.outputCommands =  cms.untracked.vstring('drop *')
process.output.outputCommands.append('keep recoCaloJets_*_*_*')
process.output.outputCommands.append('keep recoPFJets_*_*_*')
process.output.outputCommands.append('keep recoGenJets_*_*_*')
process.output.outputCommands.append('keep recoBasicJets_*_*_*')
process.output.outputCommands.append('keep *_*_*_JETRECO')
process.output.outputCommands.append('keep *_trackRefsForJets_*_*')
process.output.outputCommands.append('keep *_generalTracks_*_*')

# jet reconstruction
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_3XY_V12::All'


from RecoJets.JetProducers.CaloJetParameters_cfi import *
#CaloJetParameters.doAreaFastjet = True
from RecoJets.JetProducers.PFJetParameters_cfi import *
#PFJetParameters.doAreaFastjet = True

process.load('RecoJets.Configuration.GenJetParticles_cff')
process.load('RecoJets.Configuration.RecoGenJets_cff')
process.load('RecoJets.Configuration.RecoJets_cff')
process.load('RecoJets.Configuration.RecoPFJets_cff')
process.load('RecoJets.JetProducers.TracksForJets_cff')
process.load('RecoJets.Configuration.RecoTrackJets_cff')
process.load('RecoJets.Configuration.JetIDProducers_cff')

#process.kt6PFJets.doRhoFastjet = True;
#process.kt6CaloJets.doRhoFastjet = True;
#process.kt6TrackJets.doRhoFastjet = True;
#process.kt6GenJets.doRhoFastjet = True;

process.recoJets = cms.Path(process.genParticlesForJets+process.recoGenJets+
                            process.recoJets+
                            process.recoPFJets+
                            process.recoTrackJets+
                            process.recoJetIds
                            )

process.recoAllJets = cms.Path(process.genParticlesForJets+process.recoAllGenJets+
                               process.recoAllJets+
                               process.recoAllPFJets+
                               process.recoAllTrackJets+
                               process.recoAllJetIds
                               )

process.recoAllJetsPUOffsetCorr = cms.Path(process.recoAllJetsPUOffsetCorr
                                           )

process.out = cms.EndPath(process.output)

# schedule
process.schedule = cms.Schedule(process.recoAllJets,process.out)

# Set the threshold for output logging to 'info'
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
#process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
#process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
#process.MessageLogger.debugModules = cms.untracked.vstring('*')
