import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO4")

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')
process.load('RecoJets.Configuration.RecoJPTJets_cff')
process.load('JetMETCorrections.Configuration.CorrectedJetProducersAllAlgos_cff')
process.load('JetMETCorrections.Configuration.CorrectedJetProducers_cff')
process.load('JetMETCorrections.Configuration.JetCorrectors_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

#################################################################  

### For 219, file from RelVal
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
   '/store/relval/CMSSW_10_6_4/RelValProdTTbar_13_pmx25ns/MINIAODSIM/PUpmx25ns_106X_upgrade2018_realistic_v9-v1/10000/87AD30D2-F673-F54C-8974-CB916CC66098.root'
   )
)
process.RECOoutput = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_JetPlusTrackZSPCorJetAntiKt4PAT_*_*"),
    fileName = cms.untracked.string('file:jptreco.root')
)

##########
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOoutput_step = cms.EndPath(process.RECOoutput)

process.load("RecoJets.JetPlusTracks.PATJetPlusTrackCorrections_cff")
process.p01=cms.Path(process.PATJetPlusTrackCorrectionsAntiKt4)

process.p1 =cms.Schedule(
                     process.p01,
                     process.endjob_step,
                     process.RECOoutput_step
)

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring

#call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)
