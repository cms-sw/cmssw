import FWCore.ParameterSet.Config as cms

process = cms.Process("PIONS")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_7_1_0_pre8/RelValTTbar/GEN-SIM-RECO/PRE_STA71_V4-v1/00000/2E5039B2-DCE2-E311-B7D7-02163E00E68B.root'
    ),
   skipEvents = cms.untracked.uint32( 0 ),
   secondaryFileNames=cms.untracked.vstring(
'/store/relval/CMSSW_7_1_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V4-v1/00000/0C2624AF-6EE2-E311-AA3F-02163E00F3C7.root',
'/store/relval/CMSSW_7_1_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V4-v1/00000/1A9C8819-35E2-E311-B624-0025904B0FBC.root',
'/store/relval/CMSSW_7_1_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V4-v1/00000/32D0F508-B9E2-E311-9644-02163E00EAE7.root',
'/store/relval/CMSSW_7_1_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V4-v1/00000/5834CA80-6FE2-E311-B0C8-02163E00E763.root',
'/store/relval/CMSSW_7_1_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V4-v1/00000/6A05B8BA-DCE2-E311-8FC0-02163E00EA08.root',
'/store/relval/CMSSW_7_1_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V4-v1/00000/6EDA0B9A-35E2-E311-8DF4-02163E00E983.root',
'/store/relval/CMSSW_7_1_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V4-v1/00000/906BFA50-73E2-E311-929F-02163E00EA92.root',
'/store/relval/CMSSW_7_1_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V4-v1/00000/9A58C741-6EE2-E311-BA26-02163E00E8E3.root',
'/store/relval/CMSSW_7_1_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V4-v1/00000/D63812F1-34E2-E311-A504-02163E00EA40.root',
'/store/relval/CMSSW_7_1_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_STA71_V4-v1/00000/E4FFB336-6EE2-E311-A773-02163E00EA1B.root'
   )
)

process.PionTracksProducer = cms.EDProducer('PionTracksProducer',
    src = cms.InputTag('generalV0Candidates','Kshort')
)


process.load("TrackPropagation.Geant4e.geantRefit_cff")
process.Geant4eTrackRefitter.src = cms.InputTag("PionTracksProducer","pionTrack","PIONS")
process.g4RefitPath = cms.Path( process.MeasurementTrackerEvent * process.geant4eTrackRefit )

from RecoVertex.V0Producer.generalV0Candidates_cff import *
process.generalV0CandidatesModified = generalV0Candidates.clone(
  trackRecoAlgorithm = cms.InputTag('Geant4eTrackRefitter'),
  selectLambdas = cms.bool(False)
)
process.v0 = cms.Path(process.generalV0CandidatesModified)
  
process.load("SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("Validation.Configuration.postValidation_cff")
process.quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')
process.quickTrackAssociatorByHits.useClusterTPAssociation = cms.bool(True)
process.load("SimTracker.TrackerHitAssociation.clusterTpAssociationProducer_cfi")
process.piPath = cms.Path(process.PionTracksProducer*process.tpClusterProducer)
process.load('Validation.RecoTrack.MultiTrackValidator_cfi')
process.multiTrackValidator.associators = ['quickTrackAssociatorByHits']
process.multiTrackValidator.label = cms.VInputTag(cms.InputTag("PionTracksProducer","pionTrack","PIONS"), cms.InputTag("Geant4eTrackRefitter"))
process.multiTrackValidator.useLogPt=cms.untracked.bool(True)
process.multiTrackValidator.minpT = cms.double(0.1)
process.multiTrackValidator.maxpT = cms.double(3000.0)
process.multiTrackValidator.nintpT = cms.int32(40)
process.multiTrackValidator.UseAssociators = cms.bool(True)
process.multiTrackValidator.outputFile="ciao.root"
process.multiTrackValidator.runStandalone=True

process.dEta05 = cms.EDAnalyzer('KsAnalyzer',
               vertex = cms.untracked.InputTag('generalV0CandidatesModified','Kshort'),
               dEtaMaxCut = cms.untracked.double(0.5)
)

process.dEta07 = process.dEta05.clone(dEtaMaxCut = cms.untracked.double(0.7))
process.dEta03 = process.dEta05.clone(dEtaMaxCut = cms.untracked.double(0.3))
process.dEta5  = process.dEta05.clone(dEtaMaxCut = cms.untracked.double(5))

process.dEtaOrig05 = cms.EDAnalyzer('KsAnalyzer',
               vertex = cms.untracked.InputTag('generalV0Candidates','Kshort'),
               dEtaMaxCut = cms.untracked.double(0.5)
)

process.dEtaOrig07 = process.dEtaOrig05.clone(dEtaMaxCut = cms.untracked.double(0.7))
process.dEtaOrig03 = process.dEtaOrig05.clone(dEtaMaxCut = cms.untracked.double(0.3))
process.dEtaOrig5  = process.dEtaOrig05.clone(dEtaMaxCut = cms.untracked.double(5))

process.TFileService = cms.Service("TFileService",
               fileName = cms.string('analyzer_JOBNUMBER.root')
)

process.analyzerPath = cms.Path(process.dEta05*process.dEta03*process.dEta07*process.dEta5*
                                process.dEtaOrig05*process.dEtaOrig03*process.dEtaOrig07*process.dEtaOrig5*
                                process.multiTrackValidator)
# Output definition
process.DQMoutput = cms.OutputModule("PoolOutputModule",
  splitLevel = cms.untracked.int32(0),
  outputCommands = process.DQMEventContent.outputCommands,
  fileName = cms.untracked.string('file:MTV_inDQM_JOB.root'),
  dataset = cms.untracked.PSet(
    filterName = cms.untracked.string(''),
    dataTier = cms.untracked.string('DQM')
  )
)
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)


#process.e=cms.EndPath(process.out)


process.schedule = cms.Schedule(process.piPath, process.g4RefitPath, process.v0, process.analyzerPath, process.endjob_step, process.DQMoutput_step) #, process.e)

