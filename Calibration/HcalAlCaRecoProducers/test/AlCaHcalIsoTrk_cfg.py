import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
process = cms.Process("ALCAISOTRACK",Run2_2018)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HcalIsoTrackX=dict()
    process.MessageLogger.HcalIsoTrack=dict()

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:/afs/cern.ch/user/h/huwang/work/public/for_Sunanda/RECO_data.root',
 )
)

process.load('RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi')
process.towerMakerAll = process.calotowermaker.clone()
process.towerMakerAll.hbheInput = cms.InputTag("hbhereco")
process.towerMakerAll.hoInput = cms.InputTag("none")
process.towerMakerAll.hfInput = cms.InputTag("none")
process.towerMakerAll.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE"))
process.towerMakerAll.AllowMissingInputs = True

process.ALCARECOStreamHcalCalIsoTrkProducerFilter = cms.OutputModule("PoolOutputModule",
                                                               SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIsoTrkProducerFilter')
        ),
                                                               dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('ALCARECO'),
        filterName = cms.untracked.string('ALCARECOHcalCalIsoTrkProducerFilter')
        ),
                                                             eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
                                                             outputCommands = process.OutALCARECOHcalCalIsoTrkProducerFilter.outputCommands,
                                                             fileName = cms.untracked.string('newPoolOutput.root'),
)

process.alcaHcalIsotrkProducer.ignoreTriggers = True
process.alcaHcalIsotrkProducer.debugEvents = [640818633, 640797426, 641251898]
process.alcaHcalIsotrkFilter.debugEvents = [640818633, 640797426, 641251898]
#process.alcaHcalIsotrkProducer.debugEvents = [641031809, 641092744, 640862532,
#                                              640874735, 641845581, 641144982,
#                                              641124886, 641240201, 640856725,
#                                              641709599, 641406943, 640794164,
#                                              641820644, 641053143, 641458751,
#                                              641554667, 641621481]
#process.alcaHcalIsotrkFilter.debugEvents = [641031809, 641092744, 640862532,
#                                            640874735, 641845581, 641144982,
#                                            641124886, 641240201, 640856725,
#                                            641709599, 641406943, 640794164,
#                                            641820644, 641053143, 641458751,
#                                            641554667, 641621481]


# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.ALCARECOStreamHcalCalIsoTrkProducerFilterOutPath = cms.EndPath(process.ALCARECOStreamHcalCalIsoTrkProducerFilter)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOHcalCalIsoTrkProducerFilter,process.endjob_step,process.ALCARECOStreamHcalCalIsoTrkProducerFilterOutPath)
