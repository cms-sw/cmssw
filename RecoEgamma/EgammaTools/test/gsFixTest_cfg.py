import FWCore.ParameterSet.Config as cms 

# set up process
process = cms.Process("GSFIX")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1000),
    limit = cms.untracked.int32(10000000)
)
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

#setup global tag
from Configuration.AlCa.GlobalTag import GlobalTag
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag = GlobalTag(process.GlobalTag, '80X_dataRun2_2016SeptRepro_v3', '') #


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(
        #"file:/opt/ppd/scratch/harper/dataFiles/DoubleEG_Run2016G-23Sep2016-v1_DiHEEPWOSS_GainSwitch_1.root",
        "file:DoubleEG_Run2016G-23Sep2016-v1_DiHEEPWOSS_GainSwitch_2.root",#/store/user/sharper/EventSkim/DiHEEPWOSS_GainSwitch/AOD/DoubleEG/Run2016G-23Sep2016-v1_AOD_DiHEEPWOSS_GainSwitch/170112_185336/0000/DoubleEG_Run2016G-23Sep2016-v1_DiHEEPWOSS_GainSwitch_10.root", 
                               )
)

#defines our sequences to remake the electrons and photons
#there are currently two sequences, egammaGainSwitchFixSequence and
#egammaGainSwitchSimpleFixSequence
#
#egammaGainSwitchFixSequence: remakes the superclusters and then can correctly
#                             redo the electrons and photons (photons not implimented)
#
#egammaGainSwitchSimpleFixSequence: just corrects electorns and photons energy and showershape
#                                   correction is ( supercluster raw energy - E(gain switched multi fit hits) + E(gain switched weights hits) ) / supercluster raw energy
#                                   does not correct photon H/E or R9

process.load("RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi")
process.bunchSpacingProducerSequence = cms.Sequence(process.bunchSpacingProducer)

process.load("RecoEgamma.EgammaTools.egammaGainSwitchFixForPAT_cff")
process.load("RecoParticleFlow.PFProducer.pfGSFixLinkerForPAT_cff")
process.load("RecoEgamma.EgammaIsolationAlgos.pfClusterIsolationRemapForPAT_cff")
process.load("RecoEgamma.ElectronIdentification.idExternalRemapForPAT_cff")

process.p = cms.Path(process.bunchSpacingProducerSequence * 
                     process.egammaGainSwitchFixSequence *
                     process.particleFlowLinks *
                     process.pfClusterIsolationSequence *
                     process.ElectronIDExternalProducerRemapSequence *
                     process.PhotonIDExternalProducerRemapSequence)
                    
#dumps the products made for easier debugging, you wouldnt normally need to do this
#edmDumpEventContent outputTest.root shows you all the products produced
#will be very slow when this is happening
process.load('Configuration.EventContent.EventContent_cff')
process.output = cms.OutputModule("PoolOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(4),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('MINIAODSIM'),
        filterName = cms.untracked.string('')
    ),
    dropMetaData = cms.untracked.string('ALL'),
    eventAutoFlushCompressedSize = cms.untracked.int32(15728640),
    fastCloning = cms.untracked.bool(False),
    fileName = cms.untracked.string('outputTest.root'),
    outputCommands = process.MINIAODSIMEventContent.outputCommands,
    overrideInputFileSplitLevels = cms.untracked.bool(True)
)
process.output.outputCommands = cms.untracked.vstring('keep *_*_*_*',
                                                           )
process.outPath = cms.EndPath(process.output)
