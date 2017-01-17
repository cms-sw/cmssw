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
#        '/store/mc/RunIISpring16DR80/ZToEE_NNPDF30_13TeV-powheg_M_200_400/RAWAODSIM/PUSpring16RAWAODSIM_80X_mcRun2_asymptotic_2016_v3-v1/50000/FAB476A3-4F13-E611-B591-008CFAF06402.root',
        #"file:/opt/ppd/scratch/harper/mcTestFiles/DoubleEG_Run273450_4C4E9B34-2D1C-E611-AACF-02163E013455_AOD.root"
        "file:/opt/ppd/scratch/harper/dataFiles/DoubleEG_Run2016G-23Sep2016-v1_DiHEEPWOSS_GainSwitch_1.root",
)
                                       
)

process.bunchSpacingProducer = cms.EDProducer("BunchSpacingProducer")
process.load("RecoEgamma.EgammaTools.ecalWeightRecHitFromSelectedDigis_cff")
process.load("RecoEcal.EgammaClusterProducers.ecalMultiAndGSWeightRecHitEB_cfi")
process.load("RecoEcal.EgammaClusterProducers.gsFixedSuperClustering_cff")
process.load("RecoEcal.EgammaClusterProducers.gsFixedRefinedBarrelSuperClusters_cfi")
process.load("RecoEcal.EgammaClusterProducers.gsBrokenToGSFixedSuperClustersMap_cfi")
process.load("RecoEgamma.EgammaElectronProducers.gsFixedGsfElectronCores_cfi")
process.load("RecoEgamma.EgammaElectronProducers.gsFixedGsfElectrons_cfi")

process.egammaGainSwitchFixSequence = cms.Sequence(
    process.ecalWeightLocalRecoFromSelectedDigis*
    process.ecalMultiAndGSWeightRecHitEB*
    process.gsFixedParticleFlowSuperClustering*
    process.gsFixedRefinedBarrelSuperClusters*
    process.gsBrokenToGSFixedSuperClustersMap*
    process.gsFixedGsfElectronCores*
    process.gsFixedGsfElectrons)
                                         

#this does the simple fix (replace the crystals, assume fraction of 1)
process.newGsfElectronsSimpleFix = cms.EDProducer("GsfEleGSCrysSimpleFixer",
                                                  oldEles = cms.InputTag("gedGsfElectrons"),
                                                  ebMultiRecHits = cms.InputTag("reducedEcalRecHitsEB"),
                                                  ebMultiAndWeightsRecHits = cms.InputTag("ecalMultiAndGSWeightRecHitEB"),
)
#this does the simple fix (replace the crystals, assume fraction of 1)
process.newPhotonsSimpleFix = cms.EDProducer("PhotonGSCrysSimpleFixer",
                                             oldPhos = cms.InputTag("gedPhotons"),
                                             ebMultiRecHits = cms.InputTag("reducedEcalRecHitsEB"),
                                             ebMultiAndWeightsRecHits = cms.InputTag("ecalMultiAndGSWeightRecHitEB"),
                                             energyTypesToFix = cms.vstring("ecal_standard","ecal_photons","regression1","regression2"),
                                             energyTypeForP4 = cms.string("regression2")
)
 
process.gsFixSeq = cms.Sequence(
    process.bunchSpacingProducer*
    process.egammaGainSwitchFixSequence*
    process.newGsfElectronsSimpleFix*
    process.newPhotonsSimpleFix)
process.p = cms.Path(process.gsFixSeq)



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
