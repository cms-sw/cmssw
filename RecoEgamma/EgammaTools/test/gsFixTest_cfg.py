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

   


process.load('RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi')
process.ecalWeightUncalibRecHit.EBdigiCollection = cms.InputTag("selectDigi","selectedEcalEBDigiCollection")
process.ecalWeightUncalibRecHit.EEdigiCollection = cms.InputTag("selectDigi","selectedEcalEEDigiCollection")
process.load('RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi')
process.ecalWeightsRecHits = process.ecalRecHit.clone()
process.ecalWeightsRecHits.EEuncalibRecHitCollection = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEE")
process.ecalWeightsRecHits.EBuncalibRecHitCollection = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEB")
process.ecalWeightsRecHits.recoverEBFE = cms.bool(False)
process.ecalWeightsRecHits.recoverEEFE = cms.bool(False)
process.ecalWeightsRecHits.killDeadChannels = cms.bool(False)


process.ecalMultiAndGSWeightsRecHitEB = \
    cms.EDProducer("CombinedRecHitCollectionProducer",
                   primaryRecHits=cms.InputTag("reducedEcalRecHitsEB"),
                   secondaryRecHits=cms.InputTag("ecalWeightsRecHits","EcalRecHitsEB"),
                   outputCollectionName=cms.string(""),
                   flagsToReplaceHit=cms.vstring("kHasSwitchToGain6","kHasSwitchToGain1")
)

process.load("RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff")
process.particleFlowRecHitECAL.producers[0].src=cms.InputTag("ecalMultiAndGSWeightsRecHitEB")
process.particleFlowRecHitECAL.producers[1].src=cms.InputTag("reducedEcalRecHitsEE")
process.particleFlowRecHitPS.producers[0].src=cms.InputTag("reducedEcalRecHitsES")
process.particleFlowClusterECAL.energyCorrector.recHitsEBLabel=cms.InputTag("ecalMultiAndGSWeightsRecHitEB")
process.particleFlowClusterECAL.energyCorrector.recHitsEELabel=cms.InputTag("reducedEcalRecHitsEE")

process.load("RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff")

process.particleFlowSuperClusterECAL.regressionConfig.ecalRecHitsEB=cms.InputTag("ecalMultiAndGSWeightsRecHitEB")
process.particleFlowSuperClusterECAL.regressionConfig.ecalRecHitsEE=cms.InputTag("reducedEcalRecHitsEE")

process.bunchSpacingProducer = cms.EDProducer("BunchSpacingProducer")

process.fixedRefinedBarrelSuperClusters = cms.EDProducer("EGRefinedSCFixer",
                                                         fixedSC=cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),
                                                         orgSC=cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel",processName=cms.InputTag.skipCurrentProcess()),
                                                         orgRefinedSC=cms.InputTag("particleFlowEGamma"),
                                                         fixedPFClusters=cms.InputTag("particleFlowClusterECAL"),
                                
                                                         )
process.oldSuperClustersToNewMap = cms.EDProducer("MapNewToOldSCs",
                                                  oldSC=cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel",processName=cms.InputTag.skipCurrentProcess()),
                                                  newSC=cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),
                                                  oldRefinedSC=cms.InputTag("particleFlowEGamma"),
                                                  newRefinedSC=cms.InputTag("fixedRefinedBarrelSuperClusters")
)

process.newGsfElectronCores = cms.EDProducer("GsfElectronCoreGSCrysFixer",
                                             orgCores=cms.InputTag("gedGsfElectronCores"),
                                             ebRecHits=cms.InputTag("reducedEcalRecHitsEB"), ##ffs, weights dont have the gains switches set properly for some reason, doesnt matter, all we need this for is the gain so we can use orginal multifit
                #                             ebRecHits=cms.InputTag("ecalWeightsRecHits","EcalRecHitsEB"), ##ffs, weights dont have the gains switches set properly for some reason
                                             oldRefinedSCToNewMap=cms.InputTag("oldSuperClustersToNewMap","refinedSCs"),
                                             oldSCToNewMap=cms.InputTag("oldSuperClustersToNewMap","parentSCs"),
)

#this module properly fixes the electrons
from RecoEgamma.EgammaTools.regressionModifier_cfi import regressionModifier
process.newGsfElectrons = cms.EDProducer("GsfElectronGSCrysFixer",
                                         newCores=cms.InputTag("newGsfElectronCores"),
                                         oldEles=cms.InputTag("gedGsfElectrons"),
                                         ebRecHits=cms.InputTag("ecalMultiAndGSWeightsRecHitEB"),
                                         newCoresToOldCoresMap=cms.InputTag("newGsfElectronCores","parentCores"),
                                         regressionConfig = regressionModifier.clone(rhoCollection=cms.InputTag("fixedGridRhoFastjetAllTmp")),

)
                                         

#this does the simple fix (replace the crystals, assume fraction of 1)
process.newGsfElectronsSimpleFix = cms.EDProducer("GsfEleGSCrysSimpleFixer",
                                                  oldEles = cms.InputTag("gedGsfElectrons"),
                                                  ebMultiRecHits = cms.InputTag("reducedEcalRecHitsEB"),
                                                  ebMultiAndWeightsRecHits = cms.InputTag("ecalMultiAndGSWeightsRecHitEB"),
)
#this does the simple fix (replace the crystals, assume fraction of 1)
process.newPhotonsSimpleFix = cms.EDProducer("PhotonGSCrysSimpleFixer",
                                             oldPhos = cms.InputTag("gedPhotons"),
                                             ebMultiRecHits = cms.InputTag("reducedEcalRecHitsEB"),
                                             ebMultiAndWeightsRecHits = cms.InputTag("ecalMultiAndGSWeightsRecHitEB"),
                                             energyTypesToFix = cms.vstring("ecal_standard","ecal_photons","regression1","regression2"),
                                             energyTypeForP4 = cms.string("regression2")
)
 
process.gsFixSeq = cms.Sequence(process.bunchSpacingProducer*
                                process.ecalWeightUncalibRecHit*
                                process.ecalWeightsRecHits*
                                process.ecalMultiAndGSWeightsRecHitEB*
                                process.pfClusteringPS*
                                process.pfClusteringECAL*
                                process.particleFlowSuperClusteringSequence*
                                process.fixedRefinedBarrelSuperClusters*
                                process.oldSuperClustersToNewMap*
                                process.newGsfElectronCores*
                                process.newGsfElectrons*
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
