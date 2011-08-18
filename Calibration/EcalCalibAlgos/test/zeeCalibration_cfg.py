import FWCore.ParameterSet.Config as cms

#IMPORTANT PRESCAlE
#prescale = 1

process = cms.Process('ZCALIB')

#process.prescaler = cms.EDFilter("Prescaler",
#                                    prescaleFactor = cms.int32(prescale),
#                                    prescaleOffset = cms.int32(0)
#                                    )
# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
#process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
#process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
#process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.load('Configuration.EventContent.EventContent_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200000)
)

from Calibration.EcalCalibAlgos.DoubleElectron_Jul05_ALCAELECTRON_cff import *
from Calibration.EcalCalibAlgos.Cert_160404_172802_cff import *

process.source = cms.Source("PoolSource",
                            fileNames = readFiles,
                            lumisToProcess = goodLumis                            
)

#process.source.inputCommands = cms.untracked.vstring("drop *", "keep *_*_*_HLT")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# Other statements
process.GlobalTag.globaltag = 'GR_R_42_V17::All' 
#process.GlobalTag.toGet = cms.VPSet(
# cms.PSet(record = cms.string("EcalIntercalibConstantsRcd"),
#          tag = cms.string("EcalIntercalibConstants_v10_offline"),
#          connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_ECAL")
#         )
#  ,cms.PSet(record = cms.string("EcalADCToGeVConstantRcd"),
#          tag = cms.string("EcalADCToGeVConstant_v10_offline"),
#          connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_ECAL")
#         )
# ,cms.PSet(record = cms.string("EcalLaserAPDPNRatiosRcd"),
##           tag = cms.string("EcalLaserAPDPNRatios_2011fit_noVPT_nolim_online"),
#           tag = cms.string("EcalLaserAPDPNRatios_test_20110625"),
#           tag = cms.string("EcalLaserAPDPNRatios_2011V3_online"),
#           tag = cms.string("EcalLaserAPDPNRatios_2011mixfit_online"),
#          connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_ECAL")
#          )
 #beam spot to arrive to very last runs after 167151
#   ,cms.PSet(record = cms.string("BeamSpotObjectsRcd"),
#          tag = cms.string("BeamSpotObjects_PCL_byLumi_v0_prompt"),
#          connect = cms.untracked.string("frontier://PromptProd/CMS_COND_31X_BEAMSPOT")
#         )



process.load("Calibration.EcalCalibAlgos.electronRecalibSCAssociator_cfi")

process.load("Calibration.EcalCalibAlgos.zeeCalibration_cff")

process.load("RecoLocalCalo.EcalRecProducers.ecalRecalibRecHit_cfi")

process.ecalRecHit.doIntercalib = cms.bool(True)
process.ecalRecHit.doLaserCorrection = cms.bool(False)
process.ecalRecHit.EBRecHitCollection = "alCaIsolatedElectrons:alcaBarrelHits"
process.ecalRecHit.EERecHitCollection = "alCaIsolatedElectrons:alcaEndcapHits"
process.ecalRecHit.EBRecalibRecHitCollection = "EcalRecHitsEB"
process.ecalRecHit.EERecalibRecHitCollection = "EcalRecHitsEE"

#hybridSuperClusters.ecalhitproducer = "recalibRechit"
#correctedHybridSuperClusters.recHitProducer = recalibRechit:EcalRecHitsEB
process.correctedHybridSuperClusters.corectedSuperClusterCollection = 'recalibSC'
process.correctedMulti5x5SuperClustersWithPreshower.corectedSuperClusterCollection = 'endcapRecalibSC'
process.multi5x5SuperClustersWithPreshower.preshRecHitProducer = cms.InputTag("alCaIsolatedElectrons","alcaPreshowerHits")
process.multi5x5PreshowerClusterShape.preshRecHitProducer = cms.InputTag("alCaIsolatedElectrons","alcaPreshowerHits")

process.electronRecalibSCAssociator.scIslandCollection = cms.string('endcapRecalibSC')
process.electronRecalibSCAssociator.scIslandProducer = cms.string('correctedMulti5x5SuperClustersWithPreshower')
process.electronRecalibSCAssociator.scProducer = cms.string('correctedHybridSuperClusters')
process.electronRecalibSCAssociator.scCollection = cms.string('recalibSC')
process.electronRecalibSCAssociator.electronProducer = 'PassingWP90'

process.looper.initialMiscalibrationBarrel = cms.untracked.string('')
process.looper.calibMode = cms.string('RING')
process.looper.initialMiscalibrationEndcap = cms.untracked.string('')
process.looper.HLTriggerResults = cms.InputTag("TriggerResults","","ZCALIB")
#process.looper.rechitCollection = cms.string('EcalRecHitsEB')
process.looper.ZCalib_CalibType = cms.untracked.string('RING')
process.looper.ZCalib_nCrystalCut = cms.untracked.int32(-1)
process.looper.maxLoops = cms.untracked.uint32(5)
process.looper.erechitProducer = cms.string('ecalRecHit')
process.looper.wantEtaCorrection = cms.untracked.bool(False)
process.looper.outputFile = cms.string('myHistograms_test.root')
process.looper.electronSelection = cms.untracked.uint32(0)
#process.looper.scProducer = cms.string('correctedHybridSuperClusters')
process.looper.rechitProducer = cms.string('ecalRecHit')
#process.looper.scIslandProducer = cms.string('electronRecalibSCAssociator')
#Setting to null value avoids reading mc infos
process.looper.mcProducer = cms.untracked.string('')
#process.looper.electronProducer = cms.string('PassingWP90')
#process.looper.scCollection = cms.string('recalibSC')

HLTPath = "HLT_Ele"
HLTProcessName = "HLT"

#electron cuts
ELECTRON_ET_CUT_MIN = 25.0
ELECTRON_CUTS = "(abs(superCluster.eta)<2.5) && (ecalEnergy*sin(superClusterPosition.theta)>" + str(ELECTRON_ET_CUT_MIN) + ")"

#mass cuts (for T&P)
MASS_CUT_MIN = 60.

##    _____ _           _                     ___    _
##   | ____| | ___  ___| |_ _ __ ___  _ __   |_ _|__| |
##   |  _| | |/ _ \/ __| __| '__/ _ \| '_ \   | |/ _` |
##   | |___| |  __/ (__| |_| | | (_) | | | |  | | (_| |
##   |_____|_|\___|\___|\__|_|  \___/|_| |_| |___\__,_|
##

process.goodElectrons = cms.EDFilter("GsfElectronRefSelector",
                                 src = cms.InputTag( 'gsfElectrons' ),
                                 cut = cms.string( ELECTRON_CUTS )
                             )

process.PassingWP90 = process.goodElectrons.clone(
    cut = cms.string(
        process.goodElectrons.cut.value() +
            " && (gsfTrack.trackerExpectedHitsInner.numberOfHits<=1 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))" #wrt std WP90 allowing 1 numberOfMissingExpectedHits
            " && (ecalEnergy*sin(superClusterPosition.theta)>" + str(ELECTRON_ET_CUT_MIN) + ")"
            " && ((isEB"
            " && ( dr03TkSumPt/p4.Pt <0.12 && dr03EcalRecHitSumEt/p4.Pt < 0.09 && dr03HcalTowerSumEt/p4.Pt  < 0.1 )"
            " && (sigmaIetaIeta<0.01)"
            " && ( -0.8<deltaPhiSuperClusterTrackAtVtx<0.8 )"
            " && ( -0.007<deltaEtaSuperClusterTrackAtVtx<0.007 )"
            " && (hadronicOverEm<0.12)"
            ")"
            " || (isEE"
            " && ( dr03TkSumPt/p4.Pt <0.07 && dr03EcalRecHitSumEt/p4.Pt < 0.07 && dr03HcalTowerSumEt/p4.Pt  < 0.07 )"
            " && (sigmaIetaIeta<0.03)"
            " && ( -0.7<deltaPhiSuperClusterTrackAtVtx<0.7 )"
            " && ( -0.009<deltaEtaSuperClusterTrackAtVtx<0.009 )"
            " && (hadronicOverEm<0.1) "
            "))"
            )
    )

process.Zele_sequence = cms.Sequence(
    process.goodElectrons * process.PassingWP90
    )


import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
process.ZEEHltFilter = copy.deepcopy(hltHighLevel)
process.ZEEHltFilter.throw = cms.bool(False)
process.ZEEHltFilter.HLTPaths = ["HLT_Ele*"]

##    ____       _
##   |  _ \ __ _(_)_ __ ___
##   | |_) / _` | | '__/ __|
##   |  __/ (_| | | |  \__ \
##   |_|   \__,_|_|_|  |___/
##
##

process.tagGsf =  cms.EDProducer("CandViewShallowCloneCombiner",
                                 decay = cms.string("PassingWP90 PassingWP90"),
                                 checkCharge = cms.bool(False),
                                 cut   = cms.string("mass > " + str(MASS_CUT_MIN))
            )
process.tagGsfCounter = cms.EDFilter("CandViewCountFilter",
                                     src = cms.InputTag("tagGsf"),
                                     minNumber = cms.uint32(1)
                                     )

process.tagGsfFilter = cms.Sequence(process.tagGsf * process.tagGsfCounter)
process.tagGsfSeq = cms.Sequence( process.ZEEHltFilter * (process.Zele_sequence) * process.tagGsfFilter )

process.zFilterPath = cms.Path( process.tagGsfSeq * process.ecalRecHit * process.hybridClusteringSequence* process.multi5x5ClusteringSequence * process.multi5x5PreshowerClusteringSequence * process.electronRecalibSCAssociator )
