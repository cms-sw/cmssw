import FWCore.ParameterSet.Config as cms

process = cms.Process("TIMECALIBANALYSISELE")

filelist = cms.untracked.vstring()
filelist.extend([
#'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/548BD0EF-5F67-E211-AB31-00261894393F.root'
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SingleElectron/RECO/22Jan2013-v1/10000/507233A4-57AC-E211-9EA6-0026189438F7.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SingleElectron/RECO/22Jan2013-v1/10000/502CC367-5EAC-E211-A498-00261894385A.root'
])

# Output - dummy
process.out = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    fileName = cms.untracked.string('file:EcalTiming_RUn2012C.root'),
    )


# gfworks: to get clustering 

# Geometry
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi") # gfwork: need this?
process.CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder")

################# Recovering Unclean Supercluster of gsf/gedgsf electons##################
##########################################################################################

### First Recover the superclusters #######################################################
process.load("RecoEcal.EgammaClusterProducers.uncleanSCRecovery_cfi")
process.uncleanSCRecovered.cleanScCollection=cms.InputTag ("correctedHybridSuperClusters")

#### Now do Electron reco sequence ######################################################
#### OR Import both as this
from RecoEgamma.EgammaElectronProducers.gsfElectronModules_cff  import * 
#### EcalDrivenseedElectrn Modules #######################################################
from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeedsModules_cff  import*

#### Clone Electron/ElectronCores
#process.mygsfElectronCores=RecoEgamma.EgammaElectronProducers.gsfElectronCores_cfi.gsfElectronCores.clone()
###################################################################################
#### 							  ##########################
####	CLONING  & CHANGING PROCESSES			  ##########################
####							  ##########################	
####################################################################################

from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeeds_cfi import *
uncleanedOnlyecalDrivenElectronSeeds = ecalDrivenElectronSeeds.clone(
    #barrelSuperClusters = cms.InputTag("uncleanedOnlyCorrectedHybridSuperClusters"),
    #endcapSuperClusters = cms.InputTag("uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower")
    barrelSuperClusters = cms.InputTag("uncleanSCRecovered:uncleanHybridSuperClusters"),
    endcapSuperClusters = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower")      ## As with Photons
)

from TrackingTools.GsfTracking.CkfElectronCandidateMaker_cff import *
uncleanedOnlyElectronCkfTrackCandidates = electronCkfTrackCandidates.clone(
    src = cms.InputTag("uncleanedOnlyecalDrivenElectronSeeds")
    )

from TrackingTools.GsfTracking.GsfElectronGsfFit_cff import *
uncleanedOnlyElectronGsfTracks = electronGsfTracks.clone(
    src = 'uncleanedOnlyElectronCkfTrackCandidates'
    )

#################### Unclean Traking infor ####################################
uncleanedOnlyTracking = cms.Sequence(uncleanedOnlyecalDrivenElectronSeeds*uncleanedOnlyElectronCkfTrackCandidates*uncleanedOnlyElectronGsfTracks)

#
# Conversions
#

from RecoEgamma.EgammaPhotonProducers.conversionTrackCandidates_cfi import *
uncleanedOnlyConversionTrackCandidates = conversionTrackCandidates.clone(
    scHybridBarrelProducer = cms.InputTag("uncleanSCRecovered:uncleanHybridSuperClusters"),   ## same as photons
    #scHybridBarrelProducer = cms.InputTag("uncleanedOnlyCorrectedHybridSuperClusters"),
    bcBarrelCollection  = cms.InputTag("hybridSuperClusters","uncleanOnlyHybridSuperClusters"),
    scIslandEndcapProducer  = cms.InputTag("uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower"),
    bcEndcapCollection  = cms.InputTag("multi5x5SuperClusters","uncleanOnlyMulti5x5EndcapBasicClusters")
    )

from RecoEgamma.EgammaPhotonProducers.ckfOutInTracksFromConversions_cfi import *
uncleanedOnlyCkfOutInTracksFromConversions = ckfOutInTracksFromConversions.clone(
    src = cms.InputTag("uncleanedOnlyConversionTrackCandidates","outInTracksFromConversions"),
    producer = cms.string('uncleanedOnlyConversionTrackCandidates'),
    ComponentName = cms.string('uncleanedOnlyCkfOutInTracksFromConversions')
    )

from RecoEgamma.EgammaPhotonProducers.ckfInOutTracksFromConversions_cfi import *
uncleanedOnlyCkfInOutTracksFromConversions = ckfInOutTracksFromConversions.clone(
    src = cms.InputTag("uncleanedOnlyConversionTrackCandidates","inOutTracksFromConversions"),
    producer = cms.string('uncleanedOnlyConversionTrackCandidates'),
    ComponentName = cms.string('uncleanedOnlyCkfInOutTracksFromConversions')
    )
############### Unclean Tracks From Conversion ##################################
uncleanedOnlyCkfTracksFromConversions = cms.Sequence(uncleanedOnlyConversionTrackCandidates*uncleanedOnlyCkfOutInTracksFromConversions*uncleanedOnlyCkfInOutTracksFromConversions)

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyGeneralConversionTrackProducer = generalConversionTrackProducer.clone()

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyInOutConversionTrackProducer = inOutConversionTrackProducer.clone(
    TrackProducer = cms.string('uncleanedOnlyCkfInOutTracksFromConversions')
    )

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyOutInConversionTrackProducer = outInConversionTrackProducer.clone(
    TrackProducer = cms.string('uncleanedOnlyCkfOutInTracksFromConversions')
    )

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyGsfConversionTrackProducer = gsfConversionTrackProducer.clone(
    TrackProducer = cms.string('uncleanedOnlyElectronGsfTracks')
    )
############# Unclean  Conversion Tracks #########################
uncleanedOnlyConversionTrackProducers  = cms.Sequence(uncleanedOnlyGeneralConversionTrackProducer*uncleanedOnlyInOutConversionTrackProducer*uncleanedOnlyOutInConversionTrackProducer*uncleanedOnlyGsfConversionTrackProducer)

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyInOutOutInConversionTrackMerger = inOutOutInConversionTrackMerger.clone(
    TrackProducer2 = cms.string('uncleanedOnlyOutInConversionTrackProducer'),
    TrackProducer1 = cms.string('uncleanedOnlyInOutConversionTrackProducer')
    )

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyGeneralInOutOutInConversionTrackMerger = generalInOutOutInConversionTrackMerger.clone(
    TrackProducer2 = cms.string('uncleanedOnlyGeneralConversionTrackProducer'),
    TrackProducer1 = cms.string('uncleanedOnlyInOutOutInConversionTrackMerger')
    )

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyGsfGeneralInOutOutInConversionTrackMerger = gsfGeneralInOutOutInConversionTrackMerger.clone(
    TrackProducer2 = cms.string('uncleanedOnlyGsfConversionTrackProducer'),
    TrackProducer1 = cms.string('uncleanedOnlyGeneralInOutOutInConversionTrackMerger')
    )

uncleanedOnlyConversionTrackMergers = cms.Sequence(uncleanedOnlyInOutOutInConversionTrackMerger*uncleanedOnlyGeneralInOutOutInConversionTrackMerger*uncleanedOnlyGsfGeneralInOutOutInConversionTrackMerger)

from RecoEgamma.EgammaPhotonProducers.allConversions_cfi import *
uncleanedOnlyAllConversions = allConversions.clone(
    #scBarrelProducer = cms.InputTag("uncleanedOnlyCorrectedHybridSuperClusters"),
    scBarrelProducer = cms.InputTag("uncleanSCRecovered:uncleanHybridSuperClusters"),
    bcBarrelCollection  = cms.InputTag("hybridSuperClusters","uncleanOnlyHybridSuperClusters"),
    scEndcapProducer = cms.InputTag("uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower"),
    bcEndcapCollection = cms.InputTag("multi5x5SuperClusters","uncleanOnlyMulti5x5EndcapBasicClusters"),
    src = cms.InputTag("uncleanedOnlyGsfGeneralInOutOutInConversionTrackMerger")
    )
### If recovering conversions #########################################################
uncleanedOnlyConversions = cms.Sequence(uncleanedOnlyCkfTracksFromConversions*uncleanedOnlyConversionTrackProducers*uncleanedOnlyConversionTrackMergers*uncleanedOnlyAllConversions)

#
# Particle Flow Tracking
#

from RecoParticleFlow.PFTracking.pfTrack_cfi import *
uncleanedOnlyPfTrack = pfTrack.clone(
    GsfTrackModuleLabel = cms.InputTag("uncleanedOnlyElectronGsfTracks")
    )

from RecoParticleFlow.PFTracking.pfConversions_cfi import *
uncleanedOnlyPfConversions = pfConversions.clone(
    conversionCollection = cms.InputTag("allConversions")
    )

from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
uncleanedOnlyPfTrackElec = pfTrackElec.clone(
    PFConversions = cms.InputTag("uncleanedOnlyPfConversions"),
    GsfTrackModuleLabel = cms.InputTag("uncleanedOnlyElectronGsfTracks"),
    PFRecTrackLabel = cms.InputTag("uncleanedOnlyPfTrack")
    )

################## Recovering Uncleaned PF track ###################################
uncleanedOnlyPfTracking = cms.Sequence(uncleanedOnlyPfTrack*uncleanedOnlyPfConversions*uncleanedOnlyPfTrackElec)

#
# Electrons
#

from RecoEgamma.EgammaElectronProducers.gsfElectronCores_cfi import *
uncleanedOnlyGsfElectronCores = ecalDrivenGsfElectronCores.clone(
    gsfTracks = cms.InputTag("uncleanedOnlyElectronGsfTracks"),
    gsfPfRecTracks = cms.InputTag("uncleanedOnlyPfTrackElec")
    )

from RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi import *
uncleanedOnlyecalDrivenGsfElectrons = ecalDrivenGsfElectrons.clone(
    gsfPfRecTracksTag = cms.InputTag("uncleanedOnlyPfTrackElec"),
    gsfElectronCoresTag = cms.InputTag("uncleanedOnlyGsfElectronCores"),
    seedsTag = cms.InputTag("uncleanedOnlyecalDrivenElectronSeeds"),
    barrelRecHitCollectionTag = cms.InputTag("reducedEcalRecHitsEB"),
    endcapRecHitCollectionTag = cms.InputTag("reducedEcalRecHitsEE")
    )

#### Clone Electron to use ReducedEcal rechits
mygsfElectrons=RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi.gsfElectrons.clone(
    previousGsfElectronsTag = cms.InputTag("uncleanedOnlyecalDrivenGsfElectrons"),
    barrelRecHitCollectionTag = cms.InputTag("reducedEcalRecHitsEB"),
    endcapRecHitCollectionTag = cms.InputTag("reducedEcalRecHitsEE"),
    seedsTag = cms.InputTag("uncleanedOnlyecalDrivenElectronSeeds"),
    gsfPfRecTracksTag = cms.InputTag("uncleanedOnlyPfTrackElec"),
    gsfElectronCoresTag = cms.InputTag("uncleanedOnlyGsfElectronCores")

)
####The unleaned Electron process ###############
uncleanedOnlyElectrons = cms.Sequence( process.uncleanSCRecovered*
                                               uncleanedOnlyGsfElectronCores*mygsfElectrons)
#uncleanedOnlyElectrons = cms.Sequence(uncleanedOnlyGsfElectronCores*uncleanedOnlyecalDrivenGsfElectrons)

#
# Whole Sequence
#
uncleanedOnlyElectronSequence = cms.Sequence(uncleanedOnlyTracking*uncleanedOnlyConversions*uncleanedOnlyPfTracking*uncleanedOnlyElectrons)


########################## Maybe needed Services? ####################################
#---Needed to Reconsctruct on the fly from uncleaned SCs without timing cut for slpikes
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('RecoEgamma.EgammaPhotonProducers.conversionTracks_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoEcal.Configuration.RecoEcal_cff")
from Configuration.StandardSequences.Reconstruction_cff import *
from RecoEcal.Configuration.RecoEcal_cff import *

##
from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import *
process.load("FWCore.MessageLogger.MessageLogger_cfi")

#########################################################################################
#########################################################################################


# pat needed to work out electron id/iso
from PhysicsTools.PatAlgos.tools.metTools import *
from PhysicsTools.PatAlgos.tools.tauTools import *
from PhysicsTools.PatAlgos.tools.jetTools import *
from PhysicsTools.PatAlgos.tools.coreTools import *
from PhysicsTools.PatAlgos.tools.pfTools import *
from PhysicsTools.PatAlgos.selectionLayer1.leptonCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.photonCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi import *


# Global Tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag( process.GlobalTag, 'GR_R_53_V18::All' )
# tag below tested in CMSSW_4_3_0_pre3
#process.GlobalTag.globaltag = 'GR_R_42_V14::All'

# this is for jan16 reprocessing - tested in CMSSW_4_3_0_pre3
#process.GlobalTag.globaltag = 'FT_R_42_V24::All'

process.load('Configuration.StandardSequences.MagneticField_38T_cff')


# Trigger
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v2_Unprescaled_cff")
import FWCore.Modules.printContent_cfi
process.dumpEv = FWCore.Modules.printContent_cfi.printContent.clone()

import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
process.gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()


process.options = cms.untracked.PSet(
SkipEvent = cms.untracked.vstring('ProductNotFound')	
)

########## NOW  DO PAT ################################################
##########             ################################################
#########Load PAT sequences
process.load("PhysicsTools.PatAlgos.patSequences_cff")
process.load("PhysicsTools.PatAlgos.tools.pfTools")
## THis is NOT MC => remove matching
removeMCMatching(process, ['All'])
## bugfix for DATA Run2011 (begin)
removeSpecificPATObjects( process, ['Taus'] )
process.patDefaultSequence.remove( process.patTaus )
process.patElectrons.isoDeposits = cms.PSet()
process.patElectrons.addElectronID = cms.bool(True)
process.patElectrons.electronIDSources = cms.PSet(
            simpleEleId95relIso= cms.InputTag("simpleEleId95relIso"),
            simpleEleId90relIso= cms.InputTag("simpleEleId90relIso"),
            simpleEleId85relIso= cms.InputTag("simpleEleId85relIso"),
            simpleEleId80relIso= cms.InputTag("simpleEleId80relIso"),
            simpleEleId70relIso= cms.InputTag("simpleEleId70relIso"),
            simpleEleId60relIso= cms.InputTag("simpleEleId60relIso"),
            simpleEleId95cIso= cms.InputTag("simpleEleId95cIso"),
            simpleEleId90cIso= cms.InputTag("simpleEleId90cIso"),
            simpleEleId85cIso= cms.InputTag("simpleEleId85cIso"),
            simpleEleId80cIso= cms.InputTag("simpleEleId80cIso"),
            simpleEleId70cIso= cms.InputTag("simpleEleId70cIso"),
            simpleEleId60cIso= cms.InputTag("simpleEleId60cIso"),
            )
###
process.load("ElectroWeakAnalysis.WENu.simpleEleIdSequence_cff")

########## The Electron ID sequence #############################
process.patElectronIDs = cms.Sequence(process.simpleEleIdSequence)

#### Make Pat Electrons Begining with Unlcean gsfCores #####################################

#import RecoEgamma.EgammaElectronProducers.python.uncleanedOnlyElectronSequence_cff.py *
process.makePatElectrons = cms.Sequence( uncleanedOnlyElectronSequence* 
                                         process.patElectronIDs *
                                         process.patElectrons)
process.makePatCandidates = cms.Sequence( process.makePatElectrons   )
process.patMyDefaultSequence = cms.Sequence(process.makePatCandidates)



# this is the ntuple producer
process.load("CalibCalorimetry.EcalTiming.ecalTimeEleTree_cfi")
process.ecalTimeEleTree.OutfileName = 'EcalTimeTree'
process.ecalTimeEleTree.muonCollection = cms.InputTag("muons")
process.ecalTimeEleTree.runNum = 999999
#process.ecalTimeTree.endcapSuperClusterCollection = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower","")



process.dumpEvContent = cms.EDAnalyzer("EventContentAnalyzer")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.p = cms.Path(
    process.patMyDefaultSequence *
    # process.dumpEvContent  *
    process.ecalTimeEleTree
    )

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 250

# dbs search --query "find file where dataset=/ExpressPhysics/BeamCommissioning09-Express-v2/FEVT and run=124020" | grep store | awk '{printf "\"%s\",\n", $1}'
process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = filelist,
    #fileNames = cms.untracked.vstring('file:input.root')
    #'/store/data/Commissioning10/MinimumBias/RAW-RECO/v9/000/135/494/A4C5C9FA-C462-DF11-BC35-003048D45F7A.root',
    #'/store/relval/CMSSW_4_2_0_pre8/EG/RECO/GR_R_42_V7_RelVal_wzEG2010A-v1/0043/069662C9-9A56-E011-9741-0018F3D096D2.root'
    #'/store/data/Run2010A/EG/RECO/v4/000/144/114/EEC21BFA-25B4-DF11-840A-001617DBD5AC.root'

   # 'file:/data/franzoni/data/Run2011A_DoubleElectron_AOD_PromptReco-v4_000_166_946_CE9FBCFF-4B98-E011-A6C3-003048F11C58.root'
 #       'file:/hdfs/cms/phedex/store/data/Run2012C/SinglePhoton/RECO/EXODisplacedPhoton-PromptSkim-v3/000/198/941/00000/0EA7C91A-B8CF-E111-9766-002481E150EA.root'

 #   )
    
 )

