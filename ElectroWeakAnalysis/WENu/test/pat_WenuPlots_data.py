import FWCore.ParameterSet.Config as cms

process = cms.Process("PAT")


process.MessageLogger = cms.Service(
        "MessageLogger",
            categories = cms.untracked.vstring('info', 'debug','cout')
            )

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)


# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    #'file:rfio:/castor/cern.ch/user/r/rompotis/DATA_STUDIES/Spring10/sample_WenuSpring10START3X_V26_S09-v1_AODSIM.root',
    "dcap://gfe02:22128/pnfs/hep.ph.ic.ac.uk/data/cms/store/data/Commissioning10/MinimumBias/RECO/May6thPDSkim2_SD_EG-v1/0135/FCC2FA5A-BB5D-DF11-8246-002618943978.root"
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

## Load additional processes
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
## global tags:
process.GlobalTag.globaltag = cms.string('GR_R_35X_V8B::All')
process.load("Configuration.StandardSequences.MagneticField_cff")


################################################################################################
###    P r e p a r a t i o n      o f    t h e    P A T    O b j e c t s   f r o m    A O D  ###
################################################################################################

## pat sequences to be loaded:
process.load("CommonTools.ParticleFlow.PF2PAT_cff")
process.load("PhysicsTools.PatAlgos.patSequences_cff")
#process.load("PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cff")
##
#
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## MET creation     <=== WARNING: YOU MAY WANT TO MODIFY THIS PART OF THE CODE       %%%%%%%%%%%%%
##                                specify the names of the MET collections that you need here %%%%
##                                                                                             #%%
## if you don't specify anything the default MET is the raw Calo MET                           #%%
process.layer1METs = process.patMETs.clone(                                                    #%%
    metSource = cms.InputTag("tcMet","","RECO"),
    addTrigMatch = cms.bool(False),
    addMuonCorrections = cms.bool(False),
    addGenMET = cms.bool(False),
    )
## specify here what you want to have on the plots! <===== MET THAT YOU WANT ON THE PLOTS  %%%%%%%
myDesiredMetCollection = 'layer1METs'
## modify the sequence of the MET creation:                                                    #%%
process.makePatMETs = cms.Sequence(process.layer1METs)
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## modify the final pat sequence: keep only electrons + METS (muons are needed for met corrections)
process.load("RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff")
#process.patElectronIsolation = cms.Sequence(process.egammaIsolationSequence)

process.patElectrons.isoDeposits = cms.PSet()
process.patElectrons.userIsolation = cms.PSet(
#       tracker = cms.PSet(
#            src = cms.InputTag("electronTrackIsolationScone"),
#        ),
#        ecal = cms.PSet(
#            src = cms.InputTag("electronEcalRecHitIsolationScone"),
#        ),
#       hcal = cms.PSet(
#            src = cms.InputTag("electronHcalTowerIsolationScone"),
#        ),
#        user = cms.VPSet(),
    )
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
##
process.patElectrons.addGenMatch = cms.bool(False)
process.patElectrons.embedGenMatch = cms.bool(False)
##
process.load("ElectroWeakAnalysis.WENu.simpleEleIdSequence_cff")
process.patElectronIDs = cms.Sequence(process.simpleEleIdSequence)
process.makePatElectrons = cms.Sequence(process.patElectronIDs*process.patElectrons)
# process.makePatMuons may be needed depending on how you calculate the MET
process.makePatCandidates = cms.Sequence(process.makePatElectrons+process.makePatMETs)
process.patDefaultSequence = cms.Sequence(process.makePatCandidates)
##
##  ################################################################################
##
##  the filter to select the candidates from the data samples
##
##
## WARNING: you may want to modify this item:
HLT_process_name = "HLT"   # 
# trigger path selection
HLT_path_name    = "HLT_Photon10_L1R" # "HLT_Ele15_LW_L1R"
# trigger filter name
HLT_filter_name  =  "hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter" #"hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter"
#
process.wenuFilter = cms.EDFilter('WenuCandidateFilter',
                                  # cfg for data
                                  dataMagneticFieldSetUp = cms.untracked.bool(True),
                                  dcsTag = cms.untracked.InputTag("scalersRawToDigi"),
                                  # cuts
                                  ETCut = cms.untracked.double(25.),
                                  METCut = cms.untracked.double(0.),
                                  # 2nd electron in W events
                                  vetoSecondElectronEvents = cms.untracked.bool(True),
                                  ETCut2ndEle = cms.untracked.double(20.),
                                  vetoSecondElectronIDType = cms.untracked.string("simpleEleId95cIso"),
                                  vetoSecondElectronIDSign = cms.untracked.string("="),
                                  vetoSecondElectronIDValue = cms.untracked.double(7.),
                                  # trigger 
                                  useTriggerInfo = cms.untracked.bool(True),
                                  triggerCollectionTag = cms.untracked.InputTag("TriggerResults","",HLT_process_name),
                                  triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","",HLT_process_name),
                                  hltpath = cms.untracked.string(HLT_path_name), 
                                  hltpathFilter = cms.untracked.InputTag(HLT_filter_name,"",HLT_process_name),
                                  electronMatched2HLT = cms.untracked.bool(True),
                                  electronMatched2HLT_DR = cms.untracked.double(0.2),
                                  # additional preselection cuts
                                  useValidFirstPXBHit = cms.untracked.bool(False),
                                  useConversionRejection = cms.untracked.bool(False),
                                  useExpectedMissingHits = cms.untracked.bool(False),
                                  maxNumberOfExpectedMissingHits = cms.untracked.int32(1),
                                  # calculate some new cuts
                                  calculateValidFirstPXBHit = cms.untracked.bool(True),
                                  calculateConversionRejection = cms.untracked.bool(True),
                                  calculateExpectedMissingHits = cms.untracked.bool(True),
                                  # electrons and MET
                                  electronCollectionTag = cms.untracked.InputTag("patElectrons","","PAT"),
                                  metCollectionTag = cms.untracked.InputTag(myDesiredMetCollection,"","PAT"),

                                  )
####################################################################################
##
## the W selection that you prefer included in another cfg
from ElectroWeakAnalysis.WENu.simpleCutBasedSpring10SelectionBlocks_cfi import *

selection_inverse = cms.PSet (
    deta_EB_inv = cms.untracked.bool(True),
    deta_EE_inv = cms.untracked.bool(True)
    )
# dummy selection for debugging
selection_dummy = cms.PSet (
    trackIso_EB = cms.untracked.double(100.),
    ecalIso_EB =  cms.untracked.double(100.),
    hcalIso_EB =  cms.untracked.double(100.),
    sihih_EB =    cms.untracked.double(0.1 ),
    dphi_EB =     cms.untracked.double(0.1 ),
    deta_EB =     cms.untracked.double(0.1 ),
    hoe_EB =      cms.untracked.double(0.1 ),
    cIso_EB =     cms.untracked.double(100.),
    
    trackIso_EE = cms.untracked.double(100.),
    ecalIso_EE =  cms.untracked.double(100.),
    hcalIso_EE =  cms.untracked.double(100.),
    sihih_EE =    cms.untracked.double(0.1 ),
    dphi_EE =     cms.untracked.double(0.1 ),
    deta_EE =     cms.untracked.double(0.1 ),
    hoe_EE =      cms.untracked.double(0.1 ),
    cIso_EE =     cms.untracked.double(100.),
    useConversionRejection = cms.untracked.bool(False),
    useExpectedMissingHits = cms.untracked.bool(False),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(99),
    )



####################################################################################
##
## and the plot creator
process.plotter = cms.EDAnalyzer('WenuPlots',
                                 # selection in use
                                 selection_80relIso,
                                 selection_inverse,
                                 # if usePrecalcID the precalculated ID will be used only
                                 usePrecalcID = cms.untracked.bool(False),
                                 usePrecalcIDType = cms.untracked.string('simpleEleId95cIso'),
                                 usePrecalcIDSign = cms.untracked.string('='),
                                 usePrecalcIDValue = cms.untracked.double(7),
                                 #
                                 wenuCollectionTag = cms.untracked.InputTag(
                                                   "wenuFilter","selectedWenuCandidates","PAT")
                                 )




process.p = cms.Path(process.patDefaultSequence +process.wenuFilter + process.plotter)



