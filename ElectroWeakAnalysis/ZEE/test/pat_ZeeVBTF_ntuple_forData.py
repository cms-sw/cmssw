##   Configuration for the production of the ICHEP VBTF ntuple
##   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
##   MC, spring10
##
##   Stilianos Kesisoglou - Institute of Nuclear Physics
##                                NCSR Demokritos
##   25 June 2010
import FWCore.ParameterSet.Config as cms

process = cms.Process("PAT")



process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)

#process.MessageLogger = cms.Service(
#        "MessageLogger",
#            categories = cms.untracked.vstring('info', 'debug','cout')
#            )


process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cerr.threshold = cms.untracked.string("INFO")

process.MessageLogger.cerr.FwkSummary = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1000000),
    limit = cms.untracked.int32(10000000)
       )
       
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
      reportEvery = cms.untracked.int32(100000),
         limit = cms.untracked.int32(10000000)
      )
      
process.options = cms.untracked.PSet(
       wantSummary = cms.untracked.bool(True)
       )




# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
# SOME DATA FILE TO BE PUT HERE
#
#   MC test (local) running on the OneElecPlusSC skim made from the Zee MC samples
#   ------------------------------------------------------------------------------
#    'rfio:/tmp/ikesisog/TestFiles/skimMC/OneElecPlusSC_1Files.root',
#    'rfio:/tmp/ikesisog/TestFiles/skimMC/OneElecPlusSC_2Files.root',
#    'rfio:/tmp/ikesisog/TestFiles/skimMC/OneElecPlusSC_3Files.root',
#
#   MC test (local) running directly on the Zee MC samples
#   ------------------------------------------------------
#    'rfio:/tmp/ikesisog/TestFiles/dSetMC/Zee/8EA9211C-7476-DF11-BC31-00304867C136.root',
#    'rfio:/tmp/ikesisog/TestFiles/dSetMC/Zee/B0737652-7076-DF11-83F8-002618943831.root',
#    'rfio:/tmp/ikesisog/TestFiles/dSetMC/Zee/14B213B3-7576-DF11-8763-003048678FA6.root',
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

## Load additional processes
# 
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

## global tags:
#
process.GlobalTag.globaltag = cms.string('GR_R_36X_V12A::All') # GLOBAL TAG FOR DATA

process.load("Configuration.StandardSequences.MagneticField_cff")


################################################################################################
###    P r e p a r a t i o n      o f    t h e    P A T    O b j e c t s   f r o m    A O D  ###
################################################################################################

## pat sequences to be loaded:
#process.load("PhysicsTools.PFCandProducer.PF2PAT_cff")

process.load("PhysicsTools.PatAlgos.patSequences_cff")

#process.load("PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cff")

##
#

## MET creation     <=== WARNING: YOU MAY WANT TO MODIFY THIS PART OF THE CODE       %%%%%%%%%%%%%
##                                specify the names of the MET collections that you need here %%%%
##                                                                                             #%%
## if you don't specify anything the default MET is the raw Calo MET                           #%%
process.caloMET = process.patMETs.clone(                                                       #%%
    metSource = cms.InputTag("met","","RECO"),
    addTrigMatch = cms.bool(False),
    addMuonCorrections = cms.bool(False),
    addGenMET = cms.bool(False),
)
process.tcMET = process.patMETs.clone(                                                         #%%
    metSource = cms.InputTag("tcMet","","RECO"),
    addTrigMatch = cms.bool(False),
    addMuonCorrections = cms.bool(False),
    addGenMET = cms.bool(False),
)
process.pfMET = process.patMETs.clone(                                                         #%%
    metSource = cms.InputTag("pfMet","","RECO"),
    addTrigMatch = cms.bool(False),
    addMuonCorrections = cms.bool(False),
    addGenMET = cms.bool(False),
)

## specify here what you want to have on the plots! <===== MET THAT YOU WANT ON THE PLOTS  %%%%%%%
myMetCollection   = 'caloMET'
myPfMetCollection =   'pfMET'
myTcMetCollection =   'tcMET'

## modify the sequence of the MET creation:                                                    #%%
process.makePatMETs = cms.Sequence(process.caloMET*process.tcMET*process.pfMET)


## modify the final pat sequence: keep only electrons + METS (muons are needed for met corrections)

process.load("RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff")
#process.patElectronIsolation = cms.Sequence(process.egammaIsolationSequence)

process.patElectrons.isoDeposits = cms.PSet()
process.patElectrons.userIsolation = cms.PSet()
process.patElectrons.addElectronID = cms.bool(True)

process.patElectrons.electronIDSources = cms.PSet(
    simpleEleId95relIso = cms.InputTag("simpleEleId95relIso"),
    simpleEleId90relIso = cms.InputTag("simpleEleId90relIso"),
    simpleEleId85relIso = cms.InputTag("simpleEleId85relIso"),
    simpleEleId80relIso = cms.InputTag("simpleEleId80relIso"),
    simpleEleId70relIso = cms.InputTag("simpleEleId70relIso"),
    simpleEleId60relIso = cms.InputTag("simpleEleId60relIso"),
    simpleEleId95cIso   = cms.InputTag("simpleEleId95cIso"),
    simpleEleId90cIso   = cms.InputTag("simpleEleId90cIso"),
    simpleEleId85cIso   = cms.InputTag("simpleEleId85cIso"),
    simpleEleId80cIso   = cms.InputTag("simpleEleId80cIso"),
    simpleEleId70cIso   = cms.InputTag("simpleEleId70cIso"),
    simpleEleId60cIso   = cms.InputTag("simpleEleId60cIso")    
)

##

process.patElectrons.addGenMatch = cms.bool(False)
process.patElectrons.embedGenMatch = cms.bool(False)
process.patElectrons.usePV = cms.bool(False)

##

process.load("ElectroWeakAnalysis.ZEE.simpleEleIdSequence_cff")

# you have to tell the ID that it is data. These are set to False for MC
process.simpleEleId95relIso.dataMagneticFieldSetUp = cms.bool(True)
process.simpleEleId90relIso.dataMagneticFieldSetUp = cms.bool(True)
process.simpleEleId85relIso.dataMagneticFieldSetUp = cms.bool(True)
process.simpleEleId80relIso.dataMagneticFieldSetUp = cms.bool(True)
process.simpleEleId70relIso.dataMagneticFieldSetUp = cms.bool(True)
process.simpleEleId60relIso.dataMagneticFieldSetUp = cms.bool(True)
process.simpleEleId95cIso.dataMagneticFieldSetUp = cms.bool(True)
process.simpleEleId90cIso.dataMagneticFieldSetUp = cms.bool(True)
process.simpleEleId85cIso.dataMagneticFieldSetUp = cms.bool(True)
process.simpleEleId80cIso.dataMagneticFieldSetUp = cms.bool(True)
process.simpleEleId70cIso.dataMagneticFieldSetUp = cms.bool(True)
process.simpleEleId60cIso.dataMagneticFieldSetUp = cms.bool(True)
#
process.patElectronIDs = cms.Sequence(process.simpleEleIdSequence)

process.makePatElectrons = cms.Sequence(process.patElectronIDs*process.patElectrons)

# process.makePatMuons may be needed depending on how you calculate the MET

process.makePatCandidates = cms.Sequence(process.makePatElectrons+process.makePatMETs)

process.patDefaultSequence = cms.Sequence(process.makePatCandidates)



##  the filter to select the candidates from the data samples


## WARNING: you may want to modify this item:
HLT_process_name = "HLT"   # REDIGI for the Spring10 production traditional MC / HLT for the powheg samples or data

# trigger path selection
HLT_path_name     = "HLT_Photon10_L1R" #= "HLT_Ele15_LW_L1R" #

# trigger filter name
HLT_filter_name  =  "hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter"

HLT_path_name_extra0   = "HLT_Photon15_L1R"
HLT_filter_name_extra0 = cms.untracked.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter","",HLT_process_name)

HLT_path_name_extra1   = "HLT_Photon15_Cleaned_L1R"
HLT_filter_name_extra1 = cms.untracked.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15CleanedHcalIsolFilter","",HLT_process_name)

HLT_path_name_extra2   = "HLT_Ele15_SW_L1R"
HLT_filter_name_extra2 = cms.untracked.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter","",HLT_process_name)


rules_Filter = cms.PSet (
    ### the input collections needed:
    electronCollectionTag = cms.untracked.InputTag("patElectrons","","PAT"),
    metCollectionTag = cms.untracked.InputTag(myMetCollection,"","PAT"),
    pfMetCollectionTag = cms.untracked.InputTag(myPfMetCollection,"","PAT"),
    tcMetCollectionTag = cms.untracked.InputTag(myTcMetCollection,"","PAT"),
    triggerCollectionTag = cms.untracked.InputTag("TriggerResults","",HLT_process_name),
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","",HLT_process_name),
    hltpath = cms.untracked.string(HLT_path_name), 
    hltpathFilter = cms.untracked.InputTag(HLT_filter_name,"",HLT_process_name),
    ebRecHits = cms.untracked.InputTag("reducedEcalRecHitsEB"),
    eeRecHits = cms.untracked.InputTag("reducedEcalRecHitsEE"),
    PrimaryVerticesCollection = cms.untracked.InputTag("offlinePrimaryVertices"),
    ### here the preselection is applied
    # fiducial cuts:
    BarrelMaxEta = cms.untracked.double(1.4442),
    EndCapMinEta = cms.untracked.double(1.560),
    EndCapMaxEta = cms.untracked.double(2.5),
    # demand ecal driven electron:
    useEcalDrivenElectrons = cms.untracked.bool(True),
    # demand offline spike cleaning with the Swiss Cross criterion:
    useSpikeRejection = cms.untracked.bool(True),
    spikeCleaningSwissCrossCut = cms.untracked.double(0.95),
    # demand geometrically matched to an HLT object with ET>15GeV
    useTriggerInfo = cms.untracked.bool(True),
    electronMatched2HLT = cms.untracked.bool(True),
    electronMatched2HLT_DR = cms.untracked.double(0.1),
    useHLTObjectETCut = cms.untracked.bool(True),
    hltObjectETCut = cms.untracked.double(15.0),
    useExtraTrigger = cms.untracked.bool(True),
    vHltpathExtra = cms.untracked.vstring(HLT_path_name_extra0,HLT_path_name_extra1,HLT_path_name_extra2),
    vHltpathFilterExtra = cms.untracked.VInputTag(HLT_filter_name_extra0, HLT_filter_name_extra1, HLT_filter_name_extra2),
    # ET Cut in the SC
    ETCut = cms.untracked.double(20.0),                                  
    METCut = cms.untracked.double(0.0),
    # For DATA set it to True, for MC set it to False
    dataMagneticFieldSetUp = cms.untracked.bool(True),
    dcsTag = cms.untracked.InputTag("scalersRawToDigi")
)


rules_Filter_Elec0 = cms.PSet (
    # Other parameters of the code - leave them as they are
    useValidFirstPXBHit0            = cms.untracked.bool(False),
    useConversionRejection0         = cms.untracked.bool(False),
    useExpectedMissingHits0         = cms.untracked.bool(False),
    maxNumberOfExpectedMissingHits0 = cms.untracked.int32(1),
    # calculate some new cuts
    calculateValidFirstPXBHit0      = cms.untracked.bool(True),
    calculateConversionRejection0   = cms.untracked.bool(True),
    calculateExpectedMissingHits0   = cms.untracked.bool(True)
)

rules_Filter_Elec1 = cms.PSet (
    # Other parameters of the code - leave them as they are
    useValidFirstPXBHit1            = cms.untracked.bool(False),
    useConversionRejection1         = cms.untracked.bool(False),
    useExpectedMissingHits1         = cms.untracked.bool(False),
    maxNumberOfExpectedMissingHits1 = cms.untracked.int32(1),
    # calculate some new cuts
    calculateValidFirstPXBHit1      = cms.untracked.bool(True),
    calculateConversionRejection1   = cms.untracked.bool(True),
    calculateExpectedMissingHits1   = cms.untracked.bool(True)
)

rules_Filter_Elec2 = cms.PSet (
    # Other parameters of the code - leave them as they are
    useValidFirstPXBHit2            = cms.untracked.bool(False),
    useConversionRejection2         = cms.untracked.bool(False),
    useExpectedMissingHits2         = cms.untracked.bool(False),
    maxNumberOfExpectedMissingHits2 = cms.untracked.int32(1),
    # calculate some new cuts
    calculateValidFirstPXBHit2      = cms.untracked.bool(True),
    calculateConversionRejection2   = cms.untracked.bool(True),
    calculateExpectedMissingHits2   = cms.untracked.bool(True)
)


process.zeeFilter = cms.EDFilter('ZeeCandidateFilter',
    rules_Filter,
    rules_Filter_Elec0,    
    rules_Filter_Elec1,    
    rules_Filter_Elec2   
)



## the Z selection that you prefer
from ElectroWeakAnalysis.ZEE.simpleCutBasedSpring10SelectionBlocks_cfi import *

selection_inverse = cms.PSet (
    deta_EB_inv = cms.untracked.bool(True),
    deta_EE_inv = cms.untracked.bool(True)
    )


rules_Plotter = cms.PSet (
    #   treat or not the elecrons with same criteria. if yes then rules_Plotter_Elec1/2 are neglected
    useSameSelectionOnBothElectrons = cms.untracked.bool(True),    
    # some extra information on the ntuple production:
    sihih0_EB = cms.untracked.double(1000.0),
    dphi0_EB  = cms.untracked.double(1000.0),
    deta0_EB  = cms.untracked.double(1000.0),
    hoe0_EB   = cms.untracked.double(1000.0),
    sihih0_EE = cms.untracked.double(1000.0),
    dphi0_EE  = cms.untracked.double(1000.0),
    deta0_EE  = cms.untracked.double(1000.0),
    hoe0_EE   = cms.untracked.double(1000.0),
    includeJetInformationInNtuples = cms.untracked.bool(True),
    caloJetCollectionTag = cms.untracked.InputTag('ak4CaloJetsL2L3'),
    pfJetCollectionTag = cms.untracked.InputTag('ak4PFJetsL2L3'),
    DRJetFromElectron = cms.untracked.double(0.3),
    #
    zeeCollectionTag = cms.untracked.InputTag("zeeFilter","selectedZeeCandidates","PAT"),
    ZEE_VBTFselectionFileName = cms.untracked.string("ZEE_VBTFselection.root"),
    ZEE_VBTFpreseleFileName = cms.untracked.string("ZEE_VBTFpreselection.root"),
    DatasetTag =  cms.untracked.int32(100),
    storeExtraInformation = cms.untracked.bool(True)
)

rules_Plotter_Elec0 = cms.PSet (
    # The selection to be used here:
    usePrecalcID0                   = cms.untracked.bool(True),
    usePrecalcIDType0               = cms.untracked.string('simpleEleId95relIso'),
    usePrecalcIDSign0               = cms.untracked.string('='),
    usePrecalcIDValue0              = cms.untracked.double(7),    
    ## preselection criteria are independent of useSameSelectionOnBothElectrons
    #  set them to False if you don't want them
    useValidFirstPXBHit0            = cms.untracked.bool(False),
    useConversionRejection0         = cms.untracked.bool(False),
    useExpectedMissingHits0         = cms.untracked.bool(False),
    maxNumberOfExpectedMissingHits0 = cms.untracked.int32(1),    
    ##
    calculateValidFirstPXBHit0      = cms.untracked.bool(False),
    calculateConversionRejection0   = cms.untracked.bool(False),
    calculateExpectedMissingHits0   = cms.untracked.bool(False)
)

rules_Plotter_Elec1 = cms.PSet (
    # The selection to be used here:
    usePrecalcID1                   = cms.untracked.bool(True),
    usePrecalcIDType1               = cms.untracked.string('simpleEleId95relIso'),
    usePrecalcIDSign1               = cms.untracked.string('='),
    usePrecalcIDValue1              = cms.untracked.double(7),    
    ## preselection criteria are independent of useSameSelectionOnBothElectrons
    #  set them to False if you don't want them
    useValidFirstPXBHit1            = cms.untracked.bool(False),
    useConversionRejection1         = cms.untracked.bool(False),
    useExpectedMissingHits1         = cms.untracked.bool(False),
    maxNumberOfExpectedMissingHits1 = cms.untracked.int32(1),    
    ##
    calculateValidFirstPXBHit1      = cms.untracked.bool(False),
    calculateConversionRejection1   = cms.untracked.bool(False),
    calculateExpectedMissingHits1   = cms.untracked.bool(False)
)

rules_Plotter_Elec2 = cms.PSet (
    # The selection to be used here:
    usePrecalcID2                   = cms.untracked.bool(True),
    usePrecalcIDType2               = cms.untracked.string('simpleEleId95relIso'),
    usePrecalcIDSign2               = cms.untracked.string('='),
    usePrecalcIDValue2              = cms.untracked.double(7),    
    ## preselection criteria are independent of useSameSelectionOnBothElectrons
    #  set them to False if you don't want them
    useValidFirstPXBHit2            = cms.untracked.bool(False),
    useConversionRejection2         = cms.untracked.bool(False),
    useExpectedMissingHits2         = cms.untracked.bool(False),
    maxNumberOfExpectedMissingHits2 = cms.untracked.int32(1),    
    ##
    calculateValidFirstPXBHit2      = cms.untracked.bool(False),
    calculateConversionRejection2   = cms.untracked.bool(False),
    calculateExpectedMissingHits2   = cms.untracked.bool(False)
)


# we need to store jet information, hence we have to produce the jets:
process.load("JetMETCorrections.Configuration.DefaultJEC_cff")

process.jetSequence = cms.Sequence( process.ak4CaloJetsL2L3  )

process.pfjetAK5Sequence = cms.Sequence( process.ak4PFJetsL2L3 )

process.ourJetSequence = cms.Sequence( process.jetSequence * process.pfjetAK5Sequence )


process.plotter = cms.EDAnalyzer('ZeePlots',
    rules_Plotter,
    rules_Plotter_Elec0,
    rules_Plotter_Elec1,
    rules_Plotter_Elec2   
)


# if you run on data then you have to do misalignment  corrections first!!!
# not to be used with MC!!

process.load("RecoEgamma.EgammaTools.correctedElectronsProducer_cfi")

process.p = cms.Path( process.gsfElectrons*process.ourJetSequence*process.patDefaultSequence*process.zeeFilter*process.plotter)
#process.p = cms.Path( process.ourJetSequence * process.patDefaultSequence +process.zeeFilter + process.plotter)


