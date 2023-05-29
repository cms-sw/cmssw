# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step1 --processName=L1REPR --conditions auto:phase2_realistic_T15 -n 10 --era Phase2C9 --eventcontent FEVTDEBUGHLT -s RAW2DIGI,L1 --datatier FEVTDEBUGHLT --geometry Extended2026D49 --fileout file:/tmp/step1_Reprocess_L1.root --no_exec --nThreads 8 --python step1_L1_Reprocess.py --filein das:/MinBias_TuneCP5_14TeV-pythia8/Phase2HLTTDRSummer20L1T-PU200_111X_mcRun4_realistic_T15_v1-v2/FEVT --customise L1Trigger/Configuration/customisePhase2.addHcalTriggerPrimitives --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9

process = cms.Process('L1REPR',Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff') # Needed for MuonTPS
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.load('L1Trigger.L1TMuonTPS.L1TTrackerPlusStubs_cfi') # Adding MuonTPS

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/00006128-B6E9-164F-86F0-A650029BF556.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/008F2D30-9CB9-3149-BACC-44E377BD4339.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/01DCAE84-9A84-BA46-A3C9-911FE21B772B.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/03354C18-02BB-BC4E-AC41-929415ACBA10.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/04B5B609-AE07-2845-A953-C7381AFA4BA3.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/053B903A-B414-B34C-8A0C-576E52ED40BB.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/0741BA70-FB5D-5F45-83BA-F21E36F4A6A0.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/07D0DC19-75A6-EF43-86AC-500087D5F044.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/0A246F64-6D90-FB4A-A363-024F172686F1.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/0A597E05-0A6F-CC4B-B091-6B48314BE330.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/0AB846BB-0972-A046-BC9A-C571E6BAC3C5.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/0BB165E7-1A72-BF45-A0BC-DFC90CA0087F.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/0CFC45A2-D468-B846-A203-777B97A43120.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/0D0AA151-EA54-0C42-9E2B-0D9A6430596F.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/0D2CB354-3047-8C41-89ED-25662E1AC832.root',
'/store/mc/Phase2HLTTDRWinter20DIGI/DYToLL_M-50_TuneCP5_14TeV-pythia8/GEN-SIM-DIGI-RAW/NoPU_pilot_110X_mcRun4_realistic_v3-v2/20000/0E673C7A-1C1F-4948-9C05-5DD69FBB4098.root'





#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/0398918A-BD7A-6C4D-8696-6F061FF08845.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/045A4B0F-D7AA-1C44-B097-F700F9C11881.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/09E56A06-CFFA-2A46-90BA-2CF4F6C7BDC8.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/0ACCD724-9CA3-FA4D-B85B-70C5A999E089.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/0E408E15-DCEB-5546-BD16-0B45D2F5D590.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/12A09A17-6734-1244-B789-A6A79829E12B.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/12E755CC-059B-D94D-B2C0-57C2C189F4DE.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/1946CACB-2903-2649-B33C-66FB7D7F3ACC.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/19667D5B-49B9-EF45-A911-EB17EAA1E67D.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/2012FA01-916D-6C4B-8CD1-2D86920B06ED.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/234CE519-87C5-DB44-9457-E679AD595E75.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/23B4B535-51AB-9945-8BD9-9CB011277E28.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/2873F537-D329-8840-8A51-A452BACB8F66.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/2CA0BA6A-B735-C54C-8D5F-BA6A0AFE3D37.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/2D5E99B7-5876-3B44-9253-4E1B22DB02B0.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/2EFCE00C-524C-F642-B951-2825189FE8A7.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/2FFD6E4F-C824-B140-B285-A65983FF54FF.root',
#'/store/mc/Phase2HLTTDRWinter20DIGI/DoubleMuon_gun_FlatPt-1To100/GEN-SIM-DIGI-RAW/NoPU_110X_mcRun4_realistic_v3-v2/30000/330FA1A0-652B-AA49-B0C9-729965D8A0E4.root'


    ),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(1)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step1 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(
        "drop *_*_*_*",
        "keep *_simCscTriggerPrimitiveDigis_*_*",
        "keep *_dtTriggerPhase2PrimitiveDigis_*_*",
        "keep *_simDtTriggerPrimitiveDigis_*_*",
        "keep *_simMuonRPCDigis_*_*",
        "keep *_simMuonME0*_*_*",
        "keep *_simMuonGEMDigis*_*_*",
        "keep *_simBmtfDigis_*_*",
        "keep *_simEmtfDigis_*_*",
        "keep *_simOmtfDigis_*_*",
        "keep *_genParticles_*_*",
        "keep *_l1tTTTracksFromTrackletEmulation_Level1TTTracks_*"
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')

# Path and EndPath definitions
process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger) # Needed for MuonTPS
#process.pL1TMuonTPS = cms.Path(process.l1TrackerPlusStubsSequence) # Adding MuonTPS
process.endjob_step = cms.EndPath(process.endOfProcess)
process.e = cms.EndPath(process.out)


#Calibrate Digis
process.load("L1Trigger.DTTriggerPhase2.CalibratedDigis_cfi")
process.CalibratedDigis.dtDigiTag = "simMuonDTDigis" 
process.CalibratedDigis.scenario = 0

#DTTriggerPhase2
process.load("L1Trigger.DTTriggerPhase2.dtTriggerPhase2PrimitiveDigis_cfi")
process.dtTriggerPhase2PrimitiveDigis.debug = False
process.dtTriggerPhase2PrimitiveDigis.dump = False
process.dtTriggerPhase2PrimitiveDigis.scenario = 0




#process.schedule = cms.Schedule(process.L1TrackTrigger_step,process.pL1TMuonTPS,process.endjob_step,process.e) # Adding MuonTPS


process.stubs = cms.EDProducer("Phase2L1TGMTStubProducer",
    verbose = cms.int32(0),
    srcCSC = cms.InputTag("simCscTriggerPrimitiveDigis"),
    srcDT = cms.InputTag("dtTriggerPhase2PrimitiveDigis"),
    srcDTTheta = cms.InputTag("simDtTriggerPrimitiveDigis"),
    srcRPC = cms.InputTag("simMuonRPCDigis"),
    Endcap =cms.PSet(                            
        verbose              = cms.uint32(0),
        minBX                = cms.int32(0),                           
        maxBX                = cms.int32(0),         
        coord1LSB            = cms.double(0.00076660156*32), 
        eta1LSB              = cms.double(7.68334e-04*32), 
        coord2LSB            = cms.double(0.00076660156*32), 
        eta2LSB              = cms.double(7.68334e-04*32),
        phiMatch             = cms.double(0.05),
        etaMatch             = cms.double(0.1)
    ),
    Barrel = cms.PSet(                         
        verbose            = cms.int32(0),
        minPhiQuality      = cms.int32(0),
        minThetaQuality    = cms.int32(0),
        minBX              = cms.int32(-100),                           
        maxBX              = cms.int32(100),                           
        phiLSB             = cms.double(0.00076660156*32),
        phiBDivider        = cms.int32(1),
        etaLSB             = cms.double(7.68334e-04*32), 
        eta_1              = cms.vint32(-1503/32,-1446/32,-1387/32,-1327/32,-1266/32,-1194/32,-1125/32,-985/32,-916/32,-839/32,-752/32,-670/32,-582/32,-489/32,-315/32,-213/32,-115/32,-49/32,49/32, 115/32, 213/32, 315/32, 489/32, 582/32, 670/32, 752/32, 839/32, 916/32, 985/32, 1125/32, 1194/32, 1266/32, 1327/32, 1387/32, 1446/32, 1503),
        eta_2              = cms.vint32(-1334/32,-1279/32,-1227/32,-1168/32,-1109/32,-1044/32,-982/32,-861/32,-793/32,-720/32,-648/32,-577/32,-496/32,-425/32,-268/32,-185/32,-97/32,-51/32,51/32, 97/32, 185/32, 268/32, 425/32, 496/32, 577/32, 648/32, 720/32, 793/32, 861/32, 982/32, 1044/32, 1109/32, 1168/32, 1227/32, 1279/32, 1334),
        eta_3              = cms.vint32(-1148/32,-1110/32,-1051/32,-1004/32,-947/32,-895/32,-839/32,-728/32,-668/32,-608/32,-546/32,-485/32,-425/32,-366/32,-222/32,-155/32,-87/32,-40/32,40/32, 87/32, 155/32, 222/32, 366/32, 425/32, 485/32, 546/32, 608/32, 668/32, 728/32, 839/32, 895/32, 947/32, 1004/32, 1051/32, 1110/32, 1148),
        coarseEta_1        = cms.vint32(0/32,758/32,1336/32),
        coarseEta_2        = cms.vint32(0/32,653/32,1168/32),
        coarseEta_3        = cms.vint32(0/32,552/32,1001/32),
        coarseEta_4        = cms.vint32(0/32,478/32,878/32),
        phiOffset          = cms.vint32(75/32,-30/32,+26/32,0)    
   )

)





process.prod = cms.EDProducer('Phase2L1TGMTProducer',
                              srcTracks = cms.InputTag("l1tTTTracksFromTrackletEmulation:Level1TTTracks"),
                              srcStubs  = cms.InputTag('stubs'),
                              srcBMTF   = cms.InputTag('simBmtfDigis','BMTF'),
                              srcEMTF   = cms.InputTag('simEmtfDigis','EMTF'),
                              srcOMTF   = cms.InputTag('simOmtfDigis','OMTF'),
                              muonBXMin = cms.int32(0),
                              muonBXMax = cms.int32(0),
                              IsoThreshold1 = cms.int32(0),
                              IsoThreshold2 = cms.int32(0),
                              IsoThreshold3 = cms.int32(0),
                              IsoThreshold4 = cms.int32(0),
                              verbose       = cms.int32(0),
                              IsodumpForHLS = cms.int32(0)
                             )
process.testpath=cms.Path(process.CalibratedDigis*process.dtTriggerPhase2PrimitiveDigis*process.stubs*process.prod)
#process.testpath=cms.Path(process.CalibratedDigis*process.dtTriggerPhase2PrimitiveDigis)
process.schedule = cms.Schedule(process.L1TrackTrigger_step,process.testpath,process.endjob_step,process.e)


from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(1)
process.options.numberOfStreams=cms.untracked.uint32(0)
process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1)

# customisation of the process.

# Automatic addition of the customisation function from L1Trigger.Configuration.customisePhase2
from L1Trigger.Configuration.customisePhase2 import addHcalTriggerPrimitives 

#call to customisation function addHcalTriggerPrimitives imported from L1Trigger.Configuration.customisePhase2
process = addHcalTriggerPrimitives(process)

# End of customisation functions

# Customisation from command line

# Automatic addition of the customisation function from L1Trigger.Configuration.customisePhase2TTNoMC # To make the cfg work
from L1Trigger.Configuration.customisePhase2TTNoMC import customisePhase2TTNoMC  # To make the cfg work

#call to customisation function customisePhase2TTNoMC imported from L1Trigger.Configuration.customisePhase2TTNoMC # To make the cfg work
process = customisePhase2TTNoMC(process) # To make the cfg work

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
