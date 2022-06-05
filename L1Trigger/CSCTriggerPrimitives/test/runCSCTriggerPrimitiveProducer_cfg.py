import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

options = VarParsing('analysis')
options.register("unpack", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when you want to unpack the CSC DAQ data.")
options.register("selectCSCs", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when you want to (un)select certain CSCs.")
options.register("maskedChambers", "", VarParsing.multiplicity.list, VarParsing.varType.string,
                 "Chambers you want to explicitly mask.")
options.register("selectedChambers", "", VarParsing.multiplicity.list, VarParsing.varType.string,
                 "Chambers you want to explicitly mask.")
options.register("unpackGEM", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when you want to unpack the GEM DAQ data.")
options.register("l1", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when you want to re-emulate the CSC trigger primitives.")
options.register("l1GEM", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when you want to re-emulate the GEM trigger primitives.")
options.register("mc", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when running on MC.")
options.register("dqm", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when you want to run the CSC DQM")
options.register("dqmGEM", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when you want to run the GEM DQM")
options.register("useEmtfGEM", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when you want to use GEM clusters from the EMTF in the DQM")
options.register("useB904ME11", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when using B904 ME1/1 data.")
options.register("useB904ME21", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when using B904 ME2/1 data (also works for ME3/1 and ME4/1).")
options.register("useB904ME234s2", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when using B904 ME1/1 data (also works for MEX/2 and ME1/3).")
options.register("run3", True, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when using Run-3 data.")
options.register("runCCLUTOTMB", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when using the CCLUT OTMB algorithm.")
options.register("runCCLUTTMB", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when using the CCLUT TMB algorithm.")
options.register("runME11ILT", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when running the GEM-CSC integrated local trigger algorithm in ME1/1.")
options.register("runME21ILT", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when running the GEM-CSC integrated local trigger algorithm in ME2/1.")
options.register("saveEdmOutput", True, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True if you want to keep the EDM ROOT after unpacking and re-emulating.")
options.register("preTriggerAnalysis", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True if you want to print out more details about CLCTs and LCTs in the offline CSC DQM module.")
options.register("dropNonMuonCollections", True, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Option to drop most non-muon collections generally considered unnecessary for GEM/CSC analysis")
options.register("dqmOutputFile", "step_DQM.root", VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Name of the DQM output file. Default: step_DQM.root")
options.parseArguments()

process_era = Run3
if not options.run3:
      process_era = Run2_2018

process = cms.Process("L1CSCTPG", process_era)
process.load("Configuration/StandardSequences/GeometryRecoDB_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
process.load('EventFilter.GEMRawToDigi.muonGEMDigis_cfi')
process.load('EventFilter.L1TRawToDigi.emtfStage2Digis_cfi')
process.load("L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi")
process.load("CalibMuon.CSCCalibration.CSCL1TPLookupTableEP_cff")
process.load('L1Trigger.L1TGEM.simGEMDigis_cff')
process.load("DQM.L1TMonitor.L1TdeCSCTPG_cfi")
process.load("DQM.L1TMonitor.L1TdeGEMTPG_cfi")

process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(options.maxEvents)
)

process.options = cms.untracked.PSet(
      SkipEvent = cms.untracked.vstring('ProductNotFound')
)

process.source = cms.Source(
      "PoolSource",
      fileNames = cms.untracked.vstring(options.inputFiles),
      inputCommands = cms.untracked.vstring(
            'keep *',
            'drop CSCDetIdCSCShowerDigiMuonDigiCollection_simCscTriggerPrimitiveDigis_*_*'
      )
)

## this line is needed to run the GEM unpacker on output from AMC13SpyReadout.py or readFile_b904_Run3.py
if options.unpackGEM:
      process.source.labelRawDataLikeMC = cms.untracked.bool(False)

## global tag (data or MC, Run-2 or Run-3)
from Configuration.AlCa.GlobalTag import GlobalTag
if options.mc:
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
      if options.run3:
            process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')
else:
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
      if options.run3:
            process.GlobalTag = GlobalTag(process.GlobalTag, '112X_dataRun3_Prompt_v5', '')

## running on unpacked data, or after running the unpacker
if not options.mc or options.unpack:
      process.cscTriggerPrimitiveDigis.CSCComparatorDigiProducer = "muonCSCDigis:MuonCSCComparatorDigi"
      process.cscTriggerPrimitiveDigis.CSCWireDigiProducer = "muonCSCDigis:MuonCSCWireDigi"

## unpacker
useB904Data = options.useB904ME11 or options.useB904ME21 or options.useB904ME234s2
if useB904Data:
      ## CSC
      process.muonCSCDigis.DisableMappingCheck = True
      process.muonCSCDigis.B904Setup = True
      process.muonCSCDigis.InputObjects = "rawDataCollectorCSC"
      if options.unpackGEM:
            process.muonCSCDigis.useGEMs = True
      ## GEM
      process.muonGEMDigis.InputLabel = "rawDataCollectorGEM"

## l1 emulator
l1csc = process.cscTriggerPrimitiveDigis
if options.l1:
      l1csc.commonParam.runCCLUT_OTMB = cms.bool(options.runCCLUTOTMB)
      l1csc.commonParam.runCCLUT_TMB = cms.bool(options.runCCLUTTMB)
      l1csc.commonParam.runME11ILT = options.runME11ILT
      l1csc.commonParam.runME21ILT = options.runME21ILT
      ## running on unpacked data, or after running the unpacker
      if (not options.mc or options.unpack):
            l1csc.CSCComparatorDigiProducer = "muonCSCDigis:MuonCSCComparatorDigi"
            l1csc.CSCWireDigiProducer = "muonCSCDigis:MuonCSCWireDigi"
            ## GEM-CSC trigger enabled
            if options.runME11ILT or options.runME21ILT:
                  l1csc.GEMPadDigiClusterProducer = "muonCSCDigis:MuonGEMPadDigiCluster"

if options.l1GEM:
      process.simMuonGEMPadDigis.InputCollection = 'muonGEMDigis'

## DQM monitor
if options.dqm:
      process.l1tdeCSCTPG.useB904ME11 = options.useB904ME11
      process.l1tdeCSCTPG.useB904ME21 = options.useB904ME21
      process.l1tdeCSCTPG.useB904ME234s2 = options.useB904ME234s2
      process.l1tdeCSCTPG.emulALCT = "cscTriggerPrimitiveDigis"
      process.l1tdeCSCTPG.emulCLCT = "cscTriggerPrimitiveDigis"
      process.l1tdeCSCTPG.emulLCT = "cscTriggerPrimitiveDigis:MPCSORTED"
      process.l1tdeCSCTPG.preTriggerAnalysis = options.preTriggerAnalysis

if options.dqmGEM:
      ## GEM pad clusters from the EMTF
      if options.useEmtfGEM:
            process.l1tdeGEMTPG.data = "emtfStage2Digis"
      ## GEM pad clusters from the CSC TPG
      else:
            process.l1tdeGEMTPG.data = "muonCSCDigis"
      ## GEM pad clusters from the GEM TPG
      process.l1tdeGEMTPG.emul = "simMuonGEMPadDigiClusters"

# Output
process.output = cms.OutputModule(
    "PoolOutputModule",
      outputCommands = cms.untracked.vstring(
            ['keep *',
             'drop *_rawDataCollector_*_*',
      ]),
      fileName = cms.untracked.string("lcts2.root"),
)

## for most studies, you don't need these collections.
## adjust as necessary
if options.dropNonMuonCollections:
      outputCom = process.output.outputCommands
      outputCom.append('drop *_rawDataCollector_*_*')
      outputCom.append('drop *_sim*al*_*_*')
      outputCom.append('drop *_hlt*al*_*_*')
      outputCom.append('drop *_g4SimHits_*al*_*')
      outputCom.append('drop *_simSi*_*_*')
      outputCom.append('drop *_hltSi*_*_*')
      outputCom.append('drop *_simBmtfDigis_*_*')
      outputCom.append('drop *_*_*BMTF*_*')
      outputCom.append('drop *_hltGtStage2ObjectMap_*_*')
      outputCom.append('drop *_simGtStage2Digis_*_*')
      outputCom.append('drop *_hltTriggerSummary*_*_*')

## DQM output
process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:{}'.format(options.dqmOutputFile)),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

## schedule and path definition
process.unpacksequence = cms.Sequence(process.muonCSCDigis)

## when unpacking data only from select chambers...
if options.selectCSCs:

      from EventFilter.CSCRawToDigi.cscDigiFilterDef_cfi import cscDigiFilterDef

      # clone the original producer
      process.preCSCDigis = process.muonCSCDigis.clone()

      # now apply the filter
      process.muonCSCDigis = cscDigiFilterDef.clone(
            stripDigiTag = "preCSCDigis:MuonCSCStripDigi",
            wireDigiTag = "preCSCDigis:MuonCSCWireDigi",
            compDigiTag = "preCSCDigis:MuonCSCComparatorDigi",
            alctDigiTag = "preCSCDigis:MuonCSCALCTDigi",
            clctDigiTag = "preCSCDigis:MuonCSCCLCTDigi",
            lctDigiTag = "preCSCDigis:MuonCSCCorrelatedLCTDigi",
            showerDigiTag = "preCSCDigis:MuonCSCShowerDigi",
            gemPadClusterDigiTag = "preCSCDigis:MuonGEMPadDigiCluster",
            maskedChambers = options.maskedChambers,
            selectedChambers = options.selectedChambers
      )

      # these 3 chambers had Phase-2 firmware loaded partially during Run-2
      # https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCOTMB2018
      process.muonCSCDigis.maskedChambers = [
            "ME+1/1/9", "ME+1/1/10", "ME+1/1/11"]

      process.unpacksequence = cms.Sequence(process.preCSCDigis * process.muonCSCDigis)

if options.unpackGEM:
      ## unpack GEM strip digis
      process.unpacksequence += process.muonGEMDigis
      ## unpack GEM pad clusters from the EMTF
      if options.useEmtfGEM:
            process.unpacksequence += process.emtfStage2Digis
process.p1 = cms.Path(process.unpacksequence)

process.l1sequence = cms.Sequence(l1csc)
if options.l1GEM:
      ## not sure if append would work for the GEM-CSC trigger
      ## maybe the modules need to come first
      process.l1sequence += process.simMuonGEMPadDigis
      process.l1sequence += process.simMuonGEMPadDigiClusters
process.p2 = cms.Path(process.l1sequence)

process.dqmsequence = cms.Sequence(process.l1tdeCSCTPG)
if options.dqmGEM:
      process.dqmsequence += process.l1tdeGEMTPG
process.p3 = cms.Path(process.dqmsequence)

process.p4 = cms.EndPath(process.DQMoutput)
process.p5 = cms.EndPath(process.output)
process.p6 = cms.EndPath(process.endOfProcess)

process.schedule = cms.Schedule()
## add the unpacker
if options.unpack:
      process.schedule.extend([process.p1])

## add the emulator
if options.l1:
      process.schedule.extend([process.p2])

## add DQM step 1
if options.dqm:
      process.schedule.extend([process.p3, process.p4])

if options.saveEdmOutput:
      process.schedule.extend([process.p5])

process.schedule.extend([process.p6])
