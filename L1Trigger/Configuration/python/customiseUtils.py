from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os
import sys
import commands


##############################################################################
# customisations for L1T utilities
#
# customisations which add utilities features such as debugging of L1T,
#    summary module, etc.
#
##############################################################################

# Unpack Stage-2 GT and GMT
def L1TTurnOffGtAndGmtEmulation(process):
    cutlist=['simDtTriggerPrimitiveDigis','simCscTriggerPrimitiveDigis','simTwinMuxDigis','simBmtfDigis','simEmtfDigis','simOmtfDigis','simGmtCaloSumDigis','simMuonQualityAdjusterDigis','simGmtStage2Digis','simGtStage2Digis']
    for b in cutlist:
        process.SimL1Emulator.remove(getattr(process,b))
    return process

# Unpack Stage-2 GT and GMT
def L1TTurnOffUnpackStage2GtAndGmt(process):
    cutlist=['gtStage2Digis','gmtStage2Digis']
    for b in cutlist:
        process.L1TRawToDigi.remove(getattr(process,b))
    return process

# Unpack Stage-2 GT and GMT
def L1TTurnOffUnpackStage2GtGmtAndCalo(process):
    cutlist=['gtStage2Digis','gmtStage2Digis','caloStage2Digis']
    for b in cutlist:
        process.L1TRawToDigi.remove(getattr(process,b))
    return process

def L1TStage1DigisSummary(process):
    print("L1T INFO:  will dump a summary of unpacked Stage1 content to screen.")
    process.load('L1Trigger.L1TCommon.l1tSummaryStage1Digis_cfi')
    process.l1tstage1summary = cms.Path(process.l1tSummaryStage1Digis)
    process.schedule.append(process.l1tstage1summary)
    return process

def L1TStage2DigisSummary(process):
    print("L1T INFO:  will dump a summary of unpacked Stage2 content to screen.")
    process.load('L1Trigger.L1TCommon.l1tSummaryStage2Digis_cfi')
    process.l1tstage2summary = cms.Path(process.l1tSummaryStage2Digis)
    process.schedule.append(process.l1tstage2summary)
    return process

def L1TStage1SimDigisSummary(process):
    print("L1T INFO:  will dump a summary of simulated Stage1 content to screen.")
    process.load('L1Trigger.L1TCommon.l1tSummaryStage1SimDigis_cfi')
    process.l1tsimstage1summary = cms.Path(process.l1tSummaryStage1SimDigis)
    process.schedule.append(process.l1tsimstage1summary)
    return process

def L1TStage2SimDigisSummary(process):
    print("L1T INFO:  will dump a summary of simulated Stage2 content to screen.")
    process.load('L1Trigger.L1TCommon.l1tSummaryStage2SimDigis_cfi')
    process.l1tsimstage2summary = cms.Path(process.l1tSummaryStage2SimDigis)
    process.schedule.append(process.l1tsimstage2summary)
    return process

def L1TGlobalDigisSummary(process):
    print("L1T INFO:  will dump a summary of unpacked L1T Global output to screen.")
    process.l1tGlobalSummary = cms.EDAnalyzer(
        'L1TGlobalSummary',
        AlgInputTag = cms.InputTag("gtStage2Digis"),
        ExtInputTag = cms.InputTag("gtStage2Digis"),
        DumpTrigResults = cms.bool(False), # per event dump of trig results
        DumpTrigSummary = cms.bool(True), # pre run dump of trig results
        )
    process.l1tglobalsummary = cms.Path(process.l1tGlobalSummary)
    process.schedule.append(process.l1tglobalsummary)
    return process

def L1TGlobalMenuXML(process):
    process.load('L1Trigger.L1TGlobal.GlobalParameters_cff')
    process.load('L1Trigger.L1TGlobal.TriggerMenu_cff')
    process.TriggerMenu.L1TriggerMenuFile = cms.string('L1Menu_Collisions2016_v2c.xml')
    return process

def L1TGlobalSimDigisSummary(process):
    print("L1T INFO:  will dump a summary of simulated L1T Global output to screen.")
    process.l1tSimGlobalSummary = cms.EDAnalyzer(
        'L1TGlobalSummary',
        AlgInputTag = cms.InputTag("simGtStage2Digis"),
        ExtInputTag = cms.InputTag("simGtStage2Digis"),
        DumpTrigResults = cms.bool(False), # per event dump of trig results
        DumpTrigSummary = cms.bool(True), # pre run dump of trig results
        )
    process.l1tsimglobalsummary = cms.Path(process.l1tSimGlobalSummary)
    process.schedule.append(process.l1tsimglobalsummary)
    return process

def L1TAddInfoOutput(process):
    process.MessageLogger = cms.Service(
        "MessageLogger",
        destinations = cms.untracked.vstring('cout','cerr'),
        cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
        cerr = cms.untracked.PSet(threshold  = cms.untracked.string('WARNING')),
        )
    return process


def L1TAddDebugOutput(process):
    print("L1T INFO:  sending debugging ouput to file l1tdebug.log")
    print("L1T INFO:  add <flags CXXFLAGS=\"-g -D=EDM_ML_DEBUG\"/> in BuildFile.xml of any package you want to debug...")
    process.MessageLogger = cms.Service(
        "MessageLogger",
        destinations = cms.untracked.vstring('l1tdebug','cerr'),
        l1tdebug = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG')),
        #debugModules = cms.untracked.vstring('caloStage1Digis'))
        cerr = cms.untracked.PSet(threshold  = cms.untracked.string('WARNING')),
        debugModules = cms.untracked.vstring('*'))
    return process

def L1TDumpEventData(process):
    print("L1T INFO:  adding EventContentAnalyzer to process schedule")
    process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
    process.l1tdumpevent = cms.Path(process.dumpED)
    process.schedule.append(process.l1tdumpevent)
    return process

def L1TDumpEventSummary(process):
    process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")
    process.l1tdumpeventsetup = cms.Path(process.dumpES)
    process.schedule.append(process.l1tdumpeventsetup)
    return process

def L1TStage2ComparisonRAWvsEMU(process):
    print("L1T INFO:  will dump a comparison of unpacked vs emulated Stage2 content to screen.")
    process.load('L1Trigger.L1TCommon.l1tComparisonStage2RAWvsEMU_cfi')
    process.l1tstage2comparison = cms.Path(process.l1tComparisonStage2RAWvsEMU)
    process.schedule.append(process.l1tstage2comparison)
    return process


def L1TGtStage2ComparisonRAWvsEMU(process):
    print("L1T INFO:  will dump a comparison of unpacked vs emulated GT Stage2 content to screen.")
    process.load('L1Trigger.L1TCommon.l1tComparisonGtStage2RAWvsEMU_cfi')
    process.l1tgtstage2comparison = cms.Path(process.l1tComparisonGtStage2RAWvsEMU)
    process.schedule.append(process.l1tgtstage2comparison)
    return process

def DropDepricatedProducts(process):
    print ("INPUT SOURCE: dropping products depricated from CMSSW_9_4_X on.")
    print ("drop l1tHGCalTowerMapBXVector_hgcalTriggerPrimitiveDigiProducer_towerMap_HLT")
    print ("drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT")
    print ("drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT")
    print ("drop l1tEMTFHit2016s_simEmtfDigis__HLT")
    print ("drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT")
    print ("drop l1tEMTFTrack2016s_simEmtfDigis__HLT")
    process.source.inputCommands = cms.untracked.vstring("keep *" 
        ,"drop l1tHGCalTowerMapBXVector_hgcalTriggerPrimitiveDigiProducer_towerMap_HLT"
        ,"drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT"
        ,"drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT"
        ,"drop l1tEMTFHit2016s_simEmtfDigis__HLT"
        ,"drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT"
        ,"drop l1tEMTFTrack2016s_simEmtfDigis__HLT"
    )
    return process

def DropOutputProducts(process):
    print ("OutputModule: dropping products.")
    print ("drop TrackingParticles_mix_MergedTrackTruth_*")
    print ("drop PixelDigiSimLinkedmDetSetVector_simSiPixelDigis_Tracker_*")
    print ("drop TrackingVertexs_mix_MergedTrackTruth_*")
    print ("drop HGCalDetIdHGCSampleHGCDataFramesSorted_mix_HGCDigisEE_HLT")
    print ("drop l1tHGCalTriggerCellBXVector_hgcalTriggerPrimitiveDigiProducer_calibratedTriggerCellsTower_L1")
    print ("drop PixelDigiSimLinkedmDetSetVector_simSiPixelDigis_Pixel_HLT")
    print ("drop l1tHGCalTowerMapBXVector_hgcalTriggerPrimitiveDigiProducer_towerMap_L1")
    print ("drop SimClusters_mix_MergedCaloTruth_HLT")
    print ("drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusteredmNewDetSetVector_TTClustersFromPhase2TrackerDigis_ClusterInclusive_HLT")
    print ("drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusterAssociationMap_TTClusterAssociatorFromPixelDigis_ClusterInclusive_HLT")
    print ("drop PixelDigiedmDetSetVector_simSiPixelDigis_Pixel_HLT")
    print ("drop Phase2TrackerDigiedmDetSetVector_mix_Tracker_HLT")
    process.FEVTDEBUGHLToutput.outputCommands = cms.untracked.vstring('keep *' 
              ,'drop TrackingParticles_mix_MergedTrackTruth_*'
              ,'drop PixelDigiSimLinkedmDetSetVector_simSiPixelDigis_Tracker_*'
              ,'drop TrackingVertexs_mix_MergedTrackTruth_*'
              ,'drop HGCalDetIdHGCSampleHGCDataFramesSorted_mix_HGCDigisEE_HLT'
              ,'drop l1tHGCalTriggerCellBXVector_hgcalTriggerPrimitiveDigiProducer_calibratedTriggerCellsTower_L1'
              ,'drop PixelDigiSimLinkedmDetSetVector_simSiPixelDigis_Pixel_HLT'
              ,'drop l1tHGCalTowerMapBXVector_hgcalTriggerPrimitiveDigiProducer_towerMap_L1'
              ,'drop SimClusters_mix_MergedCaloTruth_HLT'
              ,'drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusteredmNewDetSetVector_TTClustersFromPhase2TrackerDigis_ClusterInclusive_HLT'
              ,'drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusterAssociationMap_TTClusterAssociatorFromPixelDigis_ClusterInclusive_HLT'
              ,'drop PixelDigiedmDetSetVector_simSiPixelDigis_Pixel_HLT'
              ,'drop Phase2TrackerDigiedmDetSetVector_mix_Tracker_HLT'
              ,"drop l1tL1TkGlbMuonParticles_L1TkGlbMuons__REPR"
    )
    return process

def L1TrackTriggerTracklet(process):
    #print "L1T INFO:  run the L1TrackStep with Tracklet."
    process.load('L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff')
    process.L1TrackTriggerTracklet_step = cms.Path(process.L1TrackletTracksWithAssociators)
    process.schedule.insert(2,process.L1TrackTriggerTracklet_step)
    return process

def L1TrackTriggerTMTT(process):
    #print "L1T INFO:  run the L1TrackStep with TMTT."
    process.load('L1Trigger.TrackFindingTMTT.TMTrackProducer_Ultimate_cff')
    process.TMTrackProducer.EnableMCtruth = cms.bool(False)
    process.TMTrackProducer.EnableHistos    = cms.bool(False)
    process.L1TrackTriggerTMTT_step = cms.Path(process.TMTrackProducer)
    process.schedule.insert(2,process.L1TrackTriggerTMTT_step)
    return process

def L1TTurnOffHGCalTPs(process):
    cutlist=['hgcalTriggerPrimitiveDigiProducer']
    for b in cutlist:
        process.SimL1Emulator.remove(getattr(process,b))
    return process

def L1TTurnOffHGCalTPs_v9(process):
    cutlist=['hgcalVFE','hgcalConcentrator','hgcalBackEndLayer1','hgcalBackEndLayer2','hgcalTowerMap','hgcalTower']
    for b in cutlist:
        process.SimL1Emulator.remove(getattr(process,b))
    return process

def appendDTChamberAgingAtL1Trigger(process):
# #############################################################################
# This function adds aging producers for DT TPs 
# by appending DTChamberMasker and the corresponding dtTriggerPrimitiveDigies 
# #############################################################################

    from SimMuon.DTDigitizer.dtChamberMasker_cfi import dtChamberMasker as _dtChamberMasker
    from L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi import dtTriggerPrimitiveDigis as _dtTriggerPrimitiveDigis

    if hasattr(process,'simDtTriggerPrimitiveDigis') and hasattr(process,'SimL1TMuonCommon') :

        sys.stderr.write("[appendDTChamberMasker] : Found simDtTriggerPrimitivesDigis, appending producer for aged DTs and corresponding TriggerPrimitives producer\n")

        process.simAgedDtTriggerDigis = _dtChamberMasker.clone()

        process.simDtTriggerPrimitiveDigis = _dtTriggerPrimitiveDigis.clone()
        process.simDtTriggerPrimitiveDigis.digiTag = "simAgedDtTriggerDigis"

        process.withAgedDtTriggerSequence = cms.Sequence(process.simAgedDtTriggerDigis + process.simDtTriggerPrimitiveDigis)
        process.SimL1TMuonCommon.replace(process.simDtTriggerPrimitiveDigis, process.withAgedDtTriggerSequence)

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.simAgedDtTriggerDigis = cms.PSet(
                 initialSeed = cms.untracked.uint32(789342),
                 engineName = cms.untracked.string('TRandom3')
                 )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                simAgedDtTriggerDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )
            
    return process

def appendCSCChamberAgingAtL1Trigger(process):
# #############################################################################
# This function adds aging producers for CSC TPs 
# by appending CSCChamberMasker and the corresponding cscTriggerPrimitiveDigies 
# #############################################################################

    from SimMuon.CSCDigitizer.cscChamberMasker_cfi import cscChamberMasker as _cscChamberMasker
    from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi import cscTriggerPrimitiveDigis as _cscTriggerPrimitiveDigis

    if hasattr(process,'simCscTriggerPrimitiveDigis') and hasattr(process,'SimL1TMuonCommon') :

        sys.stderr.write("[appendCSCChamberMasker] : Found simCscTriggerPrimitivesDigis, appending producer for aged CSCs and corresponding TriggerPrimitives producer\n")

        process.simAgedMuonCSCDigis = _cscChamberMasker.clone()
        process.simAgedMuonCSCDigis.stripDigiTag = cms.InputTag("simMuonCSCDigis", "MuonCSCStripDigi")
        process.simAgedMuonCSCDigis.wireDigiTag = cms.InputTag("simMuonCSCDigis", "MuonCSCWireDigi") 
        process.simAgedMuonCSCDigis.comparatorDigiTag = cms.InputTag("simMuonCSCDigis", "MuonCSCComparatorDigi")
        process.simAgedMuonCSCDigis.rpcDigiTag = cms.InputTag("simMuonCSCDigis", "MuonCSCRPCDigi") 
        process.simAgedMuonCSCDigis.alctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis", "", \
                                                        processName = cms.InputTag.skipCurrentProcess())
        process.simAgedMuonCSCDigis.clctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis", "", \
                                                        processName = cms.InputTag.skipCurrentProcess())

        process.simCscTriggerPrimitiveDigis = _cscTriggerPrimitiveDigis.clone()
        process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'simMuonCSCDigis', 'MuonCSCComparatorDigi' )
        process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'simAgedMuonCSCDigis', 'MuonCSCWireDigi' )

        process.withAgedCscTriggerSequence = cms.Sequence(process.simAgedMuonCSCDigis + process.simCscTriggerPrimitiveDigis)
        process.SimL1TMuonCommon.replace(process.simCscTriggerPrimitiveDigis, process.withAgedCscTriggerSequence)

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.simAgedMuonCSCDigis = cms.PSet(
                 initialSeed = cms.untracked.uint32(789342),
                 engineName = cms.untracked.string('TRandom3')
                 )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                simAgedMuonCSCDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )
            
    return process

def appendRPCChamberAgingAtL1Trigger(process):
# #############################################################################
# This function adds aging producers for RPC Digis 
# #############################################################################

    from SimMuon.RPCDigitizer.rpcChamberMasker_cfi import rpcChamberMasker as _rpcChamberMasker

    if hasattr(process,'simTwinMuxDigis') and hasattr(process,'SimL1TMuon') :

        sys.stderr.write("[appendRPCChamberMasker] : Found simTwinMuxDigis, prepending producer for aged RPC\n")

        process.simMuonRPCDigis = _rpcChamberMasker.clone()
        process.simMuonRPCDigis.digiTag = cms.InputTag('simMuonRPCDigis', \
                                                        processName = cms.InputTag.skipCurrentProcess())


        process.withAgedRpcTriggerSequence = cms.Sequence(process.simMuonRPCDigis + process.simTwinMuxDigis )
        process.SimL1TMuon.replace(process.simTwinMuxDigis, process.withAgedRpcTriggerSequence)

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.simMuonRPCDigis = cms.PSet(
                 initialSeed = cms.untracked.uint32(789342),
                 engineName = cms.untracked.string('TRandom3')
                 )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                simMuonRPCDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )
            
    return process

def appendGEMChamberAgingAtL1Trigger(process):
# #############################################################################
# This function adds aging producers for GEM Digis 
# #############################################################################
    from SimMuon.GEMDigitizer.gemChamberMasker_cfi import gemChamberMasker as _gemChamberMasker
    from SimMuon.GEMDigitizer.muonGEMPadDigis_cfi import simMuonGEMPadDigis
    from SimMuon.GEMDigitizer.muonGEMPadDigiClusters_cfi import simMuonGEMPadDigiClusters


    if hasattr(process,'simCscTriggerPrimitiveDigis') and hasattr(process,'SimL1TMuon') :

        sys.stderr.write("[appendGEMChamberMasker] : Found simCscTriggerPrimitiveDigis, prepending producer for aged GEM\n")

        process.simMuonGEMPadDigis = simMuonGEMPadDigis.clone()
        process.simMuonGEMPadDigiClusters = simMuonGEMPadDigiClusters.clone()
        process.simMuonGEMDigis = _gemChamberMasker.clone()
        process.simMuonGEMDigis.digiTag =  cms.InputTag("simMuonGEMDigis", \
                                                        processName = cms.InputTag.skipCurrentProcess())

        process.withAgedGEMDigiSequence = cms.Sequence( process.simMuonGEMDigis \
                                                        + process.simMuonGEMPadDigis \
                                                        + process.simMuonGEMPadDigiClusters \
                                                        + process.simCscTriggerPrimitiveDigis)

        process.SimL1TMuon.replace(process.simCscTriggerPrimitiveDigis, process.withAgedGEMDigiSequence)

    return process
