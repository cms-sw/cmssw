import FWCore.ParameterSet.Config as cms
process = cms.Process("L1TMuonEmulation")
import os
import sys
import commands

process.load("FWCore.MessageLogger.MessageLogger_cfi")


VERBOSE = False
SAMPLE = "zmumu"  # "relval"##"minbias"
EDM_OUT = True
# min bias: 23635 => 3477 passed L1TMuonFilter (~6.7%), zmumu ~84%
NEVENTS = 50
if VERBOSE:
    process.MessageLogger = cms.Service("MessageLogger",
                                        suppressInfo=cms.untracked.vstring('AfterSource', 'PostModule'),
                                        destinations=cms.untracked.vstring('detailedInfo', 'critical', 'cout'),
                                        categories=cms.untracked.vstring(
                                            'CondDBESSource', 'EventSetupDependency', 'Geometry', 'MuonGeom', 'GetManyWithoutRegistration', 'GetByLabelWithoutRegistration', 'Alignment', 'SiStripBackPlaneCorrectionDepESProducer', 'SiStripLorentzAngleDepESProducer', 'SiStripQualityESProducer', 'TRACKER', 'HCAL'
                                        ),
                                        critical=cms.untracked.PSet(
                                            threshold=cms.untracked.string('ERROR')
                                        ),
                                        detailedInfo=cms.untracked.PSet(
                                            threshold=cms.untracked.string('INFO'),
                                            CondDBESSource=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                            EventSetupDependency=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                            Geometry=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                            MuonGeom=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                            Alignment=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                            GetManyWithoutRegistration=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                            GetByLabelWithoutRegistration=cms.untracked.PSet(limit=cms.untracked.int32(0))

                                        ),
                                        cout=cms.untracked.PSet(
                                            threshold=cms.untracked.string('WARNING'),
                                            CondDBESSource=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                            EventSetupDependency=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                            Geometry=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                            MuonGeom=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                            Alignment=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                            GetManyWithoutRegistration=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                            GetByLabelWithoutRegistration=cms.untracked.PSet(limit=cms.untracked.int32(0))
                                        ),
                                        )

if not VERBOSE:
    process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)

process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(True))

fnames = ['/store/relval/CMSSW_7_5_0_pre1/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_74_V7-v1/00000/16BF2D14-83E3-E411-B212-003048FFD756.root',
          '/store/relval/CMSSW_7_5_0_pre1/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_74_V7-v1/00000/26833213-83E3-E411-9238-0025905B8590.root',
          '/store/relval/CMSSW_7_5_0_pre1/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_74_V7-v1/00000/5E967412-83E3-E411-9DA0-003048FFD756.root',
          '/store/relval/CMSSW_7_5_0_pre1/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_74_V7-v1/00000/686FB705-83E3-E411-A8FC-003048FF9AC6.root',
          '/store/relval/CMSSW_7_5_0_pre1/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_74_V7-v1/00000/8E6F7913-83E3-E411-B72F-0025905A48BB.root']

if SAMPLE == "zmumu":
    fnames = ['root://xrootd.unl.edu//store/mc/Fall13dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/GEN-SIM-RAW/tsg_PU20bx25_POSTLS162_V2-v1/20000/B61E1FCD-A077-E311-8B65-001E673974EA.root',
              'root://xrootd.unl.edu//store/mc/Fall13dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/GEN-SIM-RAW/tsg_PU20bx25_POSTLS162_V2-v1/10000/0023D81B-2980-E311-85A1-001E67398C0F.root',
              'root://xrootd.unl.edu//store/mc/Fall13dr/DYToMuMu_M-50_Tune4C_13TeV-pythia8/GEN-SIM-RAW/tsg_PU20bx25_POSTLS162_V2-v1/10000/248FB042-3080-E311-A346-001E67397D00.root']
elif SAMPLE == "minbias":
    fnames = ['root://xrootd.unl.edu//store/mc/Fall13dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/tsg_PU20bx25_POSTLS162_V2-v1/00000/00276D94-AA88-E311-9C90-0025905A6060.root',
              'root://xrootd.unl.edu//store/mc/Fall13dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/tsg_PU20bx25_POSTLS162_V2-v1/00000/004F8058-6F88-E311-B971-0025905A6094.root',
              'root://xrootd.unl.edu//store/mc/Fall13dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/tsg_PU20bx25_POSTLS162_V2-v1/00000/005C8F98-C288-E311-ADF1-0026189438BD.root',
              'root://xrootd.unl.edu//store/mc/Fall13dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/tsg_PU20bx25_POSTLS162_V2-v1/00000/006A1FB8-7D88-E311-B61B-0025905A60A0.root']

process.source = cms.Source(
    'PoolSource',
    fileNames=cms.untracked.vstring(
        fnames
    )
)

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(NEVENTS))

# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')
############################

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
process.load('L1Trigger.L1TMuonTrackFinderEndCap.L1TMuonTriggerPrimitiveProducer_cfi')

path = "L1Trigger/L1TMuonTrackFinderOverlap/data/"
# OMTF emulator configuration
# OMTF emulator configuration
process.load('L1Trigger.L1TMuonTrackFinderOverlap.OMTFProducer_cfi')

process.L1TMuonEndcapTrackFinder = cms.EDProducer(
    'L1TMuonUpgradedTrackFinder',
    primitiveSrcs = cms.VInputTag(
    cms.InputTag('L1TMuonTriggerPrimitives', 'CSC'),
    cms.InputTag('L1TMuonTriggerPrimitives', 'DT'),
    cms.InputTag('L1TMuonTriggerPrimitives', 'RPC')
    ),
)

# BMTF Emulator
process.bmtfEmulator = cms.EDProducer("BMTrackFinder",
                                      CSCStub_Source=cms.InputTag("simCsctfTrackDigis"),
                                      DTDigi_Source=cms.InputTag("simDtTriggerPrimitiveDigis"),
                                      Debug=cms.untracked.int32(0)

                                      )

process.MicroGMTCaloInputProducer = cms.EDProducer("l1t::MicroGMTCaloInputProducer",
                                               caloStage2Layer2Label=cms.InputTag("caloStage2Layer1Digis"),
)
# WORKAROUNDS FOR WRONG SCALES / MISSING COLLECTIONS:
process.bmtfConverter = cms.EDProducer("l1t::BMTFConverter",)

# Adjust input tags if running on GEN-SIM-RAW (have to re-digi)
if SAMPLE == "zmumu" or SAMPLE == "minbias":
    process.L1TMuonTriggerPrimitives.CSC.src = cms.InputTag('simCscTriggerPrimitiveDigis')

process.L1MuonFilter = cms.EDFilter("SelectL1Muons",)
process.GenMuonFilter = cms.EDFilter("SelectGenMuons",)

process.load("L1TriggerDPG.L1Ntuples.l1NtupleProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1RecoTreeProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1ExtraTreeProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1MuonRecoTreeProducer_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1MuonUpgradeTreeProducer_cfi")

process.load("L1Trigger.L1TMuon.microgmtemulator_cfi")

process.microGMTEmulator.overlapTFInput = cms.InputTag("omtfEmulator", "OMTF")
process.l1MuonUpgradeTreeProducer.omtfTag = cms.InputTag("omtfEmulator", "OMTF")
process.microGMTEmulator.forwardTFInput = cms.InputTag("L1TMuonEndcapTrackFinder", "EMUTF")
process.l1MuonUpgradeTreeProducer.emtfTag = cms.InputTag("L1TMuonEndcapTrackFinder", "EMUTF")
process.microGMTEmulator.barrelTFInput = cms.InputTag("bmtfConverter", "ConvBMTFMuons")
process.l1MuonUpgradeTreeProducer.bmtfTag = cms.InputTag("bmtfConverter", "ConvBMTFMuons")
process.microGMTEmulator.triggerTowerInput = cms.InputTag("MicroGMTCaloInputProducer", "TriggerTowerSums")
process.l1MuonUpgradeTreeProducer.calo2x2Tag = cms.InputTag("MicroGMTCaloInputProducer", "TriggerTower2x2s")
process.l1MuonUpgradeTreeProducer.caloTag = cms.InputTag("caloStage2Layer1Digis")

# disable pre-loaded cancel-out lookup tables (they currently contain only 0)
process.microGMTEmulator.OvlNegSingleMatchQualLUTSettings.filename = cms.string("")
process.microGMTEmulator.OvlPosSingleMatchQualLUTSettings.filename = cms.string("")
process.microGMTEmulator.FOPosMatchQualLUTSettings.filename = cms.string("")
process.microGMTEmulator.FONegMatchQualLUTSettings.filename = cms.string("")
process.microGMTEmulator.BrlSingleMatchQualLUTSettings.filename = cms.string("")
process.microGMTEmulator.BOPosMatchQualLUTSettings.filename = cms.string("")
process.microGMTEmulator.BONegMatchQualLUTSettings.filename = cms.string("")

# output file
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string(
                                       '/afs/cern.ch/work/j/jlingema/private/l1ntuples_upgrade/l1ntuple_{sample}_n.root'.format(sample=SAMPLE))
                                   )

process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
from SLHCUpgradeSimulations.Configuration.muonCustoms import customise_csc_PostLS1
process = customise_csc_PostLS1(process)

# upgrade calo stage 2
process.load('L1Trigger.L1TCalorimeter.L1TCaloStage2_PPFromRaw_cff')

# analysis
process.l1NtupleProducer.hltSource = cms.InputTag("none")
process.l1NtupleProducer.gtSource = cms.InputTag("none")
process.l1NtupleProducer.gctCentralJetsSource = cms.InputTag("none")
process.l1NtupleProducer.gctNonIsoEmSource = cms.InputTag("none")
process.l1NtupleProducer.gctForwardJetsSource = cms.InputTag("none")
process.l1NtupleProducer.gctIsoEmSource = cms.InputTag("none")
process.l1NtupleProducer.gctEnergySumsSource = cms.InputTag("none")
process.l1NtupleProducer.gctTauJetsSource = cms.InputTag("none")
process.l1NtupleProducer.gctIsoTauJetsSource = cms.InputTag("none")
process.l1NtupleProducer.rctSource = cms.InputTag("none")
process.l1NtupleProducer.dttfSource = cms.InputTag("none")
process.l1NtupleProducer.ecalSource = cms.InputTag("none")
process.l1NtupleProducer.hcalSource = cms.InputTag("none")
process.l1NtupleProducer.csctfTrkSource = cms.InputTag("none")
process.l1NtupleProducer.csctfLCTSource = cms.InputTag("none")
process.l1NtupleProducer.csctfLCTSource = cms.InputTag("none")
process.l1NtupleProducer.generatorSource = cms.InputTag("genParticles")
process.l1NtupleProducer.csctfDTStubsSource = cms.InputTag("none")


process.L1ReEmulSeq = cms.Sequence(process.SimL1Emulator
                                   + process.ecalDigis
                                   + process.hcalDigis
                                   + process.gtDigis
                                   + process.gtEvmDigis
                                   + process.csctfDigis
                                   + process.dttfDigis
                                   )

process.L1NtupleSeq = cms.Sequence(process.l1NtupleProducer + process.l1MuonUpgradeTreeProducer)
    # +process.l1extraParticles
    # +process.l1ExtraTreeProducer
    # +process.l1GtTriggerMenuLite
    # +process.l1MenuTreeProducer
    # +process.l1RecoTreeProducer
    # +process.l1MuonRecoTreeProducer

process.L1TMuonSeq = cms.Sequence(
    process.L1TMuonTriggerPrimitives
    + process.bmtfEmulator
    + process.bmtfConverter
    + process.omtfEmulator
    + process.L1TMuonEndcapTrackFinder
    + process.L1TCaloStage2_PPFromRaw
    + process.MicroGMTCaloInputProducer
    + process.microGMTEmulator
)

  # type: L1MuDTChambPhContainer
  # module label: simDtTriggerPrimitiveDigis
  # product instance name: ''
  # process name: ''
process.MuonFilter = cms.Sequence()


if SAMPLE == "minbias":
    process.MuonFilter = cms.Sequence(process.L1MuonFilter)
else:
    process.MuonFilter = cms.Sequence(process.GenMuonFilter)

process.L1TMuonPath = cms.Path(process.L1ReEmulSeq + process.L1TMuonSeq + process.MuonFilter * process.L1NtupleSeq)

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands=cms.untracked.vstring(
                                   'drop *',
                                   'keep *_*_*_L1TMuonEmulation'),
                               fileName=cms.untracked.string("l1tmuon_test.root"),
                               )


process.schedule = cms.Schedule(process.L1TMuonPath)
if EDM_OUT:
    process.output_step = cms.EndPath(process.out)
    process.schedule.extend([process.output_step])
