#!/usr/bin/env python
import sys

"""
The parameters can be changed by adding commandline arguments of the form
::

    create_plots.py n=-1 pu=low

or ::

    create_plots.py n=-1:wfile=foo.root

The latter can be used to change parameters in crab.
"""

n = 1
legacy = False
data = False
debug = False

do_digi = False
do_reco = False

aod = False
raw = True
reco = True
reemul = True


# Argument parsing
# vvv

if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
    sys.argv.pop(0)
if len(sys.argv) == 2 and ':' in sys.argv[1]:
    argv = sys.argv[1].split(':')
else:
    argv = sys.argv[1:]

for arg in argv:
    (k, v) = map(str.strip, arg.split('='))
    if k not in globals():
        raise "Unknown argument '%s'!" % (k,)
    if isinstance(globals()[k], bool):
        globals()[k] = v.lower() in ('y', 'yes', 'true', 't', '1')
    elif isinstance(globals()[k], int):
        globals()[k] = int(v)
    else:
        globals()[k] = v


mc = not data
reemul = reemul or (mc and raw)

import FWCore.ParameterSet.Config as cms

process = cms.Process('L1UPGRADE')
process.load('FWCore.MessageLogger.MessageLogger_cfi')
if do_reco:
    process.MessageLogger.cerr.FwkReport.reportEvery = 1
else:
    process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.MessageLogger.L1GtTrigReport=dict()

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(n))

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')
if data:
    process.GlobalTag.globaltag = cms.string('GR_P_V40::All')
else:
    process.GlobalTag.globaltag = cms.string('POSTLS161_V12::All')

process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
if mc:
    process.load('Configuration.StandardSequences.Reconstruction_cff')
else:
    process.load('Configuration.StandardSequences.Reconstruction_Data_cff')

if do_digi:
    process.load('Configuration.StandardSequences.Digi_cff')
    process.load('Configuration.StandardSequences.SimL1Emulator_cff')
    process.load('HLTrigger.Configuration.HLT_GRun_cff')
    process.load('Configuration.StandardSequences.DigiToRaw_cff')
    process.load('Configuration.StandardSequences.L1Reco_cff')
    process.load('SimGeneral.MixingModule.mixNoPU_cfi')

    process.digi = cms.Path(process.pdigi)
    process.l1sim = cms.Path(process.SimL1Emulator)
    process.d2raw = cms.Path(process.DigiToRaw)
    process.l1reco = cms.Path(process.L1Reco)
if do_reco:
    process.GlobalTag.toGet = cms.VPSet(
            cms.PSet(record = cms.string('EcalSampleMaskRcd'),
                tag = cms.string('EcalSampleMask_offline'),
                # connect = cms.untracked.string('oracle://cms_orcoff_prep/CMS_COND_ECAL'),
                connect = cms.untracked.string('frontier://FrontierPrep/CMS_COND_ECAL'),
                )
            )
    process.GlobalTag.DBParameters.authenticationPath="/afs/cern.ch/cms/DB/conddb"
    process.ecalGlobalUncalibRecHit.kPoorRecoFlagEB = cms.bool(False)
    process.ecalGlobalUncalibRecHit.kPoorRecoFlagEE = cms.bool(False)

if data:
    process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
else:
    process.load('Configuration.StandardSequences.RawToDigi_cff')

process.raw2digi = cms.Path(process.RawToDigi)
process.reco = cms.Path(process.reconstruction)

process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')


# =====================


process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")
if reemul:
    process.l1GtTrigReport.L1GtRecordInputTag = "simGtDigis"
else:
    process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"
process.l1GtTrigReport.PrintVerbosity = 1
process.report = cms.Path(process.l1GtTrigReport)

import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
process.ZeroBiasAve = hlt.triggerResultsFilter.clone()
process.ZeroBiasAve.triggerConditions = cms.vstring('L1_ZeroBias',)
process.ZeroBiasAve.hltResults = cms.InputTag( "TriggerResults", "", "HLT" )
process.ZeroBiasAve.l1tResults = cms.InputTag("gtDigis")
process.ZeroBiasAve.throw = cms.bool( True )
process.zerobias = cms.Path(process.ZeroBiasAve)

if reemul:
    process.load('HLTrigger.Configuration.HLT_FULL_cff')
    process.load('Configuration.StandardSequences.SimL1Emulator_cff')
    process.load('EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi')

    import L1Trigger.Configuration.L1Trigger_custom
    process = L1Trigger.Configuration.L1Trigger_custom.customiseL1GtEmulatorFromRaw(process)
    process = L1Trigger.Configuration.L1Trigger_custom.customiseResetPrescalesAndMasks(process)

    import HLTrigger.Configuration.customizeHLTforL1Emulator
    process = HLTrigger.Configuration.customizeHLTforL1Emulator.switchToL1Emulator(
            process, False, 'minPt', 'minPt', False, True, False, True)
    process = HLTrigger.Configuration.customizeHLTforL1Emulator.switchToSimGtReEmulGctDigis(process)
    process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)

    process.unpacker = cms.Path(process.HLTL1UnpackerSequence)
    process.l1unpack = cms.Path(process.l1GtUnpack)

process.load('L1Trigger.L1ExtraFromDigis.l1extraParticles_cff')
# process.l1extraParticles.forwardJetSource = cms.InputTag('gctReEmulDigis', 'forJets')
# process.l1extraParticles.centralJetSource = cms.InputTag('gctReEmulDigis', 'cenJets')
# process.l1extraParticles.tauJetSource = cms.InputTag('gctReEmulDigis', 'tauJets')
process.l1extra = cms.Path(process.l1extraParticles)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.pdump = cms.Path(process.dump)


### ES Prefer preferences

#process.es_prefer_calotowerconstituentsmapbuilder = cms.ESPrefer("CaloTowerConstituentsMapBuilder","caloTowerConstituentsMapBuilder")
process.es_prefer_trackerNumberingGeometry = cms.ESPrefer("TrackerGeometricDetESModule","trackerNumberingGeometryDB")
#process.es_prefer_ttrhbwr = cms.ESPrefer("TkTransientTrackingRecHitBuilderESProducer","ttrhbwr")
#process.es_prefer_ttrhbwor = cms.ESPrefer("TkTransientTrackingRecHitBuilderESProducer","ttrhbwor")

process.schedule = cms.Schedule()


### Override the L1 menu
outputFile = 'l1_nugun.root'
dumpFile   = 'dumpedConfig.py'

if legacy:
    process.load('L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi')
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile = 'L1Menu_Collisions2012_v3_L1T_Scales_20101224_Imp0_0x102b.xml'

    process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff')
    process.es_prefer_l1GtParameters = cms.ESPrefer('L1GtTriggerMenuXmlProducer','l1GtTriggerMenuXml')
else:
    process.load('L1Trigger.L1TGlobal.TriggerMenuXml_cfi')
    process.TriggerMenuXml.TriggerMenuLuminosity = 'startup'
    #process.TriggerMenuXml.DefXmlFile = 'L1_Example_Menu_2013.xml'
    process.TriggerMenuXml.DefXmlFile = 'L1Menu_Reference_2014.xml'

    process.load('L1Trigger.L1TGlobal.L1uGtTriggerMenuConfig_cff')
    process.es_prefer_l1GtParameters = cms.ESPrefer('l1t::TriggerMenuXmlProducer','TriggerMenuXml')

    outputFile = 'l1_nugun_newL1MenuParser.root'
    dumpFile   = 'dumpedConfig_newL1MenuParser.py'


### Useful for debugging (sometimes)
#process.Tracer = cms.Service("Tracer")

if do_digi:
    process.schedule.extend([process.digi, process.l1sim, process.d2raw])
if raw or do_reco:
    process.schedule.append(process.raw2digi)
    process.schedule.append(process.l1extra)
if data:
    process.schedule.append(process.zerobias)
if do_digi:
    process.schedule.append(process.l1reco)
if reemul:
    process.schedule.append(process.unpacker)
    process.schedule.append(process.l1unpack)
if debug:
    process.schedule.append(process.pdump)
if do_reco:
    process.schedule.append(process.reco)

if raw:
    process.schedule.append(process.report)



readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",
                             fileNames = readFiles,
                             secondaryFileNames = secFiles
                             )

readFiles.extend( [
    "/store/user/puigh/RelValTTbar_GEN-SIM-DIGI-RAW-HLTDEBUG_START70_V2_amend-v4_00000_3A11157B-ED51-E311-BA75-003048679080.root",
    #"/store/user/puigh/RelValTTbar_GEN-SIM-DIGI-RAW-HLTDEBUG_START70_V2_amend-v4_00000_1A20137C-E651-E311-A9C6-00304867BFAA.root",
    #"/store/user/puigh/RelValTTbar_GEN-SIM-DIGI-RAW-HLTDEBUG_START70_V2_amend-v4_00000_2EFD8C7A-E651-E311-8C92-002354EF3BE3.root"
    #'/store/user/puigh/Neutrino_Pt2to20_gun_UpgradeL1TDR-PU50_POSTLS161_V12-v1_A4FFBC76-5B39-E211-999D-0030487F1BE5.root',
    ##'root://xrootd.unl.edu//store/mc/Summer12/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradeL1TDR-PU50_POSTLS161_V12-v1/00000/002EF512-2A39-E211-9B80-0030487F1A47.root',
    ##'/store/mc/Summer12/Neutrino_Pt2to20_gun/GEN-SIM-DIGI-RAW/UpgradeL1TDR-PU50_POSTLS161_V12-v1/00000/002EF512-2A39-E211-9B80-0030487F1A47.root',
        ] )


process.output = cms.OutputModule( "PoolOutputModule"
                                , fileName       = cms.untracked.string( 'delete.root' )
                                , SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring([]) )
                                , outputCommands = cms.untracked.vstring( 'keep *'
                                        )
                                )
process.output.fileName = outputFile

process.options = cms.untracked.PSet()

process.outpath = cms.EndPath(process.output)

#process.schedule.append(process.outpath)

# Spit out filter efficiency at the end.
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

#outfile = open(dumpFile,'w')
#print >> outfile,process.dumpPython()
#outfile.close()
