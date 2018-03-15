# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: reco -s RAW2DIGI,RECO --filein file:pickevents.root --fileout file:useless.root --conditions 92X_dataRun2_HLT_v7 --eventcontent FEVTDEBUG
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        #"/store/data/Run2017F/ZeroBias/RAW/v1/000/305/757/00000/041E9F8E-ACBA-E711-8BC9-02163E0134E0.root"
        "file:Run2017F_HTMHT_305862_1ACCE666.root"
        ),
                            secondaryFileNames = cms.untracked.vstring()
                            )

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('reco nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data_promptlike', '')

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.hcalDigis.UnpackZDC = cms.untracked.bool(False)

import RecoLocalCalo.HcalRecProducers.HBHEMethod3Parameters_cfi as method3
import RecoLocalCalo.HcalRecProducers.HBHEMethod2Parameters_cfi as method2
import RecoLocalCalo.HcalRecProducers.HBHEMethod0Parameters_cfi as method0
import RecoLocalCalo.HcalRecProducers.HBHEMahiParameters_cfi as mahi


process.flat = cms.EDAnalyzer('HBHEReconstructionDebugger',
                              algorithm = cms.PSet(
        # Parameters for "Method 3" (non-keyword arguments have to go first)                                        
        method3.m3Parameters,
        method2.m2Parameters,
        method0.m0Parameters,
        mahi.mahiParameters,
        
        Class = cms.string("SimpleHBHEPhase1Algo"),
        
        # Time shift (in ns) to add to TDC timing (for QIE11)                                                       
        tdcTimeShift = cms.double(0.0),
        
        # Parameters for "Method 0"                                                                                 
        firstSampleShift = cms.int32(0),
        
        # Use "Method 2"?                                                                                           
        useM2 = cms.bool(False),
        
        # Use "Method 3"?                                                                                           
        useM3 = cms.bool(True),
        
        # Use Mahi?                                                                                                 
        useMahi = cms.bool(True)
        )
                              )

process.hcalLocalRecoSequence.remove(process.zdcreco)
process.hcalLocalRecoSequence.remove(process.hfprereco)
process.hcalLocalRecoSequence.remove(process.horeco)
process.hcalLocalRecoSequence.remove(process.hfreco)

process.hbheprereco.setNegativeFlagsQIE8 = cms.bool(False)
process.hbheprereco.setNoiseFlagsQIE8 = cms.bool(False)
process.hbheprereco.setPulseShapeFlagsQIE8 = cms.bool(False)
process.hbheprereco.setLegacyFlagsQIE8 = cms.bool(False)

process.hbheprereco.processQIE11 = cms.bool(True)
process.hbheprereco.processQIE8  = cms.bool(True)

process.hbheprereco.saveInfos    = cms.bool(True)
process.hbheprereco.makeRecHits  = cms.bool(False)

process.hbheprereco.algorithm.Class = cms.string("SimpleHBHEPhase1AlgoDebug")

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.hcalDigis)
process.reconstruction_step = cms.Path(process.hbheprereco)
process.endjob_step = cms.EndPath(process.endOfProcess)

process.dump=cms.EDAnalyzer('EventContentAnalyzer')
process.dump_step = cms.Path(process.dump)


process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("phasescantest.root")
    )

process.flat_step = cms.Path(process.flat)


# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,
                                process.reconstruction_step,
                                process.flat_step,
                                process.endjob_step)
#process.FEVTDEBUGoutput_step)

#from Configuration.DataProcessing.Utils import addMonitoring
#process = addMonitoring(process)
#if 'FastTimerService' in process.__dict__:
#    del process.FastTimerService
#
#process.load("HLTrigger.Timer.FastTimerService_cfi")

