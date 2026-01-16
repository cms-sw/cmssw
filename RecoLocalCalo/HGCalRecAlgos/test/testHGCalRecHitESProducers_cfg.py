# Author: Izaak Neutelings (March 2023)
# Instructions:
#   export SCRAM_ARCH="el9_amd64_gcc12"
#   cmsrel CMSSW_15_1_0_pre1
#   cd CMSSW_15_1_0_pre1/src/
#   cmsenv
#   #git cms-addpkg Geometry/HGCalMapping
#   #git clone -b CalibSurrOffsetMap git@github.com:RSalvatico/Geometry-HGCalMapping.git Geometry/HGCalMapping/data
#   git clone https://gitlab.cern.ch/hgcal-dpg/hgcal-comm.git HGCalCommissioning
#   scram b -j8
#   cmsRun $CMSSW_BASE/src/RecoLocalCalo/HGCalRecAlgos/test/testHGCalRecHitESProducers_cfg.py
# Sources:
#   https://gitlab.cern.ch/hgcal-dpg/hgcal-comm/-/blob/master/Configuration/test/step_RAW2DIGI.py?ref_type=heads
#   https://gitlab.cern.ch/hgcal-dpg/hgcal-comm/-/blob/master/Configuration/python/SysValEras_cff.py?ref_type=heads
#   https://github.com/CMS-HGCAL/cmssw/blob/hgcal-condformat-HGCalNANO-13_2_0_pre3/DPGAnalysis/HGCalTools/python/tb2023_cfi.py
#   https://github.com/CMS-HGCAL/cmssw/blob/dev/hackathon_base_CMSSW_14_1_0_pre0/SimCalorimetry/HGCalSimProducers/test/hgcalRealistiDigis_cfg.py
import os
import FWCore.ParameterSet.Config as cms

# USER OPTIONS
from FWCore.ParameterSet.VarParsing import VarParsing
configdir = "/eos/cms/store/group/dpg_hgcal/tb_hgcal/DPG/calibrations/SepTB2024"
options = VarParsing('standard')
options.register('geometry', 'ExtendedRun4D104', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 info="geometry to use")
options.register('maxchans', 500, mytype=VarParsing.varType.int,
                 info="maximum number of channels to print out")
options.register('maxmods', 8, mytype=VarParsing.varType.int,
                 info="maximum number of modules to print out")
options.register('maxfeds', 30, mytype=VarParsing.varType.int,
                 info="maximum number of FED IDs to test")
options.register('fedconfig', f"{configdir}/config/config_feds_hackathon.json", mytype=VarParsing.varType.int,
                 info="Path to configuration (JSON format)")
options.register('modconfig', f"{configdir}/config/config_econds_hackathon.json", mytype=VarParsing.varType.string,
                 info="Path to configuration (JSON format)")
options.register('params',
                 #f"{configdir}/level0_calib_Relay1727210224.json",
                 f"{configdir}/level0_calib_hackathon.json",
                 mytype=VarParsing.varType.string,
                 info="Path to calibration parameters (JSON format)")
options.register('energyloss',
                 f"{configdir}/../EnergyLoss/hgcal_energyloss_v16.json",
                 mytype=VarParsing.varType.string,
                 info="Path to calibration parameters (JSON format)")
options.register('modules',
                 # see https://github.com/cms-data/Geometry-HGCalMapping
                 # or https://gitlab.cern.ch/hgcal-dpg/hgcal-comm/-/tree/master/Configuration/data/ModuleMaps
                 #"Geometry/HGCalMapping/data/ModuleMaps/modulelocator_test.txt", # test beam with six modules
                 f"Geometry/HGCalMapping/data/ModuleMaps/modulelocator_Sep2024TBv2.txt", # 3 layers (9 modules)
                 mytype=VarParsing.varType.string,
                 info="Path to module mapper. Absolute, or relative to CMSSW src directory")
options.register('sicells', "", mytype=VarParsing.varType.string,
                 info="Path to Si cell mapper. Absolute, or relative to CMSSW src directory")
options.register('sipmcells', "", mytype=VarParsing.varType.string,
                 info="Path to SiPM-on-tile cell mapper. Absolute, or relative to CMSSW src directory")
options.parseArguments()
relpath = os.path.join(os.environ.get('CMSSW_BASE',''),"src")
if options.params.startswith('/eos/'):
  options.params = os.path.relpath(options.params,relpath)
if options.energyloss.startswith('/eos/'):
  options.energyloss = os.path.relpath(options.energyloss,relpath)
if len(options.files)==0:
  options.files=['file:/eos/cms/store/group/dpg_hgcal/comm_hgcal/psilva/hackhathon/23234.103_TTbar_14TeV+2026D94Aging3000/step2.root']
  #options.files=['file:/eos/cms/store/group/dpg_hgcal/comm_hgcal/psilva/hackhathon/23234.103_TTbar_14TeV+2026D94Aging3000/step2.root']
  #options.files=['file:/afs/cern.ch/user/y/yumiao/public/HGCAL_Raw_Data_Handling/Data/Digis/testFakeDigisSoA.root']
print(f">>> Geometry:      {options.geometry!r}")
print(f">>> Input files:   {options.files!r}")
print(f">>> Module map:    {options.modules!r}")
print(f">>> SiCell map:    {options.sicells!r}")
print(f">>> SipmCell map:  {options.sipmcells!r}")
print(f">>> FED config:    {options.fedconfig!r}")
print(f">>> ECON-D config: {options.modconfig!r}")
print(f">>> Calib params:  {options.params!r}")
print(f">>> Energy loss:   {options.energyloss!r}")

# PROCESS
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9 as Era_Phase2
process = cms.Process('HGCalRecHitESProducersTest',Era_Phase2)

# GLOBAL TAG
from Configuration.AlCa.GlobalTag import GlobalTag
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# INPUT
process.source = cms.Source(
  "PoolSource",
  fileNames=cms.untracked.vstring(options.files),
  duplicateCheckMode=cms.untracked.string("noDuplicateCheck")
)
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )
process.maxEvents.input = 1

# MESSAGE LOGGER
process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.debugModules = ['*'] #"hgCalCalibrationESProducer", "hgCalConfigurationESProducer"]
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.noTimeStamps = True
process.MessageLogger.HGCalConfigurationESProducer = { } # enable logger
process.MessageLogger.HGCalCalibrationESProducer = { }
process.MessageLogger.search_modkey = { }
process.MessageLogger.search_fedkey = { }
process.MessageLogger.check_keys = { }
process.MessageLogger.cerr.FwkReport.reportEvery = 500

# GEOMETRY
process.load(f"Configuration.Geometry.Geometry{options.geometry}Reco_cff")
process.load(f"Configuration.Geometry.Geometry{options.geometry}_cff")

# MAPPING & INDEXING
# https://github.com/cms-sw/cmssw/blob/master/Geometry/HGCalMapping/python/hgcalmapping_cff.py
from Geometry.HGCalMapping.hgcalmapping_cff import customise_hgcalmapper
kwargs = { k: getattr(options,k) for k in ['modules','sicells','sipmcells'] if getattr(options,k)!='' }
customise_hgcalmapper(process, **kwargs)

# GLOBAL CONFIGURATION ESProducers (for unpacker)
#process.load("RecoLocalCalo.HGCalRecAlgos.HGCalConfigurationESProducer")
#process.load("RecoLocalCalo.HGCalRecAlgos.hgCalConfigurationESProducer_cfi")
process.hgcalConfigESProducer = cms.ESSource( # ESProducer to load configurations for unpacker
  'HGCalConfigurationESProducer',
  fedjson=cms.string(options.fedconfig),  # JSON with FED configuration parameters
  modjson=cms.string(options.modconfig),  # JSON with ECON-D configuration parameters
  #passthroughMode=cms.int32(0),          # ignore mismatch
  #cbHeaderMarker=cms.int32(0x5f),        # capture block
  #slinkHeaderMarker=cms.int32(0x2a),     # S-link
  #econdHeaderMarker=cms.int32(0x154),    # ECON-D
  #charMode=cms.int32(1),
  indexSource=cms.ESInputTag('hgCalMappingESProducer','')
)

# CALIBRATIONS & CONFIGURATION Alpaka ESProducers
process.load('Configuration.StandardSequences.Accelerators_cff')
#process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')
#process.load('HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA_cfi')
process.hgcalCalibParamESProducer = cms.ESProducer( # ESProducer to load calibration parameters from JSON file, like pedestals
  'hgcalrechit::HGCalCalibrationESProducer@alpaka',
  filename=cms.FileInPath(options.params),
  filenameEnergyLoss=cms.FileInPath(options.energyloss),
  indexSource=cms.ESInputTag('hgCalMappingESProducer',''),
  mapSource=cms.ESInputTag('hgCalMappingModuleESProducer','')
)

# MAIN PROCESS
process.testHGCalRecHitESProducers = cms.EDProducer(
  'HGCalRecHitESProducersTest@alpaka',
  #'alpaka_cuda_async::TestHGCalRecHitProducer', # GPU
  #'alpaka_serial_sync::TestHGCalRecHitProducer', # CPU
  indexSource=cms.ESInputTag('hgCalMappingESProducer', ''),
  configSource=cms.ESInputTag('hgcalConfigESProducer', ''),
  calibParamSource=cms.ESInputTag('hgcalCalibParamESProducer', ''),
  maxchans=cms.int32(options.maxchans),  # maximum number of channels to print out
  maxmods=cms.int32(options.maxmods),    # maximum number of modules to print out
  maxfeds=cms.int32(options.maxfeds),    # maximum number of FED IDs to test
  #fedjson=cms.string(options.fedconfig),  # JSON with FED configuration parameters
  fedjson=cms.string(""), # use hardcoded JSON instead
)
process.p = cms.Path(process.testHGCalRecHitESProducers)

#### PRINT available records (for debugging)
###process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")
###process.dump = cms.Path(process.dumpES)

# OUTPUT
process.output = cms.OutputModule(
  'PoolOutputModule',
  fileName = cms.untracked.string(options.output),
  #outputCommands = cms.untracked.vstring('drop *','keep *_*_*_REALDIGI')
)
process.output_path = cms.EndPath(process.output)

