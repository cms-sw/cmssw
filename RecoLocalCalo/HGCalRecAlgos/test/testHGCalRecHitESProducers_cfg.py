# Author: Izaak Neutelings (March 2023)
# Instructions:
#   export SCRAM_ARCH="el9_amd64_gcc12"
#   cmsrel CMSSW_14_1_0_pre4
#   cd CMSSW_14_1_0_pre4/src/
#   cmsenv
#   git cms-merge-topic -u CMS-HGCAL:dev/hackathon_base_CMSSW_14_1_X
#   git clone https://github.com/pfs/Geometry-HGCalMapping.git  $CMSSW_BASE/src/Geometry/HGCalMapping/data
#   git clone https://gitlab.cern.ch/hgcal-dpg/hgcal-comm.git HGCalCommissioning
#   scram b -j8
#   cmsRun $CMSSW_BASE/src/RecoLocalCalo/HGCalRecAlgos/test/testHGCalRecHitESProducers_cfg.py
# Sources:
#   https://github.com/CMS-HGCAL/cmssw/blob/hgcal-condformat-HGCalNANO-13_2_0_pre3/DPGAnalysis/HGCalTools/python/tb2023_cfi.py
#   https://gitlab.cern.ch/hgcal-dpg/hgcal-comm/-/blob/master/SystemTestEventFilters/test/test_raw2reco.py
#   https://github.com/CMS-HGCAL/cmssw/blob/dev/hackathon_base_CMSSW_14_1_0_pre0/SimCalorimetry/HGCalSimProducers/test/hgcalRealistiDigis_cfg.py
import os
import FWCore.ParameterSet.Config as cms

# USER OPTIONS
from FWCore.ParameterSet.VarParsing import VarParsing
datadir = os.path.join(os.environ.get('CMSSW_BASE',''),"src/HGCalCommissioning/LocalCalibration/data")
options = VarParsing('standard')
options.register('geometry', 'Extended2026D94', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 info="geometry to use")
options.register('fedconfig',f"{datadir}/config_feds.json",mytype=VarParsing.varType.string,
                 info="Path to configuration (JSON format)")
options.register('modconfig',f"{datadir}/config_econds.json",mytype=VarParsing.varType.string,
                 info="Path to configuration (JSON format)")
options.register('params',f"{datadir}/level0_calib_params.json",mytype=VarParsing.varType.string,
                 info="Path to calibration parameters (JSON format)")
options.register('modules',
                 #"Geometry/HGCalMapping/data/ModuleMaps/modulelocator_test.txt", # test beam with six modules
                 #"Geometry/HGCalMapping/data/ModuleMaps/modulelocator_test_2mods.txt", # only first two modules, fedId=49
                 "HGCalCommissioning/Configuration/data/ModuleMaps/modulelocator_test_2mods.txt", # fedId=0
                 mytype=VarParsing.varType.string,
                 info="Path to module mapper. Absolute, or relative to CMSSW src directory")
options.register('sicells','Geometry/HGCalMapping/data/CellMaps/WaferCellMapTraces.txt',mytype=VarParsing.varType.string,
                 info="Path to Si cell mapper. Absolute, or relative to CMSSW src directory")
options.register('sipmcells','Geometry/HGCalMapping/data/CellMaps/channels_sipmontile.hgcal.txt',mytype=VarParsing.varType.string,
                 info="Path to SiPM-on-tile cell mapper. Absolute, or relative to CMSSW src directory")
options.parseArguments()
if len(options.files)==0:
  options.files=['file:/eos/cms/store/group/dpg_hgcal/comm_hgcal/psilva/hackhathon/23234.103_TTbar_14TeV+2026D94Aging3000/step2.root']
  #options.files=['file:/eos/cms/store/group/dpg_hgcal/comm_hgcal/psilva/hackhathon/23234.103_TTbar_14TeV+2026D94Aging3000/step2.root']
  #options.files=['file:/afs/cern.ch/user/y/yumiao/public/HGCAL_Raw_Data_Handling/Data/Digis/testFakeDigisSoA.root']
  #options.files=['file:/home/hgcdaq00/CMSSW/data/23234.103_TTbar_14TeV+2026D94Aging3000/step2.root'] # on DAQ PC
  #options.files=['file:/home/hgcdaq00/CMSSW/data/testFakeDigisSoA.root'] # on DAQ PC
print(f">>> Geometry:      {options.geometry!r}")
print(f">>> Input files:   {options.files!r}")
print(f">>> Module map:    {options.modules!r}")
print(f">>> SiCell map:    {options.sicells!r}")
print(f">>> SipmCell map:  {options.sipmcells!r}")
print(f">>> FED config:    {options.fedconfig!r}")
print(f">>> ECON-D config: {options.modconfig!r}")
print(f">>> Calib params:  {options.params!r}")

# PROCESS
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9 as Era_Phase2
process = cms.Process('TestHGCalRecHitESProducers',Era_Phase2)

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
  fileNames = cms.untracked.vstring(options.files),
  duplicateCheckMode = cms.untracked.string("noDuplicateCheck")
)
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )
process.maxEvents.input = 1

# MESSAGE LOGGER
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 500

# GEOMETRY
process.load(f"Configuration.Geometry.Geometry{options.geometry}Reco_cff")
process.load(f"Configuration.Geometry.Geometry{options.geometry}_cff")
#process.load('Geometry.HGCalMapping.hgCalMappingIndexESSource_cfi') # old
process.load('Geometry.HGCalMapping.hgCalMappingESProducer_cfi')
process.hgCalMappingESProducer.si = cms.FileInPath(options.sicells)
process.hgCalMappingESProducer.sipm = cms.FileInPath(options.sipmcells)
process.hgCalMappingESProducer.modules = cms.FileInPath(options.modules)

# GLOBAL CONFIGURATION ESProducers (for unpacker)
#process.load("RecoLocalCalo.HGCalRecAlgos.HGCalConfigurationESProducer")
#process.load("RecoLocalCalo.HGCalRecAlgos.hgCalConfigurationESProducer_cfi")
process.hgcalConfigESProducer = cms.ESSource( # ESProducer to load configurations for unpacker
  'HGCalConfigurationESProducer',
  fedjson=cms.string(options.fedconfig), # JSON with FED configuration parameters
  modjson=cms.string(options.modconfig), # JSON with ECON-D configuration parameters
  #passthroughMode=cms.int32(0),          # ignore mismatch
  #cbHeaderMarker=cms.int32(0x7f),        # capture block
  ##cbHeaderMarker=cms.int32(0x5f),        # capture block
  #slinkHeaderMarker=cms.int32(0x55),     # S-link
  ##slinkHeaderMarker=cms.int32(0x2a),     # S-link
  #econdHeaderMarker=cms.int32(0x154),    # ECON-D
  #charMode=cms.int32(1),
  indexSource=cms.ESInputTag('hgCalMappingESProducer','')
)

# CALIBRATIONS & CONFIGURATION Alpaka ESProducers
process.load('Configuration.StandardSequences.Accelerators_cff')
#process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')
#process.load('HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA_cfi')
process.hgcalConfigParamESProducer = cms.ESProducer( # ESProducer to load configurations parameters from YAML file, like gain
  'hgcalrechit::HGCalConfigurationESProducer@alpaka',
  gain=cms.int32(1), # to switch between 80, 160, 320 fC calibration
  #charMode=cms.int32(1),
  indexSource=cms.ESInputTag('hgCalMappingESProducer','')
)
process.hgcalCalibParamESProducer = cms.ESProducer( # ESProducer to load calibration parameters from JSON file, like pedestals
  'hgcalrechit::HGCalCalibrationESProducer@alpaka',
  filename=cms.string(options.params),
  indexSource=cms.ESInputTag('hgCalMappingESProducer',''),
  configSource=cms.ESInputTag(''),
)

# MAIN PROCESS
process.testHGCalRecHitESProducers = cms.EDProducer(
  'TestHGCalRecHitESProducers@alpaka',
  #'alpaka_cuda_async::TestHGCalRecHitProducer', # GPU
  #'alpaka_serial_sync::TestHGCalRecHitProducer', # CPU
  indexSource=cms.ESInputTag('hgCalMappingESProducer', ''),
  configSource=cms.ESInputTag('hgcalConfigESProducer', ''),
  configParamSource=cms.ESInputTag('hgcalConfigParamESProducer', ''),
  calibParamSource=cms.ESInputTag('hgcalCalibParamESProducer', ''),
)
process.t = cms.Task(process.testHGCalRecHitESProducers)
process.p = cms.Path(process.t)

# OUTPUT
process.output = cms.OutputModule(
  'PoolOutputModule',
  fileName = cms.untracked.string(options.output),
  #outputCommands = cms.untracked.vstring('drop *','keep *_*_*_REALDIGI')
)
process.output_path = cms.EndPath(process.output)

