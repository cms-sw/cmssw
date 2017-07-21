import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("CTPPSReconstructionChainTest", eras.ctpps_2016)

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING')
  )
)

# raw data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    # run 274199, fill 4961, 29 May 2016 (before TS2)
    '/store/data/Run2016B/DoubleEG/RAW/v2/000/274/199/00000/04985451-9B26-E611-BEB9-02163E013859.root',
    #'root://eostotem.cern.ch//eos/totem/user/j/jkaspar/04C8034A-9626-E611-9B6E-02163E011F93.root'

    # run 283877, fill 5442, 23 Oct 2016 (after TS2)
    '/store/data/Run2016H/HLTPhysics/RAW/v1/000/283/877/00000/F28F8896-999B-E611-93D8-02163E013706.root',

    # test file for 2017 mapping (vertical RPs only)
    'root://eostotem.cern.ch//eos/totem/data/ctpps/run290874.root'
  ),

  inputCommands = cms.untracked.vstring(
    'drop *',
    'keep FEDRawDataCollection_*_*_*'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(8000)
)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

process.p = cms.Path(
  process.ctppsRawToDigi *
  process.recoCTPPS
)

# output configuration
from RecoCTPPS.Configuration.RecoCTPPS_EventContent_cff import RecoCTPPSAOD
process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("file:./AOD.root"),
  outputCommands = RecoCTPPSAOD.outputCommands
)

process.outpath = cms.EndPath(process.output)
