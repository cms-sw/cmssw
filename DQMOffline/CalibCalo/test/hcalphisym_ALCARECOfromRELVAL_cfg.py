# Auto generated configuration file
# using: 
# Revision: 1.109 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step23_HcalCalMinBias -s ALCA:Configuration/StandardSequences/AlCaRecoStream_HcalCalMinBias_cff:HcalCalMinBias+DQM -n 1000 --filein file:raw.root --conditions FrontierConditions_GlobalTag,IDEAL_V9::All --eventcontent RECO --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('ALCA')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/AlCaRecoStream_HcalCalMinBias_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')
process.load('DQMOffline/CalibCalo/MonitorAlCaHcalPhisym_cfi')



process.hcalminbiasHLT.HLTPaths = ['HLT_MinBiasEcal']
process.hfrecoNoise.firstSample = 1
process.hcalDigiAlCaMB.InputLabel = 'rawDataCollector'



process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.109 $'),
    annotation = cms.untracked.string('step23_HcalCalMinBias nevts:1000'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
 '/store/relval/CMSSW_3_1_0_pre3/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/14BE5305-F509-DE11-B712-000423D9870C.root'
)
)

# Additional output definition
process.ALCARECOStreamHcalCalMinBias = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalMinBias')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep HBHERecHitsSorted_hbherecoMB_*_*', 
        'keep HORecHitsSorted_horecoMB_*_*', 
        'keep HFRecHitsSorted_hfrecoMB_*_*', 
        'keep HBHERecHitsSorted_hbherecoNoise_*_*', 
        'keep HORecHitsSorted_horecoNoise_*_*', 
        'keep HFRecHitsSorted_hfrecoNoise_*_*'),
    fileName = cms.untracked.string('ALCARECOHcalCalMinBias.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOHcalCalMinBias'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)

# Other statements
process.GlobalTag.globaltag = 'IDEAL_30X::All'

# Path and EndPath definitions
process.pathALCARECODQM = cms.Path(process.HcalPhiSymMon + process.MEtoEDMConverter)
## process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
process.pathALCARECOHcalCalMinBias = cms.Path(process.seqALCARECOHcalCalMinBias)
process.endjob_step = cms.Path(process.endOfProcess)
process.ALCARECOStreamHcalCalMinBiasOutPath = cms.EndPath(process.ALCARECOStreamHcalCalMinBias)
process.pathHcalPhiSymMon = cms.Path(process.HcalPhiSymMon)
# Schedule definition
# process.schedule = cms.Schedule(process.pathALCARECODQM,process.pathHcalPhiSymMon,process.pathALCARECOHcalCalMinBias,process.endjob_step,process.ALCARECOStreamHcalCalMinBiasOutPath)
process.schedule = cms.Schedule(process.pathALCARECOHcalCalMinBias,process.pathALCARECODQM,process.endjob_step,process.ALCARECOStreamHcalCalMinBiasOutPath)
