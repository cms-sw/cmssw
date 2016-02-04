import FWCore.ParameterSet.Config as cms

process = cms.Process('RAW2DIGI')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
#process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger = cms.Service("MessageLogger",
    log = cms.untracked.PSet( threshold = cms.untracked.string("DEBUG") ),
    debugModules = cms.untracked.vstring("l1RpcEmulDigis"),
    destinations = cms.untracked.vstring('log')
)


process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('test nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.options = cms.untracked.PSet(

)
# Input source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
             'file:/scratch/scratch0/tfruboes/2010.11.testPulses/CMSSW_3_8_4_patch3/src/20101129_genForPulse/res/gen_2_1_Uxo.root'
            ,'file:/scratch/scratch0/tfruboes/2010.11.testPulses/CMSSW_3_8_4_patch3/src/20101129_genForPulse/res/gen_3_1_l25.root'
            ,'file:/scratch/scratch0/tfruboes/2010.11.testPulses/CMSSW_3_8_4_patch3/src/20101129_genForPulse/res/gen_4_1_WIf.root'
            ,'file:/scratch/scratch0/tfruboes/2010.11.testPulses/CMSSW_3_8_4_patch3/src/20101129_genForPulse/res/gen_5_1_UzR.root'
            ,'file:/scratch/scratch0/tfruboes/2010.11.testPulses/CMSSW_3_8_4_patch3/src/20101129_genForPulse/res/gen_6_1_TJp.root'
            ,'file:/scratch/scratch0/tfruboes/2010.11.testPulses/CMSSW_3_8_4_patch3/src/20101129_genForPulse/res/gen_7_1_NNC.root'
            ,'file:/scratch/scratch0/tfruboes/2010.11.testPulses/CMSSW_3_8_4_patch3/src/20101129_genForPulse/res/gen_9_1_6oi.root'
  )
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('test_RAW2DIGI.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RAW')
    )
)
########################
#
# pulse specific path
#
########################
process.load("L1Trigger.RPCTrigger.l1RpcEmulDigis_cfi")
process.l1RpcEmulDigis.label = cms.string('muonRPCDigis')
process.l1RpcEmulDigis.RPCTriggerDebug = 1


process.xmlWritter = cms.EDAnalyzer("LinkDataXMLWriter",
    digisSource = cms.InputTag("muonRPCDigis"),
    xmlDir = cms.string('testBxData.xml')
)

process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConfig_cff")
#process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeDefinition_cff")
process.load("L1Trigger.RPCTrigger.RPCConeConfig_cff")

process.rpcconf.filedir = cms.untracked.string('v6/') # backslash na koncu wazny
process.es_prefer_rpcPats = cms.ESPrefer("RPCTriggerConfig","rpcconf") 
process.xmlSeq = cms.Sequence(process.l1RpcEmulDigis*process.xmlWritter)


#process.xmlSeq = cms.Sequence(process.l1RpcEmulDigis)
process.xmlStep = cms.Path(process.xmlSeq)
#######################
#######################
#######################
# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'START38_V13::All'

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.endjob_step = cms.Path(process.endOfProcess)
#process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
#process.schedule = cms.Schedule(process.raw2digi_step,process.endjob_step,process.RAWSIMoutput_step)
process.schedule = cms.Schedule(process.raw2digi_step, process.xmlStep ,process.endjob_step )
