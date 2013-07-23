import FWCore.ParameterSet.Config as cms


process = cms.Process("makeSD")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('SD for Stopped HSCP in No Beam data'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Configuration/Skimming/test/SDmaker_StoppedHSCP_cfg.py,v $')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.GlobalTag.globaltag = "GR_R_37X_V6A::All"  


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/F85204EE-EB40-DF11-8F71-001A64789D1C.root'
        )
)
process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

import HLTrigger.HLTfilters.hltHighLevelDev_cfi


process.StoppedHSCP = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
process.StoppedHSCP.HLTPaths = (
"HLT_StoppedHSCP_8E29",
)
process.StoppedHSCP.HLTPathsPrescales  = cms.vuint32(1,)
process.StoppedHSCP.HLTOverallPrescale = cms.uint32(1)
process.StoppedHSCP.throw = False
process.StoppedHSCP.andOr = True

process.filterStoppedHSCP = cms.Path(process.StoppedHSCP)



process.outputStoppedHSCP = cms.OutputModule("PoolOutputModule",
                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filterStoppedHSCP')),                               
                                          dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('SD_StoppedHSCP')),
                                          outputCommands = process.RECOEventContent.outputCommands,
                                          fileName = cms.untracked.string('SD_StoppedHSCP.root')
                                          )

process.this_is_the_end = cms.EndPath(
process.outputStoppedHSCP
)
