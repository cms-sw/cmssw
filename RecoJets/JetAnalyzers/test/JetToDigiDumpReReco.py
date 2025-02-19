# JetToDigiDump for MC, by Amnon Harel (based on Auto generated configuration file)
# tested using: CMSSW_2_1_6
# Revision: 1.0
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('step2 nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    
"/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0003/E42C45C7-9E82-DD11-9212-001617C3B76E.root",
"/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0003/9058E3D9-9882-DD11-8033-001617C3B78C.root",
"/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0003/522DF455-9882-DD11-A8BE-000423D98BC4.root",
"/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0003/32E40262-9B82-DD11-8E34-000423D94990.root",
"/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0003/06BDA52F-9782-DD11-833B-000423D6C8E6.root",
"/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0003/9003AA86-9382-DD11-92C9-000423D98C20.root",
"/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0003/C6434683-9382-DD11-8FD5-000423D98E54.root",
"/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0003/2678826A-9582-DD11-9120-001617C3B64C.root",
"/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0003/FC9DB286-9382-DD11-85F8-000423D99B3E.root",
"/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0003/BE32796C-9582-DD11-88D5-001617DBCF90.root"
    
    )
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('step2_RAW2DIGI_RECO.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string(''),
        filterName = cms.untracked.string('')
    )
)

# Additional output definition

# Event content print out
process.dump = cms.EDAnalyzer("EventContentAnalyzer")



# Run jet2digi dump
process.jetdigi = cms.EDAnalyzer("JetToDigiDump", 
                                 DumpLevel = cms.string("Digis"),
                                 CaloJetAlg = cms.string("sisCone5CaloJets"),
                                 DebugLevel = cms.int32 (99),
                                 ShowECal = cms.bool (True)
                                 )

# Other statements
process.GlobalTag.globaltag = 'STARTUP_V5::All'

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.dump_step = cms.Path(process.dump)
process.jetdigi_step = cms.Path(process.jetdigi)
process.out_step = cms.EndPath(process.output)

# Schedule definition
#process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.dump_step,process.out_step)
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.jetdigi_step)
