import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) ) # only one event needed to extract trigger information (-1 for all events)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_2_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0033/BAA28B05-224F-E011-92F7-0026189438D3.root'
        #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_11_3/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v2/0007/A8FACACC-2F4F-E011-B535-002618943906.root'
        #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_1_3/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V2-v1/0037/D27F9CA1-C851-E011-92BC-003048679266.root'
        #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_2_0_pre5/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V3-v1/0009/1289680D-5D3C-E011-83B4-002618943894.root'
        #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_2_0_pre6/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V4-v1/0020/78B8F51E-5245-E011-BA20-003048679294.root'
        #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_2_0_pre8/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V7-v1/0042/58FD542D-8B56-E011-8AC3-003048678B38.root'
        'rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre1/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_42_V7-v1/0048/246521DE-F058-E011-9F36-002618943978.root'
    )
)

process.demo = cms.EDAnalyzer('EmDQMFeeder',
    processname = cms.string("HLT")
)


process.p = cms.Path(process.demo)
