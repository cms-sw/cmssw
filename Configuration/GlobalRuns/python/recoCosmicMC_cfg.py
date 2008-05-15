# The following comments couldn't be translated into the new config version:

#    service = Tracer {}

import FWCore.ParameterSet.Config as cms

process = cms.Process("Rec3")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

# service = Timing {}
# use Fake cond for MC
process.load("Configuration.StandardSequences.FakeConditions_cff")

# Conditions (Global Tag is used here):
#include "Configuration/GlobalRuns/data/FrontierConditionsGRGlobalTag.cff"
#replace GlobalTag.connect = "oracle://cms_orcoff_int2r/CMS_COND_GENERAL"
#replace GlobalTag.DBParameters.authenticationPath="/afs/cern.ch/cms/DB/conddb"
#replace GlobalTag.timetype = "runnumber"
#replace GlobalTag.globaltag = "GRUMM_V4::All"
# Magnetic fiuld: force mag field to be 0.0 tesla
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

# reconstruction sequence for Global Run
process.load("Configuration.GlobalRuns.ReconstructionGR_cff")

# offline raw to digi for real data
#include "Configuration/GlobalRuns/data/RawToDigiGR.cff"
# offline raw to digi for MC
process.load("Configuration.StandardSequences.RawToDigi_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.Timing = cms.Service("Timing")

process.source = cms.Source("PoolSource",
    #       untracked vstring fileNames = {'file:GlobalMar08_37965_A.root'}
    fileNames = cms.untracked.vstring('file:gen_sim_digi_raw_100k_200pre9.root')
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO')
    ),
    fileName = cms.untracked.string('reco-gr.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false

)
process.allPath = cms.Path(process.RawToDigi*process.reconstructionGR)
process.outpath = cms.EndPath(process.FEVT)

