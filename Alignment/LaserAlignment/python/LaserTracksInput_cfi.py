import FWCore.ParameterSet.Config as cms

# use event sample containing laser tracks
#
source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(10),
    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring('castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.0.SIM-DIGI-RECO.root', 
        'castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.1.SIM-DIGI-RECO.root', 
        'castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.2.SIM-DIGI-RECO.root', 
        'castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.3.SIM-DIGI-RECO.root', 
        'castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.4.SIM-DIGI-RECO.root', 
        'castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.6.SIM-DIGI-RECO.root', 
        'castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.7.SIM-DIGI-RECO.root', 
        'castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.8.SIM-DIGI-RECO.root', 
        'castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.9.SIM-DIGI-RECO.root', 
        'castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.11.SIM-DIGI-RECO.root', 
        'castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.12.SIM-DIGI-RECO.root', 
        'castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.13.SIM-DIGI-RECO.root', 
        'castor:/castor/cern.ch/user/m/mthomas/CMSSW_1_3_1/LaserAlignment/LaserAlignmentAnalytical.16.SIM-DIGI-RECO.root')
)

maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

