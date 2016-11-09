import FWCore.ParameterSet.Config as cms

process = cms.Process("JERDBLocalReader")

process.load('Configuration.StandardSequences.Services_cff')
process.load("JetMETCorrections.Modules.JetResolutionESProducer_cfi")

from CondCore.DBCommon.CondDBSetup_cfi import *

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.source = cms.Source("EmptySource")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
        CondDBSetup,
        toGet = cms.VPSet(
            # Resolution
            cms.PSet(
                record = cms.string('JetResolutionRcd'),
                tag    = cms.string('JetResolutionObject_Summer15_V0_MC_JER_AK4PFchs'),
                label  = cms.untracked.string('AK4PFchs')
                ),

            # Scale factors
            cms.PSet(
                record = cms.string('JetResolutionScaleFactorRcd'),
                tag    = cms.string('JetResolutionObject_Summer12_V1_MC_JER_SF_AK5PFchs'),
                label  = cms.untracked.string('AK5PFchs')
                ),
            ),
        connect = cms.string('sqlite:Summer15_V0_MC_JER.db')
        )


process.demo1 = cms.EDAnalyzer('JetResolutionDBReader', 
        era = cms.untracked.string('Summer15_V0_MC_JER'),
        label = cms.untracked.string('AK4PFchs'),
        dump = cms.untracked.bool(True),
        saveFile = cms.untracked.bool(True)
        )

process.demo2 = cms.EDAnalyzer('JetResolutionScaleFactorDBReader', 
        era = cms.untracked.string('Summer12_V1_MC_JER_SF'),
        label = cms.untracked.string('AK5PFchs'),
        dump = cms.untracked.bool(True),
        saveFile = cms.untracked.bool(True)
        )

process.p = cms.Path(process.demo1 * process.demo2)
