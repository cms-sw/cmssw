import FWCore.ParameterSet.Config as cms 
process = cms.Process('jerdb')

process.load('CondCore.DBCommon.CondDBCommon_cfi')
process.CondDBCommon.connect = 'sqlite_file:Summer15_V0_MC_JER.db'
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source('EmptySource')
process.PoolDBOutputService = cms.Service('PoolDBOutputService',
        process.CondDBCommon,
        toPut = cms.VPSet(
            cms.PSet(
                record = cms.string('Summer15_V0_MC_JER_AK4PFchs'),
                tag    = cms.string('JetResolutionObject_Summer15_V0_MC_JER_AK4PFchs'),
                label  = cms.string('AK4PFchs')
                ),
            cms.PSet(
                record = cms.string('Summer12_V1_MC_JER_SF_AK5PFchs'),
                tag    = cms.string('JetResolutionObject_Summer12_V1_MC_JER_SF_AK5PFchs'),
                label  = cms.string('AK5PFchs')
                ),
            )
        )

process.dbWriterAK4PF_JER = cms.EDAnalyzer('JetResolutionDBWriter',
        record = cms.untracked.string('Summer15_V0_MC_JER_AK4PFchs'),
        file = cms.untracked.FileInPath('CondFormats/JetMETObjects/data/Summer15_V0_MC_JER_AK4PFchs.txt')
        )

process.dbWriterAK4PF_JER_SF = cms.EDAnalyzer('JetResolutionDBWriter',
        record = cms.untracked.string('Summer12_V1_MC_JER_SF_AK5PFchs'),
        file = cms.untracked.FileInPath('CondFormats/JetMETObjects/data/Summer12_V1_MC_JER_SF_AK5PFchs.txt')
        )

process.p = cms.Path(process.dbWriterAK4PF_JER * process.dbWriterAK4PF_JER_SF)
