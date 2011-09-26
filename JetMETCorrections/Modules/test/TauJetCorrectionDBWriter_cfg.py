import FWCore.ParameterSet.Config as cms 

process = cms.Process('TauJecSQLliteWriter') 

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load('CondCore.DBCommon.CondDBCommon_cfi')

process.CondDBCommon.connect = 'sqlite_file:TauJec11_V1.db'

payloads = [
    # generic tau-jet energy corrections parameters,
    # not specific to any reconstructed tau decay mode 
    'AK5tauHPSlooseCombDBcorr',
    # tau-jet energy corrections parameters specific to one-prong, no pi0 decay mode
    'AK5tauHPSlooseCombDBcorrOneProng0Pi0',
    # tau-jet energy corrections parameters specific to one-prong, one pi0 decay mode
    'AK5tauHPSlooseCombDBcorrOneProng1Pi0',
    # tau-jet energy corrections parameters specific to one-prong, two pi0 decay mode
    'AK5tauHPSlooseCombDBcorrOneProng2Pi0',
    # tau-jet energy corrections parameters specific to three-prong, no pi0 decay mode
    'AK5tauHPSlooseCombDBcorrThreeProng0Pi0'
]    

process.dbWriterSequence = cms.Sequence()

PoolDBOutputService_toPut = []

for payload in payloads:
    dbWriter = cms.EDAnalyzer('JetCorrectorDBWriter', 
        era  = cms.untracked.string('TauJec11V1'), 
        algo = cms.untracked.string(payload) 
    )
    dbWriterName = "dbWriter%s" % payload
    setattr(process, dbWriterName, dbWriter)
    process.dbWriterSequence += getattr(process, dbWriterName)

    PoolDBOutputService_toPut.append(cms.PSet(
        record = cms.string(payload),
        tag    = cms.string('JetCorrectorParametersCollection_TauJec11_V1_%s' % payload), 
        label  = cms.string(payload)
    ))

process.PoolDBOutputService = cms.Service('PoolDBOutputService', 
    process.CondDBCommon, 
    toPut = cms.VPSet(PoolDBOutputService_toPut)
) 

process.p = cms.Path(process.dbWriterSequence)
