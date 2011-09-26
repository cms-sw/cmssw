import FWCore.ParameterSet.Config as cms

process = cms.Process("TauJecSQLliteReader")

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START42_V13::All'

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

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

process.dbReaderSequence = cms.Sequence()

PoolDBESSource_toGet = []

for payload in payloads:
    dbReader = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string(payload),
        globalTag      = cms.untracked.string('START42_V13'),  
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(True)
    )
    dbReaderName = "dbReader%s" % payload
    setattr(process, dbReaderName, dbReader)
    process.dbReaderSequence += getattr(process, dbReaderName)

    PoolDBESSource_toGet.append(cms.PSet(
        record = cms.string('JetCorrectionsRecord'),
        tag    = cms.string('JetCorrectorParametersCollection_TauJec11_V1_%s' % payload),
        label  = cms.untracked.string(payload)
    ))

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.SQLliteInput = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(PoolDBESSource_toGet),
    connect = cms.string('sqlite_fip:JetMETCorrections/Modules/test/TauJec11_V1.db')
)
process.es_prefer_jec = cms.ESPrefer('PoolDBESSource', 'SQLliteInput')

process.p = cms.Path(process.dbReaderSequence)
