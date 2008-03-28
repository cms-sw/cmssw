import FWCore.ParameterSet.Config as cms

L1DBParams = cms.PSet(
    validItems = cms.VPSet(cms.PSet(
        record = cms.string('L1JetEtScaleRcd'),
        data = cms.vstring('L1CaloEtScale')
    )),
    catalog = cms.string('file:test.xml'),
    # What tag to use to load main L1TriggerKey
    tag = cms.string('current'),
    # DB connection
    connect = cms.string('sqlite_file:l1config.db')
)

