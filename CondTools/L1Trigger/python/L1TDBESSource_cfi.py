import FWCore.ParameterSet.Config as cms

from CondTools.L1Trigger.L1DBParams_cfi import *
L1TDBESSource = cms.ESSource("L1TDBESSource",
    # Here you should specify what data to put where. Similar as in
    # PoolDBESS ource. You can also put L1TriggerKeyRcd with value L1TriggerKey, but
    # system will load it even if you do not put it here.
    # In other words, putting key record here is just a waste of keystrokes, but you can do that
    toLoad = cms.VPSet(),
    catalog = cms.string('file:test.xml'),
    # What tag to use to load main L1TriggerKey
    tag = cms.string('current'),
    # Connection strings
    connect = cms.string('sqlite_file:l1config.db')
)

L1TDBESSource.toLoad.extend(L1DBParams.validItems)

