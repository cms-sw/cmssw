import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var,P3Vars

hbheRecHitTable = cms.EDProducer("HBHERecHitFlatTableProducer",
    src       = cms.InputTag("hbhereco"),
    cut       = cms.string(""), 
    name      = cms.string("RecHitHBHE"),
    doc       = cms.string("HCAL barrel and endcap rec hits"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the object
    variables = cms.PSet(
                    detId  = Var('detid().rawId()', 'int', precision=-1, doc='detId'),
                    energy = Var('energy', 'float', precision=14, doc='energy'),
                    time   = Var('time', 'float', precision=14, doc='hit time'),
                    ieta   = Var('id().ieta()', 'int', precision=-1, doc='ieta'),
                    iphi   = Var('id().iphi()', 'int', precision=-1, doc='iphi'),
                    depth  = Var('id().depth()', 'int', precision=-1, doc='depth')
                )
)

hfRecHitTable = cms.EDProducer("HFRecHitFlatTableProducer",
    src       = cms.InputTag("hfreco"),
    cut       = cms.string(""), 
    name      = cms.string("RecHitHF"),
    doc       = cms.string("HCAL forward (HF) rec hits"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the object
    variables = cms.PSet(
                    detId  = Var('detid().rawId()', 'int', precision=-1, doc='detId'),
                    energy = Var('energy', 'float', precision=14, doc='energy'),
                    time   = Var('time', 'float', precision=14, doc='hit time'),
                    ieta   = Var('id().ieta()', 'int', precision=-1, doc='ieta'),
                    iphi   = Var('id().iphi()', 'int', precision=-1, doc='iphi'),
                    depth  = Var('id().depth()', 'int', precision=-1, doc='depth')
                )
)

hoRecHitTable = cms.EDProducer("HORecHitFlatTableProducer",
    src       = cms.InputTag("horeco"),
    cut       = cms.string(""), 
    name      = cms.string("RecHitHO"),
    doc       = cms.string("HCAL outer (HO) rec hits"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the object
    variables = cms.PSet(
                    detId  = Var('detid().rawId()', 'int', precision=-1, doc='detId'),
                    energy = Var('energy', 'float', precision=14, doc='energy'),
                    time   = Var('time', 'float', precision=14, doc='hit time'),
                    ieta   = Var('id().ieta()', 'int', precision=-1, doc='ieta'),
                    iphi   = Var('id().iphi()', 'int', precision=-1, doc='iphi'),
                    depth  = Var('id().depth()', 'int', precision=-1, doc='depth')
                )
)

hcalRecHitTableSeq = cms.Sequence(
    hbheRecHitTable
    + hfRecHitTable
    + hoRecHitTable
)

hcalRecHitTableTask = cms.Task(
    hbheRecHitTable,
    hfRecHitTable,
    hoRecHitTable,
)
