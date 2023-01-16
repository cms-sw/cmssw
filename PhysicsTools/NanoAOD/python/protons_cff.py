import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.genProtonTable_cfi import genProtonTable as _genproton
from PhysicsTools.NanoAOD.simpleProtonTrackFlatTableProducer_cfi import simpleProtonTrackFlatTableProducer

singleRPProtons = True

protonTable = cms.EDProducer("ProtonProducer",
                             tagRecoProtonsSingle = cms.InputTag("ctppsProtons", "singleRP"),
                             tagRecoProtonsMulti  = cms.InputTag("ctppsProtons", "multiRP"),
                             tagTrackLite         = cms.InputTag("ctppsLocalTrackLiteProducer"),
                             storeSingleRPProtons = cms.bool(singleRPProtons)
)

multiRPTable = simpleProtonTrackFlatTableProducer.clone(
    src = cms.InputTag("ctppsProtons","multiRP"),
    name = cms.string("Proton_multiRP"),
    doc  = cms.string("bon"),
    skipNonExistingSrc = cms.bool(True),#is this safe?
    variables = cms.PSet(
        xi = Var("xi",float,doc="xi or dp/p",precision=12),
        thetaX = Var("thetaX",float,doc="theta x",precision=13),
        thetaY = Var("thetaY",float,doc="theta y",precision=13),
        t = Var("t",float,doc="Mandelstam variable t",precision=13),
        time = Var("time()",float,doc="time",precision=16),
        timeUnc = Var("timeError",float,doc="time uncertainty",precision=13),
    ),
    externalVariables = cms.PSet(
        arm = ExtVar("protonTable:arm", "uint8", doc="0 = sector45, 1 = sector56"),
    ),
)

singleRPTable = simpleProtonTrackFlatTableProducer.clone(
    src = cms.InputTag("ctppsProtons","singleRP"),
    name = cms.string("Proton_singleRP"),
    doc  = cms.string("bon"),
    skipNonExistingSrc = cms.bool(True),#is this safe?
    variables = cms.PSet(
        xi = Var("xi",float,doc="xi or dp/p",precision=12),
        thetaY = Var("thetaY",float,doc="th y",precision=10),
    ),
    externalVariables = cms.PSet(
        decRPId = ExtVar("protonTable:protonRPId", "int16",doc="Detector ID"),
    ),
)

protonTablesTask = cms.Task(protonTable,multiRPTable)
if singleRPProtons: protonTablesTask.add(singleRPTable)

# GEN-level signal/PU protons collection
genProtonTable = _genproton.clone(
    cut = cms.string('(pdgId == 2212) && (abs(pz) > 5200) && (abs(pz) < 6467.5)') # xi in [0.015, 0.2]
)

genProtonTablesTask = cms.Task(genProtonTable)
