import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.genProtonTable_cfi import genProtonTable as _genproton
from PhysicsTools.NanoAOD.nano_eras_cff import *
from RecoPPS.ProtonReconstruction.ppsFilteredProtonProducer_cfi import *

singleRPProtons = True

filteredProtons = ppsFilteredProtonProducer.clone(
    protons_single_rp = cms.PSet(
        include = cms.bool(singleRPProtons)
    )
)

protonTable = cms.EDProducer("ProtonProducer",
                             tagRecoProtonsMulti  = cms.InputTag("filteredProtons", "multiRP"),
                             tagTrackLite         = cms.InputTag("ctppsLocalTrackLiteProducer"),
                             storeSingleRPProtons = cms.bool(singleRPProtons)
)
protonTable.tagRecoProtonsSingle = cms.InputTag("filteredProtons" if singleRPProtons else "ctppsProtons","singleRP")


multiRPTable = cms.EDProducer("SimpleProtonTrackFlatTableProducer",
    src = cms.InputTag("filteredProtons","multiRP"),
    cut = cms.string(""),
    name = cms.string("Proton_multiRP"),
    doc  = cms.string("bon"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    skipNonExistingSrc = cms.bool(True),
    variables = cms.PSet(
        xi = Var("xi",float,doc="xi or dp/p",precision=12),
        thetaX = Var("thetaX",float,doc="theta x",precision=13),
        thetaY = Var("thetaY",float,doc="theta y",precision=13),
        t = Var("t",float,doc="Mandelstam variable t",precision=13),
        time = Var("time()",float,doc="time",precision=16),
        timeUnc = Var("timeError",float,doc="time uncertainty",precision=13),
    ),
    externalVariables = cms.PSet(
        arm = ExtVar("protonTable:arm",int,doc="0 = sector45, 1 = sector56"),
    ),
)

singleRPTable = cms.EDProducer("SimpleProtonTrackFlatTableProducer",
    src = cms.InputTag("filteredProtons","singleRP"),
    cut = cms.string(""),
    name = cms.string("Proton_singleRP"),
    doc  = cms.string("bon"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    skipNonExistingSrc = cms.bool(True),
    variables = cms.PSet(
        xi = Var("xi",float,doc="xi or dp/p",precision=12),
        thetaY = Var("thetaY",float,doc="th y",precision=10),
    ),
    externalVariables = cms.PSet(
        decRPId = ExtVar("protonTable:protonRPId",int,doc="Detector ID",precision=8),
    ),
)

protonTablesTask = cms.Task(filteredProtons,protonTable,multiRPTable)
if singleRPProtons: protonTablesTask.add(singleRPTable)

# GEN-level signal/PU protons collection
genProtonTable = _genproton.clone(
    cut = cms.string('(pdgId == 2212) && (abs(pz) > 5200) && (abs(pz) < 6467.5)') # xi in [0.015, 0.2]
)

genProtonTablesTask = cms.Task(genProtonTable)

for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2, run2_nanoAOD_94X2016, run2_nanoAOD_102Xv1:
    modifier.toReplaceWith(protonTablesTask, cms.Task())
    # input GEN-level PU protons collection only introduced for UL and 12_X_Y
    modifier.toReplaceWith(genProtonTablesTask, cms.Task())
