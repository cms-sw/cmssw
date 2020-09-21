import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

protonTable = cms.EDProducer("ProtonProducer",
                             tagRecoProtonsSingle = cms.InputTag("ctppsProtons", "singleRP"),
                             tagRecoProtonsMulti  = cms.InputTag("ctppsProtons", "multiRP"),
                             tagTrackLite         = cms.InputTag("ctppsLocalTrackLiteProducer")
)

singleRPTable = cms.EDProducer("SimpleProtonTrackFlatTableProducer",
    src = cms.InputTag("ctppsProtons","singleRP"),
    cut = cms.string(""),
    name = cms.string("Proton_singleRP"),
    doc  = cms.string("bon"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        xi = Var("xi",float,doc="xi or dp/p",precision=12),
        xiUnc = Var("xiError",float,doc="uncertainty on xi or dp/p",precision=8),
        thetaY = Var("thetaY",float,doc="th y",precision=13),
        thetaYUnc = Var("thetaYError",float,doc="theta Y uncertainty",precision=8),
    ),
    externalVariables = cms.PSet(
        decRPId = ExtVar("protonTable:protonRPId",int,doc="Detector ID",precision=8), 
        sector45 = ExtVar("protonTable:singleRPsector45",bool,doc="LHC sector 45"),
        sector56 = ExtVar("protonTable:singleRPsector56",bool,doc="LHC sector 56"),
    ),
)

multiRPTable = cms.EDProducer("SimpleProtonTrackFlatTableProducer",
    src = cms.InputTag("ctppsProtons","multiRP"),
    cut = cms.string(""),
    name = cms.string("Proton_multiRP"),
    doc  = cms.string("bon"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        xi = Var("xi",float,doc="xi or dp/p",precision=12),
        xiUnc = Var("xiError",float,doc="uncertainty on xi or dp/p",precision=8),
        vtxY = Var("vy()",float,doc="vertex y",precision=11),
        vtxYUnc = Var("vyError",float,doc="vy Error",precision=8),
        pt = Var("pt",float,doc="pt",precision=15),
        thetaX = Var("thetaX",float,doc="theta x",precision=13),
        thetaXUnc = Var("thetaXError",float,doc="theta X uncertainty",precision=8),
        thetaY = Var("thetaY",float,doc="theta y",precision=13),
        thetaYUnc = Var("thetaYError",float,doc="theta Y uncertainty",precision=8),
        chi2 = Var("chi2",float,doc="chi 2",precision=15),
        ndof = Var("ndof()",int, doc="n dof", precision=8),
        t = Var("t",float,doc="t",precision=13),
        validFit = Var("validFit",bool,doc="valid Fit"),
        time = Var("time()",float,doc="time",precision=16),
        timeUnc = Var("timeError",float,doc="time uncertainty",precision=13),
    ),
    externalVariables = cms.PSet(
        sector45 = ExtVar("protonTable:multiRPsector45",bool,doc="LHC sector 45"),
        sector56 = ExtVar("protonTable:multiRPsector56",bool,doc="LHC sector 56"),
    ),
)

protonTables = cms.Sequence(    
    protonTable
    +singleRPTable
    +multiRPTable
)
