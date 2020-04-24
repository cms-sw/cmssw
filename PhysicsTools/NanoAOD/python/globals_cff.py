import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

rhoTable = cms.EDProducer("GlobalVariablesTableProducer",
    variables = cms.PSet(
        fixedGridRhoFastjetAll = ExtVar( cms.InputTag("fixedGridRhoFastjetAll"), "double", doc = "rho from all PF Candidates, used e.g. for JECs" ),
        fixedGridRhoFastjetCentralNeutral = ExtVar( cms.InputTag("fixedGridRhoFastjetCentralNeutral"), "double", doc = "rho from neutral PF Candidates with |eta| < 2.5, used e.g. for rho corrections of some lepton isolations" ),
        fixedGridRhoFastjetCentralCalo = ExtVar( cms.InputTag("fixedGridRhoFastjetCentralCalo"), "double", doc = "rho from calo towers with |eta| < 2.5, used e.g. egamma PFCluster isolation" ),
    )
)

puTable = cms.EDProducer("NPUTablesProducer",
        src = cms.InputTag("slimmedAddPileupInfo"),
)

genTable  = cms.EDProducer("SimpleGenEventFlatTableProducer",
        src = cms.InputTag("generator"),
        cut = cms.string(""), 
        name= cms.string("Generator"),
        doc = cms.string("Generator information"),
        singleton = cms.bool(True), 
        extension = cms.bool(False),
    variables = cms.PSet(
        x1 = Var( "?hasPDF?pdf().x.first:-1", float, doc="x1 fraction of proton momentum carried by the first parton",precision=14 ),
        x2 = Var( "?hasPDF?pdf().x.second:-1", float, doc="x2 fraction of proton momentum carried by the second parton",precision=14 ),
        xpdf1 = Var( "?hasPDF?pdf().xPDF.first:-1", float, doc="x*pdf(x) for the first parton", precision=14 ),
        xpdf2 = Var( "?hasPDF?pdf().xPDF.second:-1", float, doc="x*pdf(x) for the second parton", precision=14 ),
        id1 = Var( "?hasPDF?pdf().id.first:-1", int, doc="id of first parton", precision=6 ),
        id2 = Var( "?hasPDF?pdf().id.second:-1", int, doc="id of second parton", precision=6 ),
        scalePDF = Var( "?hasPDF?pdf().scalePDF:-1", float, doc="Q2 scale for PDF", precision=14 ),
        binvar = Var("?hasBinningValues()?binningValues()[0]:-1", float, doc="MC generation binning value", precision=14),
        weight = Var("weight()", float,doc="MC generator weight", precision=14),
        ),
)

globalTables = cms.Sequence(rhoTable)
globalTablesMC = cms.Sequence(puTable+genTable)
