import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.globalVariablesTableProducer_cfi import globalVariablesTableProducer
from PhysicsTools.NanoAOD.simpleBeamspotFlatTableProducer_cfi import simpleBeamspotFlatTableProducer
from PhysicsTools.NanoAOD.simpleGenEventFlatTableProducer_cfi import simpleGenEventFlatTableProducer
from PhysicsTools.NanoAOD.simpleGenFilterFlatTableProducerLumi_cfi import simpleGenFilterFlatTableProducerLumi

beamSpotTable = simpleBeamspotFlatTableProducer.clone(
    src = cms.InputTag("offlineBeamSpot"),
    name = cms.string("BeamSpot"),
    doc = cms.string("offlineBeamSpot, the offline reconstructed beamspot"),
    variables = cms.PSet(
       type = Var("type()","int8",doc="BeamSpot type (Unknown = -1, Fake = 0, LHC = 1, Tracker = 2)"),
       z = Var("position().z()",float,doc="BeamSpot center, z coordinate (cm)",precision=-1),
       zError = Var("z0Error()",float,doc="Error on BeamSpot center, z coordinate (cm)",precision=-1),
       sigmaZ = Var("sigmaZ()",float,doc="Width of BeamSpot in z (cm)",precision=-1),
       sigmaZError = Var("sigmaZ0Error()",float,doc="Error on width of BeamSpot in z (cm)",precision=-1),
    ),
)

rhoTable = globalVariablesTableProducer.clone(
    name = cms.string("Rho"),
    variables = cms.PSet(
        fixedGridRhoAll = ExtVar( cms.InputTag("fixedGridRhoAll"), "double", doc = "rho from all PF Candidates, no foreground removal (for isolation of prompt photons)" ),
        fixedGridRhoFastjetAll = ExtVar( cms.InputTag("fixedGridRhoFastjetAll"), "double", doc = "rho from all PF Candidates, used e.g. for JECs" ),
        fixedGridRhoFastjetCentralNeutral = ExtVar( cms.InputTag("fixedGridRhoFastjetCentralNeutral"), "double", doc = "rho from neutral PF Candidates with |eta| < 2.5, used e.g. for rho corrections of some lepton isolations" ),
        fixedGridRhoFastjetCentralCalo = ExtVar( cms.InputTag("fixedGridRhoFastjetCentralCalo"), "double", doc = "rho from calo towers with |eta| < 2.5, used e.g. egamma PFCluster isolation" ),
        fixedGridRhoFastjetCentral = ExtVar( cms.InputTag("fixedGridRhoFastjetCentral"), "double", doc = "rho from all PF Candidates for central region, used e.g. for JECs" ),
        fixedGridRhoFastjetCentralChargedPileUp = ExtVar( cms.InputTag("fixedGridRhoFastjetCentralChargedPileUp"), "double", doc = "rho from charged PF Candidates for central region, used e.g. for JECs" ),
    )
)

puTable = cms.EDProducer("NPUTablesProducer",
        src = cms.InputTag("slimmedAddPileupInfo"),
        pvsrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
        zbins = cms.vdouble( [0.0,1.7,2.6,3.0,3.5,4.2,5.2,6.0,7.5,9.0,12.0] ),
        savePtHatMax = cms.bool(False),
)

genTable  = simpleGenEventFlatTableProducer.clone(
    src = cms.InputTag("generator"),
    name= cms.string("Generator"),
    doc = cms.string("Generator information"),
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

genFilterTable = simpleGenFilterFlatTableProducerLumi.clone(
    src = cms.InputTag("genFilterEfficiencyProducer"),
    name= cms.string("GenFilter"),
    doc = cms.string("Generator filter information"),
    variables = cms.PSet(
        numEventsTotal        = Var("numEventsTotal()",        int,   doc="generator filter: total number of events",  precision=6),
        numEventsPassed       = Var("numEventsPassed()",       int,   doc="generator filter: passed number of events", precision=6),
        filterEfficiency      = Var("filterEfficiency()",      float, doc="generator filter: efficiency",              precision=14),
        filterEfficiencyError = Var("filterEfficiencyError()", float, doc="generator filter: efficiency error",        precision=14),
    ),
)

globalTablesTask = cms.Task(beamSpotTable, rhoTable)
globalTablesMCTask = cms.Task(puTable,genTable,genFilterTable)
