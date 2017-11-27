import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

rhoTable = cms.EDProducer("GlobalVariablesTableProducer",
    variables = cms.PSet(
        fixedGridRhoFastjetAll = ExtVar( cms.InputTag("fixedGridRhoFastjetAll"), "double", doc = "rho from all PF Candidates, used e.g. for JECs" ),
        fixedGridRhoFastjetCentralNeutral = ExtVar( cms.InputTag("fixedGridRhoFastjetCentralNeutral"), "double", doc = "rho from neutral PF Candidates with |eta| < 2.5, used e.g. for rho corrections of some lepton isolations" ),
        fixedGridRhoFastjetCentralCalo = ExtVar( cms.InputTag("fixedGridRhoFastjetCentralCalo"), "double", doc = "rho from calo towers with |eta| < 2.5, used e.g. egamma PFCluster isolation" ),
    )
)

puTable = cms.EDProducer("SimplePileupFlatTableProducer",
        src = cms.InputTag("slimmedAddPileupInfo"),
        cut = cms.string("getBunchCrossing()==0"), # save only the pileup of the in-time bunch crossing
        name= cms.string("Pileup"),
        doc = cms.string("pileup information for bunch crossing 0"),
        singleton = cms.bool(False), # slimmedAddPileupInfo collection has all the BXs, but only BX=0 is saved
        extension = cms.bool(False),
    variables = cms.PSet(
        nTrueInt = Var( "getTrueNumInteractions()", int, doc="the true mean number of the poisson distribution for this event from which the number of interactions each bunch crossing has been sampled" ),
        nPU = Var( "getPU_NumInteractions()", int, doc="the number of pileup interactions that have been added to the event in the current bunch crossing" ),
        ),
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
	#AR: not sure what "xPDF" is wrt to just "x"
        #x1PDF = Var( "?hasPDF?pdf().xPDF.first:-1", float, doc="x1 fraction of proton momentum carried by the first parton" ),
        #x2PDF = Var( "?hasPDF?pdf().xPDF.second:-1", float, doc="x2 fraction of proton momentum carried by the second parton" ),
        ),
)

globalTables = cms.Sequence(rhoTable)
globalTablesMC = cms.Sequence(puTable+genTable)
