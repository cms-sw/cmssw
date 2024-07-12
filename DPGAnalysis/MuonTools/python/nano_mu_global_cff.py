import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import lhcInfoTable

lumiTable = cms.EDProducer("SimpleOnlineLuminosityFlatTableProducer",
    src = cms.InputTag("onlineMetaDataDigis"),
    name = cms.string("lumi"),
    doc  = cms.string("Online luminosity information"),
    variables = cms.PSet(
        instLumi = Var( "instLumi()", "double", doc = "Instantaneous luminosity"),
        avgPileUp = Var( "avgPileUp()", "double", doc = "Average PU")
    )
)

globalTables = cms.Sequence(lumiTable + lhcInfoTable)