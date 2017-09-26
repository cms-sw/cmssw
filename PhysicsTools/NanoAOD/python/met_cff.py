import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *



##################### User floats producers, selectors ##########################



##################### Tables for final output and docs ##########################
metTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("slimmedMETs"),
    name = cms.string("MET"),
    doc = cms.string("slimmedMET, type-1 corrected PF MET"),
    singleton = cms.bool(True),  # there's always exactly one MET per event
    extension = cms.bool(False), # this is the main table for the MET
    variables = cms.PSet(PTVars,
       sumEt = Var("sumEt()", float, doc="scalar sum of Et",precision=10),
       rawEt = Var("uncorPt", float, doc="raw (uncorrected) scalar sum of Et",precision=10),
       covXX = Var("getSignificanceMatrix().At(0,0)",float,doc="xx element of met covariance matrix", precision=8),
       covXY = Var("getSignificanceMatrix().At(0,1)",float,doc="xy element of met covariance matrix", precision=8),
       covYY = Var("getSignificanceMatrix().At(1,1)",float,doc="yy element of met covariance matrix", precision=8),
       significance = Var("metSignificance()", float, doc="MET significance",precision=10),

    ),
)

caloMetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = metTable.src,
    name = cms.string("METCalo"),
    doc = cms.string("Calo MET"),
    singleton = cms.bool(True),  # there's always exactly one MET per event
    extension = cms.bool(False), # this is the main table for the MET
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
       pt  = Var("caloMETPt",  float, precision=10),
       phi = Var("caloMETPhi", float, precision=10),
       sumEt = Var("caloMETSumEt", float, doc="scalar sum of Et", precision=10),
    ),
)

puppiMetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("slimmedMETsPuppi"),
    name = cms.string("METpuppi"),
    doc = cms.string("PUPPI  MET"),
    singleton = cms.bool(True),  # there's always exactly one MET per event
    extension = cms.bool(False), # this is the main table for the MET
    variables = cms.PSet(PTVars,
       sumEt = Var("sumEt()", float, doc="scalar sum of Et",precision=10),
    ),
)

metMCTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = metTable.src,
    name = cms.string("MET"),
    doc = cms.string("Calo MET"),
    singleton = cms.bool(True),  # there's always exactly one MET per event
    extension = cms.bool(True),  # we add variables to the MET
    variables = cms.PSet(
       genPt  = Var("genMET.pt",  float, precision=10),
       genPhi = Var("genMET.phi", float, precision=10),
    ),
)

#metSequence = cms.Sequence()
metTables = cms.Sequence( metTable + caloMetTable + puppiMetTable )
metMC = cms.Sequence( metMCTable )

