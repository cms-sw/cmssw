import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *



##################### User floats producers, selectors ##########################
## this can be merged with chsFor soft activity if we keep the same selection
chsForTkMet = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string('charge()!=0 && pvAssociationQuality()>=5 && vertexRef().key()==0'))
tkMet = cms.EDProducer("PFMETProducer",
    src = cms.InputTag("chsForTkMet"),
    alias = cms.string('tkMet'),
    globalThreshold = cms.double(0.0),
    calculateSignificance = cms.bool(False),
)



##################### Tables for final output and docs ##########################
metTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("slimmedMETs"),
    name = cms.string("MET"),
    doc = cms.string("slimmedMET, type-1 corrected PF MET"),
    singleton = cms.bool(True),  # there's always exactly one MET per event
    extension = cms.bool(False), # this is the main table for the MET
    variables = cms.PSet(PTVars,
       sumEt = Var("sumEt()", float, doc="scalar sum of Et",precision=10),
       covXX = Var("getSignificanceMatrix().At(0,0)",float,doc="xx element of met covariance matrix", precision=8),
       covXY = Var("getSignificanceMatrix().At(0,1)",float,doc="xy element of met covariance matrix", precision=8),
       covYY = Var("getSignificanceMatrix().At(1,1)",float,doc="yy element of met covariance matrix", precision=8),
       significance = Var("metSignificance()", float, doc="MET significance",precision=10),
       MetUnclustEnUpDeltaX = Var("shiftedPx('UnclusteredEnUp')-px()", float, doc="Delta (METx_mod-METx) Unclustered Energy Up",precision=10),
       MetUnclustEnUpDeltaY = Var("shiftedPy('UnclusteredEnUp')-py()", float, doc="Delta (METy_mod-METy) Unclustered Energy Up",precision=10),

    ),
)


rawMetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = metTable.src,
    name = cms.string("RawMET"),
    doc = cms.string("raw PF MET"),
    singleton = cms.bool(True),  # there's always exactly one MET per event
    extension = cms.bool(False), # this is the main table for the MET
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
       pt  = Var("uncorPt",  float, doc="pt", precision=10),
       phi = Var("uncorPhi", float, doc="phi", precision=10),
       sumEt = Var("uncorSumEt", float, doc="scalar sum of Et", precision=10),
    ),
)


caloMetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = metTable.src,
    name = cms.string("CaloMET"),
    doc = cms.string("Offline CaloMET (muon corrected)"),
    singleton = cms.bool(True),  # there's always exactly one MET per event
    extension = cms.bool(False), # this is the main table for the MET
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
       pt  = Var("caloMETPt",  float, doc="pt", precision=10),
       phi = Var("caloMETPhi", float, doc="phi", precision=10),
       sumEt = Var("caloMETSumEt", float, doc="scalar sum of Et", precision=10),
    ),
)

puppiMetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("slimmedMETsPuppi"),
    name = cms.string("PuppiMET"),
    doc = cms.string("PUPPI  MET"),
    singleton = cms.bool(True),  # there's always exactly one MET per event
    extension = cms.bool(False), # this is the main table for the MET
    variables = cms.PSet(PTVars,
       sumEt = Var("sumEt()", float, doc="scalar sum of Et",precision=10),
    ),
)

tkMetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("tkMet"),
    name = cms.string("TkMET"),
    doc = cms.string("Track MET computed with tracks from PV0 ( pvAssociationQuality()>=5 ) "),
    singleton = cms.bool(True),  # there's always exactly one MET per event
    extension = cms.bool(False), # this is the main table for the TkMET
    variables = cms.PSet(PTVars,
       sumEt = Var("sumEt()", float, doc="scalar sum of Et",precision=10),
    ),
)


metMCTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = metTable.src,
    name = cms.string("GenMET"),
    doc = cms.string("Gen MET"),
    singleton = cms.bool(True),  
    extension = cms.bool(False),
    variables = cms.PSet(
       pt  = Var("genMET.pt",  float, doc="pt", precision=10),
       phi = Var("genMET.phi", float, doc="phi", precision=10),
    ),
)



metSequence = cms.Sequence(chsForTkMet+tkMet)
metTables = cms.Sequence( metTable + rawMetTable + caloMetTable + puppiMetTable + tkMetTable)
metMC = cms.Sequence( metMCTable )

