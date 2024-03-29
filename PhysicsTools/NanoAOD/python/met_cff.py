import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleSingletonCandidateFlatTableProducer_cfi import simpleSingletonCandidateFlatTableProducer

##################### Tables for final output and docs ##########################
pfmetTable = simpleSingletonCandidateFlatTableProducer.clone(
    src = cms.InputTag("slimmedMETs"),
    name = cms.string("PFMET"),
    doc = cms.string("slimmedMET, type-1 corrected PF MET"),
    variables = cms.PSet(PTVars,
       covXX = Var("getSignificanceMatrix().At(0,0)",float,doc="xx element of met covariance matrix", precision=8, lazyEval=True),
       covXY = Var("getSignificanceMatrix().At(0,1)",float,doc="xy element of met covariance matrix", precision=8, lazyEval=True),
       covYY = Var("getSignificanceMatrix().At(1,1)",float,doc="yy element of met covariance matrix", precision=8, lazyEval=True),
       significance = Var("metSignificance()", float, doc="MET significance",precision=10, lazyEval=True),
       sumEt = Var("sumEt()", float, doc="scalar sum of Et",precision=10, lazyEval=True), 
       sumPtUnclustered = Var("metSumPtUnclustered()", float, doc="sumPt used for MET significance",precision=10, lazyEval=True),
       ptUnclusteredUp = Var("shiftedPt('UnclusteredEnUp')", float, doc="Unclustered up pt",precision=10, lazyEval=True),
       ptUnclusteredDown = Var("shiftedPt('UnclusteredEnDown')", float, doc="Unclustered down pt",precision=10, lazyEval=True),
       phiUnclusteredUp = Var("shiftedPhi('UnclusteredEnUp')", float, doc="Unclustered up phi",precision=10, lazyEval=True),
       phiUnclusteredDown = Var("shiftedPhi('UnclusteredEnDown')", float, doc="Unclustered down phi",precision=10, lazyEval=True),
    ),
)


rawMetTable = simpleSingletonCandidateFlatTableProducer.clone(
    src = pfmetTable.src,
    name = cms.string("RawPFMET"),
    doc = cms.string("raw PF MET"),
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
       pt  = Var("uncorPt",  float, doc="pt", precision=10, lazyEval=True),
       phi = Var("uncorPhi", float, doc="phi", precision=10, lazyEval=True),
       sumEt = Var("uncorSumEt", float, doc="scalar sum of Et", precision=10, lazyEval=True),
    ),
)


caloMetTable = simpleSingletonCandidateFlatTableProducer.clone(
    src = pfmetTable.src,
    name = cms.string("CaloMET"),
    doc = cms.string("Offline CaloMET (muon corrected)"),
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
       pt  = Var("caloMETPt",  float, doc="pt", precision=10, lazyEval=True),
       phi = Var("caloMETPhi", float, doc="phi", precision=10, lazyEval=True),
       sumEt = Var("caloMETSumEt", float, doc="scalar sum of Et", precision=10, lazyEval=True),
    ),
)

puppiMetTable = simpleSingletonCandidateFlatTableProducer.clone(
    src = cms.InputTag("slimmedMETsPuppi"),
    name = cms.string("PuppiMET"),
    doc = cms.string("PUPPI  MET"),
    variables = cms.PSet(PTVars,
       covXX = Var("getSignificanceMatrix().At(0,0)",float,doc="xx element of met covariance matrix", precision=8, lazyEval=True),
       covXY = Var("getSignificanceMatrix().At(0,1)",float,doc="xy element of met covariance matrix", precision=8, lazyEval=True),
       covYY = Var("getSignificanceMatrix().At(1,1)",float,doc="yy element of met covariance matrix", precision=8, lazyEval=True),
       significance = Var("metSignificance()", float, doc="MET significance",precision=10, lazyEval=True),
       sumEt = Var("sumEt()", float, doc="scalar sum of Et",precision=10, lazyEval=True),
       sumPtUnclustered = Var("metSumPtUnclustered()", float, doc="sumPt used for MET significance",precision=10, lazyEval=True),
       ptUnclusteredUp = Var("shiftedPt('UnclusteredEnUp')", float, doc="Unclustered up pt",precision=10, lazyEval=True),
       ptUnclusteredDown = Var("shiftedPt('UnclusteredEnDown')", float, doc="Unclustered down pt",precision=10, lazyEval=True),
       phiUnclusteredUp = Var("shiftedPhi('UnclusteredEnUp')", float, doc="Unclustered up phi",precision=10, lazyEval=True),
       phiUnclusteredDown = Var("shiftedPhi('UnclusteredEnDown')", float, doc="Unclustered down phi",precision=10, lazyEval=True),
    ),
)

rawPuppiMetTable = simpleSingletonCandidateFlatTableProducer.clone(
    src = puppiMetTable.src,
    name = cms.string("RawPuppiMET"),
    doc = cms.string("raw Puppi MET"),
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
       pt  = Var("uncorPt",  float, doc="pt", precision=10, lazyEval=True),
       phi = Var("uncorPhi", float, doc="phi", precision=10, lazyEval=True),
       sumEt = Var("uncorSumEt", float, doc="scalar sum of Et", precision=10, lazyEval=True),
    ),)


trkMetTable = simpleSingletonCandidateFlatTableProducer.clone(
    src = pfmetTable.src,
    name = cms.string("TrkMET"),
    doc = cms.string("Track MET computed with tracks from PV0 ( pvAssociationQuality()>=4 ) "),
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
       pt = Var("corPt('RawTrk')", float, doc="raw track MET pt",precision=10, lazyEval=True),
       phi = Var("corPhi('RawTrk')", float, doc="raw track MET phi",precision=10, lazyEval=True),
       sumEt = Var("corSumEt('RawTrk')", float, doc="raw track scalar sum of Et",precision=10, lazyEval=True),
    ),
)

deepMetResolutionTuneTable = simpleSingletonCandidateFlatTableProducer.clone(
    # current deepMets are saved in slimmedMETs in MiniAOD,
    # in the same way as chsMet/TrkMET
    src = pfmetTable.src,
    name = cms.string("DeepMETResolutionTune"),
    doc = cms.string("Deep MET trained with resolution tune"),
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
        pt = Var("corPt('RawDeepResolutionTune')", float, doc="DeepMET ResolutionTune pt",precision=-1, lazyEval=True),
        phi = Var("corPhi('RawDeepResolutionTune')", float, doc="DeepmET ResolutionTune phi",precision=12, lazyEval=True),
    ),
)

deepMetResponseTuneTable = simpleSingletonCandidateFlatTableProducer.clone(
    src = pfmetTable.src,
    name = cms.string("DeepMETResponseTune"),
    doc = cms.string("Deep MET trained with extra response tune"),
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
        pt = Var("corPt('RawDeepResponseTune')", float, doc="DeepMET ResponseTune pt",precision=-1, lazyEval=True),
        phi = Var("corPhi('RawDeepResponseTune')", float, doc="DeepMET ResponseTune phi",precision=12, lazyEval=True),
    ),
)

metMCTable = simpleSingletonCandidateFlatTableProducer.clone(
    src = pfmetTable.src,
    name = cms.string("GenMET"),
    doc = cms.string("Gen MET"),
    variables = cms.PSet(
       pt  = Var("genMET.pt",  float, doc="pt", precision=10, lazyEval=True),
       phi = Var("genMET.phi", float, doc="phi", precision=10, lazyEval=True),
    ),
)


metTablesTask = cms.Task(pfmetTable, rawMetTable, caloMetTable, puppiMetTable, rawPuppiMetTable, trkMetTable, 
        deepMetResolutionTuneTable, deepMetResponseTuneTable )
metMCTask = cms.Task( metMCTable )
