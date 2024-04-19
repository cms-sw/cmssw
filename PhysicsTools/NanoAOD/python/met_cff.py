import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simplePATMETFlatTableProducer_cfi import simplePATMETFlatTableProducer

simpleSingletonPATMETFlatTableProducer = simplePATMETFlatTableProducer.clone(
    singleton = cms.bool(True),
    cut = None,
    lazyEval = None
)

##################### Tables for final output and docs ##########################
pfmetTable = simpleSingletonPATMETFlatTableProducer.clone(
    src = cms.InputTag("slimmedMETs"),
    name = cms.string("PFMET"),
    doc = cms.string("slimmedMET, type-1 corrected PF MET"),
    variables = cms.PSet(PTVars,
       covXX = Var("getSignificanceMatrix().At(0,0)",float,doc="xx element of met covariance matrix", precision=8),
       covXY = Var("getSignificanceMatrix().At(0,1)",float,doc="xy element of met covariance matrix", precision=8),
       covYY = Var("getSignificanceMatrix().At(1,1)",float,doc="yy element of met covariance matrix", precision=8),
       significance = Var("metSignificance()", float, doc="MET significance",precision=10),
       sumEt = Var("sumEt()", float, doc="scalar sum of Et",precision=10), 
       sumPtUnclustered = Var("metSumPtUnclustered()", float, doc="sumPt used for MET significance",precision=10),
       ptUnclusteredUp = Var("shiftedPt('UnclusteredEnUp')", float, doc="Unclustered up pt",precision=10),
       ptUnclusteredDown = Var("shiftedPt('UnclusteredEnDown')", float, doc="Unclustered down pt",precision=10),
       phiUnclusteredUp = Var("shiftedPhi('UnclusteredEnUp')", float, doc="Unclustered up phi",precision=10),
       phiUnclusteredDown = Var("shiftedPhi('UnclusteredEnDown')", float, doc="Unclustered down phi",precision=10),
    ),
)


rawMetTable = simpleSingletonPATMETFlatTableProducer.clone(
    src = pfmetTable.src,
    name = cms.string("RawPFMET"),
    doc = cms.string("raw PF MET"),
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
       pt  = Var("uncorPt",  float, doc="pt", precision=10),
       phi = Var("uncorPhi", float, doc="phi", precision=10),
       sumEt = Var("uncorSumEt", float, doc="scalar sum of Et", precision=10),
    ),
)


caloMetTable = simpleSingletonPATMETFlatTableProducer.clone(
    src = pfmetTable.src,
    name = cms.string("CaloMET"),
    doc = cms.string("Offline CaloMET (muon corrected)"),
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
       pt  = Var("caloMETPt",  float, doc="pt", precision=10),
       phi = Var("caloMETPhi", float, doc="phi", precision=10),
       sumEt = Var("caloMETSumEt", float, doc="scalar sum of Et", precision=10),
    ),
)

puppiMetTable = simpleSingletonPATMETFlatTableProducer.clone(
    src = cms.InputTag("slimmedMETsPuppi"),
    name = cms.string("PuppiMET"),
    doc = cms.string("PUPPI  MET"),
    variables = cms.PSet(PTVars,
       covXX = Var("getSignificanceMatrix().At(0,0)",float,doc="xx element of met covariance matrix", precision=8),
       covXY = Var("getSignificanceMatrix().At(0,1)",float,doc="xy element of met covariance matrix", precision=8),
       covYY = Var("getSignificanceMatrix().At(1,1)",float,doc="yy element of met covariance matrix", precision=8),
       significance = Var("metSignificance()", float, doc="MET significance",precision=10),
       sumEt = Var("sumEt()", float, doc="scalar sum of Et",precision=10),
       sumPtUnclustered = Var("metSumPtUnclustered()", float, doc="sumPt used for MET significance",precision=10),
       ptUnclusteredUp = Var("shiftedPt('UnclusteredEnUp')", float, doc="Unclustered up pt",precision=10),
       ptUnclusteredDown = Var("shiftedPt('UnclusteredEnDown')", float, doc="Unclustered down pt",precision=10),
       phiUnclusteredUp = Var("shiftedPhi('UnclusteredEnUp')", float, doc="Unclustered up phi",precision=10),
       phiUnclusteredDown = Var("shiftedPhi('UnclusteredEnDown')", float, doc="Unclustered down phi",precision=10),
    ),
)

rawPuppiMetTable = simpleSingletonPATMETFlatTableProducer.clone(
    src = puppiMetTable.src,
    name = cms.string("RawPuppiMET"),
    doc = cms.string("raw Puppi MET"),
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
       pt  = Var("uncorPt",  float, doc="pt", precision=10),
       phi = Var("uncorPhi", float, doc="phi", precision=10),
       sumEt = Var("uncorSumEt", float, doc="scalar sum of Et", precision=10),
    ),)


trkMetTable = simpleSingletonPATMETFlatTableProducer.clone(
    src = pfmetTable.src,
    name = cms.string("TrkMET"),
    doc = cms.string("Track MET computed with tracks from PV0 ( pvAssociationQuality()>=4 ) "),
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
       pt = Var("corPt('RawTrk')", float, doc="raw track MET pt",precision=10),
       phi = Var("corPhi('RawTrk')", float, doc="raw track MET phi",precision=10),
       sumEt = Var("corSumEt('RawTrk')", float, doc="raw track scalar sum of Et",precision=10),
    ),
)

deepMetResolutionTuneTable = simpleSingletonPATMETFlatTableProducer.clone(
    # current deepMets are saved in slimmedMETs in MiniAOD,
    # in the same way as chsMet/TrkMET
    src = pfmetTable.src,
    name = cms.string("DeepMETResolutionTune"),
    doc = cms.string("Deep MET trained with resolution tune"),
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
        pt = Var("corPt('RawDeepResolutionTune')", float, doc="DeepMET ResolutionTune pt",precision=-1),
        phi = Var("corPhi('RawDeepResolutionTune')", float, doc="DeepmET ResolutionTune phi",precision=12),
    ),
)

deepMetResponseTuneTable = simpleSingletonPATMETFlatTableProducer.clone(
    src = pfmetTable.src,
    name = cms.string("DeepMETResponseTune"),
    doc = cms.string("Deep MET trained with extra response tune"),
    variables = cms.PSet(#NOTA BENE: we don't copy PTVars here!
        pt = Var("corPt('RawDeepResponseTune')", float, doc="DeepMET ResponseTune pt",precision=-1),
        phi = Var("corPhi('RawDeepResponseTune')", float, doc="DeepMET ResponseTune phi",precision=12),
    ),
)

metMCTable = simpleSingletonPATMETFlatTableProducer.clone(
    src = pfmetTable.src,
    name = cms.string("GenMET"),
    doc = cms.string("Gen MET"),
    variables = cms.PSet(
       pt  = Var("genMET.pt",  float, doc="pt", precision=10),
       phi = Var("genMET.phi", float, doc="phi", precision=10),
    ),
)


metTablesTask = cms.Task(pfmetTable, rawMetTable, caloMetTable, puppiMetTable, rawPuppiMetTable, trkMetTable, 
        deepMetResolutionTuneTable, deepMetResponseTuneTable )
metMCTask = cms.Task( metMCTable )
