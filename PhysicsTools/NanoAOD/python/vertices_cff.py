import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *



##################### User floats producers, selectors ##########################


##################### Tables for final output and docs ##########################
vertexTable = cms.EDProducer("VertexTableProducer",
    pvSrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
    goodPvCut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"), 
    svSrc = cms.InputTag("slimmedSecondaryVertices"),
    svCut = cms.string(""),
    dlenMin = cms.double(0),
    dlenSigMin = cms.double(3),
    pvName = cms.string("PV"),
    svName = cms.string("SV"),
    svDoc  = cms.string("secondary vertices from IVF algorithm"),
)

svCandidateTable =  cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("vertexTable"),
    cut = cms.string(""),  #DO NOT further cut here, use vertexTable.svCut
    name = cms.string("SV"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(True), 
    variables = cms.PSet(P4Vars,
        x   = Var("position().x()", float, doc = "secondary vertex X position, in cm",precision=10),
        y   = Var("position().y()", float, doc = "secondary vertex Y position, in cm",precision=10),
        z   = Var("position().z()", float, doc = "secondary vertex Z position, in cm",precision=14),
        ndof    = Var("vertexNdof()", float, doc = "number of degrees of freedom",precision=8),
        chi2    = Var("vertexNormalizedChi2()", float, doc = "reduced chi2, i.e. chi/ndof",precision=8),
        ntracks = Var("numberOfDaughters()", "uint8", doc = "number of tracks"),
    ),
)
svCandidateTable.variables.pt.precision=10
svCandidateTable.variables.phi.precision=12


#before cross linking
vertexSequence = cms.Sequence()
#after cross linkining
vertexTables = cms.Sequence( vertexTable+svCandidateTable)

