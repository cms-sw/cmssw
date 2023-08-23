import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer


##################### User floats producers, selectors ##########################


##################### Tables for final output and docs ##########################
vertexTable = cms.EDProducer("VertexTableProducer",
    pvSrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
    goodPvCut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"), 
    svSrc = cms.InputTag("linkedObjects", "vertices"),
    svCut = cms.string(""),  # careful: adding a cut here would make the collection matching inconsistent with the SV table
    dlenMin = cms.double(0),
    dlenSigMin = cms.double(3),
    pvName = cms.string("PV"),
    svName = cms.string("SV"),
    svDoc  = cms.string("secondary vertices from IVF algorithm"),
)

svCandidateTable =  simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("vertexTable"),
    name = cms.string("SV"),
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
vertexTask = cms.Task()
#after cross linkining
vertexTablesTask = cms.Task( vertexTable, svCandidateTable )

