import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

# Primary Vertex Producer: Defines the collection of primary vertices to be included in the NANOAOD
primaryVertexBPH = cms.EDProducer(
    "PrimaryVertexMerger",
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    minNdof  = cms.double(4.0),  # Minimum degrees of freedom for vertex
    maxZ     = cms.double(24.0), # Maximum Z coordinate [cm]
    maxRho   = cms.double(2.0)   # Maximum transverse distance from beamline [cm]
)

# Primary Vertex Table: Flattens the primary vertex information into a table format suitable for NANOAOD
primaryVertexBPHTable = cms.EDProducer(
    "SimpleCandidateFlatTableProducer",
    src  = cms.InputTag("primaryVertexBPH"),
    cut  = cms.string(""), # No additional selection applied to vertices
    name = cms.string("PrimaryVertex"),
    doc  = cms.string("Offline primary vertices after basic selection"),
    singleton = cms.bool(False), # Variable number of entries
    extension = cms.bool(False), # This is the main table for primary vertices
    variables = cms.PSet(
        x       = Var("position().x()", float, doc="x coordinate of vertex position [cm]", precision=10),
        y       = Var("position().y()", float, doc="y coordinate of vertex position [cm]", precision=10),
        z       = Var("position().z()", float, doc="z coordinate of vertex position [cm]", precision=10),
        ndof    = Var("vertexNdof()", float, doc="Number of degrees of freedom of the vertex fit", precision=10),
        chi2    = Var("vertexChi2()", float, doc="Chi-squared of the vertex fit", precision=10),
        trkSize = Var("numberOfDaughters()", int, doc="Number of associated tracks", precision=10),
        covXX   = Var("vertexCovariance(0, 0)", float, doc="Covariance of x with x", precision=10),
        covYY   = Var("vertexCovariance(1, 1)", float, doc="Covariance of y with y", precision=10),
        covZZ   = Var("vertexCovariance(2, 2)", float, doc="Covariance of z with z", precision=10),
        covXY   = Var("vertexCovariance(0, 1)", float, doc="Covariance of x with y", precision=10),
        covXZ   = Var("vertexCovariance(0, 2)", float, doc="Covariance of x with z", precision=10),
        covYZ   = Var("vertexCovariance(1, 2)", float, doc="Covariance of y with z", precision=10),
    ),
)

primaryVertexBPHSequence = cms.Sequence(primaryVertexBPH)
primaryVertexBPHTables   = cms.Sequence(primaryVertexBPHTable)
