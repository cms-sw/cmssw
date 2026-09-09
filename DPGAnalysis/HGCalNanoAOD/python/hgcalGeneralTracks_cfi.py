import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

trackMuonInfo = cms.EDProducer("TrackMuonInfoProducer",
    tracks = cms.InputTag("generalTracks"),
    muons  = cms.InputTag("muons")
)

# General Tracks table for NanoAOD 
generalTracksTable = cms.EDProducer(
    "SimpleTrackFlatTableProducer",
    skipNonExistingSrc=cms.bool(True),
    src=cms.InputTag("generalTracks"),
    cut=cms.string(""),
    name=cms.string("GeneralTrack"),
    doc=cms.string("General reconstructed tracks"),
    extension=cms.bool(False),
    variables=cms.PSet(
        # Basic kinematics
        pt=Var("pt()", "float", doc="track p_T [GeV]"),
        p=Var("p()", "float", doc="track momentum magnitude [GeV]"),
        eta=Var("eta()", "float", doc="track pseudorapidity"),
        phi=Var("phi()", "float", doc="track phi angle [rad]"),
        charge=Var("charge()", "int", doc="track charge"),
        trackLambda=Var("lambda()", "float", doc="track lambda"),
        
        # Position 
        vx=Var("vx()", "float", doc="track vertex x [cm]"),
        vy=Var("vy()", "float", doc="track vertex y [cm]"),
        vz=Var("vz()", "float", doc="track vertex z [cm]"),
        
        # Quality metrics
        nhits=Var("numberOfValidHits()", "uint16", doc="number of valid hits"),
        missingOuterHits=Var("missingOuterHits()", "uint8", doc="number of missing outer hits"),
        
        # Error parameters 
        ptErr=Var("ptError()", "float", doc="track p_T error [GeV]"),
        etaErr=Var("etaError()", "float", doc="track eta error"),
        phiErr=Var("phiError()", "float", doc="track phi error"),
        lambdaErr=Var("lambdaError()", "float", doc="track lambda error"),
        qoverpErr=Var("qoverpError()", "float", doc="q/p error"),
    ),
    externalVariables = cms.PSet(
        isMuon = ExtVar(cms.InputTag("trackMuonInfo", "isMuon"),"int", doc="1 if PFMuonAlgo::isMuon, 0 otherwise, -1 if no muon"),
        isTrackerMuon = ExtVar(cms.InputTag("trackMuonInfo", "isTrackerMuon"),"int", doc="1 if tracker muon, -1 if no muon"),
        muon_dt_hits = ExtVar(cms.InputTag("trackMuonInfo", "muonDtHits"),"int", doc="ST muon DT hits (-1 if no muon)"),
        muon_csc_hits = ExtVar(cms.InputTag("trackMuonInfo", "muonCscHits"), "int", doc="ST muon CSC hits (-1 if no muon)"),
        muon_type = ExtVar(cms.InputTag("trackMuonInfo", "muonType"), "int", doc="muon type (-1 if no muon)"),
    )
)

# Sequence for general tracks
hgcalGeneralTracksTableSequence = cms.Sequence(trackMuonInfo + generalTracksTable)
