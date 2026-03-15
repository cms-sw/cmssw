import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import nanoMetadata

hltUpgradeNanoTask = cms.Task(nanoMetadata)

hltTiclCandidateTable = cms.EDProducer(
    "TICLCandidateTableProducer",
    skipNonExistingSrc=cms.bool(True),
    src=cms.InputTag("hltTiclCandidate"),
    cut=cms.string(""),
    name=cms.string("hltTICLCandidates"),
    doc=cms.string("TICLCandidates"),
    singleton=cms.bool(False),  # the number of entries is variable
    variables=cms.PSet(
        raw_energy=Var("rawEnergy", "float",
                       doc="Raw Energy of the TICLCandidate [GeV]"),
        pt=Var(
            "pt", "float", doc="TICLCandidate pT, computed from trackster raw energy and direction or from associated track [GeV]"),
        p=Var(
            "p", "float", doc="TICLCandidate momentum magnitude, computed from trackster raw energy and direction or from associated track [GeV]"),
        px=Var(
            "px", "float", doc="TICLCandidate x component of mementum, computed from trackster raw energy and direction or from associated track [GeV]"),
        py=Var(
            "py", "float", doc="TICLCandidate y component of mementum, computed from trackster raw energy and direction or from associated track [GeV]"),
        pz=Var(
            "pz", "float", doc="TICLCandidate z component of mementum, computed from trackster raw energy and direction or from associated track [GeV]"),
        energy=Var("energy", "float",
                             doc="Energy of the TICLCandidate, computed from the raw energy of the associated Trackster or from the associated track"),
        eta=Var(
            "eta", "float", doc="TICLCandidate pseudorapidity, derived from p4 built from associated Tracksters or from associated Track"),
        phi=Var(
            "phi", "float", doc="TICLCandidate phi, derived from p4 built from associated Tracksters or from associated Track"),
        mass=Var(
            "mass", "float", doc="TICLCandidate mass"),
        pdgID=Var(
            "pdgId", "int", doc="TICLCandidate assigned pdgID"),
        charge=Var(
            "charge", "int", doc="TICLCandidate assigned charge"),
        time=Var("time", "float", doc="TICLCandidate time, obtained from combination HGCAL and MTD time (offline) or HGCAL only (HLT)"),
        timeError=Var("timeError", "float",
                      doc="Trackster HGCAL time error"),
        MTDtime=Var("MTDtime", "float",
                    doc="TICLCandidate associated MTDTime, meaningful only for offline reconstruction"),
        MTDtimeError=Var("MTDtimeError", "float",
                         doc="Trackster associated MTD time error, meaningful only for offline reconstruction"),
        trackIdx=Var("trackPtr().key", "int",
                     doc="Index of hltGeneralTrack associated with TICLCandidate")
    ),
)


hltSimTiclCandidateTable = cms.EDProducer(
    "TICLCandidateTableProducer",
    skipNonExistingSrc=cms.bool(True),
    src=cms.InputTag("hltTiclSimTracksters"),
    cut=cms.string(""),
    name=cms.string("hltSimTICLCandidates"),
    doc=cms.string("SimTICLCandidates"),
    singleton=cms.bool(False),  # the number of entries is variable
    variables=cms.PSet(
        raw_energy=Var("rawEnergy", "float",
                       doc="Raw Energy of the TICLCandidate [GeV]"),
        pt=Var(
            "pt", "float", doc="TICLCandidate pT, computed from trackster raw energy and direction or from associated track [GeV]"),
        p=Var(
            "p", "float", doc="TICLCandidate momentum magnitude, computed from trackster raw energy and direction or from associated track [GeV]"),
        px=Var(
            "px", "float", doc="TICLCandidate x component of mementum, computed from trackster raw energy and direction or from associated track [GeV]"),
        py=Var(
            "py", "float", doc="TICLCandidate y component of mementum, computed from trackster raw energy and direction or from associated track [GeV]"),
        pz=Var(
            "pz", "float", doc="TICLCandidate z component of mementum, computed from trackster raw energy and direction or from associated track [GeV]"),
        energy=Var("energy", "float",
                             doc="Energy of the TICLCandidate, computed from the raw energy of the associated Trackster or from the associated track"),
        eta=Var(
            "eta", "float", doc="TICLCandidate pseudorapidity, derived from p4 built from associated Tracksters or from associated Track"),
        phi=Var(
            "phi", "float", doc="TICLCandidate phi, derived from p4 built from associated Tracksters or from associated Track"),
        mass=Var(
            "mass", "float", doc="TICLCandidate mass"),
        pdgID=Var(
            "pdgId", "int", doc="TICLCandidate assigned pdgID"),
        charge=Var(
            "charge", "int", doc="TICLCandidate assigned charge"),
        time=Var("time", "float", doc="TICLCandidate time, obtained from combination HGCAL and MTD time (offline) or HGCAL only (HLT)"),
        timeError=Var("timeError", "float",
                      doc="Trackster HGCAL time error"),
        MTDtime=Var("MTDtime", "float",
                    doc="TICLCandidate associated MTDTime, meaningful only for offline reconstruction"),
        MTDtimeError=Var("MTDtimeError", "float",
                         doc="Trackster associated MTD time error, meaningful only for offline reconstruction"),
        trackIdx=Var("trackPtr().key", "int",
                     doc="Index of hltGeneralTrack associated with TICLCandidate")
    ),
)
hltTiclCandidateExtraTable = cms.EDProducer(
    "TICLCandidateExtraTableProducer",
    src = cms.InputTag("hltTiclCandidate"),
    name = cms.string("Candidate2Tracksters"),
    doc = cms.string("TICLCandidates extra table with linked Tracksters"),
    collectionVariables = cms.PSet(
        tracksters = cms.PSet(
            name = cms.string("Candidate2TrackstersIndices"),
            doc = cms.string("Tracksters linked to TICLCandidates"),
            useCount = cms.bool(True),
            useOffset = cms.bool(False),
            variables = cms.PSet() 
        ),
    ),
)

hltSimTiclCandidateExtraTable = cms.EDProducer(
    "TICLCandidateExtraTableProducer",
    src = cms.InputTag("hltTiclSimTracksters"),
    name = cms.string("SimCandidate2Tracksters"),
    doc = cms.string("TICLCandidates extra table with linked Tracksters"),
    collectionVariables = cms.PSet(
        tracksters = cms.PSet(
            name = cms.string("SimCandidate2TrackstersIndices"),
            doc = cms.string("Tracksters linked to SimTICLCandidates"),
            useCount = cms.bool(True),
            useOffset = cms.bool(False),
            variables = cms.PSet() 
        ),
    ),
)


