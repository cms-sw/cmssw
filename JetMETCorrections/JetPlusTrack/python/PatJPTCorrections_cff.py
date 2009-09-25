import FWCore.ParameterSet.Config as cms

# -------------------- Uncorrected PAT jets --------------------

uncorrectedLayer1JetsIC5 = cms.EDProducer(
    "RawPATJetProducer",
    JetCollection = cms.InputTag("allLayer1JetsIC5"), 
    )

uncorrectedLayer1JetsSC5 = cms.EDProducer(
    "RawPATJetProducer",
    JetCollection = cms.InputTag("allLayer1JetsSC5"), 
    )

uncorrectedLayer1JetsAK5 = cms.EDProducer(
    "RawPATJetProducer",
    JetCollection = cms.InputTag("allLayer1JetsAK5"), 
    )

# -------------------- ZSP Corrections --------------------

from JetMETCorrections.Configuration.ZSPJetCorrections219_cff import *

# Services

PatZSPCorrectorIC5Calo = ZSPJetCorrectorIcone5.clone()
PatZSPCorrectorIC5Calo.label = cms.string('PatZSPCorrectorIC5Calo')

PatZSPCorrectorSC5Calo = ZSPJetCorrectorIcone5.clone()
PatZSPCorrectorSC5Calo.label = cms.string('PatZSPCorrectorSC5Calo')

PatZSPCorrectorAK5Calo = ZSPJetCorrectorIcone5.clone()
PatZSPCorrectorAK5Calo.label = cms.string('PatZSPCorrectorAK5Calo')

# Producers

PatZSPCorJetIC5Calo = cms.EDProducer(
    "PatJetCorrectionProducer",
    src = cms.InputTag("uncorrectedLayer1JetsIC5"),
    correctors = cms.vstring('PatZSPCorrectorIC5Calo'),
    alias = cms.untracked.string('PatZSPCorJetIC5Calo'),
    )

PatZSPCorJetSC5Calo = cms.EDProducer(
    "PatJetCorrectionProducer",
    src = cms.InputTag("uncorrectedLayer1JetsSC5"),
    correctors = cms.vstring('PatZSPCorrectorSC5Calo'),
    alias = cms.untracked.string('PatZSPCorJetSC5Calo'),
    )

PatZSPCorJetAK5Calo = cms.EDProducer(
    "PatJetCorrectionProducer",
    src = cms.InputTag("uncorrectedLayer1JetsAK5"),
    correctors = cms.vstring('PatZSPCorrectorAK5Calo'),
    alias = cms.untracked.string('PatZSPCorJetAK5Calo'),
    )

# -------------------- Jet-Tracks Association --------------------

# IC5

from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import*
PatZSPvertexIC5Calo = iterativeCone5JetTracksAssociatorAtVertex.clone() 
PatZSPvertexIC5Calo.jets = cms.InputTag("PatZSPCorJetIC5Calo")
PatZSPcalofaceIC5Calo = iterativeCone5JetTracksAssociatorAtCaloFace.clone()
PatZSPcalofaceIC5Calo.jets = cms.InputTag("PatZSPCorJetIC5Calo")
PatZSPextenderIC5Calo = iterativeCone5JetExtender.clone() 
PatZSPextenderIC5Calo.jets = cms.InputTag("PatZSPCorJetIC5Calo")
PatZSPextenderIC5Calo.jet2TracksAtCALO = cms.InputTag("ZSPcalofaceIC5Calo")
PatZSPextenderIC5Calo.jet2TracksAtVX = cms.InputTag("ZSPvertexIC5Calo")

# SC5

from RecoJets.JetAssociationProducers.sisCone5JTA_cff import*
PatZSPvertexSC5Calo = sisCone5JetTracksAssociatorAtVertex.clone() 
PatZSPvertexSC5Calo.jets = cms.InputTag("PatZSPCorJetSC5Calo")
PatZSPcalofaceSC5Calo = sisCone5JetTracksAssociatorAtCaloFace.clone()
PatZSPcalofaceSC5Calo.jets = cms.InputTag("PatZSPCorJetSC5Calo")
PatZSPextenderSC5Calo = sisCone5JetExtender.clone() 
PatZSPextenderSC5Calo.jets = cms.InputTag("PatZSPCorJetSC5Calo")
PatZSPextenderSC5Calo.jet2TracksAtCALO = cms.InputTag("ZSPcalofaceSC5Calo")
PatZSPextenderSC5Calo.jet2TracksAtVX = cms.InputTag("ZSPvertexSC5Calo")

# AK5

from RecoJets.JetAssociationProducers.ak5JTA_cff import*
PatZSPvertexAK5Calo = ak5JetTracksAssociatorAtVertex.clone() 
PatZSPvertexAK5Calo.jets = cms.InputTag("PatZSPCorJetAK5Calo")
PatZSPcalofaceAK5Calo = ak5JetTracksAssociatorAtCaloFace.clone()
PatZSPcalofaceAK5Calo.jets = cms.InputTag("PatZSPCorJetAK5Calo")
PatZSPextenderAK5Calo = ak5JetExtender.clone() 
PatZSPextenderAK5Calo.jets = cms.InputTag("PatZSPCorJetAK5Calo")
PatZSPextenderAK5Calo.jet2TracksAtCALO = cms.InputTag("ZSPcalofaceAK5Calo")
PatZSPextenderAK5Calo.jet2TracksAtVX = cms.InputTag("ZSPvertexAK5Calo")

# Sequences

PatZSPjtaIC5Calo = cms.Sequence( PatZSPvertexIC5Calo * PatZSPcalofaceIC5Calo * PatZSPextenderIC5Calo )
PatZSPjtaSC5Calo = cms.Sequence( PatZSPvertexSC5Calo * PatZSPcalofaceSC5Calo * PatZSPextenderSC5Calo )
PatZSPjtaAK5Calo = cms.Sequence( PatZSPvertexAK5Calo * PatZSPcalofaceAK5Calo * PatZSPextenderAK5Calo )

# -------------------- JPT Corrections --------------------

from JetMETCorrections.Configuration.JetPlusTrackCorrections_cfi import *

# Services

PatJPTCorrectorIC5Calo = cms.ESSource(
    "PatJPTCorrectionService",
    cms.PSet(JPTZSPCorrectorICone5),
    label = cms.string('PatJPTCorrectorIC5Calo'),
    UsePatCollections = cms.bool(True),
    AllowOnTheFly     = cms.bool(True),
    Tracks            = cms.InputTag("generalTracks"),
    Propagator        = cms.string('SteppingHelixPropagatorAlong'),
    ConeSize          = cms.double(0.5),
    )
PatJPTCorrectorIC5Calo.JetSplitMerge = cms.int32(0)
PatJPTCorrectorIC5Calo.Muons = cms.InputTag("cleanLayer1Muons")
PatJPTCorrectorIC5Calo.Electrons = cms.InputTag("cleanLayer1Electrons")
PatJPTCorrectorIC5Calo.ElectronIds = cms.InputTag("eidTight")
PatJPTCorrectorIC5Calo.JetTracksAssociationAtVertex = cms.InputTag("PatZSPvertexIC5Calo")
PatJPTCorrectorIC5Calo.JetTracksAssociationAtCaloFace = cms.InputTag("PatZSPcalofaceIC5Calo")

PatJPTCorrectorSC5Calo = cms.ESSource(
    "PatJPTCorrectionService",
    cms.PSet(JPTZSPCorrectorICone5),
    label = cms.string('PatJPTCorrectorSC5Calo'),
    UsePatCollections = cms.bool(True),
    AllowOnTheFly     = cms.bool(True),
    Tracks            = cms.InputTag("generalTracks"),
    Propagator        = cms.string('SteppingHelixPropagatorAlong'),
    ConeSize          = cms.double(0.5),
    )
PatJPTCorrectorSC5Calo.JetSplitMerge = cms.int32(1)
PatJPTCorrectorSC5Calo.Muons = cms.InputTag("cleanLayer1Muons")
PatJPTCorrectorSC5Calo.Electrons = cms.InputTag("cleanLayer1Electrons")
PatJPTCorrectorSC5Calo.ElectronIds = cms.InputTag("eidTight")
PatJPTCorrectorSC5Calo.JetTracksAssociationAtVertex = cms.InputTag("PatZSPvertexSC5Calo")
PatJPTCorrectorSC5Calo.JetTracksAssociationAtCaloFace = cms.InputTag("PatZSPcalofaceSC5Calo")

PatJPTCorrectorAK5Calo = cms.ESSource(
    "PatJPTCorrectionService",
    cms.PSet(JPTZSPCorrectorICone5),
    label = cms.string('PatJPTCorrectorAK5Calo'),
    UsePatCollections = cms.bool(True),
    AllowOnTheFly     = cms.bool(True),
    Tracks            = cms.InputTag("generalTracks"),
    Propagator        = cms.string('SteppingHelixPropagatorAlong'),
    ConeSize          = cms.double(0.5),
    )
PatJPTCorrectorAK5Calo.JetSplitMerge = cms.int32(2)
PatJPTCorrectorAK5Calo.Muons = cms.InputTag("cleanLayer1Muons")
PatJPTCorrectorAK5Calo.Electrons = cms.InputTag("cleanLayer1Electrons")
PatJPTCorrectorAK5Calo.ElectronIds = cms.InputTag("eidTight")
PatJPTCorrectorAK5Calo.JetTracksAssociationAtVertex = cms.InputTag("PatZSPvertexAK5Calo")
PatJPTCorrectorAK5Calo.JetTracksAssociationAtCaloFace = cms.InputTag("PatZSPcalofaceAK5Calo")

# Producers

PatJPTCorJetIC5Calo = cms.EDProducer(
    "PatJetCorrectionProducer",
    src = cms.InputTag("PatZSPCorJetIC5Calo"),
    correctors = cms.vstring('PatJPTCorrectorIC5Calo'),
    alias = cms.untracked.string('PatJPTCorJetIC5Calo')
    )

PatJPTCorJetSC5Calo = cms.EDProducer(
    "PatJetCorrectionProducer",
    src = cms.InputTag("PatZSPCorJetSC5Calo"),
    correctors = cms.vstring('PatJPTCorrectorSC5Calo'),
    alias = cms.untracked.string('PatJPTCorJetSC5Calo')
    )

PatJPTCorJetAK5Calo = cms.EDProducer(
    "PatJetCorrectionProducer",
    src = cms.InputTag("PatZSPCorJetAK5Calo"),
    correctors = cms.vstring('PatJPTCorrectorAK5Calo'),
    alias = cms.untracked.string('PatJPTCorJetAK5Calo')
    )

# Sequences

PatJPTCorrectionsIC5 = cms.Sequence(
    uncorrectedLayer1JetsIC5 *
    PatZSPCorJetIC5Calo *
    PatZSPjtaIC5Calo *
    PatJPTCorJetIC5Calo
    )

PatJPTCorrectionsSC5 = cms.Sequence(
    uncorrectedLayer1JetsSC5 *
    PatZSPCorJetSC5Calo *
    PatZSPjtaSC5Calo *
    PatJPTCorJetSC5Calo
    )

PatJPTCorrectionsAK5 = cms.Sequence(
    uncorrectedLayer1JetsAK5 *
    PatZSPCorJetAK5Calo *
    PatZSPjtaAK5Calo *
    PatJPTCorJetAK5Calo
    )
