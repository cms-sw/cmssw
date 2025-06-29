import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

hltTauTable = cms.EDProducer("SimplePFTauCandidateFlatTableProducer",
      skipNonExistingSrc = cms.bool(True),
      src = cms.InputTag("hltHpsPFTauProducer"),
      name = cms.string("hltHpsPFTau"),
      cut = cms.string(""),
      doc = cms.string("HLT HPS Taus information"),
      singleton = cms.bool(False),
      extension = cms.bool(False),
      variables = cms.PSet(
        P4Vars,
        # taken from https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/taus_cff.py#L61 
        leadTkPtOverTauPt = Var("?leadChargedHadrCand.isNonnull()?leadChargedHadrCand.pt/pt:1",float, doc="pt of the leading track divided by tau pt",precision=10),
        leadTkDeltaEta = Var("?leadChargedHadrCand.isNonnull()?(leadChargedHadrCand.eta - eta):0",float, doc="eta of the leading track, minus tau eta",precision=8),
        leadTkDeltaPhi = Var("?leadChargedHadrCand.isNonnull()?deltaPhi(leadChargedHadrCand.phi, phi):0",float, doc="phi of the leading track, minus tau phi",precision=8),
        # taken from https://github.com/cms-tau-pog/TauMLTools/blob/00bd9416f3198d7aa19ff9799037c14f2fa14514/Production/python/customiseHLT.py#L88 
        charge = Var("charge", int, doc="electric charge"),
        vx = Var("vx", float, doc='x coordinate of vertex position'),
        vy = Var("vy", float, doc='y coordinate of vertex position'),
        vz = Var("vz", float, doc='z coordinate of vertex position'),
        pdgId = Var("pdgId", int, doc='PDG identifier'),
        dz = Var("? leadPFCand.trackRef.isNonnull && leadPFCand.trackRef.isAvailable ? leadPFCand.trackRef.dz : -999 ", float, doc='lead PF Candidate dz'),
        dzError = Var("? leadPFCand.trackRef.isNonnull && leadPFCand.trackRef.isAvailable ? leadPFCand.trackRef.dzError : -999 ", float, doc='lead PF Candidate dz Error'),
        decayMode = Var("decayMode", int, doc='tau decay mode'),
        # # source: DataFormats/TauReco/interface/PFTau.h
        # ## variables available in PF tau
        emFraction = Var("emFraction", float, doc = " Ecal/Hcal Cluster Energy"),
        hcalTotOverPLead = Var("hcalTotOverPLead", float, doc = " total Hcal Cluster E / leadPFChargedHadron P"),
        signalConeSize = Var("signalConeSize", float, doc = "Size of signal cone"),
        # variables available in PF jets
        jetIsValid = Var("jetRef.isNonnull && jetRef.isAvailable", bool, doc = "jet is valid"),
      ),
)

hltTauExtTable = cms.EDProducer("HLTTauTableProducer",
                                tableName = cms.string("hltHpsPFTau"),
                                skipNonExistingSrc = cms.bool(True),
                                taus = cms.InputTag( "hltHpsPFTauProducer" ),
                                deepTauVSe = cms.InputTag("hltHpsPFTauDeepTauProducer", "VSe"),
                                deepTauVSmu = cms.InputTag("hltHpsPFTauDeepTauProducer", "VSmu"),
                                deepTauVSjet = cms.InputTag("hltHpsPFTauDeepTauProducer", "VSjet"),
                                tauTransverseImpactParameters = cms.InputTag( "hltHpsPFTauTransverseImpactParametersForDeepTau" ),
                                precision = cms.int32(7),
                                )
