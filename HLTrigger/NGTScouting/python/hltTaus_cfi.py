import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

hltTauTable = cms.EDProducer("SimplePFTauCandidateFlatTableProducer",
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
        # source: DataFormats/JetReco/interface/PFJet.h
        ## FIXME below does not work - members not found 
        # chargedHadronEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedHadronEnergy : -999.", float, doc = "chargedHadronEnergy"),
        # neutralHadronEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.neutralHadronEnergy : -999.", float, doc = "neutralHadronEnergy"),
        # photonEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.photonEnergy : -999.", float, doc = "photonEnergy"),
        # muonEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.muonEnergy : -999.", float, doc = "muonEnergy"),
        # chargedHadronMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedHadronMultiplicity : -999.", float, doc = "chargedHadronMultiplicity"),
        # neutralHadronMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.neutralHadronMultiplicity : -999.", float, doc = "neutralHadronMultiplicity"),
        # photonMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.photonMultiplicity : -999.", float, doc = "photonMultiplicity"),
        # muonMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.muonMultiplicity : -999.", float, doc = "muonMultiplicity"),
        # chargedMuEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedMuEnergy : -999.", float, doc = "chargedMuEnergy"),
        # neutralEmEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.neutralEmEnergy : -999.", float, doc = "neutralEmEnergy"),
        # chargedMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedMultiplicity : -999.", float, doc = "chargedMultiplicity"),
        # neutralMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.neutralMultiplicity : -999.", float, doc = "neutralMultiplicity"),
      ),
      externalVariables = cms.PSet(
        ## FIXME does not work despite DiMediumTau path using this! 
        ## https://github.com/cms-sw/cmssw/blob/48e0354ea37072eeacc8237a4a79e7ad34b9b0ae/HLTrigger/Configuration/python/HLT_75e33/modules/hltHpsSelectedPFTausMediumDitauWPDeepTau_cfi.py#L6
        deepTauVSjet = ExtVar(cms.InputTag("hltHpsPFTauDeepTauProducer","VSjet"), float, doc="deeptau VSjet", precision=10),
      ),
)