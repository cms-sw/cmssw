#!/bin/env python
from math import *
import ROOT
#from CMGTools.TTHAnalysis.signedSip import *
from PhysicsTools.Heppy.analyzers.core.autovars import *
from PhysicsTools.HeppyCore.utils.deltar import deltaR

objectFloat = NTupleObjectType("builtInType", variables = [
    NTupleVariable("",    lambda x : x),
])
objectInt = NTupleObjectType("builtInType", variables = [
    NTupleVariable("",    lambda x : x,int),
])

twoVectorType = NTupleObjectType("twoVector", variables = [
    NTupleVariable("pt",    lambda x : x.pt()),
    NTupleVariable("phi",   lambda x : x.phi()),
])

fourVectorType = NTupleObjectType("fourVector", variables = [
    NTupleVariable("pt",    lambda x : x.pt()),
    NTupleVariable("eta",   lambda x : x.eta()),
    NTupleVariable("phi",   lambda x : x.phi()),
    NTupleVariable("mass",  lambda x : x.mass()),
    NTupleVariable("p4",    lambda x : x, "TLorentzVector", default=ROOT.reco.Particle.LorentzVector(0.,0.,0.,0.), filler = lambda vector, obj: vector.SetPtEtaPhiM(obj.pt(), obj.eta(), obj.phi(), obj.mass())),
    #               ^^^^------- Note: p4 normally is not saved unless 'saveTLorentzVectors' is enabled in the tree producer
])
particleType = NTupleObjectType("particle", baseObjectTypes = [ fourVectorType ], variables = [
    NTupleVariable("pdgId",   lambda x : x.pdgId(), int),
])

##------------------------------------------  
## LEPTON
##------------------------------------------  

### BASIC VERSION WITH ONLY MAIN LEPTON ID CRITERIA
leptonType = NTupleObjectType("lepton", baseObjectTypes = [ particleType ], variables = [
    NTupleVariable("charge",   lambda x : x.charge(), int),
    # Identification
    NTupleVariable("tightId",     lambda x : x.tightId(), int, help="POG Tight ID (for electrons it's configured in the analyzer)"),
    NTupleVariable("eleCutIdCSA14_25ns_v1",     lambda x : (1*x.electronID("POG_Cuts_ID_CSA14_25ns_v1_Veto") + 1*x.electronID("POG_Cuts_ID_CSA14_25ns_v1_Loose") + 1*x.electronID("POG_Cuts_ID_CSA14_25ns_v1_Medium") + 1*x.electronID("POG_Cuts_ID_CSA14_25ns_v1_Tight")) if abs(x.pdgId()) == 11 else -1, int, help="Electron cut-based id (POG CSA14_25ns_v1): 0=none, 1=veto, 2=loose, 3=medium, 4=tight"),
    NTupleVariable("eleCutIdCSA14_50ns_v1",     lambda x : (1*x.electronID("POG_Cuts_ID_CSA14_50ns_v1_Veto") + 1*x.electronID("POG_Cuts_ID_CSA14_50ns_v1_Loose") + 1*x.electronID("POG_Cuts_ID_CSA14_50ns_v1_Medium") + 1*x.electronID("POG_Cuts_ID_CSA14_50ns_v1_Tight")) if abs(x.pdgId()) == 11 else -1, int, help="Electron cut-based id (POG CSA14_50ns_v1): 0=none, 1=veto, 2=loose, 3=medium, 4=tight"),
    # Impact parameter
    NTupleVariable("dxy",   lambda x : x.dxy(), help="d_{xy} with respect to PV, in cm (with sign)"),
    NTupleVariable("dz",    lambda x : x.dz() , help="d_{z} with respect to PV, in cm (with sign)"),
    NTupleVariable("edxy",  lambda x : x.edB(), help="#sigma(d_{xy}) with respect to PV, in cm"),
    NTupleVariable("edz",   lambda x : x.gsfTrack().dzError() if abs(x.pdgId())==11 else x.innerTrack().dzError() , help="#sigma(d_{z}) with respect to PV, in cm"),
    NTupleVariable("ip3d",  lambda x : x.ip3D() , help="d_{3d} with respect to PV, in cm (absolute value)"),
    NTupleVariable("sip3d",  lambda x : x.sip3D(), help="S_{ip3d} with respect to PV (significance)"),
    # Conversion rejection
    NTupleVariable("convVeto",    lambda x : x.passConversionVeto() if abs(x.pdgId())==11 else 1, int, help="Conversion veto (always true for muons)"),
    NTupleVariable("lostHits",    lambda x : (x.gsfTrack() if abs(x.pdgId())==11 else x.innerTrack()).hitPattern().numberOfLostHits(ROOT.reco.HitPattern.MISSING_INNER_HITS), int, help="Number of lost hits on inner track"),
    # Isolations with the two radia
    NTupleVariable("relIso03",  lambda x : x.relIso03, help="PF Rel Iso, R=0.3, pile-up corrected"),
    NTupleVariable("relIso04",  lambda x : x.relIso04, help="PF Rel Iso, R=0.4, pile-up corrected"),
    # Charge flip rejection criteria
    NTupleVariable("tightCharge",  lambda lepton : ( lepton.isGsfCtfScPixChargeConsistent() + lepton.isGsfScPixChargeConsistent() ) if abs(lepton.pdgId()) == 11 else 2*(lepton.innerTrack().ptError()/lepton.innerTrack().pt() < 0.2), int, help="Tight charge criteria: for electrons, 2 if isGsfCtfScPixChargeConsistent, 1 if only isGsfScPixChargeConsistent, 0 otherwise; for muons, 2 if ptError/pt < 0.20, 0 otherwise "),
    # MC-match info
    NTupleVariable("mcMatchId",  lambda x : x.mcMatchId, int, mcOnly=True, help="Match to source from hard scatter (pdgId of heaviest particle in chain, 25 for H, 6 for t, 23/24 for W/Z), zero if non-prompt or fake"),
    NTupleVariable("mcMatchAny", lambda x : x.mcMatchAny, int, mcOnly=True, help="Match to any final state leptons: 0 if unmatched, 1 if light flavour (including prompt), 4 if charm, 5 if bottom"),
    NTupleVariable("mcMatchTau", lambda x : x.mcMatchTau, int, mcOnly=True, help="True if the leptons comes from a tau"),
])

### EXTENDED VERSION WITH INDIVIUAL DISCRIMINATING VARIABLES
leptonTypeExtra = NTupleObjectType("leptonExtra", baseObjectTypes = [ leptonType ], variables = [
    # Extra isolation variables
    NTupleVariable("chargedHadRelIso03",  lambda x : x.chargedHadronIsoR(0.3)/x.pt(), help="PF Rel Iso, R=0.3, charged hadrons only"),
    NTupleVariable("chargedHadRelIso04",  lambda x : x.chargedHadronIsoR(0.4)/x.pt(), help="PF Rel Iso, R=0.4, charged hadrons only"),
    # Extra muon ID working points
    NTupleVariable("softMuonId", lambda x : x.muonID("POG_ID_Soft") if abs(x.pdgId())==13 else 1, int, help="Muon POG Soft id"),
    NTupleVariable("pfMuonId",   lambda x : x.muonID("POG_ID_Loose") if abs(x.pdgId())==13 else 1, int, help="Muon POG Loose id"),
    NTupleVariable("mediumMuonId",   lambda x : x.muonID("POG_ID_Medium") if abs(x.pdgId())==13 else 1, int, help="Muon POG Medium id"),
    # Extra electron ID working points
    NTupleVariable("eleCutId2012_full5x5",     lambda x : (1*x.electronID("POG_Cuts_ID_2012_full5x5_Veto") + 1*x.electronID("POG_Cuts_ID_2012_full5x5_Loose") + 1*x.electronID("POG_Cuts_ID_2012_full5x5_Medium") + 1*x.electronID("POG_Cuts_ID_2012_full5x5_Tight")) if abs(x.pdgId()) == 11 else -1, int, help="Electron cut-based id (POG 2012, full5x5 shapes): 0=none, 1=veto, 2=loose, 3=medium, 4=tight"),
    # Extra tracker-related variables
    NTupleVariable("trackerLayers", lambda x : (x.track() if abs(x.pdgId())==13 else x.gsfTrack()).hitPattern().trackerLayersWithMeasurement(), int, help="Tracker Layers"),
    NTupleVariable("pixelLayers", lambda x : (x.track() if abs(x.pdgId())==13 else x.gsfTrack()).hitPattern().pixelLayersWithMeasurement(), int, help="Pixel Layers"),
    NTupleVariable("trackerHits", lambda x : (x.track() if abs(x.pdgId())==13 else x.gsfTrack()).hitPattern().numberOfValidTrackerHits(), int, help="Tracker hits"),
    NTupleVariable("lostOuterHits",    lambda x : (x.gsfTrack() if abs(x.pdgId())==11 else x.innerTrack()).hitPattern().numberOfLostHits(ROOT.reco.HitPattern.MISSING_OUTER_HITS), int, help="Number of lost hits on inner track"),
    NTupleVariable("innerTrackValidHitFraction", lambda x : (x.gsfTrack() if abs(x.pdgId())==11 else x.innerTrack()).validFraction(), help="fraction of valid hits on inner track"), 
    NTupleVariable("innerTrackChi2",      lambda x : (x.gsfTrack() if abs(x.pdgId())==11 else x.innerTrack()).normalizedChi2(), help="Inner track normalized chi2"), 
    # Extra muon ID variables
    NTupleVariable("nStations",    lambda lepton : lepton.numberOfMatchedStations() if abs(lepton.pdgId()) == 13 else 4, help="Number of matched muons stations (4 for electrons)"),
    NTupleVariable("caloCompatibility",      lambda lepton : lepton.caloCompatibility() if abs(lepton.pdgId()) == 13 else 0, help="Calorimetric compatibility"), 
    NTupleVariable("globalTrackChi2",      lambda lepton : lepton.globalTrack().normalizedChi2() if abs(lepton.pdgId()) == 13 and lepton.globalTrack().isNonnull() else 0, help="Global track normalized chi2"), 
    NTupleVariable("trkKink",      lambda lepton : lepton.combinedQuality().trkKink if abs(lepton.pdgId()) == 13 else 0, help="Tracker kink-finder"), 
    NTupleVariable("segmentCompatibility", lambda lepton : lepton.segmentCompatibility() if abs(lepton.pdgId()) == 13 else 0, help="Segment-based compatibility"), 
    NTupleVariable("chi2LocalPosition",    lambda lepton : lepton.combinedQuality().chi2LocalPosition if abs(lepton.pdgId()) == 13 else 0, help="Tracker-Muon matching in position"), 
    NTupleVariable("chi2LocalMomentum",    lambda lepton : lepton.combinedQuality().chi2LocalMomentum if abs(lepton.pdgId()) == 13 else 0, help="Tracker-Muon matching in momentum"), 
    NTupleVariable("glbTrackProbability",  lambda lepton : lepton.combinedQuality().glbTrackProbability if abs(lepton.pdgId()) == 13 else 0, help="Global track pseudo-probability"), 
    # Extra electron ID variables
    NTupleVariable("sigmaIEtaIEta",  lambda x : x.full5x5_sigmaIetaIeta() if abs(x.pdgId())==11 else 0, help="Electron sigma(ieta ieta), with full5x5 cluster shapes"),
    NTupleVariable("dEtaScTrkIn",    lambda x : x.deltaEtaSuperClusterTrackAtVtx() if abs(x.pdgId())==11 else 0, help="Electron deltaEtaSuperClusterTrackAtVtx (without absolute value!)"),
    NTupleVariable("dPhiScTrkIn",    lambda x : x.deltaPhiSuperClusterTrackAtVtx() if abs(x.pdgId())==11 else 0, help="Electron deltaPhiSuperClusterTrackAtVtx (without absolute value!)"),
    NTupleVariable("hadronicOverEm", lambda x : x.hadronicOverEm() if abs(x.pdgId())==11 else 0, help="Electron hadronicOverEm"),
    NTupleVariable("eInvMinusPInv",  lambda x : ((1.0/x.ecalEnergy() - x.eSuperClusterOverP()/x.ecalEnergy()) if x.ecalEnergy()>0. else 9e9) if abs(x.pdgId())==11 else 0, help="Electron 1/E - 1/p  (without absolute value!)"),
])
 

##------------------------------------------  
## TAU
##------------------------------------------  

tauType = NTupleObjectType("tau",  baseObjectTypes = [ particleType ], variables = [
    NTupleVariable("charge",   lambda x : x.charge(), int),
    NTupleVariable("dxy",   lambda x : x.dxy(), help="d_{xy} of lead track with respect to PV, in cm (with sign)"),
    NTupleVariable("dz",    lambda x : x.dz() , help="d_{z} of lead track with respect to PV, in cm (with sign)"),
    #NTupleVariable("idMVA2", lambda x : x.idMVA2, int, help="1,2,3 if the tau passes the loose, medium, tight WP of the By<X>IsolationMVA2 discriminator"),
    NTupleVariable("idCI3hit", lambda x : x.idCI3hit, int, help="1,2,3 if the tau passes the loose, medium, tight WP of the By<X>CombinedIsolationDBSumPtCorr3Hits discriminator"),
    NTupleVariable("isoCI3hit",  lambda x : x.tauID("byCombinedIsolationDeltaBetaCorrRaw3Hits"), help="byCombinedIsolationDeltaBetaCorrRaw3Hits raw output discriminator"),
    #NTupleVariable("isoMVA2",  lambda x : x.tauID("byIsolationMVA2raw"), help="ByIsolationMVA2 raw output discriminator"),
    # MC-match info
    NTupleVariable("mcMatchId",  lambda x : x.mcMatchId, int, mcOnly=True, help="Match to source from hard scatter (pdgId of heaviest particle in chain, 25 for H, 6 for t, 23/24 for W/Z), zero if non-prompt or fake"),
])

##------------------------------------------  
##  ISOTRACK
##------------------------------------------  

isoTrackType = NTupleObjectType("isoTrack",  baseObjectTypes = [ particleType ], variables = [
    NTupleVariable("charge",   lambda x : x.charge(), int),
    NTupleVariable("dz",    lambda x : x.dz() , help="d_{z} of lead track with respect to PV, in cm (with sign)"),
    NTupleVariable("absIso",  lambda x : x.absIso, float, mcOnly=False, help="abs charged iso with condition for isolation such that Min(0.2*pt, 8 GeV)"),
    NTupleVariable("mcMatchId",  lambda x : x.mcMatchId, int, mcOnly=True, help="Match to source from hard scatter (pdgId of heaviest particle in chain, 25 for H, 6 for t, 23/24 for W/Z), zero if non-prompt or fake"),
])


##------------------------------------------  
## PHOTON
##------------------------------------------  

photonType = NTupleObjectType("gamma", baseObjectTypes = [ particleType ], variables = [
    NTupleVariable("idCutBased", lambda x : x.idCutBased, int, help="1,2,3 if the gamma passes the loose, medium, tight WP of PhotonCutBasedID"),
    NTupleVariable("hOverE",  lambda x : x.hOVERe(), float, help="hoverE for photons"),
    NTupleVariable("r9",  lambda x : x.r9(), float, help="r9 for photons"),
    NTupleVariable("sigmaIetaIeta",  lambda x : x.sigmaIetaIeta(), float, help="sigmaIetaIeta for photons"),
    NTupleVariable("chHadIso",  lambda x : x.chargedHadronIso(), float, help="chargedHadronIsolation for photons"),
    NTupleVariable("neuHadIso",  lambda x : x.neutralHadronIso(), float, help="neutralHadronIsolation for photons"),
    NTupleVariable("phIso",  lambda x : x.photonIso(), float, help="gammaIsolation for photons"),
    NTupleVariable("mcMatchId",  lambda x : x.mcMatchId, int, mcOnly=True, help="Match to source from hard scatter (pdgId of heaviest particle in chain, 25 for H, 6 for t, 23/24 for W/Z), zero if non-prompt or fake"),
])

##------------------------------------------  
## JET
##------------------------------------------  

jetType = NTupleObjectType("jet",  baseObjectTypes = [ fourVectorType ], variables = [
    NTupleVariable("id",    lambda x : x.jetID("POG_PFID") , int, mcOnly=False,help="POG Loose jet ID"),
    NTupleVariable("puId", lambda x : getattr(x, 'puJetIdPassed', -99), int,     mcOnly=False, help="puId (full MVA, loose WP, 5.3.X training on AK5PFchs: the only thing that is available now)"),
    NTupleVariable("btagCSV",   lambda x : x.btag('combinedInclusiveSecondaryVertexV2BJetTags'), help="CSV-IVF v2 discriminator"),
    NTupleVariable("btagCMVA",  lambda x : x.btag('combinedMVABJetTags'), help="CMVA discriminator"),
    NTupleVariable("rawPt",  lambda x : x.pt() * x.rawFactor(), help="p_{T} before JEC"),
    NTupleVariable("mcPt",   lambda x : x.mcJet.pt() if getattr(x,"mcJet",None) else 0., mcOnly=True, help="p_{T} of associated gen jet"),
    NTupleVariable("mcFlavour", lambda x : x.partonFlavour(), int,     mcOnly=True, help="parton flavour (physics definition, i.e. including b's from shower)"),
    NTupleVariable("mcMatchId",  lambda x : getattr(x, 'mcMatchId', -99), int, mcOnly=True, help="Match to source from hard scatter (pdgId of heaviest particle in chain, 25 for H, 6 for t, 23/24 for W/Z), zero if non-prompt or fake"),
])
jetTypeExtra = NTupleObjectType("jetExtra",  baseObjectTypes = [ jetType ], variables = [
    NTupleVariable("area",   lambda x : x.jetArea(), help="Catchment area of jet"),
    # QG variables:
    NTupleVariable("qgl",   lambda x : getattr(x,'qgl', 0) , float, mcOnly=False,help="QG Likelihood"),
    NTupleVariable("ptd",   lambda x : getattr(x,'ptd', 0), float, mcOnly=False,help="QG input variable: ptD"),
    NTupleVariable("axis2",   lambda x : getattr(x,'axis2', 0) , float, mcOnly=False,help="QG input variable: axis2"),
    NTupleVariable("mult",   lambda x : getattr(x,'mult', 0) , int, mcOnly=False,help="QG input variable: total multiplicity"),
    NTupleVariable("partonId", lambda x : getattr(x,'partonId', 0), int,     mcOnly=True, help="parton flavour (manually matching to status 23 particles)"),
    NTupleVariable("partonMotherId", lambda x : getattr(x,'partonMotherId', 0), int,     mcOnly=True, help="parton flavour (manually matching to status 23 particles)"),
    NTupleVariable("nLeptons",   lambda x : len(x.leptons) if  hasattr(x,'leptons') else  0 , float, mcOnly=False,help="Number of associated leptons"),
])

      
##------------------------------------------  
## MET
##------------------------------------------  
  
metType = NTupleObjectType("met", baseObjectTypes = [ fourVectorType ], variables = [
    NTupleVariable("sumEt", lambda x : x.sumEt() ),
    NTupleVariable("genPt",  lambda x : x.genMET().pt() , mcOnly=True ),
    NTupleVariable("genPhi", lambda x : x.genMET().phi(), mcOnly=True ),
    NTupleVariable("genEta", lambda x : x.genMET().eta(), mcOnly=True ),
])

##------------------------------------------  
## GENPARTICLE
##------------------------------------------  

genParticleType = NTupleObjectType("genParticle", baseObjectTypes = [ particleType ], mcOnly=True, variables = [
    NTupleVariable("charge",   lambda x : x.threeCharge()/3.0, float),
    NTupleVariable("status",   lambda x : x.status(),int),
])
genParticleWithMotherId = NTupleObjectType("genParticleWithMotherId", baseObjectTypes = [ genParticleType ], mcOnly=True, variables = [
    NTupleVariable("motherId", lambda x : x.mother(0).pdgId() if x.mother(0) else 0, int, help="pdgId of the mother of the particle"),
    NTupleVariable("grandmotherId", lambda x : x.mother(0).mother(0).pdgId() if x.mother(0) and x.mother(0).mother(0) else 0, int, help="pdgId of the grandmother of the particle")
])
genParticleWithAncestryType = NTupleObjectType("genParticleWithAncestry", baseObjectTypes = [ genParticleType ], mcOnly=True, variables = [
    NTupleVariable("motherId", lambda x : x.motherId, int, help="pdgId of the mother of the particle"),
    NTupleVariable("grandmotherId", lambda x : x.grandmotherId, int, help="pdgId of the grandmother of the particle"),
    NTupleVariable("sourceId", lambda x : x.sourceId, int, help="origin of the particle (heaviest ancestor): 6=t, 25=h, 23/24=W/Z"),
])
genParticleWithLinksType = NTupleObjectType("genParticleWithLinks", baseObjectTypes = [ genParticleWithAncestryType ], mcOnly=True, variables = [
    NTupleVariable("motherIndex", lambda x : x.motherIndex, int, help="index of the mother in the generatorSummary")
])

