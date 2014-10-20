#!/bin/env python
from math import *
import ROOT
#from CMGTools.TTHAnalysis.signedSip import *
from PhysicsTools.Heppy.analyzers.core.AutoFillNTupleObjects import *
from PhysicsTools.HeppyCore.utils.deltar import deltaR

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

leptonType = NTupleObjectType("lepton", baseObjectTypes = [ particleType ], variables = [
    NTupleVariable("charge",   lambda x : x.charge(), int),
    NTupleVariable("dxy",   lambda x : x.dxy(), help="d_{xy} with respect to PV, in cm (with sign)"),
    NTupleVariable("dz",    lambda x : x.dz() , help="d_{z} with respect to PV, in cm (with sign)"),
    NTupleVariable("edxy",  lambda x : x.edB(), help="#sigma(d_{xy}) with respect to PV, in cm"),
    NTupleVariable("edz",   lambda x : x.gsfTrack().dzError() if abs(x.pdgId())==11 else x.innerTrack().dzError() , help="#sigma(d_{z}) with respect to PV, in cm"),
#    NTupleVariable("ip3d",  lambda x : abs(signedIp3D(x)) , help="d_{3d} with respect to PV, in cm (absolute value)"),
    NTupleVariable("sip3d",  lambda x : x.sip3D(), help="S_{ip3d} with respect to PV (absolute value)"),
    NTupleVariable("tightId",     lambda x : x.tightId(), int, help="POG Tight ID (based on triggering MVA value for electrons, boolean for muons)"),
    NTupleVariable("convVeto",    lambda x : x.passConversionVeto() if abs(x.pdgId())==11 else 1, int, help="Conversion veto (always true for muons)"),
    NTupleVariable("lostHits",    lambda x : x.gsfTrack().trackerExpectedHitsInner().numberOfLostHits() if abs(x.pdgId())==11 else x.innerTrack().trackerExpectedHitsInner().numberOfLostHits(), int, help="Number of lost hits on inner track")
])
leptonTypeTTH = NTupleObjectType("leptonTTH", baseObjectTypes = [ leptonType ], variables = [
    NTupleVariable("tightCharge", lambda lepton : ( lepton.isGsfCtfScPixChargeConsistent() + lepton.isGsfScPixChargeConsistent() ) if abs(lepton.pdgId()) == 11 else 2*(lepton.innerTrack().ptError()/lepton.innerTrack().pt() < 0.2), int, help="Tight charge criteria"),
    NTupleVariable("relIso",  lambda x : x.relIso(dBetaFactor=0.5), help="PF Iso, R=0.4, with deltaBeta correction"),
    NTupleVariable("chargedRelIso", lambda x : x.chargedHadronIso()/x.pt(), help="PF Iso from charged hadrons only, R=0.4"),
    NTupleVariable("mvaId",      lambda lepton : lepton.mvaNonTrigV0() if abs(lepton.pdgId()) == 11 else 1, help="EGamma POG MVA ID for non-triggering electrons (as HZZ); 1 for muons"),
    NTupleVariable("mvaIdTrig",  lambda lepton : lepton.mvaTrigV0()    if abs(lepton.pdgId()) == 11 else 1, help="EGamma POG MVA ID for triggering electrons; 1 for muons"),
    NTupleVariable("mva",        lambda lepton : lepton.mvaValue, help="Lepton MVA (ttH version)"),
    NTupleVariable("jetPtRatio", lambda lepton : lepton.pt()/lepton.jet.pt()),
    NTupleVariable("jetBTagCSV", lambda lepton : lepton.jet.btag('combinedSecondaryVertexBJetTags') if hasattr(lepton.jet, 'btag') else -99),
    NTupleVariable("jetDR",      lambda lepton : deltaR(lepton.eta(),lepton.phi(),lepton.jet.eta(),lepton.jet.phi())),
    NTupleVariable("mcMatchId",  lambda x : x.mcMatchId, int, mcOnly=True, help="Match to source from hard scatter (25 for H, 6 for t, 23/24 for W/Z)"),
    NTupleVariable("mcMatchAny",  lambda x : x.mcMatchAny, int, mcOnly=True, help="Match to any final state leptons: -mcMatchId if prompt, 0 if unmatched, 1 if light flavour, 2 if heavy flavour (b)"),
    NTupleVariable("mcMatchTau",  lambda x : x.mcMatchTau, int, mcOnly=True, help="True if the leptons comes from a tau"),
    NTupleVariable("convVetoFull", lambda x : (x.passConversionVeto() and x.gsfTrack().trackerExpectedHitsInner().numberOfLostHits() == 0) if abs(x.pdgId())==11 else 1, int, help="Conv veto + no missing hits for electrons, always true for muons."),
])
leptonTypeSusy = NTupleObjectType("leptonSusy", baseObjectTypes = [ leptonType ], variables = [
    # Loose id 
    NTupleVariable("looseIdSusy", lambda x : x.looseIdSusy if hasattr(x, 'looseIdSusy') else -1, int, help="Loose ID for Susy ntuples (always true on selected leptons)"),
    # Isolations with the two radia
    NTupleVariable("relIso03",  lambda x : x.relIso03, help="PF Rel Iso, R=0.3, with deltaBeta correction for muons and rho corrections for electrons"),
    NTupleVariable("relIso04",  lambda x : x.relIso04, help="PF Rel Iso, R=0.4, with deltaBeta correction for muons and rho corrections for electrons"),
    NTupleVariable("chargedHadRelIso03",  lambda x : x.chargedHadronIso(0.3)/x.pt(), help="PF Rel Iso, R=0.3, charged hadrons only"),
    NTupleVariable("chargedHadRelIso04",  lambda x : x.chargedHadronIso(0.4)/x.pt(), help="PF Rel Iso, R=0.4, charged hadrons only"),
    # Extra electron id variables
    NTupleVariable("convVetoFull", lambda x : (x.passConversionVeto() and x.gsfTrack().trackerExpectedHitsInner().numberOfLostHits() == 0) if abs(x.pdgId())==11 else 1, int, help="Conv veto + no missing hits for electrons, always true for muons."),
    #NTupleVariable("eleCutId2012",     lambda x : (1*x.electronID("POG_Cuts_ID_2012_Veto") + 1*x.electronID("POG_Cuts_ID_2012_Loose") + 1*x.electronID("POG_Cuts_ID_2012_Medium") + 1*x.electronID("POG_Cuts_ID_2012_Tight")) if abs(x.pdgId()) == 11 else -1, int, help="Electron cut-based id (POG 2012): 0=none, 1=veto, 2=loose, 3=medium, 4=tight"),
    #NTupleVariable("eleCutId2012_full5x5",     lambda x : (1*x.electronID("POG_Cuts_ID_2012_full5x5_Veto") + 1*x.electronID("POG_Cuts_ID_2012_full5x5_Loose") + 1*x.electronID("POG_Cuts_ID_2012_full5x5_Medium") + 1*x.electronID("POG_Cuts_ID_2012_full5x5_Tight")) if abs(x.pdgId()) == 11 else -1, int, help="Electron cut-based id (POG 2012, full5x5 shapes): 0=none, 1=veto, 2=loose, 3=medium, 4=tight"),
    NTupleVariable("eleCutIdCSA14_25ns_v1",     lambda x : (1*x.electronID("POG_Cuts_ID_CSA14_25ns_v1_Veto") + 1*x.electronID("POG_Cuts_ID_CSA14_25ns_v1_Loose") + 1*x.electronID("POG_Cuts_ID_CSA14_25ns_v1_Medium") + 1*x.electronID("POG_Cuts_ID_CSA14_25ns_v1_Tight")) if abs(x.pdgId()) == 11 else -1, int, help="Electron cut-based id (POG CSA14_25ns_v1): 0=none, 1=veto, 2=loose, 3=medium, 4=tight"),
    NTupleVariable("eleCutIdCSA14_50ns_v1",     lambda x : (1*x.electronID("POG_Cuts_ID_CSA14_50ns_v1_Veto") + 1*x.electronID("POG_Cuts_ID_CSA14_50ns_v1_Loose") + 1*x.electronID("POG_Cuts_ID_CSA14_50ns_v1_Medium") + 1*x.electronID("POG_Cuts_ID_CSA14_50ns_v1_Tight")) if abs(x.pdgId()) == 11 else -1, int, help="Electron cut-based id (POG CSA14_50ns_v1): 0=none, 1=veto, 2=loose, 3=medium, 4=tight"),
    #NTupleVariable("eleMVAId",     lambda x : (x.electronID("POG_MVA_ID_NonTrig") + 2*x.electronID("POG_MVA_ID_Trig")) if abs(x.pdgId()) == 11 else -1, int, help="Electron mva id working point: 0=none, 1=non-trig, 2=trig, 3=both"),
    NTupleVariable("eleMVAId",     lambda x : (x.electronID("POG_MVA_ID_NonTrig_full5x5") + 2*x.electronID("POG_MVA_ID_Trig_full5x5")) if abs(x.pdgId()) == 11 else -1, int, help="Electron mva id working point (2012, full5x5 shapes): 0=none, 1=non-trig, 2=trig, 3=both"),
    NTupleVariable("tightCharge",  lambda lepton : ( lepton.isGsfCtfScPixChargeConsistent() + lepton.isGsfScPixChargeConsistent() ) if abs(lepton.pdgId()) == 11 else 2*(lepton.innerTrack().ptError()/lepton.innerTrack().pt() < 0.2), int, help="Tight charge criteria"),
    #NTupleVariable("mvaId",        lambda lepton : lepton.mvaNonTrigV0() if abs(lepton.pdgId()) == 11 else 1, help="EGamma POG MVA ID for non-triggering electrons (as HZZ); 1 for muons"),
    NTupleVariable("mvaId",         lambda lepton : lepton.mvaNonTrigV0(full5x5=True) if abs(lepton.pdgId()) == 11 else 1, help="EGamma POG MVA ID for non-triggering electrons (as HZZ); 1 for muons"),
    NTupleVariable("mvaIdTrig",     lambda lepton : lepton.mvaTrigV0(full5x5=True)    if abs(lepton.pdgId()) == 11 else 1, help="EGamma POG MVA ID for triggering electrons; 1 for muons"),
    # Muon-speficic info
    NTupleVariable("nStations",    lambda lepton : lepton.numberOfMatchedStations() if abs(lepton.pdgId()) == 13 else 4, help="Number of matched muons stations (4 for electrons)"),
    NTupleVariable("trkKink",      lambda lepton : lepton.combinedQuality().trkKink if abs(lepton.pdgId()) == 13 else 0, help="Tracker kink-finder"), 
    NTupleVariable("caloCompatibility",      lambda lepton : lepton.caloCompatibility() if abs(lepton.pdgId()) == 13 else 0, help="Calorimetric compatibility"), 
    NTupleVariable("globalTrackChi2",      lambda lepton : lepton.globalTrack().normalizedChi2() if abs(lepton.pdgId()) == 13 and lepton.globalTrack().isNonnull() else 0, help="Global track normalized chi2"), 
    # Extra tracker-related id variables
    NTupleVariable("trackerLayers", lambda x : (x.track() if abs(x.pdgId())==13 else x.gsfTrack()).hitPattern().trackerLayersWithMeasurement(), int, help="Tracker Layers"),
    NTupleVariable("pixelLayers", lambda x : (x.track() if abs(x.pdgId())==13 else x.gsfTrack()).hitPattern().pixelLayersWithMeasurement(), int, help="Pixel Layers"),
    # TTH-id related variables
    NTupleVariable("mvaTTH",     lambda lepton : lepton.mvaValue if hasattr(lepton,'mvaValue') else -1, help="Lepton MVA (ttH version)"),
    NTupleVariable("jetPtRatio", lambda lepton : lepton.pt()/lepton.jet.pt() if hasattr(lepton,'jet') else -1, help="pt(lepton)/pt(nearest jet)"),
    NTupleVariable("jetBTagCSV", lambda lepton : lepton.jet.btag('combinedSecondaryVertexBJetTags') if hasattr(lepton,'jet') and hasattr(lepton.jet, 'btag') else -99, help="btag of nearest jet"),
    NTupleVariable("jetDR",      lambda lepton : deltaR(lepton.eta(),lepton.phi(),lepton.jet.eta(),lepton.jet.phi()) if hasattr(lepton,'jet') else -1, help="deltaR(lepton, nearest jet)"),
    # MC-match info
    NTupleVariable("mcMatchId",  lambda x : x.mcMatchId, int, mcOnly=True, help="Match to source from hard scatter (25 for H, 6 for t, 23/24 for W/Z)"),
    NTupleVariable("mcMatchAny",  lambda x : x.mcMatchAny, int, mcOnly=True, help="Match to any final state leptons: -mcMatchId if prompt, 0 if unmatched, 1 if light flavour, 2 if heavy flavour (b)"),
    NTupleVariable("mcMatchTau",  lambda x : x.mcMatchTau, int, mcOnly=True, help="True if the leptons comes from a tau"),
])

leptonTypeSusyExtra = NTupleObjectType("leptonSusyExtra", baseObjectTypes = [ leptonTypeSusy ], variables = [
    # IVF variables
    NTupleVariable("hasSV",   lambda x : (2 if getattr(x,'ivfAssoc','') == "byref" else (0 if getattr(x,'ivf',None) == None else 1)), int, help="2 if lepton track is from a SV, 1 if loosely matched, 0 if no SV found."),
    NTupleVariable("svRedPt", lambda x : getattr(x, 'ivfRedPt', 0), help="pT of associated SV, removing the lepton track"),
    NTupleVariable("svRedM",  lambda x : getattr(x, 'ivfRedM', 0), help="mass of associated SV, removing the lepton track"),
    NTupleVariable("svLepSip3d", lambda x : getattr(x, 'ivfSip3d', 0), help="sip3d of lepton wrt SV"),
    NTupleVariable("svSip3d", lambda x : x.ivf.d3d.significance() if getattr(x,'ivf',None) != None else -99, help="S_{ip3d} of associated SV"),
    NTupleVariable("svNTracks", lambda x : x.ivf.numberOfDaughters() if getattr(x,'ivf',None) != None else -99, help="Number of tracks of associated SV"),
    NTupleVariable("svChi2n", lambda x : x.vertexChi2()/x.ivf.vertexNdof() if getattr(x,'ivf',None) != None else -99, help="Normalized chi2 of associated SV"),
    NTupleVariable("svDxy", lambda x : x.ivf.dxy.value() if getattr(x,'ivf',None) != None else -99, help="dxy of associated SV"),
    NTupleVariable("svM", lambda x : x.ivf.mass() if getattr(x,'ivf',None) != None else -99, help="mass of associated SV"),
    NTupleVariable("svPt", lambda x : x.ivf.pt() if getattr(x,'ivf',None) != None else -99, help="pt of associated SV"),
    NTupleVariable("svMCMatchFraction", lambda x : x.ivf.mcMatchFraction if getattr(x,'ivf',None) != None else -99, mcOnly=True, help="Fraction of mc-matched tracks from b/c matched to a single hadron (if >= 2 tracks found), for associated SV"),
    # MVA muon ID variables
    NTupleVariable("segmentCompatibility",      lambda lepton : lepton.segmentCompatibility() if abs(lepton.pdgId()) == 13 else 0, help="Segment-based compatibility"), 
    NTupleVariable("innerTrackValidHitFraction",      lambda lepton : lepton.innerTrack().validFraction() if abs(lepton.pdgId()) == 13 else 0, help="Calorimetric compatibility"), 
    NTupleVariable("chi2LocalPosition",      lambda lepton : lepton.combinedQuality().chi2LocalPosition if abs(lepton.pdgId()) == 13 else 0, help="Calorimetric compatibility"), 
    NTupleVariable("chi2LocalMomentum",      lambda lepton : lepton.combinedQuality().chi2LocalMomentum if abs(lepton.pdgId()) == 13 else 0, help="Calorimetric compatibility"), 
    NTupleVariable("glbTrackProbability",      lambda lepton : lepton.combinedQuality().glbTrackProbability if abs(lepton.pdgId()) == 13 else 0, help="Calorimetric compatibility"), 
    NTupleVariable("trackerHits", lambda x : (x.track() if abs(x.pdgId())==13 else x.gsfTrack()).hitPattern().numberOfValidTrackerHits(), int, help="Tracker hits"),
    NTupleVariable("lostOuterHits",    lambda x : x.gsfTrack().trackerExpectedHitsOuter().numberOfLostHits() if abs(x.pdgId())==11 else x.innerTrack().trackerExpectedHitsOuter().numberOfLostHits(), int, help="Number of lost hits on inner track"),
    # extra muon variables not used in MVA ID
    NTupleVariable("caloEMEnergy",      lambda lepton : lepton.calEnergy().em if abs(lepton.pdgId()) == 13 else 0, help="Calorimetric compatibility"), 
    NTupleVariable("caloHadEnergy",      lambda lepton : lepton.calEnergy().had if abs(lepton.pdgId()) == 13 else 0, help="Calorimetric compatibility"), 
    NTupleVariable("innerTrackChi2",      lambda lepton : lepton.innerTrack().normalizedChi2() if abs(lepton.pdgId()) == 13 and lepton.innerTrack().isNonnull() else 0, help="Inner track normalized chi2"), 
    NTupleVariable("stationsWithAnyHits",      lambda lepton : lepton.outerTrack().hitPattern().muonStationsWithAnyHits() if abs(lepton.pdgId()) == 13 and lepton.outerTrack().isNonnull() else 0, help="Outer track stations with any hits"), 
    NTupleVariable("stationsWithValidHits",      lambda lepton : lepton.outerTrack().hitPattern().muonStationsWithValidHits() if abs(lepton.pdgId()) == 13 and lepton.outerTrack().isNonnull() else 0, help="Outer track stations with valid hits"), 
    NTupleVariable("stationsWithValidHitsGlbTrack",      lambda lepton : lepton.globalTrack().hitPattern().muonStationsWithValidHits() if abs(lepton.pdgId()) == 13 and lepton.globalTrack().isNonnull() else 0, help="Global track stations with valid hits"), 
])

leptonTypeSusyFR = NTupleObjectType("leptonSusyFR", baseObjectTypes = [ leptonTypeTTH ], variables = [
    NTupleVariable("relIso03",  lambda x : x.relIso03, help="PF Rel Iso, R, with deltaBeta correction"),
    NTupleVariable("looseFakeId", lambda x : x.looseFakeId, int, help="Loose ID for Susy Fake Rate exercise"),
    NTupleVariable("tightFakeId", lambda x : x.tightFakeId, int, help="Tight ID for Susy Fake Rate exercise"),
    #NTupleVariable("rho",  lambda x : x.rho if hasattr(x, 'rho') else -99.0, help="PF Rel Iso, R, with deltaBeta correction"),
    #NTupleVariable("EA",  lambda x : x.EffectiveArea if hasattr(x, 'EffectiveArea') else -99.0, help="PF Rel Iso, R, with deltaBeta correction"),
    #NTupleVariable("relIso03_ch",  lambda x : x.chargedHadronIso(0.3) if abs(x.pdgId())==11 else 0, help="PF Rel Iso, R, with deltaBeta correction"),
    #NTupleVariable("relIso03_nh",  lambda x : x.neutralHadronIso(0.3) if abs(x.pdgId())==11 else 0, help="PF Rel Iso, R, with deltaBeta correction"),
    #NTupleVariable("relIso03_ph",  lambda x : x.photonIso(0.3)        if abs(x.pdgId())==11 else 0, help="PF Rel Iso, R, with deltaBeta correction"),
])
leptonTypeFull = NTupleObjectType("leptonFull", baseObjectTypes = [ leptonTypeSusy ], variables = [
    NTupleVariable("pfMuonId",    lambda x : x.muonID("POG_ID_Loose") if abs(x.pdgId())==13 else 1, int, help="Muon POG Loose id"),
    NTupleVariable("softMuonId",    lambda x : x.muonID("POG_ID_Soft") if abs(x.pdgId())==13 else 1, int, help="Muon POG Soft id"),
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
])
tauTypeTTH = NTupleObjectType("tauTTH", baseObjectTypes = [ tauType ], variables = [
    NTupleVariable("mcMatchId",  lambda x : x.mcMatchId, int, mcOnly=True, help="Match to source from hard scatter (25 for H, 6 for t, 23/24 for W/Z)"),
])
tauTypeSusy = NTupleObjectType("tauSusy", baseObjectTypes = [ tauType ], variables = [
    NTupleVariable("mcMatchId",  lambda x : x.mcMatchId, int, mcOnly=True, help="Match to source from hard scatter (25 for H, 6 for t, 23/24 for W/Z)"),
])

##------------------------------------------  
##  ISOTRACK
##------------------------------------------  

isoTrackType = NTupleObjectType("isoTrack",  baseObjectTypes = [ particleType ], variables = [
    NTupleVariable("charge",   lambda x : x.charge(), int),
    NTupleVariable("dz",    lambda x : x.dz() , help="d_{z} of lead track with respect to PV, in cm (with sign)"),
    NTupleVariable("absIso",  lambda x : x.absIso, float, mcOnly=False, help="abs charged iso with condition for isolation such that Min(0.2*pt, 8 GeV)"),
    NTupleVariable("mcMatchId",  lambda x : x.mcMatchId, int, mcOnly=True, help="Match to source from hard scatter (25 for H, 6 for t, 23/24 for W/Z)"),
])


##------------------------------------------  
## PHOTON
##------------------------------------------  

photonTypeSusy = NTupleObjectType("gamma", baseObjectTypes = [ particleType ], variables = [
    NTupleVariable("idCutBased", lambda x : x.idCutBased, int, help="1,2,3 if the gamma passes the loose, medium, tight WP of PhotonCutBasedID"),
    NTupleVariable("hOverE",  lambda x : x.hOVERe(), float, help="hoverE for photons"),
    NTupleVariable("r9",  lambda x : x.r9(), float, help="r9 for photons"),
    NTupleVariable("sigmaIetaIeta",  lambda x : x.sigmaIetaIeta(), float, help="sigmaIetaIeta for photons"),
    NTupleVariable("chHadIso",  lambda x : x.chargedHadronIso(), float, help="chargedHadronIsolation for photons"),
    NTupleVariable("neuHadIso",  lambda x : x.neutralHadronIso(), float, help="neutralHadronIsolation for photons"),
    NTupleVariable("phIso",  lambda x : x.photonIso(), float, help="gammaIsolation for photons"),
    NTupleVariable("mcMatchId",  lambda x : x.mcMatchId, int, mcOnly=True, help="Match to source from hard scatter (25 for H, 6 for t, 23/24 for W/Z)"),
])

##------------------------------------------  
## JET
##------------------------------------------  

jetType = NTupleObjectType("jet",  baseObjectTypes = [ fourVectorType ], variables = [
    NTupleVariable("btagCSV",   lambda x : x.btag('combinedSecondaryVertexBJetTags'), help="CSV discriminator"),
    NTupleVariable("rawPt",     lambda x : x.pt() * x.rawFactor(), help="p_{T} before JEC"),
    NTupleVariable("mcPt",      lambda x : x.mcJet.pt() if x.mcJet else 0., mcOnly=True, help="p_{T} of associated gen jet"),
    NTupleVariable("mcFlavour", lambda x : x.partonFlavour(), int,     mcOnly=True, help="parton flavour (physics definition, i.e. including b's from shower)"),
    #NTupleVariable("quarkGluonID", lambda x : x.QG,  mcOnly=False),
])
jetTypeTTH = NTupleObjectType("jetTTH",  baseObjectTypes = [ jetType ], variables = [
    NTupleVariable("mcMatchId",    lambda x : x.mcMatchId,   int, mcOnly=True, help="Match to source from hard scatter (25 for H, 6 for t, 23/24 for W/Z)"),
    NTupleVariable("mcMatchFlav",  lambda x : x.mcMatchFlav, int, mcOnly=True, help="Flavour of associated parton from hard scatter (if any)"),
])
jetTypeSusy = NTupleObjectType("jetSusy",  baseObjectTypes = [ jetType ], variables = [
    NTupleVariable("mcMatchId",    lambda x : x.mcMatchId,   int, mcOnly=True, help="Match to source from hard scatter (25 for H, 6 for t, 23/24 for W/Z)"),
    NTupleVariable("mcMatchFlav",  lambda x : x.mcMatchFlav, int, mcOnly=True, help="Flavour of associated parton from hard scatter (if any)"),
    NTupleVariable("area",   lambda x : x.jetArea(), help="Catchment area of jet"),
    NTupleVariable("puId", lambda x : x.puJetIdPassed, int,     mcOnly=False, help="puId (full MVA, loose WP, 5.3.X training on AK5PFchs: the only thing that is available now)"),
    #NTupleVariable("PuId_full", lambda x : x.puId("full"), int,     mcOnly=False, help="puId full: returns an integeger containing 3 bits, one for each working point (loose-bit2, medium-bit1, tight-bit0)"),
    #NTupleVariable("PuId_simple", lambda x : x.puId("simple"), int,    mcOnly=False, help="puId simple: returns an integeger containing 3 bits, one for each working point (loose-bit2, medium-bit1, tight-bit0)"),
    #NTupleVariable("PuId_cut_based", lambda x : x.puId("cut-based"), int,    mcOnly=False, help="puId cut-based: returns an integeger containing 3 bits, one for each working point (loose-bit2, medium-bit1, tight-bit0)"),
    NTupleVariable("id",    lambda x : x.jetID("POG_PFID") , int, mcOnly=False,help="POG Loose jet ID"),
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
genParticleWithSourceType = NTupleObjectType("genParticleWithSource", baseObjectTypes = [ genParticleType ], mcOnly=True, variables = [
    NTupleVariable("sourceId", lambda x : x.sourceId, int, help="origin of the particle: 6=t, 25=h, 23/24=W/Z")
])
genParticleWithMotherId = NTupleObjectType("genParticleWithMotherId", baseObjectTypes = [ genParticleType ], mcOnly=True, variables = [
    #NTupleVariable("motherId", lambda x : x.motherId, int, help="pdgId of the mother of the particle")
    NTupleVariable("motherId", lambda x : x.mother(0).pdgId() if x.mother(0) else 0, int, help="pdgId of the mother of the particle"),
    NTupleVariable("grandmaId", lambda x : x.mother(0).mother(0).pdgId() if x.mother(0) and x.mother(0).mother(0) else 0, int, help="pdgId of the grandmother of the particle")
])

##------------------------------------------  
## SECONDARY VERTEX CANDIDATE
##------------------------------------------  
  
svType = NTupleObjectType("sv", baseObjectTypes = [ fourVectorType ], variables = [
    NTupleVariable("charge",   lambda x : x.charge(), int),
    NTupleVariable("ntracks", lambda x : x.numberOfDaughters(), int, help="Number of tracks (with weight > 0.5)"),
    NTupleVariable("chi2", lambda x : x.vertexChi2(), help="Chi2 of the vertex fit"),
    NTupleVariable("ndof", lambda x : x.vertexNdof(), help="Degrees of freedom of the fit, ndof = (2*ntracks - 3)" ),
    NTupleVariable("dxy",  lambda x : x.dxy.value(), help="Transverse distance from the PV [cm]"),
    NTupleVariable("edxy", lambda x : x.dxy.error(), help="Uncertainty on the transverse distance from the PV [cm]"),
    NTupleVariable("ip3d",  lambda x : x.d3d.value(), help="3D distance from the PV [cm]"),
    NTupleVariable("eip3d", lambda x : x.d3d.error(), help="Uncertainty on the 3D distance from the PV [cm]"),
    NTupleVariable("sip3d", lambda x : x.d3d.significance(), help="S_{ip3d} with respect to PV (absolute value)"),
    NTupleVariable("cosTheta", lambda x : x.cosTheta, help="Cosine of the angle between the 3D displacement and the momentum"),
    NTupleVariable("jetPt",  lambda x : x.jet.pt() if x.jet != None else 0, help="pT of associated jet"),
    NTupleVariable("jetBTag",  lambda x : x.jet.btag('combinedSecondaryVertexBJetTags') if x.jet != None else -99, help="CSV b-tag of associated jet"),
    NTupleVariable("mcMatchNTracks", lambda x : x.mcMatchNTracks, int, mcOnly=True, help="Number of mc-matched tracks in SV"),
    NTupleVariable("mcMatchNTracksHF", lambda x : x.mcMatchNTracksHF, int, mcOnly=True, help="Number of mc-matched tracks from b/c in SV"),
    NTupleVariable("mcMatchFraction", lambda x : x.mcMatchFraction, mcOnly=True, help="Fraction of mc-matched tracks from b/c matched to a single hadron (or -1 if mcMatchNTracksHF < 2)"),
    NTupleVariable("mcFlavFirst", lambda x : x.mcFlavFirst, int, mcOnly=True, help="Flavour of last ancestor with maximum number of matched daughters"),
    NTupleVariable("mcFlavHeaviest", lambda x : x.mcFlavHeaviest, int, mcOnly=True, help="Flavour of heaviest hadron with maximum number of matched daughters"),
])

heavyFlavourHadronType = NTupleObjectType("heavyFlavourHadron", baseObjectTypes = [ genParticleType ], variables = [
    NTupleVariable("flav", lambda x : x.flav, int, mcOnly=True, help="Flavour"),
    NTupleVariable("sourceId", lambda x : x.sourceId, int, mcOnly=True, help="pdgId of heaviest mother particle (stopping at the first one heaviest than 175 GeV)"),
    NTupleVariable("svMass",   lambda x : x.sv.mass() if x.sv else 0, help="SV: mass"),
    NTupleVariable("svPt",   lambda x : x.sv.pt() if x.sv else 0, help="SV: pt"),
    NTupleVariable("svCharge",   lambda x : x.sv.charge() if x.sv else -99., int, help="SV: charge"),
    NTupleVariable("svNtracks", lambda x : x.sv.numberOfDaughters() if x.sv else 0, int, help="SV: Number of tracks (with weight > 0.5)"),
    NTupleVariable("svChi2", lambda x : x.sv.vertexChi2() if x.sv else -99., help="SV: Chi2 of the vertex fit"),
    NTupleVariable("svNdof", lambda x : x.sv.vertexNdof() if x.sv else -99., help="SV: Degrees of freedom of the fit, ndof = (2*ntracks - 3)" ),
    NTupleVariable("svDxy",  lambda x : x.sv.dxy.value() if x.sv else -99., help="SV: Transverse distance from the PV [cm]"),
    NTupleVariable("svEdxy", lambda x : x.sv.dxy.error() if x.sv else -99., help="SV: Uncertainty on the transverse distance from the PV [cm]"),
    NTupleVariable("svIp3d",  lambda x : x.sv.d3d.value() if x.sv else -99., help="SV: 3D distance from the PV [cm]"),
    NTupleVariable("svEip3d", lambda x : x.sv.d3d.error() if x.sv else -99., help="SV: Uncertainty on the 3D distance from the PV [cm]"),
    NTupleVariable("svSip3d", lambda x : x.sv.d3d.significance() if x.sv else -99., help="SV: S_{ip3d} with respect to PV (absolute value)"),
    NTupleVariable("svCosTheta", lambda x : x.sv.cosTheta if x.sv else -99., help="SV: Cosine of the angle between the 3D displacement and the momentum"),
    NTupleVariable("jetPt",  lambda x : x.jet.pt() if x.jet != None else 0, help="Jet: pT"),
    NTupleVariable("jetBTag",  lambda x : x.jet.btag('combinedSecondaryVertexBJetTags') if x.jet != None else -99, help="CSV b-tag of associated jet"),
])


