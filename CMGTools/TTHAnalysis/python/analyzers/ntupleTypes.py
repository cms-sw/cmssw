#!/bin/env python
from math import *
from PhysicsTools.Heppy.analyzers.core.autovars import NTupleObjectType  
from PhysicsTools.Heppy.analyzers.objects.autophobj import  *
from PhysicsTools.HeppyCore.utils.deltar import deltaR

from CMGTools.TTHAnalysis.signedSip import *

##------------------------------------------  
## LEPTON
##------------------------------------------  

leptonTypeSusy = NTupleObjectType("leptonSusy", baseObjectTypes = [ leptonType ], variables = [
    #NTupleVariable("eleMVAId",     lambda x : (x.electronID("POG_MVA_ID_NonTrig_full5x5") + 2*x.electronID("POG_MVA_ID_Trig_full5x5")) if abs(x.pdgId()) == 11 else -1, int, help="Electron mva id working point (2012, full5x5 shapes): 0=none, 1=non-trig, 2=trig, 3=both"),
    #NTupleVariable("mvaId",         lambda lepton : lepton.mvaNonTrigV0(full5x5=True) if abs(lepton.pdgId()) == 11 else 1, help="EGamma POG MVA ID for non-triggering electrons (as HZZ); 1 for muons"),
    #NTupleVariable("mvaIdTrig",     lambda lepton : lepton.mvaTrigV0(full5x5=True)    if abs(lepton.pdgId()) == 11 else 1, help="EGamma POG MVA ID for triggering electrons; 1 for muons"),
    NTupleVariable("mvaIdPhys14",   lambda lepton : lepton.mvaRun2("NonTrigPhys14") if abs(lepton.pdgId()) == 11 else 1, help="EGamma POG MVA ID for non-triggering electrons, Phys14 re-training; 1 for muons"),
    # Lepton MVA-id related variables
    NTupleVariable("mvaTTH",    lambda lepton : getattr(lepton, 'mvaValueTTH', -1), help="Lepton MVA (TTH version)"),
    NTupleVariable("mvaSusy",    lambda lepton : getattr(lepton, 'mvaValueSusy', -1), help="Lepton MVA (SUSY version)"),
    NTupleVariable("jetPtRatio", lambda lepton : lepton.pt()/lepton.jet.pt() if hasattr(lepton,'jet') else -1, help="pt(lepton)/pt(nearest jet)"),
    NTupleVariable("jetPtRel", lambda lepton : ptRelv1(lepton.p4(),lepton.jet.p4()) if hasattr(lepton,'jet') else -1, help="pt of the lepton transverse to the jet axis (subtracting the lepton)"),
    NTupleVariable("jetBTagCSV", lambda lepton : lepton.jet.btag('pfCombinedInclusiveSecondaryVertexV2BJetTags') if hasattr(lepton,'jet') and hasattr(lepton.jet, 'btag') else -99, help="CSV btag of nearest jet"),
    NTupleVariable("jetBTagCMVA", lambda lepton : lepton.jet.btag('pfCombinedMVABJetTags') if hasattr(lepton,'jet') and hasattr(lepton.jet, 'btag') else -99, help="CMA btag of nearest jet"),
    NTupleVariable("jetDR",      lambda lepton : deltaR(lepton.eta(),lepton.phi(),lepton.jet.eta(),lepton.jet.phi()) if hasattr(lepton,'jet') else -1, help="deltaR(lepton, nearest jet)"),
])


leptonTypeSusyExtra = NTupleObjectType("leptonSusyExtra", baseObjectTypes = [ leptonTypeSusy, leptonTypeExtra ], variables = [
    NTupleVariable("miniRelIsoCharged",   lambda x : getattr(x,'miniAbsIsoCharged',-99)/x.pt()),
    NTupleVariable("miniRelIsoNeutral",   lambda x : getattr(x,'miniAbsIsoNeutral',-99)/x.pt()),
    # IVF variables
    NTupleVariable("hasSV",   lambda x : (2 if getattr(x,'ivfAssoc','') == "byref" else (0 if getattr(x,'ivf',None) == None else 1)), int, help="2 if lepton track is from a SV, 1 if loosely matched, 0 if no SV found."),
    NTupleVariable("svRedPt", lambda x : getattr(x, 'ivfRedPt', 0), help="pT of associated SV, removing the lepton track"),
    NTupleVariable("svRedM",  lambda x : getattr(x, 'ivfRedM', 0), help="mass of associated SV, removing the lepton track"),
    NTupleVariable("svLepSip3d", lambda x : getattr(x, 'ivfSip3d', 0), help="sip3d of lepton wrt SV"),
    NTupleVariable("svSip3d", lambda x : x.ivf.d3d.significance() if getattr(x,'ivf',None) != None else -99, help="S_{ip3d} of associated SV"),
    NTupleVariable("svNTracks", lambda x : x.ivf.numberOfDaughters() if getattr(x,'ivf',None) != None else -99, help="Number of tracks of associated SV"),
    NTupleVariable("svChi2n", lambda x : x.ivf.vertexChi2()/x.ivf.vertexNdof() if getattr(x,'ivf',None) != None else -99, help="Normalized chi2 of associated SV"),
    NTupleVariable("svDxy", lambda x : x.ivf.dxy.value() if getattr(x,'ivf',None) != None else -99, help="dxy of associated SV"),
    NTupleVariable("svMass", lambda x : x.ivf.mass() if getattr(x,'ivf',None) != None else -99, help="mass of associated SV"),
    NTupleVariable("svPt", lambda x : x.ivf.pt() if getattr(x,'ivf',None) != None else -99, help="pt of associated SV"),
    NTupleVariable("svMCMatchFraction", lambda x : x.ivf.mcMatchFraction if getattr(x,'ivf',None) != None else -99, mcOnly=True, help="Fraction of mc-matched tracks from b/c matched to a single hadron (if >= 2 tracks found), for associated SV"),
    NTupleVariable("svMva", lambda x : x.ivf.mva if getattr(x,'ivf',None) != None else -99, help="mva value of associated SV"),
    # Additional jet-lepton related variables
    NTupleVariable("jetNDau",    lambda lepton : lepton.jet.numberOfDaughters() if hasattr(lepton,'jet') and lepton.jet != lepton else -1, help="n daughters of nearest jet"),
    NTupleVariable("jetNDauCharged",    lambda lepton : sum(x.charge()!=0 for x in lepton.jet.daughterPtrVector()) if hasattr(lepton,'jet') and lepton.jet != lepton else -1, help="n charged daughters of nearest jet"),
    NTupleVariable("jetNDauPV",    lambda lepton : sum(x.charge()!=0 and x.fromPV()==3 for x in lepton.jet.daughterPtrVector()) if hasattr(lepton,'jet') and lepton.jet != lepton else -1, help="n charged daughters from PV of nearest jet"),
    NTupleVariable("jetNDauNotPV",    lambda lepton : sum(x.charge()!=0 and x.fromPV()<=2 for x in lepton.jet.daughterPtrVector()) if hasattr(lepton,'jet') and lepton.jet != lepton else -1, help="n charged daughters from PV of nearest jet"),
    NTupleVariable("jetmaxSignedSip3D",    lambda lepton :  maxSignedSip3Djettracks(lepton), help="max signed Sip3D among jet's tracks"),
    NTupleVariable("jetmaxSip3D",    lambda lepton :   maxSip3Djettracks(lepton), help="max Sip3D among jet's tracks"),
    NTupleVariable("jetmaxSignedSip2D",    lambda lepton  : maxSignedSip2Djettracks(lepton) , help="max signed Sip2D among jet's tracks"),
    NTupleVariable("jetmaxSip2D",    lambda lepton :   maxSip2Djettracks(lepton), help="max Sip2D among jet's tracks"),
    NTupleVariable("jetPtRelv0",   lambda lepton : ptRel(lepton.p4(),lepton.jet.p4()) if hasattr(lepton,'jet') else -1, help="pt of the lepton transverse to the jet axis (not subtracting the lepton)"),
    NTupleVariable("jetMass",      lambda lepton : lepton.jet.mass() if hasattr(lepton,'jet') else -1, help="Mass of associated jet"),
    NTupleVariable("jetPrunedMass",      lambda lepton : getattr(lepton.jet, 'prunedP4', lepton.jet.p4()).M() if hasattr(lepton,'jet') else -1, help="Pruned mass of associated jet"),
    NTupleVariable("jetDecDR",      lambda lepton : lepton.jetDecDR if hasattr(lepton,'jetDecDR') else -1, help="deltaR(lepton, nearest jet) after declustering"),
    NTupleVariable("jetDecPtRel", lambda lepton : lepton.jetDecPtRel if hasattr(lepton,'jetDecPtRel') else -1, help="pt of the lepton transverse to the jet axis (subtracting the lepton), after declustering"),
    NTupleVariable("jetDecPtRatio", lambda lepton :  lepton.jetDecPtRatio if hasattr(lepton,'jetDecPtRatio') else -1, help="pt(lepton)/pt(nearest jet) after declustering"),
    NTupleVariable("jetDecPrunedMass", lambda lepton :  lepton.jetDecPrunedMass if hasattr(lepton,'jetDecPrunedMass') else -1, help="pt(lepton)/pt(nearest jet) after declustering and pruning"),
    NTupleVariable("jetDecPrunedPtRatio", lambda lepton :  lepton.jetDecPrunedPtRatio if hasattr(lepton,'jetDecPrunedPtRatio') else -1, help="pt(lepton)/pt(nearest jet) after declustering and pruning"),
    NTupleVariable("jetDec02DR",      lambda lepton : lepton.jetDec02DR if hasattr(lepton,'jetDec02DR') else -1, help="deltaR(lepton, nearest jet) after declustering 02"),
    NTupleVariable("jetDec02PtRel", lambda lepton : lepton.jetDec02PtRel if hasattr(lepton,'jetDec02PtRel') else -1, help="pt of the lepton transverse to the jet axis (subtracting the lepton), after declustering 02"),
    NTupleVariable("jetDec02PtRatio", lambda lepton :  lepton.jetDec02PtRatio if hasattr(lepton,'jetDec02PtRatio') else -1, help="pt(lepton)/pt(nearest jet) after declustering 02"),
    NTupleVariable("jetDec02PrunedPtRatio", lambda lepton :  lepton.jetDec02PrunedPtRatio if hasattr(lepton,'jetDec02PrunedPtRatio') else -1, help="pt(lepton)/pt(nearest jet) after declustering 02 and pruning"),
    NTupleVariable("jetDec02PrunedMass", lambda lepton :  lepton.jetDec02PrunedMass if hasattr(lepton,'jetDec02PrunedMass') else -1, help="pt(lepton)/pt(nearest jet) after declustering 02 and pruning"),

])


##------------------------------------------  
## TAU
##------------------------------------------  

tauTypeSusy = NTupleObjectType("tauSusy",  baseObjectTypes = [ tauType ], variables = [
])

##------------------------------------------  
##  ISOTRACK
##------------------------------------------  

isoTrackTypeSusy = NTupleObjectType("isoTrackSusy",  baseObjectTypes = [ isoTrackType ], variables = [
])


##------------------------------------------  
## PHOTON
##------------------------------------------  

photonTypeSusy = NTupleObjectType("gammaSusy", baseObjectTypes = [ photonType ], variables = [
    NTupleVariable("genIso04",  lambda x : getattr(x, 'genIso04', -1.0), float, mcOnly=True, help="sum pt of all status 1 particles within DeltaR = 0.4 of the photon"),
    NTupleVariable("genIso",  lambda x : getattr(x, 'genIso03', -1.0), float, mcOnly=True, help="sum pt of all status 1 particles within DeltaR = 0.3 of the photon"),
    NTupleVariable("chHadIsoRC04",  lambda x : getattr(x, 'chHadIsoRC04', -1.0), float, mcOnly=False, help="charged iso 0.4 in a random cone 90 degrees in phi from photon"),
    NTupleVariable("chHadIsoRC",  lambda x : getattr(x, 'chHadIsoRC03', -1.0), float, mcOnly=False, help="charged iso 0.3 in a random cone 90 degrees in phi from photon"),
])

##------------------------------------------  
## JET
##------------------------------------------  

jetTypeSusy = NTupleObjectType("jetSusy",  baseObjectTypes = [ jetTypeExtra ], variables = [
    NTupleVariable("mcMatchFlav",  lambda x : x.mcMatchFlav, int, mcOnly=True, help="Flavour of associated parton from hard scatter (if any)"),
])

jetTypeSusyExtra = NTupleObjectType("jetSusyExtra",  baseObjectTypes = [ jetTypeSusy ], variables = [
    NTupleVariable("prunedMass", lambda x : x.prunedP4.M() if hasattr(x,'prunedP4') else x.mass(), float, help="Pruned mass"),
    NTupleVariable("mcNumPartons", lambda x : getattr(x,'mcNumPartons',-1),int, mcOnly=True, help="Number of matched partons (quarks, photons)"),
    NTupleVariable("mcNumLeptons", lambda x : getattr(x,'mcNumLeptons',-1),int, mcOnly=True, help="Number of matched leptons"),
    NTupleVariable("mcNumTaus", lambda x : getattr(x,'mcNumTaus',-1),int, mcOnly=True, help="Number of matched taus"),
    NTupleVariable("mcAnyPartonMass", lambda x : getattr(x,"mcAnyPartonMass",-1),float, mcOnly=True, help="Mass of associated partons, leptons, taus"),
    NTupleVariable("nSubJets", lambda x : getattr(x, "nSubJets", 0), int, help="Number of subjets (kt, R=0.2)"), 
    NTupleVariable("nSubJets25", lambda x : getattr(x, "nSubJets25", 0), int, help="Number of subjets with pt > 25 (kt, R=0.2)"), 
    NTupleVariable("nSubJets30", lambda x : getattr(x, "nSubJets30", 0), int, help="Number of subjets with pt > 30 (kt, R=0.2)"), 
    NTupleVariable("nSubJets40", lambda x : getattr(x, "nSubJets40", 0), int, help="Number of subjets with pt > 40 (kt, R=0.2)"), 
    NTupleVariable("nSubJetsZ01", lambda x : getattr(x, "nSubJetsZ01", 0), int, help="Number of subjets with pt > 0.1 * pt(jet) (kt, R=0.2)"), 
    # --------------- 
    NTupleVariable("chHEF", lambda x : x.chargedHadronEnergyFraction(), float, mcOnly = False, help="chargedHadronEnergyFraction (relative to uncorrected jet energy)"),
    NTupleVariable("neHEF", lambda x : x.neutralHadronEnergyFraction(), float, mcOnly = False,help="neutralHadronEnergyFraction (relative to uncorrected jet energy)"),
    NTupleVariable("phEF", lambda x : x.photonEnergyFraction(), float, mcOnly = False,help="photonEnergyFraction (relative to corrected jet energy)"),
    NTupleVariable("eEF", lambda x : x.electronEnergyFraction(), float, mcOnly = False,help="electronEnergyFraction (relative to corrected jet energy)"),
    NTupleVariable("muEF", lambda x : x.muonEnergyFraction(), float, mcOnly = False,help="muonEnergyFraction (relative to corrected jet energy)"),
    NTupleVariable("HFHEF", lambda x : x.HFHadronEnergyFraction(), float, mcOnly = False,help="HFHadronEnergyFraction (relative to corrected jet energy)"),
    NTupleVariable("HFEMEF", lambda x : x.HFEMEnergyFraction(), float, mcOnly = False,help="HFEMEnergyFraction (relative to corrected jet energy)"),
    NTupleVariable("chHMult", lambda x : x.chargedHadronMultiplicity(), int, mcOnly = False,help="chargedHadronMultiplicity from PFJet.h"),
    NTupleVariable("neHMult", lambda x : x.neutralHadronMultiplicity(), int, mcOnly = False,help="neutralHadronMultiplicity from PFJet.h"),
    NTupleVariable("phMult", lambda x : x.photonMultiplicity(), int, mcOnly = False,help="photonMultiplicity from PFJet.h"),
    NTupleVariable("eMult", lambda x : x.electronMultiplicity(), int, mcOnly = False,help="electronMultiplicity from PFJet.h"),
    NTupleVariable("muMult", lambda x : x.muonMultiplicity(), int, mcOnly = False,help="muonMultiplicity from PFJet.h"),
    NTupleVariable("HFHMult", lambda x : x.HFHadronMultiplicity(), int, mcOnly = False,help="HFHadronMultiplicity from PFJet.h"),
    NTupleVariable("HFEMMult", lambda x : x.HFEMMultiplicity(), int, mcOnly = False,help="HFEMMultiplicity from PFJet.h"),
])

fatJetType = NTupleObjectType("fatJet",  baseObjectTypes = [ jetType ], variables = [
    NTupleVariable("prunedMass",  lambda x : x.userFloat("ak8PFJetsCHSPrunedLinks"),  float, help="pruned mass"),
    NTupleVariable("trimmedMass", lambda x : x.userFloat("ak8PFJetsCHSTrimmedLinks"), float, help="trimmed mass"),
    NTupleVariable("filteredMass", lambda x : x.userFloat("ak8PFJetsCHSFilteredLinks"), float, help="filtered mass"),
    NTupleVariable("tau1", lambda x : x.userFloat("NjettinessAK8:tau1"), float, help="1-subjettiness"),
    NTupleVariable("tau2", lambda x : x.userFloat("NjettinessAK8:tau2"), float, help="2-subjettiness"),
    NTupleVariable("tau3", lambda x : x.userFloat("NjettinessAK8:tau3"), float, help="3-subjettiness"),
    NTupleVariable("topMass", lambda x : (x.tagInfo("caTop").properties().topMass if x.tagInfo("caTop") else -99), float, help="CA8 jet topMass"),
    NTupleVariable("minMass", lambda x : (x.tagInfo("caTop").properties().minMass if x.tagInfo("caTop") else -99), float, help="CA8 jet minMass"),
    NTupleVariable("nSubJets", lambda x : (x.tagInfo("caTop").properties().nSubJets if x.tagInfo("caTop") else -99), float, help="CA8 jet nSubJets"),
])
      
##------------------------------------------  
## MET
##------------------------------------------  
  
metTypeSusy = NTupleObjectType("metSusy", baseObjectTypes = [ metType ], variables = [
])

##------------------------------------------  
## GENPARTICLE
##------------------------------------------  


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
    NTupleVariable("mva", lambda x : x.mva, help="MVA discriminator"),
    NTupleVariable("jetPt",  lambda x : x.jet.pt() if x.jet != None else 0, help="pT of associated jet"),
    NTupleVariable("jetBTagCSV",   lambda x : x.jet.btag('pfCombinedInclusiveSecondaryVertexV2BJetTags') if x.jet != None else -99, help="CSV b-tag of associated jet"),
    NTupleVariable("jetBTagCMVA",  lambda x : x.jet.btag('pfCombinedMVABJetTags') if x.jet != None else -99, help="CMVA b-tag of associated jet"),
    NTupleVariable("mcMatchNTracks", lambda x : getattr(x, 'mcMatchNTracks', -1), int, mcOnly=True, help="Number of mc-matched tracks in SV"),
    NTupleVariable("mcMatchNTracksHF", lambda x : getattr(x, 'mcMatchNTracksHF', -1), int, mcOnly=True, help="Number of mc-matched tracks from b/c in SV"),
    NTupleVariable("mcMatchFraction", lambda x : getattr(x, 'mcMatchFraction', -1), mcOnly=True, help="Fraction of mc-matched tracks from b/c matched to a single hadron (or -1 if mcMatchNTracksHF < 2)"),
    NTupleVariable("mcFlavFirst", lambda x : getattr(x,'mcFlavFirst', -1), int, mcOnly=True, help="Flavour of last ancestor with maximum number of matched daughters"),
    NTupleVariable("mcFlavHeaviest", lambda x : getattr(x,'mcFlavHeaviest', -1), int, mcOnly=True, help="Flavour of heaviest hadron with maximum number of matched daughters"),
    NTupleVariable("maxDxyTracks", lambda x : x.maxDxyTracks, help="highest |dxy| of vertex tracks"),
    NTupleVariable("secDxyTracks", lambda x : x.secDxyTracks, help="second highest |dxy| of vertex tracks"),
    NTupleVariable("maxD3dTracks", lambda x : x.maxD3dTracks, help="highest |ip3D| of vertex tracks"),
    NTupleVariable("secD3dTracks", lambda x : x.secD3dTracks, help="second highest |ip3D| of vertex tracks"),

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
    NTupleVariable("svMva", lambda x : x.sv.mva if x.sv else -99., help="SV: mva value"),
    NTupleVariable("jetPt",  lambda x : x.jet.pt() if x.jet != None else 0, help="Jet: pT"),
    NTupleVariable("jetBTagCSV",  lambda x : x.jet.btag('pfCombinedInclusiveSecondaryVertexV2BJetTags') if x.jet != None else -99, help="CSV b-tag of associated jet"),
    NTupleVariable("jetBTagCMVA",  lambda x : x.jet.btag('pfCombinedMVABJetTags') if x.jet != None else -99, help="CMVA b-tag of associated jet"),
    
])


def ptRel(p4,axis):
    a = ROOT.TVector3(axis.Vect().X(),axis.Vect().Y(),axis.Vect().Z())
    o = ROOT.TLorentzVector(p4.Px(),p4.Py(),p4.Pz(),p4.E())
    return o.Perp(a)
def ptRelv1(p4,axis):
    axis = axis - p4
    a = ROOT.TVector3(axis.Vect().X(),axis.Vect().Y(),axis.Vect().Z())
    o = ROOT.TLorentzVector(p4.Px(),p4.Py(),p4.Pz(),p4.E())
    return o.Perp(a)
