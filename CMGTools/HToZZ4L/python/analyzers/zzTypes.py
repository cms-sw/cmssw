from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer  import * 

leptonTypeHZZ = NTupleObjectType("leptonHZZ", baseObjectTypes = [ leptonTypeExtra ], variables = [
    #NTupleVariable("mvaIdPhys14",   lambda lepton : lepton.mvaRun2("NonTrigPhys14") if abs(lepton.pdgId()) == 11 else 1, help="EGamma POG MVA ID for non-triggering electrons, Phys14 re-training; 1 for muons"),
    NTupleVariable("mvaIdPhys14",   lambda lepton : lepton.mvaRun2("NonTrigPhys14Fix") if abs(lepton.pdgId()) == 11 else 1, help="EGamma POG MVA ID for non-triggering electrons, Phys14 re-training; 1 for muons"),
    # Extra isolation variables
    NTupleVariable("chargedHadIso04",   lambda x : x.chargedHadronIsoR(0.4),   help="PF Abs Iso, R=0.4, charged hadrons only"),
    NTupleVariable("neutralHadIso04",   lambda x : x.neutralHadronIsoR(0.4),   help="PF Abs Iso, R=0.4, neutral hadrons only"),
    NTupleVariable("photonIso04",       lambda x : x.photonIsoR(0.4),          help="PF Abs Iso, R=0.4, photons only"),
    NTupleVariable("puChargedHadIso04", lambda x : x.puChargedHadronIsoR(0.4), help="PF Abs Iso, R=0.4, pileup charged hadrons only"),
    NTupleVariable("rho",               lambda x : x.rho,                             help="rho for isolation"),
    NTupleVariable("EffectiveArea04",   lambda x : x.EffectiveArea04,                 help="EA for isolation"),
])

fsrPhotonTypeHZZ = NTupleObjectType("fsrPhotonHZZ", baseObjectTypes = [ particleType ], variables = [
    NTupleVariable("chargedHadIso",   lambda x : getattr(x,'absIsoCH',-1.0),   help="PF Abs Iso, R=0.3, charged hadrons only"),
    NTupleVariable("photonIso",       lambda x : getattr(x,'absIsoPH',-1.0),   help="PF Abs Iso, R=0.3, photons only"),
    NTupleVariable("neutralHadIso",   lambda x : getattr(x,'absIsoNH',-1.0),   help="PF Abs Iso, R=0.3, neutral hadrons only"),
    NTupleVariable("puChargedHadIso", lambda x : getattr(x,'absIsoPU',-1.0),   help="PF Abs Iso, R=0.3, pileup charged hadrons only"),
    NTupleVariable("relIso",          lambda x : getattr(x,'relIso', -1.0),    help="PF Rel Iso, R=0.3, charged + netural had + pileup"),
    NTupleSubObject("closestLepton",  lambda x : x.globalClosestLepton, particleType),
    NTupleVariable("closestLeptonDR", lambda x : deltaR(x.eta(),x.phi(),x.globalClosestLepton.eta(),x.globalClosestLepton.phi())),
]) 

ZZType = NTupleObjectType("ZZType", baseObjectTypes=[fourVectorType], variables = [
    NTupleVariable("hasFSR",   lambda x : x.hasFSR(), int),
    NTupleVariable("z1_hasFSR",   lambda x : x.leg1.hasFSR(), int),
    NTupleVariable("z2_hasFSR",   lambda x : x.leg2.hasFSR(), int),
    NTupleSubObject("z1",  lambda x : x.leg1,fourVectorType),
    NTupleSubObject("z2",  lambda x : x.leg2,fourVectorType),
    NTupleSubObject("z1_l1",  lambda x : x.leg1.leg1,leptonTypeHZZ),
    NTupleSubObject("z1_l2",  lambda x : x.leg1.leg2,leptonTypeHZZ),
    NTupleSubObject("z2_l1",  lambda x : x.leg2.leg1,leptonTypeHZZ),
    NTupleSubObject("z2_l2",  lambda x : x.leg2.leg2,leptonTypeHZZ),
    NTupleVariable("mll_12",   lambda x : (x.leg1.leg1.p4()+x.leg1.leg2.p4()).M()),
    NTupleVariable("mll_13",   lambda x : (x.leg1.leg1.p4()+x.leg2.leg1.p4()).M()),
    NTupleVariable("mll_14",   lambda x : (x.leg1.leg1.p4()+x.leg2.leg2.p4()).M()),
    NTupleVariable("mll_23",   lambda x : (x.leg1.leg2.p4()+x.leg2.leg1.p4()).M()),
    NTupleVariable("mll_24",   lambda x : (x.leg1.leg2.p4()+x.leg2.leg2.p4()).M()),
    NTupleVariable("mll_34",   lambda x : (x.leg2.leg1.p4()+x.leg2.leg2.p4()).M()),
    # -------
    NTupleVariable("z1_pho_pt",  lambda x : (x.leg1.fsrPhoton.pt()  if x.leg1.hasFSR() else -99.0) ),
    NTupleVariable("z1_pho_eta", lambda x : (x.leg1.fsrPhoton.eta() if x.leg1.hasFSR() else -99.0) ),
    NTupleVariable("z1_pho_phi", lambda x : (x.leg1.fsrPhoton.phi() if x.leg1.hasFSR() else -99.0) ),
    NTupleVariable("z2_pho_pt",  lambda x : (x.leg2.fsrPhoton.pt()  if x.leg2.hasFSR() else -99.0) ),
    NTupleVariable("z2_pho_eta", lambda x : (x.leg2.fsrPhoton.eta() if x.leg2.hasFSR() else -99.0) ),
    NTupleVariable("z2_pho_phi", lambda x : (x.leg2.fsrPhoton.phi() if x.leg2.hasFSR() else -99.0) ),
    # -------
    NTupleVariable("KD",   lambda x : getattr(x, 'KD', -1.0), help="MELA KD"),
    NTupleVariable("MELAcosthetastar", lambda x : x.melaAngles.costhetastar if hasattr(x,'melaAngles') else -99.0, help="MELA angle costhetastar"),
    NTupleVariable("MELAcostheta1", lambda x : x.melaAngles.costheta1 if hasattr(x,'melaAngles') else -99.0, help="MELA angle costheta1"),
    NTupleVariable("MELAcostheta2", lambda x : x.melaAngles.costheta2 if hasattr(x,'melaAngles') else -99.0, help="MELA angle costheta2"),
    NTupleVariable("MELAphi", lambda x : x.melaAngles.phi if hasattr(x,'melaAngles') else -99.0, help="MELA angle phi"),
    NTupleVariable("MELAphistar1", lambda x : x.melaAngles.phistar1 if hasattr(x,'melaAngles') else -99.0, help="MELA angle phistar1"),

])


ZType = NTupleObjectType("ZType", baseObjectTypes=[fourVectorType], variables = [
    NTupleVariable("hasFSR",   lambda x : x.hasFSR(), int),
    NTupleSubObject("l1",  lambda x : x.leg1,leptonTypeHZZ),
    NTupleSubObject("l2",  lambda x : x.leg2,leptonTypeHZZ),
    # -------
    NTupleVariable("pho_pt",  lambda x : (x.fsrPhoton.pt()  if x.hasFSR() else -99.0) ),
    NTupleVariable("pho_eta", lambda x : (x.fsrPhoton.eta() if x.hasFSR() else -99.0) ),
    NTupleVariable("pho_phi", lambda x : (x.fsrPhoton.phi() if x.hasFSR() else -99.0) ),
])


