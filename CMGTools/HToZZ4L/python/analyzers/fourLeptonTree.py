from CMGTools.HToZZ4L.analyzers.zzTypes import *

hzz_globalVariables = [
    NTupleVariable("rho",  lambda ev: ev.rho, float, help="kt6PFJets rho"),
    NTupleVariable("nVert",  lambda ev: len(ev.goodVertices), int, help="Number of good vertices"),
    NTupleVariable("nJet30", lambda ev: len([j for j in ev.cleanJets]), int, help="Number of jets with pt > 30"),
]

hzz_globalObjects = {
    "met" : NTupleObject("met", metType)
}


hzz_collections = {
    "bestFourLeptonsSignal"  : NTupleCollection("zz",    ZZType, 1, help="Four Lepton Candidates"),    
    "bestFourLeptons2P2F"    : NTupleCollection("zz2P2F",ZZType, 1, help="Four Lepton Candidates 2Pass 2 Fail"),    
    "bestFourLeptons3P1F"    : NTupleCollection("zz3P1F",ZZType, 1, help="Four Lepton Candidates 3 Pass 1 Fail"),   
    "bestFourLeptonsSS"      : NTupleCollection("zzSS",  ZZType, 1, help="Four Lepton Candidates SS"),   
    "bestFourLeptonsRelaxIdIso" : NTupleCollection("zzRelII",  ZZType, 8, help="Four Lepton Candidates (relax id, iso)"),   
    # ---------------
    "selectedLeptons" : NTupleCollection("Lep",    leptonTypeHZZ, 10, help="Leptons after the preselection"),
    "cleanJets"       : NTupleCollection("Jet",     jetTypeExtra, 10, help="Cental jets after full selection and cleaning, sorted by pt"),
    "discardedJets"   : NTupleCollection("DiscJet", jetTypeExtra,  5, help="Jets discarted in the jet-lepton cleaning"),
    "fsrPhotonsNoIso" : NTupleCollection("FSR",    fsrPhotonTypeHZZ, 10, help="Photons for FSR recovery (isolation not applied)"),
}
