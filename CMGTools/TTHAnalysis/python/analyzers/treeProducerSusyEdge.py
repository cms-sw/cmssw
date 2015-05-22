from CMGTools.TTHAnalysis.analyzers.treeProducerSusyCore import *
from CMGTools.TTHAnalysis.analyzers.ntupleTypes import *

# including the multilepton analyzer and all its stuff
from CMGTools.TTHAnalysis.analyzers.treeProducerSusyMultilepton import *

susyJZBEdge_globalVariables = susyMultilepton_globalVariables + [
    
    NTupleVariable("l1l2_m", lambda ev : ev.l1l2_m, help="Invariant mass of two leading leptons"),
    NTupleVariable("l1l2_pt", lambda ev : ev.l1l2_pt, help="Pt of the two leading leptons"),
    NTupleVariable("l1l2_eta", lambda ev : ev.l1l2_eta, help="Eta of the two leading leptons"),
    NTupleVariable("l1l2_phi", lambda ev : ev.l1l2_phi, help="Phi of the two leading leptons"),
    NTupleVariable("l1l2_DR", lambda ev : ev.l1l2_DR, help="DR of the two leading leptons"),
    NTupleVariable("genl1l2_m", lambda ev : ev.genl1l2_m, help="Invariant mass of two leading gen leptons"),
    NTupleVariable("genl1l2_pt", lambda ev : ev.genl1l2_pt, help="Pt of the two gen leading leptons"),
    NTupleVariable("genl1l2_eta", lambda ev : ev.genl1l2_eta, help="Eta of the two gen leading leptons"),
    NTupleVariable("genl1l2_phi", lambda ev : ev.genl1l2_phi, help="Phi of the two gen leading leptons"),
    NTupleVariable("genl1l2_DR", lambda ev : ev.genl1l2_DR, help="DR of the two gen leading leptons"),
    NTupleVariable("jzb", lambda ev : ev.jzb, help="JZB variable"),
]


susyJZBEdge_globalObjects = susyCore_globalObjects.copy()

susyJZBEdge_collections = susyMultilepton_collections.copy()
susyJZBEdge_collections.update({


        "genleps"         : NTupleCollection("genLep",     genParticleWithLinksType, 10, help="Generated leptons (e/mu) from W/Z decays"),                                             
        #"selectedLeptons" : NTupleCollection("lep", leptonTypeSusy, 50, help="Leptons after the preselection", filter=lambda l : l.pt()>10 ),
        "cleanJetsAll"       : NTupleCollection("jet", jetTypeSusy, 100, help="all jets (w/ x-cleaning, w/ ID applied w/o PUID applied pt>10 |eta|<5.2) , sorted by pt", filter=lambda l : l.pt()>10  ),
        "selectedPhotons"    : NTupleCollection("gamma", photonTypeSusy, 50, help="photons with pt>20 and loose cut based ID"),
        "generatorSummary" : NTupleCollection("GenPart", genParticleWithLinksType, 100 , help="Hard scattering particles, with ancestry and links"),

})
        
        
