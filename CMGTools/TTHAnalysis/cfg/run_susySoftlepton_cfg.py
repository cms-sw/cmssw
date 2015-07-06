##########################################################
##   CONFIGURATION FOR SOFT DILEPTON + MONOJET TREES    ##
## skim condition: >= 2 loose leptons, no pt cuts or id ##
##########################################################

import CMGTools.RootTools.fwlite.Config as cfg
from CMGTools.RootTools.fwlite.Config import printComps
from CMGTools.RootTools.RootTools import *

#Load all analyzers
from CMGTools.TTHAnalysis.analyzers.susyCore_modules_cff import * 

# Redefine what I need

# --- LEPTON DEFINITION ---
ttHLepAna.inclusive_muon_pt  = 3
ttHLepAna.loose_muon_pt  = 3
ttHLepAna.loose_muon_relIso = 99.0
ttHLepAna.loose_muon_absIso = 10.0
ttHLepAna.inclusive_electron_pt  = 5
ttHLepAna.loose_electron_pt  = 5
ttHLepAna.loose_electron_relIso = 99.0
ttHLepAna.loose_electron_absIso = 10.0


# --- LEPTON SKIMMING ---
ttHLepSkim.minLeptons = 2
ttHLepSkim.maxLeptons = 999
ttHLepSkim.ptCuts = [5,3]

# --- JET-LEPTON CLEANING ---
ttHJetAna.minLepPt = 20 
# otherwise with only absIso cut at 10 GeV and no relIso we risk cleaning away good jets

# --- JET-MET SKIMMING ---
ttHJetMETSkim.jetPtCuts = [100,]
ttHJetMETSkim.metCut    = 100

# Event Analyzer for susy multi-lepton (at the moment, it's the TTH one)
ttHEventAna = cfg.Analyzer(
    'ttHLepEventAnalyzer',
    minJets25 = 0,
    )


from CMGTools.RootTools.samples.samples_8TeV_v517 import triggers_1mu, triggers_mumu, triggers_ee, triggers_mue, triggers_MET150, triggers_HT650, triggers_HTMET 
# Tree Producer
treeProducer = cfg.Analyzer(
    'treeProducerSusySoftlepton',
    vectorTree = True,
    PDFWeights = PDFWeights,
    triggerBits = {
            'SingleMu' : triggers_1mu,
            'DoubleMu' : triggers_mumu,
            'DoubleEl' : [ t for t in triggers_ee if "Ele15_Ele8_Ele5" not in t ],
            'TripleEl' : [ t for t in triggers_ee if "Ele15_Ele8_Ele5"     in t ],
            'MuEG'     : [ t for t in triggers_mue if "Mu" in t and "Ele" in t ],
            'MET15'    : triggers_MET150,
            'HT650'    : triggers_HT650,
            'triggers_HTMET' : triggers_HTMET,
        }
    )


#-------- SAMPLES AND TRIGGERS -----------
from CMGTools.RootTools.samples.samples_13TeV_CSA14 import * 

selectedComponents = [ TTJets_MSDecaysCKM_central_PU_S14_POSTLS170, SingleMu, DoubleElectron, TTHToWW_PUS14, DYJetsToLL_M50_PU20bx25, TTJets_PUS14 ]

#-------- SEQUENCE

sequence = cfg.Sequence(susyCoreSequence+[
    ttHEventAna,
    treeProducer,
    ])


#-------- HOW TO RUN
test = 1
if test==1:
    # test a single component, using a single thread.
    comp = selectedComponents[0]
    comp.files = comp.files[:4]
    selectedComponents = [comp]
    comp.splitFactor = 1
elif test==2:    
    # test all components (1 thread per component).
    for comp in selectedComponents:
        comp.splitFactor = 1
        comp.files = comp.files[:1]



config = cfg.Config( components = selectedComponents,
                     sequence = sequence )

printComps(config.components, True)
