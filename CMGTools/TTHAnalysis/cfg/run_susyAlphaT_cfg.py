import PhysicsTools.HeppyCore.framework.config as cfg
from CMGTools.RootTools.RootTools import *

#Load all analyzers
from CMGTools.TTHAnalysis.analyzers.susyCore_modules_cff import *

##------------------------------------------
## Redefine analyzer parameters
##------------------------------------------

# Muons
#------------------------------
lepAna.loose_muon_pt               = 10.,
lepAna.loose_muon_eta              = 2.5,
lepAna.loose_muon_dxy              = 0.5
lepAna.loose_muon_dz               = 1.0
#lepAna.loose_muon_relIso           = 0.15

# Electrons
#------------------------------
lepAna.loose_electron_id           = "POG_Cuts_ID_2012_Veto"
lepAna.loose_electron_pt           = 10
lepAna.loose_electron_eta          = 2.5
lepAna.loose_electron_dxy          = 0.5
lepAna.loose_electron_dz           = 0.
# lepAna.loose_electron_relIso       = 0.15
# lepAna.loose_electron_lostHits     = 999 # no cut
# lepAna.inclusive_electron_lostHits = 999 # no cut
# lepAna.ele_isoCorr                 = "deltaBeta"
# lepAna.ele_tightId                 = "Cuts_2012"

# Photons
#------------------------------
photonAna.ptMin                        = 25,
photonAna.epaMax                       = 2.5,

# Taus 
#------------------------------
tauAna.etaMax         = 2.3
tauAna.dxyMax         = 99999.
tauAna.dzMax          = 99999.
tauAna.vetoLeptons    = False
tauAna.vetoLeptonsPOG = True


# Jets (for event variables do apply the jetID and not PUID yet)
#------------------------------
jetAna.relaxJetId      = False
jetAna.doPuId          = False
jetAna.jetEta          = 5.
jetAna.jetEtaCentral   = 3.
jetAna.jetPt           = 50.
jetAna.recalibrateJets = False
jetAna.jetLepDR        = 0.4


# Energy sums
#------------------------------
# NOTE: Currently energy sums are calculated with 40 GeV jets (ttHCoreEventAnalyzer.py)
#       However, the input collection is cleanjets which have a 50 GeV cut so this is a labeling problem

metAna.doMetNoMu=True
metAna.doMetNoPhoton=True

# Jet-MET based Skim (generic, but requirements depend on the final state)
from CMGTools.TTHAnalysis.analyzers.ttHJetMETSkimmer import ttHJetMETSkimmer
ttHJetMETSkim = cfg.Analyzer(
   ttHJetMETSkimmer, name='ttHJetMETSkimmer',
   jets      = "cleanJets", # jet collection to use
   jetPtCuts = [],  # e.g. [60,40,30,20] to require at least four jets with pt > 60,40,30,20
   jetVetoPt =  0,  # if non-zero, veto additional jets with pt > veto beyond the ones in jetPtCuts
   metCut    =  0,  # MET cut
   htCut     = ('htJet40j', 0), # cut on HT defined with only jets and pt cut 40, at zero; i.e. no cut
                                # see ttHCoreEventAnalyzer for alternative definitions
   mhtCut    = ('mhtJet40', 0), # cut on MHT defined with all leptons, and jets with pt > 40.
   nBJet     = ('CSVv2IVFM', 0, "jet.pt() > 30"),     # require at least 0 jets passing CSV medium and pt > 30
   )

ttHJetMETSkim.htCut       = ('htJet50j', 0)
ttHJetMETSkim.mhtCut      = ('htJet40j', 0)
ttHJetMETSkim.nBJet       = ('CSVM', 0, "jet.pt() > 50")     # require at least 0 jets passing CSVM and pt > 50

##------------------------------------------
##  ISOLATED TRACK
##------------------------------------------

isoTrackAna.setOff=False
isoTrackAna.doRelIsolation = True

##------------------------------------------
##  ALPHAT VARIABLES
##------------------------------------------

from PhysicsTools.Heppy.analyzers.eventtopology.AlphaTAnalyzer import AlphaTAnalyzer
from CMGTools.TTHAnalysis.analyzers.ttHAlphaTControlAnalyzer import ttHAlphaTControlAnalyzer
# Tree Producer
ttHAlphaTAna = cfg.Analyzer(
            AlphaTAnalyzer, name='AlphaTAnalyzer'
            )

ttHAlphaTControlAna = cfg.Analyzer(
                        ttHAlphaTControlAnalyzer,name='ttHAlphaTControlAnalyzer'
                        )
##------------------------------------------
##  EXTRA GEN STUFF
##------------------------------------------

# Gen Info Analyzer
from CMGTools.TTHAnalysis.analyzers.ttHGenBinningAnalyzer import ttHGenBinningAnalyzer
ttHGenBinAna = cfg.Analyzer(
    ttHGenBinningAnalyzer, name = 'ttHGenBinningAnalyzer'
    )


#------------------------------------------
##  PRODUCER
##------------------------------------------
from CMGTools.TTHAnalysis.analyzers.treeProducerSusyAlphaT import * 
## Tree Producer
treeProducer = cfg.Analyzer(
     AutoFillTreeProducer, name='treeProducerSusyAlphaT',
     vectorTree = True,
     saveTLorentzVectors = False,  # can set to True to get also the TLorentzVectors, but trees will be bigger
     PDFWeights = PDFWeights,
     globalVariables = susyAlphaT_globalVariables,
     globalObjects = susyAlphaT_globalObjects,
     collections = susyAlphaT_collections,
)

#-------- SEQUENCE

sequence = cfg.Sequence(susyCoreSequence + [
                        ttHAlphaTAna,
                        ttHAlphaTControlAna,
                        ttHGenBinAna,
                        treeProducer,
                        ])


#-------- SAMPLES AND TRIGGERS -----------
from CMGTools.RootTools.samples.samples_13TeV_74X import *


# Selected samples as defined on the AlphaT twiki


#selectedComponents = [ SingleMu, DoubleElectron, TTHToWW_PUS14, DYJetsToLL_M50_PU20bx25, TTJets_PUS14 ]
selectedComponents = []
selectedComponents.extend( [TTJets_50ns] )






#-------- HOW TO RUN
test = 1

# Test a single component, using a single thread.
#--------------------------------------------------
if test==1:
    comp               =  TTJets_50ns 
    #comp.files = ['/afs/cern.ch/work/p/pandolf/CMSSW_7_0_6_patch1_2/src/CMGTools/TTHAnalysis/cfg/pickevents.root']
    comp.files         = comp.files[:1]
    
    selectedComponents = [comp]
    comp.splitFactor   = 1
#--------------------------------------------------

# Test all components (1 thread per component).
#--------------------------------------------------
elif test==2:
    for comp in selectedComponents:
        comp.splitFactor = 1
        comp.files       = comp.files[:1]
#--------------------------------------------------

# Run on local files
#--------------------------------------------------
elif test==4:
    comp = TTJets_PU20bx25
#    comp.name = 'TTJets'
    #    comp.files = [ '/store/mc/Spring14miniaod/TT_Tune4C_13TeV-pythia8-tauola/MINIAODSIM/PU20bx25_POSTLS170_V5-v1/00000/063013AD-9907-E411-8135-0026189438BD.root' ]

    comp.files = [ '/afs/cern.ch/user/m/mbaber/WORK/public/CSA14Samples/TT_Tune4C_13TeV-pythia8_PU20bx25.root' ]

    selectedComponents = [comp]
    comp.splitFactor = 1
#--------------------------------------------------


#config = cfg.Config( components = selectedComponents,
#                     sequence = sequence )

#printComps(config.components, True)
        # the following is declared in case this cfg is used in input to the heppy.py script

from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
config = cfg.Config( components = selectedComponents,
                     sequence = sequence,
                     services = [],  
                     events_class = Events)

