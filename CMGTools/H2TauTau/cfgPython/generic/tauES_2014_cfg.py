import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.HeppyCore.framework.config import printComps
from PhysicsTools.HeppyCore.framework.heppy import getHeppyOption

from PhysicsTools.Heppy.analyzers.objects.TauAnalyzer import TauAnalyzer

from CMGTools.H2TauTau.proto.analyzers.TauTreeProducer import TauTreeProducer
from CMGTools.H2TauTau.proto.analyzers.TauGenTreeProducer import TauGenTreeProducer

# from CMGTools.H2TauTau.proto.samples.phys14.connector import httConnector
from CMGTools.TTHAnalysis.samples.samples_13TeV_PHYS14 import DYJetsToLL_M50, TTH
from CMGTools.TTHAnalysis.samples.ComponentCreator import ComponentCreator

# common configuration and sequence
from CMGTools.H2TauTau.htt_ntuple_base_cff import genAna, vertexAna


# Get all heppy options; set via "-o production" or "-o production=True"

# production = True run on batch, production = False (or unset) run locally
production = getHeppyOption('production')


treeProducer = cfg.Analyzer(
    TauTreeProducer,
    name='TauTreeProducer'
)

genTreeProducer = cfg.Analyzer(
    TauGenTreeProducer,
    name='TauGenTreeProducer'
)

tauAna = cfg.Analyzer(
    TauAnalyzer,
    name='TauAnalyzer',
    ptMin=20,
    etaMax=9999,
    dxyMax=1000.,
    dzMax=0.2,
    vetoLeptons=False,
    leptonVetoDR=0.4,
    decayModeID="decayModeFindingNewDMs",  # ignored if not set or ""
    tauID="decayModeFindingNewDMs",
    vetoLeptonsPOG=False, # If True, the following two IDs are required
    tauAntiMuonID="againstMuonLoose3",
    tauAntiElectronID="againstElectronLooseMVA5",
    tauLooseID="decayModeFinding",
)

# ###################################################
# ### CONNECT SAMPLES TO THEIR ALIASES AND FILES  ###
# ###################################################
# my_connect = httConnector('htt_6mar15_manzoni_nom', 'CMS',
#                           '.*root', 'mt', production=production)
# my_connect.connect()
# MC_list = my_connect.MC_list


creator = ComponentCreator()
ggh125 = creator.makeMCComponent("GGH125", "/GluGluToHToTauTau_M-125_13TeV-powheg-pythia6/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 1.0)

###################################################
###             SET COMPONENTS BY HAND          ###
###################################################
selectedComponents = [TTH] # [ggh125]
sequence = cfg.Sequence([
    genAna,
    vertexAna,
    tauAna,
    treeProducer,
    genTreeProducer
])

if not production:
    cache = True
    comp = selectedComponents[0]
    selectedComponents = [comp]
    comp.splitFactor = 1
    comp.fineSplitFactor = 1
    comp.files = comp.files[:1]


# the following is declared in case this cfg is used in input to the
# heppy.py script
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
config = cfg.Config(components=selectedComponents,
                    sequence=sequence,
                    services=[],
                    events_class=Events
                    )

printComps(config.components, True)


def modCfgForPlot(config):
    config.components = []
