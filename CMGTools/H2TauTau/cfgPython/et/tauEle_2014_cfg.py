import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.HeppyCore.framework.config import printComps

# Tau-tau analyzers
from CMGTools.H2TauTau.proto.analyzers.TauEleAnalyzer import TauEleAnalyzer
from CMGTools.H2TauTau.proto.analyzers.H2TauTauTreeProducerTauEle import H2TauTauTreeProducerTauEle
from CMGTools.H2TauTau.proto.analyzers.DYLLReweighterTauEle import DYLLReweighterTauEle
from CMGTools.H2TauTau.proto.analyzers.TauDecayModeWeighter import TauDecayModeWeighter
from CMGTools.H2TauTau.proto.analyzers.TauFakeRateWeighter import TauFakeRateWeighter
from CMGTools.H2TauTau.proto.analyzers.LeptonWeighter import LeptonWeighter
from CMGTools.H2TauTau.proto.analyzers.SVfitProducer import SVfitProducer

# common configuration and sequence
from CMGTools.H2TauTau.htt_ntuple_base_cff import commonSequence, genAna, dyJetsFakeAna, puFileData, puFileMC, eventSelector

# e-tau specific configuration settings

# 'Nom', 'Up', 'Down', or None
shift = None
syncntuple = True
computeSVfit = True
production = False  # production = True run on batch, production = False run locally

# When ready, include weights from CMGTools.H2TauTau.proto.weights.weighttable

# hlt_tauEffWeight_mc = 'effTau_eTau_MC_2012ABCDSummer13'
# hlt_tauEffWeight = 'effTau_eTau_Data_2012ABCDSummer13'
# hlt_eleEffWeight_mc = 'effEle_eTau_MC_2012ABCD'
# hlt_eleEffWeight = 'effEle_eTau_Data_2012ABCDSummer13'

hlt_tauEffWeight_mc = None
hlt_tauEffWeight = None
hlt_eleEffWeight_mc = None
hlt_eleEffWeight = None

dyJetsFakeAna.channel = 'et'

# Define e-tau specific modules

tauEleAna = cfg.Analyzer(
    TauEleAnalyzer,
    name='TauEleAnalyzer',
    pt1=20,
    eta1=2.3,
    iso1=1.5,
    looseiso1 = 9999.,
    pt2=23,
    eta2=2.1,
    iso2=0.1,
    looseiso2 = 9999.,
    m_min=10,
    m_max=99999,
    dR_min=0.5,
    verbose=False
)

dyLLReweighterTauEle = cfg.Analyzer(
    DYLLReweighterTauEle,
    name='DYLLReweighterTauEle',
    # 2012
    W1p0PB=1.,  # 1.37, # weight for 1 prong 0 Pi Barrel
    W1p0PE=1.,  # 1.11,
    W1p1PB=1.,  # 2.18,
    W1p1PE=1.,  # 0.47,
    verbose=False
)

tauDecayModeWeighter = cfg.Analyzer(
    TauDecayModeWeighter,
    name='TauDecayModeWeighter',
)

tauFakeRateWeighter = cfg.Analyzer(
    TauFakeRateWeighter,
    name='TauFakeRateWeighter'
)

tauWeighter = cfg.Analyzer(
    LeptonWeighter,
    name='LeptonWeighter_tau',
    effWeight = None,
    effWeightMC = None,
    lepton = 'leg1',
    verbose = False,
    disable = True,
    )

eleWeighter = cfg.Analyzer(
    LeptonWeighter,
    name='LeptonWeighter_ele',
    effWeight = None,
    effWeightMC = None,
    lepton = 'leg2',
    verbose = False,
    disable = True,
    idWeight = None,
    isoWeight = None
    )

treeProducer = cfg.Analyzer(
    H2TauTauTreeProducerTauEle,
    name='H2TauTauTreeProducerTauEle'
)

syncTreeProducer = cfg.Analyzer(
    H2TauTauTreeProducerTauEle,
    name='H2TauTauSyncTreeProducerTauEle',
    varStyle='sync',
    skimFunction='event.isSignal'
    )


svfitProducer = cfg.Analyzer(
    SVfitProducer,
    name='SVfitProducer',
    #integration='VEGAS',
    integration='MarkovChain',
    #verbose=True,
    #order='21', # muon first, tau second
    l1type='tau',
    l2type='ele'
    )

###################################################
### CONNECT SAMPLES TO THEIR ALIASES AND FILES  ###
###################################################
from CMGTools.H2TauTau.proto.samples.phys14.connector import httConnector
my_connect = httConnector('htt_6mar15_manzoni_nom', 'htautau_group',
                          '.*root', 'et', production=production)
my_connect.connect()
MC_list = my_connect.MC_list

###################################################
###              ASSIGN PU to MC                ###
###################################################
for mc in MC_list:
    mc.puFileData = puFileData
    mc.puFileMC = puFileMC

###################################################
###             SET COMPONENTS BY HAND          ###
###################################################
selectedComponents = MC_list
# selectedComponents = mc_dict['HiggsGGH125']
# for c in selectedComponents : c.splitFactor *= 5

###################################################
###                  SEQUENCE                   ###
###################################################
sequence = commonSequence
sequence.insert(sequence.index(genAna), tauEleAna)
sequence.append(tauDecayModeWeighter)
sequence.append(tauFakeRateWeighter)
sequence.append(tauWeighter)
sequence.append(eleWeighter)
sequence.insert(sequence.index(dyJetsFakeAna) + 1, dyLLReweighterTauEle)
if computeSVfit:
    sequence.append(svfitProducer)
sequence.append(treeProducer)
if syncntuple:
    sequence.append(syncTreeProducer)

###################################################
###             CHERRY PICK EVENTS              ###
###################################################
# eventSelector.toSelect = []
# sequence.insert(0, eventSelector)

###################################################
###            SET BATCH OR LOCAL               ###
###################################################
if not production:
    cache = True
    comp = my_connect.mc_dict['HiggsGGH125']
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
