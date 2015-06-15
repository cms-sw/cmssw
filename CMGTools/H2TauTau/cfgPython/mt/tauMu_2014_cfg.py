import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.HeppyCore.framework.config import printComps
from PhysicsTools.HeppyCore.framework.heppy import getHeppyOption

# Tau-tau analyzers
from CMGTools.H2TauTau.proto.analyzers.TauMuAnalyzer import TauMuAnalyzer
from CMGTools.H2TauTau.proto.analyzers.H2TauTauTreeProducerTauMu import H2TauTauTreeProducerTauMu
from CMGTools.H2TauTau.proto.analyzers.TauDecayModeWeighter import TauDecayModeWeighter
from CMGTools.H2TauTau.proto.analyzers.TauFakeRateWeighter import TauFakeRateWeighter
from CMGTools.H2TauTau.proto.analyzers.LeptonWeighter import LeptonWeighter
from CMGTools.H2TauTau.proto.analyzers.SVfitProducer import SVfitProducer

from CMGTools.H2TauTau.proto.samples.phys14.connector import httConnector

# common configuration and sequence
from CMGTools.H2TauTau.htt_ntuple_base_cff import commonSequence, genAna, dyJetsFakeAna, puFileData, puFileMC, eventSelector


# Get all heppy options; set via "-o production" or "-o production=True"

# production = True run on batch, production = False (or unset) run locally
production = getHeppyOption('production')

# mu-tau specific configuration settings

# 'Nom', 'Up', 'Down', or None
shift = None
syncntuple = True
computeSVfit = True


# When ready, include weights from CMGTools.H2TauTau.proto.weights.weighttable

# mc_tauEffWeight_mc = 'effTau_muTau_MC_2012ABCDSummer13'
# mc_muEffWeight_mc = 'effMu_muTau_MC_2012ABCD'
# mc_tauEffWeight = 'effTau_muTau_Data_2012ABCDSummer13'
# mc_muEffWeight = 'effMu_muTau_Data_2012ABCDSummer13'

mc_tauEffWeight_mc = None
mc_muEffWeight_mc = None
mc_tauEffWeight = None
mc_muEffWeight = None

dyJetsFakeAna.channel = 'mt'

# Define mu-tau specific modules

tauMuAna = cfg.Analyzer(
    TauMuAnalyzer,
    name='TauMuAnalyzer',
    pt1=20,
    eta1=2.3,
    iso1=1.5,
    looseiso1=9999.,
    pt2=18,
    eta2=2.1,
    iso2=0.1,
    looseiso2=9999.,
    m_min=10,
    m_max=99999,
    dR_min=0.5,
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
    effWeight=None,
    effWeightMC=None,
    lepton='leg1',
    verbose=False,
    disable=True,
)

muonWeighter = cfg.Analyzer(
    LeptonWeighter,
    name='LeptonWeighter_mu',
    effWeight=None,
    effWeightMC=None,
    lepton='leg2',
    verbose=False,
    disable=True,
    idWeight=None,
    isoWeight=None
)

treeProducer = cfg.Analyzer(
    H2TauTauTreeProducerTauMu,
    name='H2TauTauTreeProducerTauMu'
)

syncTreeProducer = cfg.Analyzer(
    H2TauTauTreeProducerTauMu,
    name='H2TauTauSyncTreeProducerTauMu',
    varStyle='sync',
    skimFunction='event.isSignal'
)

svfitProducer = cfg.Analyzer(
    SVfitProducer,
    name='SVfitProducer',
    # integration='VEGAS',
    integration='MarkovChain',
    # verbose=True,
    # order='21', # muon first, tau second
    l1type='tau',
    l2type='muon'
)

###################################################
### CONNECT SAMPLES TO THEIR ALIASES AND FILES  ###
###################################################
my_connect = httConnector('htt_6mar15_manzoni_nom', 'htautau_group',
                          '.*root', 'mt', production=production)
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
sequence.insert(sequence.index(genAna), tauMuAna)
sequence.append(tauDecayModeWeighter)
sequence.append(tauFakeRateWeighter)
sequence.append(tauWeighter)
sequence.append(muonWeighter)
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
    comp = my_connect.mc_dict['TTJets']
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
