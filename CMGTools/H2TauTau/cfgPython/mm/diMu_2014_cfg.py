import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.HeppyCore.framework.config import printComps

# Tau-tau analyzers
from CMGTools.H2TauTau.proto.analyzers.MuMuAnalyzer import MuMuAnalyzer
from CMGTools.H2TauTau.proto.analyzers.H2TauTauTreeProducerMuMu import H2TauTauTreeProducerMuMu
from CMGTools.H2TauTau.proto.analyzers.LeptonWeighter import LeptonWeighter
from CMGTools.H2TauTau.proto.analyzers.SVfitProducer import SVfitProducer
from CMGTools.H2TauTau.proto.analyzers.MVAMETProducer import MVAMETProducer

# common configuration and sequence
from CMGTools.H2TauTau.htt_ntuple_base_cff import commonSequence, genAna, dyJetsFakeAna, puFileData, puFileMC


### mu-tau specific configuration settings

# 'Nom', 'Up', 'Down', or None
shift = None
syncntuple = False
computeSVfit = False

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

### Define mu-tau specific modules

MuMuAna = cfg.Analyzer(
    MuMuAnalyzer,
    name='MuMuAnalyzer',
    pt1 = 20,
    eta1 = 2.3,
    iso1 = 0.1,
    pt2 = 20,
    eta2 = 2.3,
    iso2 = 0.1,
    m_min = 10,
    m_max = 99999,
    dR_min = 0.5,
    # triggerMap = pathsAndFilters,
    verbose = True
)

muonWeighter1 = cfg.Analyzer(
    LeptonWeighter,
    name='LeptonWeighter_mu_1',
    effWeight = None,
    effWeightMC = None,
    lepton = 'leg1',
    verbose = True,
    disable = True,
    )

muonWeighter2 = cfg.Analyzer(
    LeptonWeighter,
    name='LeptonWeighter_mu_2',
    effWeight = None,
    effWeightMC = None,
    lepton = 'leg2',
    verbose = True,
    disable = True,
    idWeight = None,
    isoWeight = None    
    )

treeProducer = cfg.Analyzer(
    H2TauTauTreeProducerMuMu,
    name='H2TauTauTreeProducerMuMu'
    )

syncTreeProducer = cfg.Analyzer(
    H2TauTauTreeProducerMuMu,
    name='H2TauTauSyncTreeProducerMuMu',
    varStyle='sync',
    skimFunction='event.isSignal'
    )

svfitProducer = cfg.Analyzer(
    SVfitProducer,
    name='SVfitProducer',
    integration='VEGAS',
    #integration='MarkovChain',
    #debug=True,
    l1type='muon',
    l2type='muon'
    )


mvaMetProducer = cfg.Analyzer(
    MVAMETProducer,
    name='MVAMETProducer'
    )
    
###################################################
### CONNECT SAMPLES TO THEIR ALIASES AND FILES  ###
###################################################
from CMGTools.H2TauTau.proto.samples.phys14.tauMu_Jan_Feb13 import MC_list, mc_dict

###################################################
###              ASSIGN PU to MC                ###
###################################################
for mc in MC_list:
    mc.puFileData = puFileData
    mc.puFileMC = puFileMC

###################################################
###             SET COMPONENTS BY HAND          ###
###################################################
selectedComponents = [mc_dict['HiggsGGH125']]
# for c in selectedComponents : c.splitFactor *= 5

###################################################
###                  SEQUENCE                   ###
###################################################
sequence = commonSequence
sequence.insert(sequence.index(genAna), MuMuAna)
sequence.append(muonWeighter1)
sequence.append(muonWeighter2)
sequence.append(mvaMetProducer)
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
# JAN - can we finally get this via command line options?
test = 1  # test = 0 run on batch, test = 1 run locally
if test == 1:
    comp = mc_dict['HiggsGGH125']
    comp.name = 'DYJets'
    selectedComponents = [comp]
    comp.splitFactor = 1
    comp.files = ['/afs/cern.ch/user/s/steggema/work/CMSSW_7_2_3/src/CMGTools/H2TauTau/prod/diMu_fullsel_tree_CMG.root']
    # comp.files = comp.files[:1]


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
