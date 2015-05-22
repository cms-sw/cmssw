"""Simple test of Colin's fwlite analysis system.

This test will run the VertexAnalyzer, the TriggerAnalyzer, and the SimpleJetAnalyzer

do the following:
alias httMultiLoop='python -i $CMSSW_BASE/src/CMGTools/RootTools/python/fwlite/MultiLoop.py'

import a few files from this sample
/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola/Summer11-PU_S4_START42_V11-v1/AODSIM/V2/PAT_CMG_V2_5_0/H2TAUTAU_Feb2
in a 2011 subdirectory so that your root files are described by the following wildcard pattern:
2011/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola/Summer11-PU_S4_START42_V11-v1/AODSIM/V2/PAT_CMG_V2_5_0/H2TAUTAU_Feb2/tauMu*fullsel*.root

Then run:

httMultiLoop Test test_fwlite_cfg.py -N 5000

Look at the results:
cat Test/DYJets/log.txt 
rootDir Test/DYJets/test.root

You can also go right to a given event:
httMultiLoop Test test_fwlite_cfg.py -i 20

then:
print loop.event
"""

import copy
import CMGTools.RootTools.fwlite.Config as cfg


period = 'Period_2011A'

baseDir = '2011'
filePattern = 'tree*.root'

# mc_triggers = 'HLT_IsoMu12_v1'
mc_triggers = []

mc_jet_scale = 1.
mc_jet_smear = 0.

mc_vertexWeight = None
mc_tauEffWeight = None
mc_muEffWeight = None

mc_effCorrFactor = 1 

# For Fall11 need to use vertexWeightFall11 for WJets and DYJets and TTJets 
# For Fall11 : trigger is applied in MC:
#   "HLT_IsoMu15_LooseIsoPFTau15_v9"

if period == 'Period_2011A':
    mc_vertexWeight = 'vertexWeight2invfb'
    mc_tauEffWeight = 'effTau2011A'
    mc_muEffWeight = 'effMu2011A'
elif period == 'Period_2011B':
    mc_vertexWeight = 'vertexWeight2011B'
    mc_tauEffWeight = 'effTau2011B'
    mc_muEffWeight = 'effMu2011B'
elif period == 'Period_2011AB':
    mc_vertexWeight = 'vertexWeight2011AB'
    mc_tauEffWeight = 'effTau2011AB'
    mc_muEffWeight = 'effMu2011AB'


ZMuMuAna = cfg.Analyzer(
    'ZMuMuAnalyzer',
    pt1 = 20,
    pt2 = 20,
    iso1 = 0.1,
    iso2 = 0.1,
    eta1 = 2,
    eta2 = 2,
    m_min = 0,
    m_max = 200
    )

triggerAna = cfg.Analyzer(
    'TriggerAnalyzer'
    )

jetAna = cfg.Analyzer(
    'SimpleJetAnalyzer',
    ptCut = 0
    )


effMuAna = cfg.Analyzer(
    'EfficiencyAnalyzer',
    # recsel = 'cuts_vbtfmuon'
    genPdgId = 13
    )

vertexAna = cfg.Analyzer(
    'VertexAnalyzer',
    vertexWeight = mc_vertexWeight,
    verbose = False
    )


#########################################################################################

data_Run2011A_May10ReReco_v1 = cfg.DataComponent(
    name = 'data_Run2011A_May10ReReco_v1',
    files ='{baseDir}/TauPlusX/Run2011A-May10ReReco-v1/AOD/V2/PAT_CMG_V2_5_0/H2TAUTAU_Feb2/{filePattern}'.format(baseDir=baseDir, filePattern=filePattern),
    intLumi = 168.597,
    triggers = ['HLT_IsoMu12_LooseIsoPFTau10_v4'] )


data_Run2011A_PromptReco_v4 = cfg.DataComponent(
    name = 'data_Run2011A_PromptReco_v4',
    files ='{baseDir}/TauPlusX/Run2011A-PromptReco-v4/AOD/V2/PAT_CMG_V2_5_0/H2TAUTAU_Feb2/{filePattern}'.format(baseDir=baseDir, filePattern=filePattern),
    intLumi = 929.748,
    triggers = ['HLT_IsoMu15_LooseIsoPFTau15_v[2,4,5,6]'],
    )

data_Run2011A_05Aug2011_v1 = cfg.DataComponent(
    name = 'data_Run2011A_05Aug2011_v1',
    files ='{baseDir}/TauPlusX/Run2011A-05Aug2011-v1/AOD/V2/PAT_CMG_V2_5_0/H2TAUTAU_Feb2/{filePattern}'.format(baseDir=baseDir, filePattern=filePattern),
    intLumi = 373.349,
    triggers = ['HLT_IsoMu15_LooseIsoPFTau15_v8'] )

data_Run2011A_PromptReco_v6 = cfg.DataComponent(
    name = 'data_Run2011A_PromptReco_v6',
    files ='{baseDir}/TauPlusX/Run2011A-PromptReco-v6/AOD/V2/PAT_CMG_V2_5_0/H2TAUTAU_Feb2/{filePattern}'.format(baseDir=baseDir, filePattern=filePattern),
    intLumi = 658.886,
    triggers = ['HLT_IsoMu15_LooseIsoPFTau15_v[8,9]'] )

data_Run2011A_03Oct2011_v1 = cfg.DataComponent(
    name = 'data_Run2011A_03Oct2011_v1',
    files ='{baseDir}/TauPlusX/Run2011A-03Oct2011-v1/AOD/V2/PAT_CMG_V2_5_0/H2TAUTAU_Feb2/{filePattern}'.format(baseDir=baseDir, filePattern=filePattern),
    intLumi = 658.886,
    triggers = ['HLT_IsoMu15_LooseIsoPFTau15_v[8,9]'] )

data_Run2011B_PromptReco_v1 = cfg.DataComponent(
    name = 'data_Run2011B_PromptReco_v1',
    files ='{baseDir}/TauPlusX/Run2011B-PromptReco-v1/AOD/V2/PAT_CMG_V2_5_0/H2TAUTAU_Feb2/{filePattern}'.format(baseDir=baseDir, filePattern=filePattern),
    intLumi = 2511.0,
    triggers = ['HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v[1,5,6]',
                'HLT_IsoMu15_LooseIsoPFTau15_v[9,10,11,12,13]'] )



#########################################################################################


DYJets = cfg.MCComponent(
    name = 'DYJets',
    files ='{baseDir}/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola/Fall11-PU_Chamonix12_START44_V10-v2/AODSIM/PAT_CMG_V3_0_0/{filePattern}'.format(baseDir=baseDir, filePattern=filePattern),
    xSection = 3048.,
    nGenEvents = 34915945,
    triggers = mc_triggers,
    # vertexWeight = mc_vertexWeight,
    # tauEffWeight = mc_tauEffWeight,
    # muEffWeight = mc_muEffWeight,    
    effCorrFactor = mc_effCorrFactor )

WJets = cfg.MCComponent(
    name = 'WJets',
    files ='{baseDir}/WJetsToLNu_TuneZ2_7TeV-madgraph-tauola/Summer11-PU_S4_START42_V11-v1/AODSIM/V2/PAT_CMG_V2_5_0/H2TAUTAU_Feb2/{filePattern}'.format(baseDir=baseDir, filePattern=filePattern),
    xSection = 31314.,
    nGenEvents = 53227112,
    triggers = mc_triggers,
    # vertexWeight = mc_vertexWeight,
    # tauEffWeight = mc_tauEffWeight,
    # muEffWeight = mc_muEffWeight,    
    effCorrFactor = mc_effCorrFactor )


TTJets = cfg.MCComponent(
    name = 'TTJets',
    files ='{baseDir}/TTJets_TuneZ2_7TeV-madgraph-tauola/Summer11-PU_S4_START42_V11-v1/AODSIM/V2/PAT_CMG_V2_5_0/H2TAUTAU_Feb2/{filePattern}'.format(baseDir=baseDir, filePattern=filePattern),
    xSection = 165.8,
    nGenEvents = 3542770,
    triggers = mc_triggers,
    # vertexWeight = mc_vertexWeight,
    # tauEffWeight = mc_tauEffWeight,
    # muEffWeight = mc_muEffWeight,    
    effCorrFactor = mc_effCorrFactor )




#########################################################################################


MC = [DYJets, WJets, TTJets]
for mc in MC:
    # could handle the weights in the same way
    mc.jetScale = mc_jet_scale
    mc.jetSmear = mc_jet_smear


data_2011A = [
    data_Run2011A_May10ReReco_v1,
    data_Run2011A_PromptReco_v4,
    data_Run2011A_05Aug2011_v1,
    data_Run2011A_03Oct2011_v1,
    ]

data_2011B = [
    data_Run2011B_PromptReco_v1
    ]


selectedComponents =  MC
if period == 'Period_2011A':
    selectedComponents.extend( data_2011A )
elif period == 'Period_2011B':
    selectedComponents.extend( data_2011B )
elif period == 'Period_2011AB':
    selectedComponents.extend( data_2011A )
    selectedComponents.extend( data_2011B )

# selectedComponents = data_2011A
# selectedComponents = [embed_Run2011A_PromptReco_v4]
selectedComponents  = [DYJets] 

sequence = cfg.Sequence( [
    # triggerAna,
    # vertexAna,
    # ZMuMuAna, 
    effMuAna
    ] )

config = cfg.Config( components = selectedComponents,
                     sequence = sequence )
