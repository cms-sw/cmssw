"""Simple test of Colin's fwlite analysis system.

This test will run the VertexAnalyzer, the TriggerAnalyzer, and the SimpleJetAnalyzer

do the following:
alias httMultiLoop='python -i $CMSSW_BASE/src/CMGTools/RootTools/python/fwlite/MultiLoop.py'

import at least the sample
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

import os
import glob
import copy
import CMGTools.RootTools.fwlite.Config as cfg
from CMGTools.Production.dataset import createDataset

def getFiles(dataset, user, pattern):
    from CMGTools.Production.datasetToSource import datasetToSource
    print 'getting files for', dataset,user,pattern
    ds = datasetToSource( user, dataset, pattern, False )
    files = ds.fileNames
    return ['root://eoscms//eos/cms%s' % f for f in files]


period = 'Period_2011B'

baseDir = 'samples'
filePattern = 'tree_CMG_*.root'
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




jetAna = cfg.Analyzer(
    'SimpleJetAnalyzer',
    ptCut = 0,
    genPtCut = 0,
    ## use pat::Jets
    jetCollection = ('selectedPatJetsAK5','std::vector<pat::Jet>'),
    ## or cmg::Jets
    ## jetCollection = ('cmgPFJetSel','std::vector<cmg::PFJet>'),
    doJetIdHisto    = False,
    applyPFLooseId  = False, 
    jetIdMva = ( 0, "%s/src/CMGTools/External/data/mva_JetID.weights.xml" % os.getenv("CMSSW_BASE"), "JetID" ),
    
    ## use gen jets in Loic's PATtuples    
    ## genJetsCollection =  ('prunedGen','std::vector<reco::GenParticle>'),
    ## use the gen jets available also in the CMG trees
    ## genJetsCollection = ('genJetSel','std::vector< cmg::PhysicsObjectWithPtr< edm::Ptr< reco::GenJet > > >'),
    ## use the gen jets present only in PATtuples 
    genJetsCollection = (('selectedPatJetsAK5','genJets'),'vector<reco::GenJet>'),
    
    ## use level2 gen leptons to make the matching
    useGenLeptons = True
    )


#########################################################################################



DYJetsFall11_PAT = cfg.MCComponent(
    name = 'DYJetsFall11',
    # new CHS added, i.e. V4
    files = getFiles('/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola/Fall11-PU_S6_START42_V14B-v1/AODSIM/V4/PAT_CMG_4_0_0', 'cmgtools', 'pat.*root')[:100],
    xSection = 3048.,
    nGenEvents = 34915945,
    triggers = [],
    effCorrFactor = 1 )


DYJetsChamonix_PAT = cfg.MCComponent(
    name = 'DYJetsChamonix',
    files = getFiles('/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola/Fall11-PU_Chamonix12_START44_V10-v2/AODSIM/PAT_CMG_V3_0_0', 'cmgtools', 'pat.*root')[:50],
    # files = createDataset('LOCAL','/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola/Fall11-PU_Chamonix12_START44_V10-v2/AODSIM/PAT_CMG_V3_0_0', '.*root', True).listOfGoodFiles(),
    xSection = 3048.,
    nGenEvents = 34915945,
    triggers = [],
    effCorrFactor = 1 )




#########################################################################################


MC = [DYJetsFall11_PAT, DYJetsChamonix_PAT]
for mc in MC:
    # could handle the weights in the same way
    mc.jetScale = mc_jet_scale
    mc.jetSmear = mc_jet_smear

DYJetsChamonix_PAT.splitFactor = 50
DYJetsFall11_PAT.splitFactor = 50

selectedComponents  = [DYJetsFall11_PAT] 

sequence = cfg.Sequence( [
    jetAna
    ] )

config = cfg.Config( components = selectedComponents,
                     sequence = sequence )
