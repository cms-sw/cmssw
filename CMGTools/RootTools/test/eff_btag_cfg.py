import copy
import glob
import CMGTools.RootTools.fwlite.Config as cfg
from CMGTools.Production.dataset import createDataset

def getFiles(dataset, user, pattern):
    from CMGTools.Production.datasetToSource import datasetToSource
    print 'getting files for', dataset,user,pattern
    ds = datasetToSource( user, dataset, pattern, True )
    files = ds.fileNames
    return ['root://eoscms//eos/cms%s' % f for f in files]


## import re

## def getCleanPatFiles(dataset, user):
##     trees = getFiles(dataset, user, 'tree.*root')
##     pats = getFiles(dataset, user, 'pat.*root')
##     pattern = re.compile('.*_(\d+)\.root') 
##     def num( file ):
##         m = pattern.match(file)
##         if m is not None:
##             return int( m.group(1) )
##     treenums = map(num, trees)
##     cleanpats = []
##     for patfile in pats:
##         n = num(patfile)
##         if n in treenums:
##             cleanpats.append(patfile)
##     return cleanpats


## import pprint

## pprint.pprint(getCleanPatFiles( '/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola/Fall11-PU_Chamonix12_START44_V10-v2/AODSIM/PAT_CMG_V3_0_0', 'cmgtools'))



def idMuon(muon):
    return muon.getSelection('cuts_vbtfmuon')

def isoLepton(lepton):
    return lepton.relIso(0.5)<0.1

def idIsoMuon(muon):
    return idMuon(muon) and isoLepton(muon)
    
def passLepton(lepton):
    return True

trigMap = { 'HLT_IsoMu15_v5':'hltSingleMuIsoL3IsoFiltered15',
            'HLT_IsoMu15_v14':'hltSingleMuIsoL3IsoFiltered15' }

effMuAnaStd = cfg.Analyzer(
    'EfficiencyAnalyzer_std',
    # recsel = 'cuts_vbtfmuon',
    recselFun = isoLepton,
    # refselFun = idMuon,
    triggerMap = trigMap, 
    instance = 'cmgMuonSelStdLep',
    type = 'std::vector<cmg::Muon>',
    genPdgId = 13
    )

effMuAnaPF = cfg.Analyzer(
    'EfficiencyAnalyzer_pf',
    # recsel = 'cuts_vbtfmuon',
    recselFun = isoLepton,
    # refselFun = idMuon,
    triggerMap = trigMap,
    instance = 'cmgMuonSel',
    type = 'std::vector<cmg::Muon>',
    genPdgId = 13
    )

triggerAna = cfg.Analyzer(
    'TriggerAnalyzer'
    )

muonAnas = [triggerAna, effMuAnaStd, effMuAnaPF]

def idElectron(electron):
    return electron.getSelection('cuts_vbtf80ID')
    # return electron.mvaDaniele()>-0.1

class idDanMVAGenerator(object):
    def __init__(self, cut):
        self.cut = cut 
    def __call__(self, electron):
        return electron.mvaDaniele()>self.cut

class idMITMVAGenerator(object):
    def __init__(self, cut):
        self.cut = cut 
    def __call__(self, electron):
        return electron.mvaMIT()>self.cut
    
effEleAnaStd = cfg.Analyzer(
    'EfficiencyAnalyzer_std',
    # recsel = 'cuts_vbtfmuon',
    recselFun = passLepton,
    # refselFun = ,
    instance = 'cmgElectronSelStdLep',
    type = 'std::vector<cmg::Electron>',
    genPdgId = 11
    )

effEleAnaPF = cfg.Analyzer(
    'EfficiencyAnalyzer_pf',
    # recsel = 'cuts_vbtfmuon',
    recselFun = passLepton,
    # refselFun = ,
    instance = 'cmgElectronSel',
    type = 'std::vector<cmg::Electron>',
    genPdgId = 11
    )

danStd = []

for cut in [0.0013, 0.0425, 0.025]:
    ana = copy.deepcopy(effEleAnaStd)
    ana.name = 'EfficiencyAnalyzer_std_dan_{cut}'.format(cut=cut)
    ana.recselFun = idDanMVAGenerator(cut)
    danStd.append( ana )

mitStd = []

for cut in [0.878, 0.942, 0.945]:
    ana = copy.deepcopy(effEleAnaStd)
    ana.name = 'EfficiencyAnalyzer_std_mit_{cut}'.format(cut=cut)
    ana.recselFun = idMITMVAGenerator(cut)
    mitStd.append( ana )


eleAnas = [effEleAnaStd, effEleAnaPF]
eleAnas.extend( danStd )
eleAnas.extend( mitStd )


def bTag(jet):
    return jet.btag(6)>0.7

jetAna = cfg.Analyzer(
    'EfficiencyAnalyzer_bjets',
    # recsel = 'cuts_vbtfmuon',
    recselFun = bTag,
    # refselFun = ,
    instance = 'cmgPFJetSel',
    type = 'std::vector<cmg::PFJet>',
    genPdgId = 5
    )

jetAnaU = cfg.Analyzer(
    'EfficiencyAnalyzer_gluonjets',
    # recsel = 'cuts_vbtfmuon',
    recselFun = bTag,
    # refselFun = ,
    instance = 'cmgPFJetSel',
    type = 'std::vector<cmg::PFJet>',
    genPdgId = 21
    )


#########################################################################################

nFiles = 20
splitFactor = 5

DYJetsFall11 = cfg.MCComponent(
    name = 'DYJetsFall11',
    files = getFiles('/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola/Fall11-PU_S6_START42_V14B-v1/AODSIM/V3/TestMVAs', 'cmgtools_group','tree.*root')[:nFiles],
    # files = getFiles('/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola/Fall11-PU_S6_START42_V14B-v1/AODSIM/V2/PAT_CMG_V2_5_0', 'cmgtools', 'tree.*root')[:20],
    xSection = 3048.,
    nGenEvents = 34915945,
    triggers = ['HLT_IsoMu15_v14'],
    effCorrFactor = 1 )


DYJetsChamonix = cfg.MCComponent(
    name = 'DYJetsChamonix',
    files = getFiles('/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola/Fall11-PU_Chamonix12_START44_V10-v2/AODSIM/PAT_CMG_V3_0_0/TestMVAs', 'cmgtools', 'tree.*root')[:nFiles],
    # files = createDataset('LOCAL','/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola/Fall11-PU_Chamonix12_START44_V10-v2/AODSIM/PAT_CMG_V3_0_0', '.*root', True).listOfGoodFiles(),
    xSection = 3048.,
    nGenEvents = 34915945,
    triggers = ['HLT_IsoMu15_v14'],
    effCorrFactor = 1 )


QCDMu = cfg.MCComponent(
    name = 'QCDMu',
    files = getFiles('/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1/AODSIM/V2/PAT_CMG_TestMVAs', 'cmgtools', 'tree.*root')[:nFiles],
    xSection = 3048., # dummy 
    nGenEvents = 34915945, # dummy 
    triggers = ['HLT_IsoMu15_v5'],
    effCorrFactor = 1 )



#########################################################################################

dummyAna = cfg.Analyzer(
    'Analyzer'
    )


selectedComponents  = [DYJetsFall11, QCDMu] 


DYJetsChamonix.splitFactor = splitFactor
DYJetsFall11.splitFactor = splitFactor
QCDMu.splitFactor = splitFactor
# QCDMu.files = QCDMu.files[:5]

test = False
if test:
    sam = DYJetsFall11
    sam.files = sam.files[:1]
    selectedComponents = [sam]
    sam.splitFactor = 1


sequence = cfg.Sequence( [jetAna, jetAnaU] )

config = cfg.Config( components = selectedComponents,
                     sequence = sequence )

