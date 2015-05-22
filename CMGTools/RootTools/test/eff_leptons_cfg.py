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



def baselineIdMuon(muon):
    return muon.numberOfValidTrackerHits() > 10 and abs(muon.dz())<0.5
    
def baselineIdIsoMuon(muon):
    return baselineIdMuon(muon) and isoLepton(muon)
    
def idMuon(muon):
    return muon.getSelection('cuts_vbtfmuon') and abs(muon.dz())<0.5

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
    # recselFun = 'trigObjs',
    recselFun = baselineIdIsoMuon,
    # recselFun = isoLepton,
    # refselFun = idMuon,
    # triggerMap = trigMap, 
    instance = 'cmgMuonSelStdLep',
    type = 'std::vector<cmg::Muon>',
    instance_gen = 'genLeptonsStatus1',
    type_gen = 'std::vector<reco::GenParticle>',
    genTrigMatch = False, 
    genPdgId = 13
    )

effMuAnaPF = cfg.Analyzer(
    'EfficiencyAnalyzer_pf',
    # recselFun = 'trigObjs',
    recselFun = isoLepton,
    # refselFun = passLepton,
    # triggerMap = trigMap,
    instance = 'cmgMuonSel',
    type = 'std::vector<cmg::Muon>',
    instance_gen = 'genLeptonsStatus1',
    type_gen = 'std::vector<reco::GenParticle>',
    genTrigMatch = False, 
    genPdgId = 13
    )

triggerAna = cfg.Analyzer(
    'TriggerAnalyzer'
    )

muonAnas = [effMuAnaStd, effMuAnaPF]
    

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

from CMGTools.H2TauTau.proto.samples.cmg_testMVAs import *

#########################################################################################

dummyAna = cfg.Analyzer(
    'Analyzer'
    )


selectedComponents  = [QCDMuH20Pt15, DYJets] 

DYJets.splitFactor = 5
QCDMuH20Pt15.splitFactor = 25

DYJets.files = DYJets.files[:10]
QCDMuH20Pt15.files = QCDMuH20Pt15.files[:50]

test = False

if test:
    sam = QCDMuH20Pt15
    sam.files = sam.files[:1]
    selectedComponents = [sam]
    sam.splitFactor = 1


sequence = cfg.Sequence( muonAnas )

config = cfg.Config( components = selectedComponents,
                     sequence = sequence )

