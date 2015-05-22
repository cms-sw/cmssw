import copy
import glob
import CMGTools.RootTools.fwlite.Config as cfg


def newIdMuon(muon):
    sel = muon.getSelection
    return sel('cuts_vbtfmuon_isGlobal') and \
           sel('cuts_vbtfmuon_isTracker') and \
           sel('cuts_vbtfmuon_numberOfValidPixelHits') and \
           sel('cuts_vbtfmuon_numberOfValidMuonHits') and \
           sel('cuts_vbtfmuon_numberOfMatches') and \
           sel('cuts_vbtfmuon_normalizedChi2') and \
           sel('cuts_vbtfmuon_dxy') and \
           muon.sourcePtr().track().pt()>10
           # muon.sourcePtr().track().hitPattern().trackerLayersWithMeasurement() > 8


def idMuon(muon):
    return muon.getSelection('cuts_vbtfmuon') 

ana = cfg.Analyzer(
    'DeltaAnalyzer',
    col1_instance = 'cmgMuonSel',
    col1_type = 'std::vector< cmg::Muon >',
    sel2 = newIdMuon,
    col2_instance = 'cmgMuonSelStdLep',
    col2_type = 'std::vector< cmg::Muon >',
    deltaR = 999999,
    gen_instance = 'genLeptonsStatus1',
    gen_type = 'std::vector<reco::GenParticle>',
    gen_pdgId = 13
    )



tree = cfg.Analyzer(
    'DeltaTreeAnalyzer',
    )

#########################################################################################

from CMGTools.H2TauTau.proto.samples.cmg_testMVAs import *

#########################################################################################


selectedComponents  = [DYJets] 

splitFactor = 14
DYJets.files = DYJets.files[:560]
DYJets.splitFactor = splitFactor
QCDMuH20Pt15.splitFactor = splitFactor
QCDMuH15to20Pt5.splitFactor = splitFactor
Hig105.splitFactor = splitFactor

test = True
if test:
    sam = DYJets
    sam.files = sam.files[:1]
    selectedComponents = [sam]
    sam.splitFactor = 1


sequence = cfg.Sequence( [ana, tree] )

config = cfg.Config( components = selectedComponents,
                     sequence = sequence )

