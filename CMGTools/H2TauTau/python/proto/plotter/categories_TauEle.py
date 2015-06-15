import copy

from CMGTools.H2TauTau.proto.plotter.categories_common import *
from CMGTools.H2TauTau.proto.plotter.cut import *

from PhysicsTools.Heppy.utils.cmsswRelease import cmsswIs44X,cmsswIs52X
from PhysicsTools.Heppy.utils.cmsswRelease import isNewerThan

pt1 = 30
pt2 = 20 # 2011
if isNewerThan('CMSSW_5_2_0'):
    pt2 = 24 #2012

# ELECTRON = lepton 1
# TAU      = lepton 2

# this has to be in sync with:
# - H2TauTau/python/proto/analyzers/TauEleAnalyzer.py
# - H2TauTau/Colin/tauEle_2012_cfg.py
# - H2TauTau/python/objects/eleCuts_cff.py
# - H2TauTau/python/objects/tauCuts_cff.py
# - H2TauTau/python/objects/tauEleCuts_cff.py

# inc_sig_tau = Cut('l1_looseMvaIso>0.5 && l1_againstElectronMVA > 0.5 && l1_againstElectronTightMVA2 > 0.5 && l1_againstElectronMedium > 0.5 && l1_againstMuonLoose > 0.5 && l1_dxy<0.045 && l1_dz<0.2 && l1_pt>{pt1}'.format(pt1=pt1))

inc_sig_tau = Cut('leptonAccept && thirdLeptonVeto && l1_threeHitIso<1.5 && l1_againstElectronMVA3Medium > 0.5 && l1_againstMuonLoose > 0.5 && l1_dxy<0.045 && l1_dz<0.2 && l1_pt>{pt1}'.format(pt1=pt1))

inc_sig_ele = Cut('l2_relIso05<0.1 && l2_tightId>0.5 && l2_dxy<0.045 && l2_dz<0.2 && l2_pt>{pt2}'.format(pt2=pt2))

passleptonvetoes = Cut('leptonAccept > 0.5 && thirdLeptonVeto > 0.5')
#inc_sig = inc_sig_ele & inc_sig_tau
inc_sig = inc_sig_ele & inc_sig_tau & passleptonvetoes
cat_Inc = str(inc_sig)

cat_Inc_AntiEleAntiTauIsoJan = str(inc_sig).replace('l2_relIso05<0.1','l2_relIso05>0.2 && l2_relIso05<0.5').replace('l1_threeHitIso<1.5', 'l1_threeHitIso>1.5 && l1_threeHitIso<10.')
cat_Inc_AntiEleIsoJan = str(inc_sig).replace('l2_relIso05<0.1','l2_relIso05>0.2 && l2_relIso05<0.5')
cat_Inc_AntiTauIsoJan = str(inc_sig).replace('l1_threeHitIso<1.5', 'l1_threeHitIso>1.5 && l1_threeHitIso<10.')

# cat_Inc_AntiTauEleIDJan = str(inc_sig).replace('l1_againstElectronMVA3Medium > 0.5', 'l1_againstElectronMVA3Medium < 0.5')

cat_Inc_AntiTauEleIDJan = str(inc_sig).replace('l1_againstElectronMVA3Medium > 0.5', 'l1_againstElectronMVA3Medium < 0.5')

cat_Inc_AntiTauEleIDJan09 = str(inc_sig).replace('l1_againstElectronMVA3Medium > 0.5', 'l1_againstElectronMVA3Medium < 0.5 && l1_againstElectronMVA3raw > 0.9')

cat_Inc_AntiTauEleIDJan0809 = str(inc_sig).replace('l1_againstElectronMVA3Medium > 0.5', 'l1_againstElectronMVA3Medium < 0.5 && l1_againstElectronMVA3raw > 0.8 && l1_againstElectronMVA3raw < 0.9')

def cutstr_signal():
    return inc_sig

categories = {
    'Xcat_Inc_AntiEleAntiTauIsoJanX':cat_Inc_AntiEleAntiTauIsoJan,
    'Xcat_Inc_AntiEleIsoJanX':cat_Inc_AntiEleIsoJan,
    'Xcat_Inc_AntiTauIsoJanX':cat_Inc_AntiTauIsoJan,
    'Xcat_Inc_AntiTauEleIDJanX':cat_Inc_AntiTauEleIDJan,
    'Xcat_Inc_AntiTauEleIDJan09X':cat_Inc_AntiTauEleIDJan09,
    'Xcat_Inc_AntiTauEleIDJan0809X':cat_Inc_AntiTauEleIDJan0809,
    'Xcat_IncX':cat_Inc,
    }

categories.update( categories_common )
