import PhysicsTools.HeppyCore.framework.config as cfg
import os




################## Triggers 


triggers_mumu = ["HLT_Mu17_Mu8_v*","HLT_Mu17_TkMu8_v*"]
triggers_ee   = ["HLT_Ele17_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_Ele8_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_v*",
                 "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*",
                 "HLT_Ele15_Ele8_Ele5_CaloIdL_TrkIdVL_v*"]

triggers_mue   = [
    "HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*",
    "HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*"
    ]

triggersMC_mumu = ["HLT_Mu17_Mu8_v*","HLT_Mu17_TkMu8_v*"]

triggersMC_ee   = ["HLT_Ele17_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_Ele8_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_v*",
                   "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*",
                   "HLT_Ele15_Ele8_Ele5_CaloIdL_TrkIdVL_v*"]

triggersMC_mue   = ["HLT_Ele17_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_Ele8_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_v*",
                    "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*",
                    "HLT_Ele15_Ele8_Ele5_CaloIdL_TrkIdVL_v*",
                    "HLT_Mu17_Mu8_v*",
                    "HLT_Mu17_TkMu8_v*",
                    "HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*",
                    "HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*"
                   ]

triggers_1mu = [ 'HLT_IsoMu24_eta2p1_v*' ]
triggersMC_1mu  = triggers_1mu;
triggersFR_1mu  = [ 'HLT_Mu5_v*', 'HLT_RelIso1p0Mu5_v*', 'HLT_Mu12_v*', 'HLT_Mu24_eta2p1_v*', 'HLT_Mu40_eta2p1_v*' ]
triggersFR_mumu = [ 'HLT_Mu17_Mu8_v*', 'HLT_Mu17_TkMu8_v*', 'HLT_Mu8_v*', 'HLT_Mu17_v*' ]
triggersFR_1e   = [ 'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*', 'HLT_Ele17_CaloIdL_CaloIsoVL_v*', 'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*', 'HLT_Ele8__CaloIdL_CaloIsoVL_v*']
triggersFR_mue  = triggers_mue[:]
triggersFR_MC = triggersFR_1mu + triggersFR_mumu + triggersFR_1e + triggersFR_mue


#####COMPONENT CREATOR

from CMGTools.RootTools.samples.ComponentCreator import ComponentCreator
kreator = ComponentCreator()

## CENTRALLY PRODUCED MINIAODs V2 (from global DBS, in T2_CH_CAF)
### PU40 bx50ns

#### Background samples

# DY inclusive (cross section from FEWZ, StandardModelCrossSectionsat13TeV)
DYJetsToLL_M50 = kreator.makeMCComponent("DYJetsToLL_M50", "/DYJetsToLL_M-50_13TeV-madgraph-pythia8-tauola_v2/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root", 2008.*3)

DYJetsToLL_M50_HT100to200 = kreator.makeMCComponent("DYJetsToLL_M50_HT100to200", "/DYJetsToLL_M-50_HT-100to200_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",194.3*1.27)
DYJetsToLL_M50_HT200to400 = kreator.makeMCComponent("DYJetsToLL_M50_HT200to400", "/DYJetsToLL_M-50_HT-200to400_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",52.24*1.27)
DYJetsToLL_M50_HT400to600 = kreator.makeMCComponent("DYJetsToLL_M50_HT400to600", "/DYJetsToLL_M-50_HT-400to600_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",52.24*1.27)
DYJetsToLL_M50_HT600toInf = kreator.makeMCComponent("DYJetsToLL_M50_HT600toInf", "/DYJetsToLL_M-50_HT-600toInf_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",2.179*1.27)
DYJetsM50HT = [
DYJetsToLL_M50_HT100to200,
DYJetsToLL_M50_HT200to400,
DYJetsToLL_M50_HT400to600,
DYJetsToLL_M50_HT600toInf,
]

ZJetsToNuNu_HT100to200 = kreator.makeMCComponent("ZJetsToNuNu_HT100to200", "/ZJetsToNuNu_HT-100to200_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",372.6*1.27)
ZJetsToNuNu_HT200to400 = kreator.makeMCComponent("ZJetsToNuNu_HT200to400", "/ZJetsToNuNu_HT-200to400_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",100.8*1.27)
ZJetsToNuNu_HT400to600 = kreator.makeMCComponent("ZJetsToNuNu_HT400to600", "/ZJetsToNuNu_HT-400to600_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",11.99*1.27)
ZJetsToNuNu_HT600toInf = kreator.makeMCComponent("ZJetsToNuNu_HT600toInf", "/ZJetsToNuNu_HT-600toInf_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",4.113*1.27)
ZJetsToNuNuHT = [
ZJetsToNuNu_HT100to200,
ZJetsToNuNu_HT200to400,
ZJetsToNuNu_HT400to600,
ZJetsToNuNu_HT600toInf,
]

# W inclusive (cross section from FEWZ, StandardModelCrossSectionsat13TeV)
WJetsToLNu = kreator.makeMCComponent("WJetsToLNu", "/WJetsToLNu_13TeV-madgraph-pythia8-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")

WJetsToLNu_HT100to200 = kreator.makeMCComponent("WJetsToLNu_HT100to200", "/WJetsToLNu_HT-100to200_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",1817*1.23)
WJetsToLNu_HT200to400 = kreator.makeMCComponent("WJetsToLNu_HT200to400", "/WJetsToLNu_HT-200to400_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",471.6*1.23)
WJetsToLNu_HT400to600 = kreator.makeMCComponent("WJetsToLNu_HT400to600", "/WJetsToLNu_HT-400to600_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",55.61*1.23)
WJetsToLNu_HT600toInf = kreator.makeMCComponent("WJetsToLNu_HT600toInf", "/WJetsToLNu_HT-600toInf_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",18.81*1.23)
WJetsToLNuHT = [
WJetsToLNu_HT100to200,
WJetsToLNu_HT200to400,
WJetsToLNu_HT400to600,
WJetsToLNu_HT600toInf,
]

GJets_HT100to200 = kreator.makeMCComponent("GJets_HT100to200", "/GJets_HT-100to200_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",1534)
GJets_HT200to400 = kreator.makeMCComponent("GJets_HT200to400", "/GJets_HT-200to400_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",489.9)
GJets_HT400to600 = kreator.makeMCComponent("GJets_HT400to600", "/GJets_HT-400to600_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",62.05)
GJets_HT600toInf = kreator.makeMCComponent("GJets_HT600toInf", "/GJets_HT-600toInf_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v2/MINIAODSIM", "CMS", ".*root",20.87)
GJetsHT = [
GJets_HT100to200,
GJets_HT200to400,
GJets_HT400to600,
GJets_HT600toInf,
]

QCD_HT250To500 = kreator.makeMCComponent("QCD_HT240To500", "/QCD_HT_250To500_13TeV-madgraph/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")
QCD_HT500To1000 = kreator.makeMCComponent("QCD_HT500To1000", "/QCD_HT-500To1000_13TeV-madgraph/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")
QCD_HT1000ToInf = kreator.makeMCComponent("QCD_HT1000ToInf", "/QCD_HT_1000ToInf_13TeV-madgraph/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")

QCDHT = [
QCD_HT250To500,
QCD_HT500To1000,
QCD_HT1000ToInf, 
]

QCD_Pt300to470 = kreator.makeMCComponent("QCD_Pt300to470", "/QCD_Pt-300to470_Tune4C_13TeV_pythia8/Spring14miniaod-141029_PU40bx50_castor_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")
QCD_Pt470to600 = kreator.makeMCComponent("QCD_Pt470to600", "/QCD_Pt-470to600_Tune4C_13TeV_pythia8/Spring14miniaod-141029_PU40bx50_castor_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")
QCD_Pt600to800 = kreator.makeMCComponent("QCD_Pt600to800", "/QCD_Pt-600to800_Tune4C_13TeV_pythia8/Spring14miniaod-141029_PU40bx50_castor_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")
QCD_Pt800to1000 = kreator.makeMCComponent("QCD_Pt800to1000", "/QCD_Pt-800to1000_Tune4C_13TeV_pythia8/Spring14miniaod-141029_PU40bx50_castor_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")
QCD_Pt1000to1400 = kreator.makeMCComponent("QCD_Pt1000to1400", "/QCD_Pt-1000to1400_Tune4C_13TeV_pythia8/Spring14miniaod-141029_PU40bx50_castor_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")
QCD_Pt1400to1800 = kreator.makeMCComponent("QCD_Pt1400to1800", "/QCD_Pt-1400to1800_Tune4C_13TeV_pythia8/Spring14miniaod-141029_PU40bx50_castor_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")
QCD_Pt1800       = kreator.makeMCComponent("QCD_Pt1800      ", "/QCD_Pt-1800_Tune4C_13TeV_pythia8/Spring14miniaod-141029_PU40bx50_castor_PLS170_V6AN2-v1/MINIAODSIM","CMS", ".*root")
QCD_Pt1800to2400 = kreator.makeMCComponent("QCD_Pt1800to2400", "/QCD_Pt-1800to2400_Tune4C_13TeV_pythia8/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")
QCD_Pt2400to3200 = kreator.makeMCComponent("QCD_Pt2400to3200", "/QCD_Pt-2400to3200_Tune4C_13TeV_pythia8/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")
QCD_Pt3200 = kreator.makeMCComponent("QCD_Pt3200", "/QCD_Pt-3200_Tune4C_13TeV_pythia8/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")
QCDPt = [
QCD_Pt300to470,
QCD_Pt470to600,
QCD_Pt600to800,
QCD_Pt800to1000,
QCD_Pt1000to1400,
QCD_Pt1400to1800,
QCD_Pt1800,
QCD_Pt1800to2400,
QCD_Pt2400to3200,
QCD_Pt3200,
]
TTJets = kreator.makeMCComponent("TTJets", "/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",809.1)

TToLeptons_tch = kreator.makeMCComponent("TToLeptons_tch", "/TToLeptons_t-channel-CSA14_Tune4C_13TeV-aMCatNLO-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root")
T_tWch = kreator.makeMCComponent("T_tWch", "/T_tW-channel-DR_Tune4C_13TeV-CSA14-powheg-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",35.6)

SingleTop = [
    TToLeptons_tch, T_tWch
]


TTZJets = kreator.makeMCComponent("TTZJets", "/TTZJets_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v3/MINIAODSIM", "CMS", ".*root",0.8565)
WZJetsTo3LNu = kreator.makeMCComponent("WZJetsTo3LNu", "/WZJetsTo3LNu_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",2.29)
ZZTo4L = kreator.makeMCComponent("ZZTo4L","/ZZTo4L_Tune4C_13TeV-powheg-pythia8/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root", 31.8*(3*0.03366**2))

#### Signal samples
SMS_T2qq_2J_mStop600_mLSP550 = kreator.makeMCComponent("SMS_T2qq_2J_mStop600_mLSP550", "/SMS-T2qq_2J_mStop-600_mLSP-550_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",1.76645)
SMS_T2qq_2J_mStop1200_mLSP100 = kreator.makeMCComponent("SMS_T2qq_2J_mStop1200_mLSP100", "/SMS-T2qq_2J_mStop-1200_mLSP-100_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",0.0162846)
SMS_T2bb_2J_mStop600_mLSP580 = kreator.makeMCComponent("SMS_T2bb_2J_mStop600_mLSP580", "/SMS-T2bb_2J_mStop-600_mLSP-580_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",0.174599)
SMS_T2bb_2J_mStop900_mLSP100 = kreator.makeMCComponent("SMS_T2bb_2J_mStop900_mLSP100", "/SMS-T2bb_2J_mStop-900_mLSP-100_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",0.0128895)
SMS_T2tt_2J_mStop500_mLSP325 = kreator.makeMCComponent("SMS_T2tt_2J_mStop500_mLSP325", "/SMS-T2tt_2J_mStop-500_mLSP-325_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",0.51848)
SMS_T2tt_2J_mStop650_mLSP325 = kreator.makeMCComponent("SMS_T2tt_2J_mStop650_mLSP325", "/SMS-T2tt_2J_mStop-650_mLSP-325_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v2/MINIAODSIM", "CMS", ".*root",0.107045)
SMS_T2tt_2J_mStop425_mLSP325 = kreator.makeMCComponent("SMS_T2tt_2J_mStop425_mLSP325", "/SMS-T2tt_2J_mStop-425_mLSP-325_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",1.31169)
SMS_T2tt_2J_mStop850_mLSP100 = kreator.makeMCComponent("SMS_T2tt_2J_mStop850_mLSP100", "/SMS-T2tt_2J_mStop-850_mLSP-100_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",0.0189612)
SMS_T1tttt_2J_mGl1500_mLSP100 = kreator.makeMCComponent("SMS_T1tttt_2J_mGl1500_mLSP100", "/SMS-T1tttt_2J_mGl-1500_mLSP-100_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",0.0141903)
SMS_T1tttt_2J_mGl1200_mLSP800 = kreator.makeMCComponent("SMS_T1tttt_2J_mGl1200_mLSP800", "/SMS-T1tttt_2J_mGl-1200_mLSP-800_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",0.0856418)
SMS_T1bbbb_2J_mGl1000_mLSP900 = kreator.makeMCComponent("SMS_T1bbbb_2J_mGl1000_mLSP900", "/SMS-T1bbbb_2J_mGl-1000_mLSP-900_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",0.325388)
SMS_T1bbbb_2J_mGl1500_mLSP100 = kreator.makeMCComponent("SMS_T1bbbb_2J_mGl1500_mLSP100", "/SMS-T1bbbb_2J_mGl-1500_mLSP-100_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v2/MINIAODSIM", "CMS", ".*root",0.0141903)
SMS_T1qqqq_2J_mGl1400_mLSP100 = kreator.makeMCComponent("SMS_T1qqqq_2J_mGl1400_mLSP100", "/SMS-T1qqqq_2J_mGl-1400_mLSP-100_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",0.0252977)
SMS_T1qqqq_2J_mGl1000_mLSP800 = kreator.makeMCComponent("SMS_T1qqqq_2J_mGl1000_mLSP800", "/SMS-T1qqqq_2J_mGl-1000_mLSP-800_Tune4C_13TeV-madgraph-tauola/Spring14miniaod-141029_PU40bx50_PLS170_V6AN2-v1/MINIAODSIM", "CMS", ".*root",0.325388)

SusySignalSamples = [
SMS_T2qq_2J_mStop600_mLSP550,
SMS_T2qq_2J_mStop1200_mLSP100,
SMS_T2bb_2J_mStop600_mLSP580,
SMS_T2bb_2J_mStop900_mLSP100,
SMS_T2tt_2J_mStop500_mLSP325,
SMS_T2tt_2J_mStop650_mLSP325,
SMS_T2tt_2J_mStop850_mLSP100,
SMS_T2tt_2J_mStop425_mLSP325,
SMS_T1tttt_2J_mGl1500_mLSP100,
SMS_T1tttt_2J_mGl1200_mLSP800,
SMS_T1bbbb_2J_mGl1000_mLSP900,
SMS_T1bbbb_2J_mGl1500_mLSP100,
SMS_T1qqqq_2J_mGl1400_mLSP100,
SMS_T1qqqq_2J_mGl1000_mLSP800,
]

mcSamplesCSA14v2 = DYJetsM50HT + ZJetsToNuNuHT + WJetsToLNuHT + GJetsHT + QCDHT +  QCDPt + SingleTop + [TTJets, TTZJets, WJetsToLNu, DYJetsToLL_M50, WZJetsTo3LNu, ZZTo4L]  + SusySignalSamples

## MORE private samples on EOS
### PU40 bx50ns
#### Signal samples
T5WW_2J_mGo1200_mCh1000_mChi800 = kreator.makeMCComponentFromEOS('T5WW_2J_mGo1200_mCh1000_mChi800', '/T5Full-1200-1000-800-Decay-MGMMatch50/', '/store/cmst3/group/susy/gpetrucc/13TeV/MINIAODSIM/%s',".*root",0.0856418)
T5WW_2J_mGo1500_mCh800_mChi100  = kreator.makeMCComponentFromEOS('T5WW_2J_mGo1500_mCh800_mChi100', '/T5Full-1500-800-100-Decay-MGMMatch50/', '/store/cmst3/group/susy/gpetrucc/13TeV/MINIAODSIM/%s',".*root",0.0141903)
T5WW_2J_mGo1400_mCh315_mChi300 = kreator.makePrivateMCComponent('T5WW_2J_mGo1400_mCh315_mChi300','', ["/store/cmst3/group/susy/alobanov/MC/MiniAOD_v2/13TeV_T5qqqqWW_Gl_1400_LSP_300_Chi_315/13TeV_T5qqqqWW_Gl_1400_LSP_300_Chi_315_MiniAOD-v2.root"],0.0252977)

T1tttt_2J_mGo1000_mStop300_mCh285_mChi280 = kreator.makeMCComponentFromEOS('T1tttt_2J_mGo1000_mStop300_mCh285_mChi280', '/T1tttt_2J_mGo1000_mStop300_mCh285_mChi280_pythia8-23bodydec/', '/store/cmst3/group/susy/gpetrucc/13TeV/MINIAODSIM/%s',".*root",0.325388)
T1tttt_2J_mGo1300_mStop300_mCh285_mChi280 = kreator.makeMCComponentFromEOS('T1tttt_2J_mGo1300_mStop300_mCh285_mChi280', '/T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_pythia8-23bodydec/', '/store/cmst3/group/susy/gpetrucc/13TeV/MINIAODSIM/%s',".*root",0.0460525)
T1tttt_2J_mGo1300_mStop300_mChi280 = kreator.makeMCComponentFromEOS('T1tttt_2J_mGo1300_mStop300_mChi280', '/T1tttt_2J_mGo1300_mStop300_mChi280_pythia8-4bodydec/', '/store/cmst3/group/susy/gpetrucc/13TeV/MINIAODSIM/%s/',".*root",0.0460525)
T1tttt_2J_mGo800_mStop300_mCh285_mChi280 = kreator.makeMCComponentFromEOS('T1tttt_2J_mGo800_mStop300_mCh285_mChi280', '/T1tttt_2J_mGo800_mStop300_mCh285_mChi280_pythia8-23bodydec/', '/store/cmst3/group/susy/gpetrucc/13TeV/MINIAODSIM/%s',".*root",1.4891)
T1tttt_2J_mGo800_mStop300_mChi280 = kreator.makeMCComponentFromEOS('T1tttt_2J_mGo800_mStop300_mChi280', '/T1tttt_2J_mGo800_mStop300_mChi280_pythia8-4bodydec/', '/store/cmst3/group/susy/gpetrucc/13TeV/MINIAODSIM/%s',".*root",1.4891)
T1tttt_2J_mGo1000_mStop300_mCh285_mChi280_dilep = kreator.makeMCComponentFromEOS('T1tttt_2J_mGo1000_mStop300_mCh285_mChi280_dilep', '/T1tttt_2J_mGo1000_mStop300_mCh285_mChi280_23bodydec_dilepfilter/', '/store/cmst3/group/susy/gpetrucc/13TeV/MINIAODSIM/%s',".*root",0.325388*(40.3244/20.9656)*(3596.7/28378))
T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_dilep = kreator.makeMCComponentFromEOS('T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_dilep', '/T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_23bodydec_dilepfilter/', '/store/cmst3/group/susy/gpetrucc/13TeV/MINIAODSIM/%s',".*root",0.0460525*(48.03625/22.0665)*(6699.05/89779))
T6ttWW_2J_mSbot600_mCh425_mChi50 = kreator.makeMCComponentFromEOS('T6ttWW_2J_mSbot600_mCh425_mChi50', '/T6ttWW_600_425_50/', '/store/cmst3/group/susy/gpetrucc/13TeV/MINIAODSIM/%s',".*root",0.174599)
T6ttWW_2J_mSbot650_mCh150_mChi50 = kreator.makeMCComponentFromEOS('T6ttWW_2J_mSbot650_mCh150_mChi50', '/T6ttWW_650_150_50/', '/store/cmst3/group/susy/gpetrucc/13TeV/MINIAODSIM/%s',".*root",0.107045)

T1ttbb_2J_mGo1500_mChi100 = kreator.makeMCComponentFromEOS('T1ttbb_2J_mGo1500_mChi100','T1ttbb_2J_mGo1500_mChi100_3bodydec_asymmDecOnly','/store/cmst3/group/susy/gpetrucc/13TeV/MINIAODSIM/%s',".*root",0.0141903)

mcPrivateSamplesCSA14v2 = [T5WW_2J_mGo1200_mCh1000_mChi800, T5WW_2J_mGo1500_mCh800_mChi100, T5WW_2J_mGo1400_mCh315_mChi300, T1tttt_2J_mGo1000_mStop300_mCh285_mChi280, T1tttt_2J_mGo1300_mStop300_mCh285_mChi280, T1tttt_2J_mGo1300_mStop300_mChi280, T1tttt_2J_mGo800_mStop300_mCh285_mChi280, T1tttt_2J_mGo800_mStop300_mChi280, T1tttt_2J_mGo1000_mStop300_mCh285_mChi280_dilep, T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_dilep, T6ttWW_2J_mSbot600_mCh425_mChi50, T6ttWW_2J_mSbot650_mCh150_mChi50,T1ttbb_2J_mGo1500_mChi100]

mcSamples = mcSamplesCSA14v2 + mcPrivateSamplesCSA14v2


#-----------DATA---------------

dataDir = os.environ['CMSSW_BASE']+"/src/CMGTools/TTHAnalysis/data"
#lumi: 12.21+7.27+0.134 = 19.62 /fb @ 8TeV

json=dataDir+'/json/Cert_Run2012ABCD_22Jan2013ReReco.json'

SingleMu = cfg.DataComponent(
    name = 'SingleMu',
    files = kreator.getFilesFromEOS('SingleMu', 
                                    '/SingleMu/Run2012D-15Apr2014-v1/AOD/MINIAOD/CMSSW_7_0_9_patch2_GR_70_V2_AN1',
                                    '/eos/cms/store/cmst3/user/cmgtools/CMG/%s'),
    intLumi = 1,
    triggers = [],
    json = json
    )

           
dataSamplesMu=[]
dataSamplesE=[]
dataSamplesMuE=[]
dataSamples1Mu=[SingleMu]
dataSamplesAll = dataSamplesMu+dataSamplesE+dataSamplesMuE+dataSamples1Mu


from CMGTools.TTHAnalysis.setup.Efficiencies import *


#Define splitting
for comp in mcSamples:
    comp.isMC = True
    comp.isData = False
    comp.splitFactor = 250 #  if comp.name in [ "WJets", "DY3JetsM50", "DY4JetsM50","W1Jets","W2Jets","W3Jets","W4Jets","TTJetsHad" ] else 100
    comp.puFileMC=dataDir+"/puProfile_Summer12_53X.root"
    comp.puFileData=dataDir+"/puProfile_Data12.root"
    comp.efficiency = eff2012

for comp in dataSamplesAll:
    comp.splitFactor = 1000
    comp.isMC = False
    comp.isData = True

if __name__ == "__main__":
   import sys
   if "test" in sys.argv:
       from CMGTools.RootTools.samples.ComponentCreator import testSamples
       testSamples(mcSamples)
