import PhysicsTools.HeppyCore.framework.config as cfg
import os


################## Triggers
from CMGTools.RootTools.samples.triggers_13TeV_PHYS14 import *



#####COMPONENT CREATOR

from CMGTools.RootTools.samples.ComponentCreator import ComponentCreator
kreator = ComponentCreator()


## CENTRALLY PRODUCED MINIAODs V2 (from global DBS, in T2_CH_CAF)

##################  PU40 bx25ns (not default, so samples have a _PU40bx25 postfix) ################## 
GGHZZ4L_PU40bx25 = kreator.makeMCComponent("GGHZZ4L_PU40bx25", "/GluGluToHToZZTo4L_M-125_13TeV-powheg-pythia6/Phys14DR-PU40bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 43.92*2.76E-04)

DYJetsMuMuM50_PtZ180_PU40bx25 = kreator.makeMCComponent("DYJetsMuMuM50_PtZ180_PU40bx25", "/DYJetsToMuMu_PtZ-180_M-50_13TeV-madgraph/Phys14DR-PU40bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root")

TT_PU40bx25 = kreator.makeMCComponent("TT_PU40bx25", "/TT_Tune4C_13TeV-pythia8-tauola/Phys14DR-PU40bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",809.1)
TTH_PU40bx25 = kreator.makeMCComponent("TTH_PU40bx25", "/TTbarH_M-125_13TeV_amcatnlo-pythia8-tauola/Phys14DR-PU40bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.5085)

mcSamplesPHYS14_PU40bx25 = [TT_PU40bx25,TTH_PU40bx25,DYJetsMuMuM50_PtZ180_PU40bx25,GGHZZ4L_PU40bx25]

################## PU4 bx25ns (no default of phys14, so no _4bx50 postfix) ##############
# inclusive samples only for the low PU scenario

TT_PU4bx50 = kreator.makeMCComponent("TT_PU4bx50", "/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU4bx50_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",809.1)
WJetsToLNu_PU4bx50 = kreator.makeMCComponent("WJetsToLNu_PU4bx50","/WJetsToLNu_13TeV-madgraph-pythia8-tauola/Phys14DR-PU4bx50_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 20508.9)
DYJetsToLL_M50_PU4bx50 = kreator.makeMCComponent("DYJetsToLL_M50_PU4bx50", "/DYJetsToLL_M-50_13TeV-madgraph-pythia8/Phys14DR-PU4bx50_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 2008.*3)

mcSamplesPHYS14_PU4bx50 = [TT_PU4bx50,WJetsToLNu_PU4bx50,DYJetsToLL_M50_PU4bx50]


################## PU20 bx25ns (default of phys14, so no postfix) ##############

#### Background samples

## Cross sections from McM (LO Madgraph)
QCD_HT_100To250 = kreator.makeMCComponent("QCD_HT_100To250", "/QCD_HT-100To250_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 28730000)
QCD_HT_250To500 = kreator.makeMCComponent("QCD_HT_250To500", "/QCD_HT_250To500_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 670500)
QCD_HT_250To500_ext1 = kreator.makeMCComponent("QCD_HT_250To500_ext1", "/QCD_HT_250To500_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1_ext1-v2/MINIAODSIM", "CMS", ".*root", 670500)
QCD_HT_500To1000 = kreator.makeMCComponent("QCD_HT_500To1000", "/QCD_HT-500To1000_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 26740)
QCD_HT_500To1000_ext1 = kreator.makeMCComponent("QCD_HT_500To1000_ext1", "/QCD_HT-500To1000_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1_ext1-v1/MINIAODSIM", "CMS", ".*root", 26740)
QCD_HT_1000ToInf_ext1 = kreator.makeMCComponent("QCD_HT_1000ToInf_ext1", "/QCD_HT_1000ToInf_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1_ext1-v1/MINIAODSIM", "CMS", ".*root", 769.7)
QCD_HT_1000ToInf = kreator.makeMCComponent("QCD_HT_1000ToInf", "/QCD_HT_1000ToInf_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 769.7)
QCDHT = [
QCD_HT_100To250,
QCD_HT_250To500,
QCD_HT_500To1000,
QCD_HT_1000ToInf,
QCD_HT_250To500_ext1,
QCD_HT_500To1000_ext1,
QCD_HT_1000ToInf_ext1
]


QCD_Pt15to30 = kreator.makeMCComponent("QCD_Pt15to30","/QCD_Pt-15to30_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 2237000000)
QCD_Pt30to50 = kreator.makeMCComponent("QCD_Pt30to50","/QCD_Pt-30to50_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 161500000)
QCD_Pt50to80 = kreator.makeMCComponent("QCD_Pt50to80","/QCD_Pt-50to80_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 22110000)
QCD_Pt80to120 = kreator.makeMCComponent("QCD_Pt80to120","/QCD_Pt-80to120_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 3000114.3)
QCD_Pt120to170 = kreator.makeMCComponent("QCD_Pt120to170","/QCD_Pt-120to170_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 493200)
QCD_Pt170to300 = kreator.makeMCComponent("QCD_Pt170to300","/QCD_Pt-170to300_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 120300)
QCD_Pt300to470 = kreator.makeMCComponent("QCD_Pt300to470","/QCD_Pt-300to470_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 7475)
QCD_Pt470to600 = kreator.makeMCComponent("QCD_Pt470to600","/QCD_Pt-470to600_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 587.1)
QCD_Pt600to800 = kreator.makeMCComponent("QCD_Pt600to800","/QCD_Pt-600to800_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 167)
QCD_Pt800to1000 = kreator.makeMCComponent("QCD_Pt800to1000","/QCD_Pt-800to1000_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 28.25)
QCD_Pt1000to1400 = kreator.makeMCComponent("QCD_Pt1000to1400","/QCD_Pt-1000to1400_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 8.195)
QCD_Pt1400to1800 = kreator.makeMCComponent("QCD_Pt1400to1800","/QCD_Pt-1400to1800_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 0.7346)
QCD_Pt1800to2400 = kreator.makeMCComponent("QCD_Pt1800to2400","/QCD_Pt-1800to2400_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 0.102, True)
QCD_Pt2400to3200 = kreator.makeMCComponent("QCD_Pt2400to3200","/QCD_Pt-2400to3200_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 0.00644, True)
QCD_Pt3200 = kreator.makeMCComponent("QCD_Pt3200","/QCD_Pt-3200_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 0.000163, True)

QCDPt = [
QCD_Pt15to30,
QCD_Pt30to50,
QCD_Pt50to80,
QCD_Pt80to120,
QCD_Pt120to170,
QCD_Pt170to300,
QCD_Pt300to470,
QCD_Pt470to600,
QCD_Pt600to800,
QCD_Pt800to1000,
QCD_Pt1000to1400,
QCD_Pt1400to1800,
QCD_Pt1800to2400,
QCD_Pt2400to3200,
QCD_Pt3200
]

# Muon-enriched QCD (cross sections and filter efficiency from McM)
QCD_Mu15 = kreator.makeMCComponent("QCD_Mu15", "/QCD_Pt-20toInf_MuEnrichedPt15_PionKaonDecay_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v3/MINIAODSIM", "CMS", ".*root", 866.6e6*0.00044);
QCD_Pt30to50_Mu5 = kreator.makeMCComponent("QCD_Pt30to50_Mu5", "/QCD_Pt-30to50_MuEnrichedPt5_PionKaonDecay_Tune4C_13TeV_pythia8/Phys14DR-AVE20BX25_tsg_PHYS14_25_V3-v2/MINIAODSIM", "CMS", ".*root", 164400000*0.0122);
QCD_Pt50to80_Mu5 = kreator.makeMCComponent("QCD_Pt50to80_Mu5", "/QCD_Pt-50to80_MuEnrichedPt5_PionKaonDecay_Tune4C_13TeV_pythia8/Phys14DR-AVE20BX25_tsg_PHYS14_25_V3-v1/MINIAODSIM", "CMS", ".*root", 21930000*0.0218);
QCD_Pt80to120_Mu5 = kreator.makeMCComponent("QCD_Pt80to120_Mu5", "/QCD_Pt-80to120_MuEnrichedPt5_PionKaonDecay_Tune4C_13TeV_pythia8/Phys14DR-AVE20BX25_tsg_PHYS14_25_V3-v1/MINIAODSIM", "CMS", ".*root", 3000000*0.0395);
QCD_Mu5 = [ QCD_Pt30to50_Mu5, QCD_Pt50to80_Mu5, QCD_Pt80to120_Mu5 ]

# Electron-enriched QCD (cross sections and filter efficiency from McM)
QCD_Pt10to20_EMEnriched = kreator.makeMCComponent("QCD_Pt10to20_EMEnriched", "/QCD_Pt-10to20_EMEnriched_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 8838e6*0.143);
QCD_Pt20to30_EMEnriched = kreator.makeMCComponent("QCD_Pt20to30_EMEnriched", "/QCD_Pt-20to30_EMEnriched_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 6773e5*0.007);
QCD_Pt30to80_EMEnriched = kreator.makeMCComponent("QCD_Pt30to80_EMEnriched", "/QCD_Pt-30to80_EMEnriched_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 1859e5*0.056);
QCD_Pt80to170_EMEnriched = kreator.makeMCComponent("QCD_Pt80to170_EMEnriched", "/QCD_Pt-80to170_EMEnriched_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 3259e3*0.158);
QCDPtEMEnriched = [
QCD_Pt10to20_EMEnriched,
QCD_Pt20to30_EMEnriched,
QCD_Pt30to80_EMEnriched,
QCD_Pt80to170_EMEnriched
]

QCD_Pt20to30_bcToE = kreator.makeMCComponent("QCD_Pt20to30_bcToE", "/QCD_Pt_20to30_bcToE_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 6759e5*0.00075);
QCD_Pt30to80_bcToE = kreator.makeMCComponent("QCD_Pt30to80_bcToE", "/QCD_Pt_30to80_bcToE_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 1859e5*0.00272);
QCD_Pt80to170_bcToE = kreator.makeMCComponent("QCD_Pt80to170_bcToE", "/QCD_Pt_80to170_bcToE_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 3495e3*0.01225);
QCD_Pt170toInf_bcToE = kreator.makeMCComponent("QCD_Pt170toInf_bcToE", "/QCD_Pt_170toInf_bcToE_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 1285e2*0.0406);
QCDPtbcToE = [
QCD_Pt20to30_bcToE,
QCD_Pt30to80_bcToE,
QCD_Pt80to170_bcToE,
QCD_Pt170toInf_bcToE
]


# W inclusive (cross section from FEWZ, StandardModelCrossSectionsat13TeV)
WJetsToLNu = kreator.makeMCComponent("WJetsToLNu","/WJetsToLNu_13TeV-madgraph-pythia8-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 20508.9)

# cross sections for WJets taken from McM LO times inclusive k-factor from FEWZ(20508.9 pb x3)/McM(50100.0) 
WJetsToLNu_HT100to200 = kreator.makeMCComponent("WJetsToLNu_HT100to200", "/WJetsToLNu_HT-100to200_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",1817.0*1.23)
WJetsToLNu_HT200to400 = kreator.makeMCComponent("WJetsToLNu_HT200to400", "/WJetsToLNu_HT-200to400_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",471.6*1.23)
WJetsToLNu_HT400to600 = kreator.makeMCComponent("WJetsToLNu_HT400to600", "/WJetsToLNu_HT-400to600_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",55.61*1.23)
WJetsToLNu_HT600toInf = kreator.makeMCComponent("WJetsToLNu_HT600toInf", "/WJetsToLNu_HT-600toInf_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",18.81*1.23)
WJetsToLNuHT = [
WJetsToLNu_HT100to200,
WJetsToLNu_HT200to400,
WJetsToLNu_HT400to600,
WJetsToLNu_HT600toInf,
]

# DY inclusive (cross section from FEWZ, StandardModelCrossSectionsat13TeV)
DYJetsToLL_M50 = kreator.makeMCComponent("DYJetsToLL_M50", "/DYJetsToLL_M-50_13TeV-madgraph-pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 2008.*3)

# DY HT bins: cross sections for DYJets taken from McM LO times inclusive k-factor from FEWZ(2008 pb x3)/McM(4746)
DYJetsToLL_M50_HT100to200 = kreator.makeMCComponent("DYJetsToLL_M50_HT100to200", "/DYJetsToLL_M-50_HT-100to200_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",194.3*1.27)
DYJetsToLL_M50_HT200to400 = kreator.makeMCComponent("DYJetsToLL_M50_HT200to400", "/DYJetsToLL_M-50_HT-200to400_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",52.24*1.27)
DYJetsToLL_M50_HT400to600 = kreator.makeMCComponent("DYJetsToLL_M50_HT400to600", "/DYJetsToLL_M-50_HT-400to600_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",6.546*1.27)
DYJetsToLL_M50_HT600toInf = kreator.makeMCComponent("DYJetsToLL_M50_HT600toInf", "/DYJetsToLL_M-50_HT-600toInf_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",2.179*1.27)
DYJetsM50HT = [
DYJetsToLL_M50_HT100to200,
DYJetsToLL_M50_HT200to400,
DYJetsToLL_M50_HT400to600,
DYJetsToLL_M50_HT600toInf,
]

DYJetsMuMuM50_PtZ180 = kreator.makeMCComponent("DYJetsMuMuM50_PtZ180", "/DYJetsToMuMu_PtZ-180_M-50_13TeV-madgraph/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v3/MINIAODSIM", "CMS", ".*root")

GJets_HT100to200 = kreator.makeMCComponent("GJets_HT100to200", "/GJets_HT-100to200_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",1534)
GJets_HT200to400 = kreator.makeMCComponent("GJets_HT200to400", "/GJets_HT-200to400_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",489.9)
GJets_HT400to600 = kreator.makeMCComponent("GJets_HT400to600", "/GJets_HT-400to600_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",62.05)
GJets_HT600toInf = kreator.makeMCComponent("GJets_HT600toInf", "/GJets_HT-600toInf_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",20.87)
GJetsHT = [
GJets_HT100to200,
GJets_HT200to400,
GJets_HT400to600,
GJets_HT600toInf,
]
ZJetsToNuNu_HT100to200 = kreator.makeMCComponent("ZJetsToNuNu_HT100to200", "/ZJetsToNuNu_HT-100to200_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",372.6*1.27)
ZJetsToNuNu_HT200to400 = kreator.makeMCComponent("ZJetsToNuNu_HT200to400", "/ZJetsToNuNu_HT-200to400_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",100.8*1.27)
ZJetsToNuNu_HT400to600 = kreator.makeMCComponent("ZJetsToNuNu_HT400to600", "/ZJetsToNuNu_HT-400to600_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root",11.99*1.27)
ZJetsToNuNu_HT600toInf = kreator.makeMCComponent("ZJetsToNuNu_HT600toInf", "/ZJetsToNuNu_HT-600toInf_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",4.113*1.27)
ZJetsToNuNuHT = [
ZJetsToNuNu_HT100to200,
ZJetsToNuNu_HT200to400,
ZJetsToNuNu_HT400to600,
ZJetsToNuNu_HT600toInf,
]

# Single top cross sections: https://twiki.cern.ch/twiki/bin/viewauth/CMS/SingleTopSigma
TToLeptons_tch = kreator.makeMCComponent("TToLeptons_tch", "/TToLeptons_t-channel-CSA14_Tune4C_13TeV-aMCatNLO-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 136.05*0.108*3) 
TToLeptons_sch = kreator.makeMCComponent("TToLeptons_sch", "/TToLeptons_s-channel-CSA14_Tune4C_13TeV-aMCatNLO-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 7.20*0.108*3)
TBarToLeptons_tch = kreator.makeMCComponent("TBarToLeptons_tch", "/TBarToLeptons_t-channel_Tune4C_CSA14_13TeV-aMCatNLO-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 80.97*0.108*3)
TBarToLeptons_sch = kreator.makeMCComponent("TBarToLeptons_sch", "/TBarToLeptons_s-channel-CSA14_Tune4C_13TeV-aMCatNLO-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",  4.16*0.108*3)
TBar_tWch = kreator.makeMCComponent("TBar_tWch", "/Tbar_tW-channel-DR_Tune4C_13TeV-CSA14-powheg-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",35.6)
T_tWch = kreator.makeMCComponent("T_tWch", "/T_tW-channel-DR_Tune4C_13TeV-CSA14-powheg-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",35.6)

SingleTop = [
    TToLeptons_tch, TToLeptons_sch, TBarToLeptons_tch, TBarToLeptons_sch, TBar_tWch, T_tWch
]

# TTbar cross section: MCFM with dynamic scale, StandardModelCrossSectionsat13TeV
TTJets = kreator.makeMCComponent("TTJets", "/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",809.1)

# TTV cross sections are from 13 TeV MG5_aMC@NLO v2.2.1, NNPDF 2.3nlo, fixed scale = mtop + 0.5*mv
TTWJets = kreator.makeMCComponent("TTWJets", "/TTWJets_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.6647)
TTZJets = kreator.makeMCComponent("TTZJets", "/TTZJets_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.8565)

# TTH cross section from LHC Higgs XS WG: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt1314TeV?rev=15
TTH = kreator.makeMCComponent("TTH", "/TTbarH_M-125_13TeV_amcatnlo-pythia8-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root",0.5085)


# cross section from StandardModelCrossSectionsat13TeV (NLO MCFM, mll > 12) times BR=(3*0.108)*(3*0.0337)
WZJetsTo3LNu = kreator.makeMCComponent("WZJetsTo3LNu", "/WZJetsTo3LNu_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",2.165)
# cross section from StandardModelCrossSectionsat13TeV (NLO MCFM, mll > 12) times BR=(3*0.0337)**2
ZZTo4L = kreator.makeMCComponent("ZZTo4L","/ZZTo4L_Tune4C_13TeV-powheg-pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 	31.8*(3*0.03366**2))

# GGH cross section from LHC Higgs XS WG: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt1314TeV?rev=15
GGHZZ4L = kreator.makeMCComponent("GGHZZ4L", "/GluGluToHToZZTo4L_M-125_13TeV-powheg-pythia6/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 43.92*2.76E-04)

GGHTT = kreator.makeMCComponent("GGHTT", "/GluGluToHToTauTau_M-125_13TeV-powheg-pythia6/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 43.92*0.0632)
VBFTT = kreator.makeMCComponent("VBFTT", "/VBF_HToTauTau_M-125_13TeV-powheg-pythia6/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 3.748*0.0632)


#### Signal samples
# cross sections from LHC SUSY Cross Section Working Group https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SUSYCrossSections
SMS_T2tt_2J_mStop850_mLSP100 = kreator.makeMCComponent("SMS_T2tt_2J_mStop850_mLSP100", "/SMS-T2tt_2J_mStop-850_mLSP-100_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.0189612)
SMS_T2tt_2J_mStop650_mLSP325 = kreator.makeMCComponent("SMS_T2tt_2J_mStop650_mLSP325", "/SMS-T2tt_2J_mStop-650_mLSP-325_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.107045)
SMS_T2tt_2J_mStop500_mLSP325 = kreator.makeMCComponent("SMS_T2tt_2J_mStop500_mLSP325", "/SMS-T2tt_2J_mStop-500_mLSP-325_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.51848)
SMS_T2tt_2J_mStop425_mLSP325 = kreator.makeMCComponent("SMS_T2tt_2J_mStop425_mLSP325", "/SMS-T2tt_2J_mStop-425_mLSP-325_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",1.31169)
SMS_T2qq_2J_mStop600_mLSP550 = kreator.makeMCComponent("SMS_T2qq_2J_mStop600_mLSP550", "/SMS-T2qq_2J_mStop-600_mLSP-550_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",1.76645)
SMS_T2qq_2J_mStop1200_mLSP100 = kreator.makeMCComponent("SMS_T2qq_2J_mStop1200_mLSP100", "/SMS-T2qq_2J_mStop-1200_mLSP-100_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.0162846)
SMS_T2bb_2J_mStop900_mLSP100 = kreator.makeMCComponent("SMS_T2bb_2J_mStop900_mLSP100", "/SMS-T2bb_2J_mStop-900_mLSP-100_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.0128895)
SMS_T2bb_2J_mStop600_mLSP580 = kreator.makeMCComponent("SMS_T2bb_2J_mStop600_mLSP580", "/SMS-T2bb_2J_mStop-600_mLSP-580_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.174599)
SMS_T1tttt_2J_mGl1500_mLSP100 = kreator.makeMCComponent("SMS_T1tttt_2J_mGl1500_mLSP100", "/SMS-T1tttt_2J_mGl-1500_mLSP-100_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.0141903)
SMS_T1tttt_2J_mGl1200_mLSP800 = kreator.makeMCComponent("SMS_T1tttt_2J_mGl1200_mLSP800", "/SMS-T1tttt_2J_mGl-1200_mLSP-800_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.0856418)
SMS_T1qqqq_2J_mGl1400_mLSP100 = kreator.makeMCComponent("SMS_T1qqqq_2J_mGl1400_mLSP100", "/SMS-T1qqqq_2J_mGl-1400_mLSP-100_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.0252977)
SMS_T1qqqq_2J_mGl1000_mLSP800 = kreator.makeMCComponent("SMS_T1qqqq_2J_mGl1000_mLSP800", "/SMS-T1qqqq_2J_mGl-1000_mLSP-800_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.325388)
SMS_T1bbbb_2J_mGl1500_mLSP100 = kreator.makeMCComponent("SMS_T1bbbb_2J_mGl1500_mLSP100", "/SMS-T1bbbb_2J_mGl-1500_mLSP-100_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.0141903)
SMS_T1bbbb_2J_mGl1000_mLSP900 = kreator.makeMCComponent("SMS_T1bbbb_2J_mGl1000_mLSP900", "/SMS-T1bbbb_2J_mGl-1000_mLSP-900_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.325388)
SusySignalSamples = [
SMS_T2tt_2J_mStop850_mLSP100, 
SMS_T2tt_2J_mStop650_mLSP325, 
SMS_T2tt_2J_mStop500_mLSP325, 
SMS_T2tt_2J_mStop425_mLSP325, 
SMS_T2qq_2J_mStop600_mLSP550, 
SMS_T2qq_2J_mStop1200_mLSP100, 
SMS_T2bb_2J_mStop900_mLSP100, 
SMS_T2bb_2J_mStop600_mLSP580, 
SMS_T1tttt_2J_mGl1500_mLSP100, 
SMS_T1tttt_2J_mGl1200_mLSP800, 
SMS_T1qqqq_2J_mGl1400_mLSP100, 
SMS_T1qqqq_2J_mGl1000_mLSP800, 
SMS_T1bbbb_2J_mGl1500_mLSP100, 
SMS_T1bbbb_2J_mGl1000_mLSP900,
]

mcSamplesPHYS14_PU20bx25 = QCDHT + QCDPt + [QCD_Mu15] + QCD_Mu5 + QCDPtEMEnriched + QCDPtbcToE + [WJetsToLNu] + WJetsToLNuHT +  [DYJetsToLL_M50, DYJetsMuMuM50_PtZ180] + DYJetsM50HT + GJetsHT + ZJetsToNuNuHT + SingleTop + [ TTJets, TTWJets, TTZJets, TTH, WZJetsTo3LNu, ZZTo4L, GGHZZ4L, GGHTT, VBFTT] + SusySignalSamples


## PRIVATE SAMPLES

GJets_HT100to200_fixPhoton = kreator.makeMCComponentFromEOS('GJets_HT100to200', '/GJets_HT-100to200_Tune4C_13TeV-madgraph-tauola/miniAOD_fixPhoton_7_2_3/150204_164703/0000/', '/store/user/mmasciov/PHYS14_fixPhoton/%s',".*root", 1534)
GJets_HT200to400_fixPhoton = kreator.makeMCComponentFromEOS('GJets_HT200to400', '/GJets_HT-200to400_Tune4C_13TeV-madgraph-tauola/miniAOD_fixPhoton_7_2_3/150204_164621/0000/', '/store/user/mmasciov/PHYS14_fixPhoton/%s',".*root", 489.9)
GJets_HT400to600_fixPhoton = kreator.makeMCComponentFromEOS('GJets_HT400to600', '/GJets_HT-400to600_Tune4C_13TeV-madgraph-tauola/miniAOD_fixPhoton_7_2_3/150204_164547/0000/', '/store/user/mmasciov/PHYS14_fixPhoton/%s',".*root", 62.05)
GJets_HT600toInf_fixPhoton = kreator.makeMCComponentFromEOS('GJets_HT600toInf', '/GJets_HT-600toInf_Tune4C_13TeV-madgraph-tauola/miniAOD_fixPhoton_7_2_3/150204_122016/0000/', '/store/user/mmasciov/PHYS14_fixPhoton/%s',".*root", 20.87)
GJets_fixPhoton = [
GJets_HT100to200_fixPhoton,
GJets_HT200to400_fixPhoton,
GJets_HT400to600_fixPhoton,
GJets_HT600toInf_fixPhoton,
]

QCD_HT_100To250_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_100To250", '/QCD_HT-100To250_13TeV-madgraph/miniAOD_fixPhoton_reco/150206_145121/0000/', '/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_reco/%s',  ".*root", 28730000)
QCD_HT_250To500_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_250To500", '/QCD_HT_250To500_13TeV-madgraph/miniAOD_fixPhoton_reco/150206_145040/0000/', '/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_reco/%s',  ".*root", 670500)
QCD_HT_250To500_ext1_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_250To500_ext1", '/QCD_HT_250To500_13TeV-madgraph/miniAOD_fixPhoton_reco/150206_144831/0000/', '/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_reco/%s', ".*root", 670500)
QCD_HT_500To1000_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_500To1000", '/QCD_HT-500To1000_13TeV-madgraph/miniAOD_fixPhoton_reco/150206_144759/0000/', '/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_reco/%s', ".*root", 26740)
QCD_HT_500To1000_ext1_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_500To1000_ext1", '/QCD_HT-500To1000_13TeV-madgraph/miniAOD_fixPhoton_reco/150206_144615/0000/', '/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_reco/%s', ".*root", 26740)
QCD_HT_1000ToInf_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_1000ToInf", '/QCD_HT_1000ToInf_13TeV-madgraph/miniAOD_fixPhoton_7_2_3/150204_172505/0000/', '/store/user/mmasciov/PHYS14_fixPhoton/%s', ".*root", 769.7)
QCD_HT_1000ToInf_ext1_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_1000ToInf_ext1", '/QCD_HT_1000ToInf_13TeV-madgraph/miniAOD_fixPhoton_7_2_3/150204_172427/0000/', '/store/user/mmasciov/PHYS14_fixPhoton/%s', ".*root", 769.7)
QCDHT_fixPhoton = [
QCD_HT_100To250_fixPhoton,
QCD_HT_250To500_fixPhoton,
QCD_HT_500To1000_fixPhoton,
QCD_HT_1000ToInf_fixPhoton,
QCD_HT_250To500_ext1_fixPhoton,
QCD_HT_500To1000_ext1_fixPhoton,
QCD_HT_1000ToInf_ext1_fixPhoton,
]


QCD_Pt170to300_fixPhoton   = kreator.makeMCComponentFromEOS("QCD_Pt170to300"  , "QCD_Pt-170to300_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_154438/0000/"  , "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 120300)
QCD_Pt300to470_fixPhoton   = kreator.makeMCComponentFromEOS("QCD_Pt300to470"  , "QCD_Pt-300to470_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_154529/0000/"  , "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 7475)
QCD_Pt470to600_fixPhoton   = kreator.makeMCComponentFromEOS("QCD_Pt470to600"  , "QCD_Pt-470to600_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_154700/0000/"  , "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 587.1)
QCD_Pt600to800_fixPhoton   = kreator.makeMCComponentFromEOS("QCD_Pt600to800"  , "QCD_Pt-600to800_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_154904/0000/"  , "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 167)
QCD_Pt800to1000_fixPhoton  = kreator.makeMCComponentFromEOS("QCD_Pt800to1000" , "QCD_Pt-800to1000_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_155003/0000/" , "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 28.25)
QCD_Pt1000to1400_fixPhoton = kreator.makeMCComponentFromEOS("QCD_Pt1000to1400", "QCD_Pt-1000to1400_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_154248/0000/", "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 8.195)
QCD_Pt1400to1800_fixPhoton = kreator.makeMCComponentFromEOS("QCD_Pt1400to1800", "QCD_Pt-1400to1800_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_154344/0000/", "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 0.7346)
QCD_Pt1800to2400_fixPhoton = kreator.makeMCComponentFromEOS("QCD_Pt1800to2400", "QCD_Pt-1800to2400_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150301_002302/0000/", "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 0.102)
QCD_Pt2400to3200_fixPhoton = kreator.makeMCComponentFromEOS("QCD_Pt2400to3200", "QCD_Pt-2400to3200_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150301_002547/0000/", "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 0.00644)
QCD_Pt3200_fixPhoton       = kreator.makeMCComponentFromEOS("QCD_Pt3200"      , "QCD_Pt-3200_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150301_002653/0000/"      , "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 0.000163)


QCDPt_fixPhoton = [
QCD_Pt170to300_fixPhoton,
QCD_Pt300to470_fixPhoton,
QCD_Pt470to600_fixPhoton,
QCD_Pt600to800_fixPhoton,
QCD_Pt800to1000_fixPhoton,
QCD_Pt1000to1400_fixPhoton,
QCD_Pt1400to1800_fixPhoton,
QCD_Pt1800to2400_fixPhoton,
QCD_Pt2400to3200_fixPhoton,
QCD_Pt3200_fixPhoton
]

T5ttttDeg_mGo1000_mStop300_mCh285_mChi280 = kreator.makeMCComponentFromEOS('T5ttttDeg_mGo1000_mStop300_mCh285_mChi280', '/T5ttttDeg_mGo1000_mStop300_mCh285_mChi280_23bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388)
T5ttttDeg_mGo1300_mStop300_mCh285_mChi280 = kreator.makeMCComponentFromEOS('T5ttttDeg_mGo1300_mStop300_mCh285_mChi280', '/T5ttttDeg_mGo1300_mStop300_mCh285_mChi280_23bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.0460525)
T5ttttDeg_mGo1000_mStop300_mChi280 = kreator.makeMCComponentFromEOS('T5ttttDeg_mGo1000_mStop300_mChi280', '/T5ttttDeg_mGo1000_mStop300_mChi280_4bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388)
T5ttttDeg_mGo1300_mStop300_mChi280 = kreator.makeMCComponentFromEOS('T5ttttDeg_mGo1300_mStop300_mChi280', '/T5ttttDeg_mGo1300_mStop300_mChi280_4bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.0460525)
T5ttttDeg_mGo1000_mStop300_mCh285_mChi280_dil = kreator.makeMCComponentFromEOS('T5ttttDeg_mGo1000_mStop300_mCh285_mChi280_dil', '/T5ttttDeg_mGo1000_mStop300_mCh285_mChi280_23bodydec_dilepfilterPt8p5_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388)
T5ttttDeg_mGo1300_mStop300_mCh285_mChi280_dil = kreator.makeMCComponentFromEOS('T5ttttDeg_mGo1300_mStop300_mCh285_mChi280_dil', '/T5ttttDeg_mGo1300_mStop300_mCh285_mChi280_23bodydec_dilepfilterPt8p5_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.0460525)
T5ttttDeg = [ T5ttttDeg_mGo1000_mStop300_mCh285_mChi280, T5ttttDeg_mGo1300_mStop300_mCh285_mChi280, T5ttttDeg_mGo1000_mStop300_mChi280, T5ttttDeg_mGo1300_mStop300_mChi280, T5ttttDeg_mGo1000_mStop300_mCh285_mChi280_dil, T5ttttDeg_mGo1300_mStop300_mCh285_mChi280_dil ]

T1ttbbWW_mGo1000_mCh725_mChi715 = kreator.makeMCComponentFromEOS('T1ttbbWW_mGo1000_mCh725_mChi715', '/T1ttbbWW_2J_mGo1000_mCh725_mChi715_3bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388)
T1ttbbWW_mGo1000_mCh725_mChi720 = kreator.makeMCComponentFromEOS('T1ttbbWW_mGo1000_mCh725_mChi720', '/T1ttbbWW_2J_mGo1000_mCh725_mChi720_3bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388)
T1ttbbWW_mGo1300_mCh300_mChi290 = kreator.makeMCComponentFromEOS('T1ttbbWW_mGo1300_mCh300_mChi290', '/T1ttbbWW_2J_mGo1300_mCh300_mChi290_3bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.0460525)
T1ttbbWW_mGo1300_mCh300_mChi295 = kreator.makeMCComponentFromEOS('T1ttbbWW_mGo1300_mCh300_mChi295', '/T1ttbbWW_2J_mGo1300_mCh300_mChi295_3bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.0460525)
T1ttbbWW = [ T1ttbbWW_mGo1000_mCh725_mChi715, T1ttbbWW_mGo1000_mCh725_mChi720, T1ttbbWW_mGo1300_mCh300_mChi290, T1ttbbWW_mGo1300_mCh300_mChi295 ]

T1ttbb_mGo1500_mChi100 = kreator.makeMCComponentFromEOS('T1ttbb_mGo1500_mChi100', '/T1ttbb_2J_mGo1500_mChi100_3bodydec_asymmDecOnly/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.0141903)
T1ttbb = [ T1ttbb_mGo1500_mChi100 ]

T6ttWW_mSbot600_mCh425_mChi50 = kreator.makeMCComponentFromEOS('T6ttWW_mSbot600_mCh425_mChi50', '/T6ttWW_600_425_50_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.174599)
T6ttWW_mSbot650_mCh150_mChi50 = kreator.makeMCComponentFromEOS('T6ttWW_mSbot650_mCh150_mChi50', '/T6ttWW_650_150_50_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.107045)
T6ttWW = [ T6ttWW_mSbot600_mCh425_mChi50, T6ttWW_mSbot650_mCh150_mChi50 ]

#SqGltttt_mGo1300_mSq1300_mChi100 = kreator.makeMCComponentFromEOS('SqGltttt_mGo1300_mSq1300_mChi100', '/13TeV_SqGltttt_Gl_1300_Sq_1300_LSP_100/', '/store/cmst3/group/susy/alobanov/MC/PHYS14/PU20_25ns/%s', '.*root')
SqGltttt = [ ] #SqGltttt_mGo1300_mSq1300_mChi100 ]

T1tttt_mGo1300_mChi100 = kreator.makeMCComponentFromEOS('T1tttt_mGo1300_mChi100', '/SMS_T1tttt_2J_mGl1300_mLSP100/', '/store/cmst3/group/susy/alobanov/MC/PHYS14/PU20_25ns/%s', '.*root', 0.0460525)
T1tttt_mGo800_mChi450 = kreator.makeMCComponentFromEOS('T1tttt_mGo800_mChi450', '/SMS_T1tttt_2J_mGl800_mLSP450/', '/store/cmst3/group/susy/alobanov/MC/PHYS14/PU20_25ns/%s', '.*root', 1.4891)
T1tttt_priv = [ T1tttt_mGo1300_mChi100, T1tttt_mGo800_mChi450 ] 

T5qqqqWWDeg_mGo1400_mCh315_mChi300 = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1400_mCh315_mChi300', '/SMS_T5qqqqWW_mGl1400_mChi315_mLSP300/', '/store/cmst3/group/susy/alobanov/MC/PHYS14/PU20_25ns/%s', '.*root', 0.0252977)
T5qqqqWWDeg_mGo1000_mCh310_mChi300 = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1000_mCh310_mChi300', '/T5qqqqWWDeg_mGo1000_mCh310_mChi300/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388) 
T5qqqqWWDeg_mGo1000_mCh310_mChi300_dilep= kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1000_mCh310_mChi300_dilep', '/T5qqqqWWDeg_mGo1000_mCh310_mChi300_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388*(0.333)*(0.333)) 
T5qqqqWWDeg_mGo1000_mCh315_mChi300 = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1000_mCh315_mChi300', '/T5qqqqWWDeg_mGo1000_mCh315_mChi300/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388) 
T5qqqqWWDeg_mGo1000_mCh315_mChi300_dilep = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1000_mCh315_mChi300_dilep', '/T5qqqqWWDeg_mGo1000_mCh315_mChi300_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388*(0.333)*(0.333)) 
T5qqqqWWDeg_mGo1000_mCh325_mChi300 = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1000_mCh325_mChi300', '/T5qqqqWWDeg_mGo1000_mCh325_mChi300/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388) 
T5qqqqWWDeg_mGo1000_mCh325_mChi300_dilep = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1000_mCh325_mChi300_dilep', '/T5qqqqWWDeg_mGo1000_mCh325_mChi300_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388*(0.324)*(0.324)) 
T5qqqqWWDeg_mGo800_mCh305_mChi300 = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo800_mCh305_mChi300', '/T5qqqqWWDeg_mGo800_mCh305_mChi300/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 1.4891) 
T5qqqqWWDeg_mGo800_mCh305_mChi300_dilep = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo800_mCh305_mChi300_dilep', '/T5qqqqWWDeg_mGo800_mCh305_mChi300_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 1.4891*(0.342)*(0.342)) 
T5qqqqWWDeg = [
    T5qqqqWWDeg_mGo1400_mCh315_mChi300,
    T5qqqqWWDeg_mGo1000_mCh310_mChi300, T5qqqqWWDeg_mGo1000_mCh315_mChi300, T5qqqqWWDeg_mGo1000_mCh325_mChi300, T5qqqqWWDeg_mGo800_mCh305_mChi300,
    T5qqqqWWDeg_mGo1000_mCh310_mChi300_dilep, T5qqqqWWDeg_mGo1000_mCh315_mChi300_dilep, T5qqqqWWDeg_mGo1000_mCh325_mChi300_dilep, T5qqqqWWDeg_mGo800_mCh305_mChi300_dilep
]

T5qqqqWW_mGo1500_mCh800_mChi100 = kreator.makeMCComponentFromEOS('T5qqqqWW_mGo1500_mCh800_mChi100', '/SMS_T5qqqqWW_Gl1500_Chi800_LSP100/', '/store/cmst3/group/susy/alobanov/MC/PHYS14/PU20_25ns/%s', '.*root', 0.0141903)
T5qqqqWW_mGo1200_mCh1000_mChi800 = kreator.makeMCComponentFromEOS('T5qqqqWW_mGo1200_mCh1000_mChi800', '/SMS_T5qqqqWW_Gl1200_Chi1000_LSP800/', '/store/cmst3/group/susy/alobanov/MC/PHYS14/PU20_25ns/%s', '.*root', 0.0856418)
T5qqqqWW_mGo1000_mCh800_mChi700 = kreator.makeMCComponentFromEOS('T5qqqqWW_mGo1000_mCh800_mChi700', '/T5qqqqWW_mGo1000_mCh800_mChi700/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388) 
T5qqqqWW_mGo1000_mCh800_mChi700_dilep= kreator.makeMCComponentFromEOS('T5qqqqWW_mGo1000_mCh800_mChi700_dilep', '/T5qqqqWW_mGo1000_mCh800_mChi700_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.325388*(3*0.108)*(3*0.108)) 
T5qqqqWW_mGo1200_mCh1000_mChi800_cmg = kreator.makeMCComponentFromEOS('T5qqqqWW_mGo1200_mCh1000_mChi800_cmg', '/T5qqqqWW_mGo1200_mCh1000_mChi800/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.0856418) 
T5qqqqWW_mGo1200_mCh1000_mChi800_dilep= kreator.makeMCComponentFromEOS('T5qqqqWW_mGo1200_mCh1000_mChi800_dilep', '/T5qqqqWW_mGo1200_mCh1000_mChi800_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/Phys14DR/MINIAODSIM/%s',".*root", 0.0856418*(3*0.108)*(3*0.108)) 

T5qqqqWW = [
    T5qqqqWW_mGo1500_mCh800_mChi100, T5qqqqWW_mGo1200_mCh1000_mChi800,
    T5qqqqWW_mGo1000_mCh800_mChi700, T5qqqqWW_mGo1200_mCh1000_mChi800_cmg,
    T5qqqqWW_mGo1000_mCh800_mChi700_dilep, T5qqqqWW_mGo1200_mCh1000_mChi800_dilep
]



# note: cross section for q~ q~ from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SUSYCrossSections13TeVsquarkantisquark (i.e. gluinos and stops decoupled)
T6qqWW_mSq950_mCh325_mChi300 = kreator.makeMCComponentFromEOS('T6qqWW_mSq950_mCh325_mChi300', '/SMS_T6qqWW_mSq950_mChi325_mLSP300/', '/store/cmst3/group/susy/alobanov/MC/PHYS14/PU20_25ns/%s', '.*root', 0.0898112)
T6qqWW = [ T6qqWW_mSq950_mCh325_mChi300 ]

mcSamplesPriv = T5ttttDeg + T1ttbbWW + T1ttbb + T6ttWW + SqGltttt + T1tttt_priv + T5qqqqWW + T5qqqqWWDeg + T6qqWW + GJets_fixPhoton + QCDHT_fixPhoton

mcSamples = mcSamplesPHYS14_PU20bx25 + mcSamplesPHYS14_PU40bx25 + mcSamplesPHYS14_PU4bx50 + mcSamplesPriv

#-----------DATA---------------

#dataDir = os.environ['CMSSW_BASE']+"/src/CMGTools/TTHAnalysis/data"
dataDir = "$CMSSW_BASE/src/CMGTools/TTHAnalysis/data"  # use environmental variable, useful for instance to run on CRAB
#lumi: 12.21+7.27+0.134 = 19.62 /fb @ 8TeV

json=dataDir+'/json/Cert_Run2012ABCD_22Jan2013ReReco.json'

SingleMu = cfg.DataComponent(
    name = 'SingleMu',
    files = kreator.getFilesFromEOS('SingleMu', 
                                    '/SingleMu/Run2012D-15Apr2014-v1/AOD/02e0a1be-c9c7-11e3-bfe2-0024e83ef644/MINIAOD/CMSSW_7_0_9_patch2_GR_70_V2_AN1',
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
