import PhysicsTools.HeppyCore.framework.config as cfg
import os

#####COMPONENT CREATOR

from CMGTools.RootTools.samples.ComponentCreator import ComponentCreator
kreator = ComponentCreator()

## ==== RelVals =====
TT_NoPU = kreator.makeMCComponent("TT_NoPU", "/RelValTTbar_13/CMSSW_7_4_0-MCRUN2_74_V7_GENSIM_7_1_15-v1/MINIAODSIM", "CMS", ".*root",809.1)
TT_bx25 = kreator.makeMCComponent("TT_bx25", "/RelValTTbar_13/CMSSW_7_4_0-PU25ns_MCRUN2_74_V7_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root",809.1)
TT_bx50 = kreator.makeMCComponent("TT_bx50", "/RelValTTbar_13/CMSSW_7_4_0-PU50ns_MCRUN2_74_V6_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root",809.1)

TTLep_NoPU = kreator.makeMCComponent("TTLep_NoPU", "/RelValTTbarLepton_13/CMSSW_7_4_0-MCRUN2_74_V7_GENSIM_7_1_15-v1/MINIAODSIM", "CMS", ".*root",809.1)

ZEE_bx50 = kreator.makeMCComponent("ZEE_bx50", "/RelValZEE_13/CMSSW_7_4_0-PU50ns_MCRUN2_74_V6_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root")
ZEE_bx25 = kreator.makeMCComponent("ZEE_bx25", "/RelValZEE_13/CMSSW_7_4_0-PU25ns_MCRUN2_74_V7_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root")
ZMM_bx25 = kreator.makeMCComponent("ZMM_bx25", "/RelValZMM_13/CMSSW_7_4_0-PU25ns_MCRUN2_74_V7_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root")
ZMM_bx50 = kreator.makeMCComponent("ZMM_bx50", "/RelValZMM_13/CMSSW_7_4_0-PU50ns_MCRUN2_74_V6_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root")
ZTT_bx25 = kreator.makeMCComponent("ZTT_bx25", "/RelValZTT_13/CMSSW_7_4_0-PU25ns_MCRUN2_74_V7_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root")
ZTT_bx50 = kreator.makeMCComponent("ZTT_bx50", "/RelValZTT_13/CMSSW_7_4_0-PU50ns_MCRUN2_74_V6_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root")

RelVals740 = [ TT_NoPU, TT_bx25, TT_bx50, TTLep_NoPU, ZEE_bx50, ZEE_bx25, ZMM_bx25, ZMM_bx50, ZTT_bx25, ZTT_bx50 ]

### ----------------------------- 25 ns ----------------------------------------
# TTbar cross section: NNLO, https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO (172.5)
TTJets = kreator.makeMCComponent("TTJets", "/TTJets_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 831.76)
TTJets_LO = kreator.makeMCComponent("TTJets_LO", "/TTJets_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 831.76)
TT_pow = kreator.makeMCComponent("TTLep_pow", "/TT_TuneCUETP8M1_13TeV-powheg-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 831.76)
TTLep_pow = kreator.makeMCComponent("TTLep_pow", "/TTTo2L2Nu_13TeV-powheg/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 831.76*((3*0.108)**2))
TTs = [ TTJets, TTJets_LO, TT_pow, TTLep_pow ]

# Single top cross sections: https://twiki.cern.ch/twiki/bin/viewauth/CMS/SingleTopSigma
TToLeptons_tch = kreator.makeMCComponent("TToLeptons_tch", "/ST_t-channel_4f_leptonDecays_13TeV-amcatnlo-pythia8_TuneCUETP8M1/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", (136.05+80.97)*0.108*3) 
TToLeptons_sch = kreator.makeMCComponent("TToLeptons_sch", "/ST_s-channel_4f_leptonDecays_13TeV-amcatnlo-pythia8_TuneCUETP8M1/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", (7.20+4.16)*0.108*3)
TBar_tWch = kreator.makeMCComponent("TBar_tWch", "/ST_tW_antitop_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root",35.6)
T_tWch = kreator.makeMCComponent("T_tWch", "/ST_tW_top_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root",35.6)

SingleTop = [
    TToLeptons_tch, TToLeptons_sch, TBar_tWch, T_tWch
]

### V+jets inclusive (from https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV)
WJetsToLNu = kreator.makeMCComponent("WJetsToLNu","/WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 3* 20508.9)
DYJetsToLL_M50 = kreator.makeMCComponent("DYJetsToLL_M50", "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v3/MINIAODSIM", "CMS", ".*root", 2008.*3)
DYJetsToLL_M50_PUflat1050 = kreator.makeMCComponent("DYJetsToLL_M50_PUflat1050", "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-AsymptFlat10to50bx25Raw_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 2008.*3)
## From McM
DYJetsToLL_M10to50 = kreator.makeMCComponent("DYJetsToLL_M10to50", "/DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 18610)

VJets = [ WJetsToLNu, DYJetsToLL_M50, DYJetsToLL_M10to50, DYJetsToLL_M50_PUflat1050 ]

### W+jets
WJetsToLNu_HT100to200 = kreator.makeMCComponent("WJetsToLNu_HT100to200", "/WJetsToLNu_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root",1292*1.23)
WJetsToLNu_HT200to400 = kreator.makeMCComponent("WJetsToLNu_HT200to400", "/WJetsToLNu_HT-200To400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root",385.9*1.23)
WJetsToLNu_HT400to600 = kreator.makeMCComponent("WJetsToLNu_HT400to600", "/WJetsToLNu_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v3/MINIAODSIM", "CMS", ".*root",47.9*1.23)
WJetsToLNu_HT600toInf = kreator.makeMCComponent("WJetsToLNu_HT600toInf", "/WJetsToLNu_HT-600ToInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root",19.9*1.23)
WJetsToLNuHT = [
WJetsToLNu_HT100to200,
WJetsToLNu_HT200to400,
WJetsToLNu_HT400to600,
WJetsToLNu_HT600toInf,
]

### GJets
GJets_HT400to600 = kreator.makeMCComponent("GJets_HT400to600", "/GJets_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root",62.05)
GJets_HT600toInf = kreator.makeMCComponent("GJets_HT600toInf", "/GJets_HT-600ToInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root",20.87)
GJetsHT = [
GJets_HT400to600,
GJets_HT600toInf,
]

### QCD
QCD_Pt80to120 = kreator.makeMCComponent("QCD_Pt80to120","/QCD_Pt_80to120_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 2762530)
QCD_Pt120to170 = kreator.makeMCComponent("QCD_Pt120to170","/QCD_Pt_120to170_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 471100)
QCD_Pt170to300 = kreator.makeMCComponent("QCD_Pt170to300","/QCD_Pt_170to300_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 117276)
QCD_Pt300to470 = kreator.makeMCComponent("QCD_Pt300to470","/QCD_Pt_300to470_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 7823)
QCD_Pt470to600 = kreator.makeMCComponent("QCD_Pt470to600","/QCD_Pt_470to600_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 648.2)
QCD_Pt800to1000 = kreator.makeMCComponent("QCD_Pt800to1000","/QCD_Pt_800to1000_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 32.293)
QCD_Pt1000to1400 = kreator.makeMCComponent("QCD_Pt1000to1400","/QCD_Pt_1000to1400_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 9.4183)
QCD_Pt1400to1800 = kreator.makeMCComponent("QCD_Pt1400to1800","/QCD_Pt_1400to1800_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 0.84265)
QCD_Pt1800to2400 = kreator.makeMCComponent("QCD_Pt1800to2400","/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 0.114943)
QCD_Pt2400to3200 = kreator.makeMCComponent("QCD_Pt2400to3200","/QCD_Pt_2400to3200_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 0.00682981)
QCD_Pt3200toInf = kreator.makeMCComponent("QCD_Pt3200","/QCD_Pt_3200toInf_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 0.000165445)

QCDPt = [
QCD_Pt80to120,
QCD_Pt120to170,
QCD_Pt170to300,
QCD_Pt300to470,
QCD_Pt470to600,
QCD_Pt800to1000,
QCD_Pt1000to1400,
QCD_Pt1400to1800,
QCD_Pt1800to2400,
QCD_Pt2400to3200,
QCD_Pt3200toInf
]

# Muon-enriched QCD (cross sections and filter efficiency from McM)
QCD_Mu15 = kreator.makeMCComponent("QCD_Mu15", "/QCD_Pt-20toInf_MuEnrichedPt15_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 720.65e6*0.00042)
QCD_Pt15to20_Mu5 = kreator.makeMCComponent("QCD_Pt15to20_Mu5", "/QCD_Pt-15to20_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 1273190000*0.003)
QCD_Pt20to30_Mu5 = kreator.makeMCComponent("QCD_Pt20to30_Mu5", "/QCD_Pt-20to30_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 558528000*0.0053)
QCD_Pt30to50_Mu5 = kreator.makeMCComponent("QCD_Pt30to50_Mu5", "/QCD_Pt-30to50_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 139803000*0.01182)
QCD_Pt50to80_Mu5 = kreator.makeMCComponent("QCD_Pt50to80_Mu5", "/QCD_Pt-50to80_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 19222500*0.02276)
QCD_Pt80to120_Mu5 = kreator.makeMCComponent("QCD_Pt80to120_Mu5", "/QCD_Pt-80to120_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 2758420*0.03844) 
QCD_Pt120to170_Mu5 = kreator.makeMCComponent("QCD_Pt120to170_Mu5", "/QCD_Pt-120to170_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 469797*0.05362)
QCD_Pt170to300_Mu5 = kreator.makeMCComponent("QCD_Pt170to300_Mu5", "/QCD_Pt-170to300_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 117989*0.07335)
QCD_Pt300to470_Mu5 = kreator.makeMCComponent("QCD_Pt300to470_Mu5", "/QCD_Pt-300to470_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 7820.25*0.10196)
QCD_Pt470to600_Mu5 = kreator.makeMCComponent("QCD_Pt470to600_Mu5", "/QCD_Pt-470to600_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 645.528*0.12242)
QCD_Pt600to800_Mu5 = kreator.makeMCComponent("QCD_Pt600to800_Mu5", "/QCD_Pt-600to800_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 187.109*0.13412)
QCD_Pt800to1000_Mu5 = kreator.makeMCComponent("QCD_Pt800to1000_Mu5", "/QCD_Pt-800to1000_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 32.3486*0.14552)
QCD_Pt1000toInf_Mu5 = kreator.makeMCComponent("QCD_Pt1000toInf_Mu5", "/QCD_Pt-1000toInf_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 10.4305*0.15544)
QCD_Mu5 = [ QCD_Pt15to20_Mu5, QCD_Pt20to30_Mu5, QCD_Pt30to50_Mu5, QCD_Pt50to80_Mu5, 
            QCD_Pt80to120_Mu5, QCD_Pt120to170_Mu5, QCD_Pt170to300_Mu5, QCD_Pt300to470_Mu5, 
            QCD_Pt470to600_Mu5, QCD_Pt600to800_Mu5, QCD_Pt800to1000_Mu5, QCD_Pt1000toInf_Mu5 ]
QCD_MuX = [ QCD_Mu15 ] + QCD_Mu5

# QCD EM-enriched + bcToE, 25 ns
QCD_Pt15to20_EMEnriched   = kreator.makeMCComponent("QCD_Pt15to20_EMEnriched","/QCD_Pt-15to20_EMEnriched_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 1273000000*0.0002)
QCD_Pt20to30_EMEnriched   = kreator.makeMCComponent("QCD_Pt20to30_EMEnriched","/QCD_Pt-20to30_EMEnriched_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 557600000*0.0096)
QCD_Pt30to50_EMEnriched   = kreator.makeMCComponent("QCD_Pt30to50_EMEnriched","/QCD_Pt-30to50_EMEnriched_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 136000000*0.073)
QCD_Pt50to80_EMEnriched   = kreator.makeMCComponent("QCD_Pt50to80_EMEnriched","/QCD_Pt-50to80_EMEnriched_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 19800000*0.146)
QCD_Pt80to120_EMEnriched  = kreator.makeMCComponent("QCD_Pt80to120_EMEnriched","/QCD_Pt-80to120_EMEnriched_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v3/MINIAODSIM", "CMS", ".*root", 2800000*0.125)
QCD_Pt120to170_EMEnriched = kreator.makeMCComponent("QCD_Pt120to170_EMEnriched","/QCD_Pt-120to170_EMEnriched_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 477000*0.132)
QCD_Pt170to300_EMEnriched = kreator.makeMCComponent("QCD_Pt170to300_EMEnriched","/QCD_Pt-170to300_EMEnriched_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 114000*0.165)
QCD_Pt300toInf_EMEnriched = kreator.makeMCComponent("QCD_Pt300toInf_EMEnriched","/QCD_Pt-300toInf_EMEnriched_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 9000*0.15)
QCDPtEMEnriched = [
QCD_Pt15to20_EMEnriched,
QCD_Pt20to30_EMEnriched,
QCD_Pt30to50_EMEnriched,
QCD_Pt50to80_EMEnriched,
QCD_Pt80to120_EMEnriched,
QCD_Pt120to170_EMEnriched,
QCD_Pt170to300_EMEnriched,
QCD_Pt300toInf_EMEnriched
]
QCD_Pt_15to20_bcToE   = kreator.makeMCComponent("QCD_Pt_15to20_bcToE","/QCD_Pt_15to20_bcToE_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 1272980000*0.0002)
QCD_Pt_20to30_bcToE   = kreator.makeMCComponent("QCD_Pt_20to30_bcToE","/QCD_Pt_20to30_bcToE_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 557627000*0.00059)
QCD_Pt_30to80_bcToE   = kreator.makeMCComponent("QCD_Pt_30to80_bcToE","/QCD_Pt_30to80_bcToE_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 159068000*0.00255)
QCD_Pt_80to170_bcToE  = kreator.makeMCComponent("QCD_Pt_80to170_bcToE","/QCD_Pt_80to170_bcToE_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 3221000*0.01183)
QCD_Pt_170to250_bcToE = kreator.makeMCComponent("QCD_Pt_170to250_bcToE","/QCD_Pt_170to250_bcToE_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 105771*0.02492)
QCD_Pt_250toInf_bcToE = kreator.makeMCComponent("QCD_Pt_250toInf_bcToE","/QCD_Pt_250toInf_bcToE_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 21094.1*0.03375)
QCDPtbcToE = [
QCD_Pt_15to20_bcToE,
QCD_Pt_20to30_bcToE,
QCD_Pt_30to80_bcToE,
QCD_Pt_80to170_bcToE,
QCD_Pt_170to250_bcToE,
QCD_Pt_250toInf_bcToE
]
QCD_ElX = QCDPtEMEnriched + QCDPtbcToE

# cross section from StandardModelCrossSectionsat13TeV NNLO times BR=(3*0.108)**2
WWTo2L2Nu = kreator.makeMCComponent("WWTo2L2Nu", "/WWTo2L2Nu_13TeV-powheg/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 118.7*((3*0.108)**2) )

# cross section from StandardModelCrossSectionsat13TeV (NLO MCFM, mll > 12); to be checked if it's really m(ll) > 12 also for Pythia sample
ZZp8 = kreator.makeMCComponent("ZZp8", "/ZZ_TuneCUETP8M1_13TeV-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v3/MINIAODSIM", "CMS", ".*root", 31.8)

# cros section from StandardModelCrossSectionsat13TeV (NLO MCFM, mll > 12); to be checked if it's really m(ll) > 12 also for Pythia sample
WZp8 = kreator.makeMCComponent("WZp8", "/WZ_TuneCUETP8M1_13TeV-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", (40.2+25.9))

DiBosons = [ WWTo2L2Nu, ZZp8, WZp8 ] 

# Higgs
# TTH cross section from LHC Higgs XS WG: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt1314TeV?rev=15
TTHnobb = kreator.makeMCComponent("TTHnobb", "/ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root",0.5085*(1-0.577))

Higgs = [ TTHnobb ]

### ==============  50 ns ========================
TTJets_50ns = kreator.makeMCComponent("TTJets_50ns", "/TTJets_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 831.76,)
TTJets_LO_50ns = kreator.makeMCComponent("TTJets_LO_50ns", "/TTJets_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 831.76)

### V+jets inclusive
DYJetsToLL_M50_50ns = kreator.makeMCComponent("DYJetsToLL_M50_50ns","/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 2008.*3)

DYJetsToLL_M10to50_50ns = kreator.makeMCComponent("DYJetsToLL_M10to50_50ns","/DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 18610)

WJetsToLNu_50ns = kreator.makeMCComponent("WJetsToLNu_50ns","/WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 20508.9*3)

# Single top cross sections: https://twiki.cern.ch/twiki/bin/viewauth/CMS/SingleTopSigma
TToLeptons_tch_50ns = kreator.makeMCComponent("TToLeptons_tch_50ns", "/ST_t-channel_4f_leptonDecays_13TeV-amcatnlo-pythia8_TuneCUETP8M1/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", (136.05+80.97)*0.108*3) 
TBar_tWch_50ns = kreator.makeMCComponent("TBar_tWch_50ns", "/ST_tW_antitop_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root",35.6)
T_tWch_50ns = kreator.makeMCComponent("T_tWch_50ns", "/ST_tW_top_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root",35.6)

SingleTop_50ns = [
    TToLeptons_tch_50ns, TBar_tWch_50ns, T_tWch_50ns
]

# cross section from StandardModelCrossSectionsat13TeV NNLO times BR=(3*0.108)**2
WWTo2L2Nu_50ns = kreator.makeMCComponent("WWTo2L2Nu_50ns", "/WWTo2L2Nu_13TeV-powheg/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 118.7*((3*0.108)**2) )

# cross section from StandardModelCrossSectionsat13TeV (NLO MCFM, mll > 12); to be checked if it's really m(ll) > 12 also for Pythia sample
ZZp8_50ns = kreator.makeMCComponent("ZZp8_50ns", "/ZZ_TuneCUETP8M1_13TeV-pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 31.8)

# cros section from StandardModelCrossSectionsat13TeV (NLO MCFM, mll > 12); to be checked if it's really m(ll) > 12 also for Pythia sample
WZp8_50ns = kreator.makeMCComponent("WZp8_50ns", "/WZ_TuneCUETP8M1_13TeV-pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", (40.2+25.9))

DiBosons_50ns = [ WWTo2L2Nu_50ns, ZZp8_50ns, WZp8_50ns ] 


### QCD, 50 ns
QCD_Pt10to15_50ns = kreator.makeMCComponent("QCD_Pt10to15_50ns","/QCD_Pt_10to15_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 5887580000)
QCD_Pt15to30_50ns = kreator.makeMCComponent("QCD_Pt15to30_50ns","/QCD_Pt_15to30_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 1837410000)
QCD_Pt30to50_50ns = kreator.makeMCComponent("QCD_Pt30to50_50ns","/QCD_Pt_30to50_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 140932000)
QCD_Pt50to80_50ns = kreator.makeMCComponent("QCD_Pt50to80_50ns","/QCD_Pt_50to80_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 19204300)
QCD_Pt80to120_50ns = kreator.makeMCComponent("QCD_Pt80to120_50ns","/QCD_Pt_80to120_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 2762530)
QCD_Pt120to170_50ns = kreator.makeMCComponent("QCD_Pt120to170_50ns","/QCD_Pt_120to170_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 471100)
QCD_Pt170to300_50ns = kreator.makeMCComponent("QCD_Pt170to300_50ns","/QCD_Pt_170to300_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 117276)
QCD_Pt300to470_50ns = kreator.makeMCComponent("QCD_Pt300to470_50ns","/QCD_Pt_300to470_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 7823)
QCD_Pt470to600_50ns = kreator.makeMCComponent("QCD_Pt470to600_50ns","/QCD_Pt_470to600_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 648.2)
QCD_Pt600to800_50ns = kreator.makeMCComponent("QCD_Pt600to800_50ns","/QCD_Pt_600to800_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 186.9)
QCD_Pt800to1000_50ns = kreator.makeMCComponent("QCD_Pt800to1000_50ns","/QCD_Pt_800to1000_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 32.293)
QCD_Pt1000to1400_50ns = kreator.makeMCComponent("QCD_Pt1000to1400_50ns","/QCD_Pt_1000to1400_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 9.4183)
QCD_Pt1400to1800_50ns = kreator.makeMCComponent("QCD_Pt1400to1800_50ns","/QCD_Pt_1400to1800_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 0.84265)
QCD_Pt1800to2400_50ns = kreator.makeMCComponent("QCD_Pt1800to2400_50ns","/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 0.114943)
QCD_Pt2400to3200_50ns = kreator.makeMCComponent("QCD_Pt2400to3200_50ns","/QCD_Pt_2400to3200_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 0.00682981)
QCD_Pt3200toInf_50ns = kreator.makeMCComponent("QCD_Pt3200_50ns","/QCD_Pt_3200toInf_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 0.000165445)

QCDPt_50ns = [
QCD_Pt10to15_50ns,
QCD_Pt15to30_50ns,
QCD_Pt30to50_50ns,
QCD_Pt50to80_50ns,
QCD_Pt80to120_50ns,
QCD_Pt120to170_50ns,
QCD_Pt170to300_50ns,
QCD_Pt300to470_50ns,
QCD_Pt470to600_50ns,
QCD_Pt600to800_50ns,
QCD_Pt800to1000_50ns,
QCD_Pt1000to1400_50ns,
QCD_Pt1400to1800_50ns,
QCD_Pt1800to2400_50ns,
QCD_Pt2400to3200_50ns,
QCD_Pt3200toInf_50ns
]

# Muon-enriched QCD, 50 ns
QCD_Mu15_50ns = kreator.makeMCComponent("QCD_Mu15_50ns", "/QCD_Pt-20toInf_MuEnrichedPt15_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 720.65e6*0.00042)
QCD_Pt15to20_Mu5_50ns = kreator.makeMCComponent("QCD_Pt15to20_Mu5_50ns", "/QCD_Pt-15to20_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v3/MINIAODSIM", "CMS", ".*root", 1273190000*0.003)
QCD_Pt20to30_Mu5_50ns = kreator.makeMCComponent("QCD_Pt20to30_Mu5_50ns", "/QCD_Pt-20to30_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 558528000*0.0053)
QCD_Pt30to50_Mu5_50ns = kreator.makeMCComponent("QCD_Pt30to50_Mu5_50ns", "/QCD_Pt-30to50_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 139803000*0.01182)
QCD_Pt50to80_Mu5_50ns = kreator.makeMCComponent("QCD_Pt50to80_Mu5_50ns", "/QCD_Pt-50to80_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 19222500*0.02276)
QCD_Pt80to120_Mu5_50ns = kreator.makeMCComponent("QCD_Pt80to120_Mu5_50ns", "/QCD_Pt-80to120_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 2758420*0.03844) 
QCD_Pt120to170_Mu5_50ns = kreator.makeMCComponent("QCD_Pt120to170_Mu5_50ns", "/QCD_Pt-120to170_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 469797*0.05362)
QCD_Pt170to300_Mu5_50ns = kreator.makeMCComponent("QCD_Pt170to300_Mu5_50ns", "/QCD_Pt-170to300_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 117989*0.07335)
QCD_Pt300to470_Mu5_50ns = kreator.makeMCComponent("QCD_Pt300to470_Mu5_50ns", "/QCD_Pt-300to470_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v3/MINIAODSIM", "CMS", ".*root", 7820.25*0.10196)
QCD_Pt470to600_Mu5_50ns = kreator.makeMCComponent("QCD_Pt470to600_Mu5_50ns", "/QCD_Pt-470to600_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v3/MINIAODSIM", "CMS", ".*root", 645.528*0.12242)
QCD_Pt600to800_Mu5_50ns = kreator.makeMCComponent("QCD_Pt600to800_Mu5_50ns", "/QCD_Pt-600to800_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 187.109*0.13412)
QCD_Pt800to1000_Mu5_50ns = kreator.makeMCComponent("QCD_Pt800to1000_Mu5_50ns", "/QCD_Pt-800to1000_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 32.3486*0.14552)
QCD_Pt1000toInf_Mu5_50ns = kreator.makeMCComponent("QCD_Pt1000toInf_Mu5_50ns", "/QCD_Pt-1000toInf_MuEnrichedPt5_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root", 10.4305*0.15544)
QCD_Mu5_50ns = [ QCD_Pt15to20_Mu5_50ns, QCD_Pt20to30_Mu5_50ns, QCD_Pt30to50_Mu5_50ns, QCD_Pt50to80_Mu5_50ns, 
            QCD_Pt80to120_Mu5_50ns, QCD_Pt120to170_Mu5_50ns, QCD_Pt170to300_Mu5_50ns, QCD_Pt300to470_Mu5_50ns, 
            QCD_Pt470to600_Mu5_50ns, QCD_Pt600to800_Mu5_50ns, QCD_Pt800to1000_Mu5_50ns, QCD_Pt1000toInf_Mu5_50ns ]
QCD_MuX_50ns = [ QCD_Mu15_50ns ] + QCD_Mu5_50ns





### ----------------------------- Zero Tesla run  ----------------------------------------

dataDir = "$CMSSW_BASE/src/CMGTools/TTHAnalysis/data"  # use environmental variable, useful for instance to run on CRAB
json=dataDir+'/json/Cert_246908-248005_13TeV_PromptReco_Collisions15_ZeroTesla_JSON.txt'
#lumi: delivered= 4.430 (/nb) recorded= 4.013 (/nb)

jetHT_0T = cfg.DataComponent(
    name = 'jetHT_0T',
    files = kreator.getFilesFromEOS('jetHT_0T',
                                    'firstData_JetHT_v2',
                                    '/store/user/pandolf/MINIAOD/%s'),
    intLumi = 4.0,
    triggers = [],
    json = None #json
    )

### ----------------------------- summary ----------------------------------------

mcSamples_Asymptotic25ns = TTs + SingleTop + VJets + WJetsToLNuHT + GJetsHT + QCDPt + DiBosons + Higgs + QCD_ElX

mcSamples_Asymptotic50ns = [ TTJets_50ns, TTJets_LO_50ns, WJetsToLNu_50ns, DYJetsToLL_M10to50_50ns, DYJetsToLL_M50_50ns ] + QCDPt_50ns + SingleTop_50ns + DiBosons_50ns + QCD_MuX_50ns

mcSamples = RelVals740 + mcSamples_Asymptotic25ns + mcSamples_Asymptotic50ns

dataSamples = [jetHT_0T]

samples = mcSamples + dataSamples

### ---------------------------------------------------------------------

from CMGTools.TTHAnalysis.setup.Efficiencies import *
dataDir = "$CMSSW_BASE/src/CMGTools/TTHAnalysis/data"

#Define splitting
for comp in mcSamples:
    comp.isMC = True
    comp.isData = False
    comp.splitFactor = 250 #  if comp.name in [ "WJets", "DY3JetsM50", "DY4JetsM50","W1Jets","W2Jets","W3Jets","W4Jets","TTJetsHad" ] else 100
    comp.puFileMC=dataDir+"/puProfile_Summer12_53X.root"
    comp.puFileData=dataDir+"/puProfile_Data12.root"
    comp.efficiency = eff2012

for comp in dataSamples:
    comp.splitFactor = 1000
    comp.isMC = False
    comp.isData = True

if __name__ == "__main__":
   import sys
   if "test" in sys.argv:
       from CMGTools.RootTools.samples.ComponentCreator import testSamples
       testSamples(samples)
