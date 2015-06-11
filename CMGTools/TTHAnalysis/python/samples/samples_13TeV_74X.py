import PhysicsTools.HeppyCore.framework.config as cfg
import os

#####COMPONENT CREATOR

from CMGTools.TTHAnalysis.samples.ComponentCreator import ComponentCreator
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

### 25 ns
### TTbar
TTJets = kreator.makeMCComponent("TTJets", "/TTJets_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 831.76, True)
TTJets_LO = kreator.makeMCComponent("TTJets_LO", "/TTJets_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 809.1)

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


### QCD
QCD_Pt80to120 = kreator.makeMCComponent("QCD_Pt80to120","/QCD_Pt_80to120_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 2762530)
QCD_Pt120to170 = kreator.makeMCComponent("QCD_Pt120to170","/QCD_Pt_120to170_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 471100)
QCD_Pt300to470 = kreator.makeMCComponent("QCD_Pt300to470","/QCD_Pt_300to470_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 7823)
QCD_Pt1000to1400 = kreator.makeMCComponent("QCD_Pt1000to1400","/QCD_Pt_1000to1400_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 9.4183)
QCD_Pt1400to1800 = kreator.makeMCComponent("QCD_Pt1400to1800","/QCD_Pt_1400to1800_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 0.84265)
QCD_Pt1800to2400 = kreator.makeMCComponent("QCD_Pt1800to2400","/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 0.114943)
QCD_Pt2400to3200 = kreator.makeMCComponent("QCD_Pt2400to3200","/QCD_Pt_2400to3200_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 0.00682981)
QCD_Pt3200toInf = kreator.makeMCComponent("QCD_Pt3200","/QCD_Pt_3200toInf_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 0.000165445)

QCDPt = [
QCD_Pt80to120,
QCD_Pt120to170,
QCD_Pt300to470,
QCD_Pt1000to1400,
QCD_Pt1400to1800,
QCD_Pt1800to2400,
QCD_Pt2400to3200,
QCD_Pt3200toInf
]

### 50 ns

###

mcSamples = RelVals740

mcSamples_Asymptotic25ns = [TTJets, TTJets_LO] + WJetsToLNuHT + QCDPt

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

if __name__ == "__main__":
   import sys
   if "test" in sys.argv:
       from CMGTools.TTHAnalysis.samples.ComponentCreator import testSamples
       testSamples(mcSamples)
