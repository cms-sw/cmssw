import PhysicsTools.HeppyCore.framework.config as cfg
import os

#####COMPONENT CREATOR
from CMGTools.TTHAnalysis.samples.ComponentCreator import ComponentCreator
kreator = ComponentCreator()

## ==== 740 RelVals =====
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

## === 741 RelVals ===
ADD_MJ = kreator.makeMCComponent("ADD_MJ", "/RelValADDMonoJet_d3MD3_13/CMSSW_7_4_1-MCRUN2_74_V9_gensim_740pre7-v1/MINIAODSIM", "CMS", ".*root")
TTLep = kreator.makeMCComponent("TTLep", "/RelValTTbarLepton_13/CMSSW_7_4_1-MCRUN2_74_V9_gensim_740pre7-v1/MINIAODSIM", "CMS", ".*root")
TTbar = kreator.makeMCComponent("TTbar", "/RelValTTbar_13/CMSSW_7_4_1-MCRUN2_74_V9_gensim71X-v1/MINIAODSIM", "CMS", ".*root")
RSGravGaGa = kreator.makeMCComponent("RSGravGaGa", "/RelValRSGravitonToGaGa_13TeV/CMSSW_7_4_1-MCRUN2_74_V9_gensim71X-v1/MINIAODSIM", "CMS", ".*root")

RelVals741 = [ ADD_MJ, TTLep, TTbar, RSGravGaGa ] 


### 25 ns
### TTbar
TTJets = kreator.makeMCComponent("TTJets", "/TTJets_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 831.76, True)
TTJets_LO = kreator.makeMCComponent("TTJets_LO", "/TTJets_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 809.1)

### V+jets inclusive
WJetsToLNu = kreator.makeMCComponent("WJetsToLNu","/WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 20508.9)
DYJetsToLL_M50 = kreator.makeMCComponent("DYJetsToLL_M50", "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-AsymptFlat10to50bx25Raw_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 2008.*3)

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
QCD_Pt170to300 = kreator.makeMCComponent("QCD_Pt170to300","/QCD_Pt_170to300_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 117276)
QCD_Pt300to470 = kreator.makeMCComponent("QCD_Pt300to470","/QCD_Pt_300to470_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v1/MINIAODSIM", "CMS", ".*root", 7823)
QCD_Pt470to600 = kreator.makeMCComponent("QCD_Pt470to600","/QCD_Pt_470to600_TuneCUETP8M1_13TeV_pythia8/RunIISpring15DR74-Asympt25ns_MCRUN2_74_V9-v2/MINIAODSIM", "CMS", ".*root", 648.2)
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
QCD_Pt1000to1400,
QCD_Pt1400to1800,
QCD_Pt1800to2400,
QCD_Pt2400to3200,
QCD_Pt3200toInf
]


### 50 ns

### TTbar
TTJets_50ns = kreator.makeMCComponent("TTJets_50ns", "/TTJets_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 831.76, True)
TTJets_LO_50ns = kreator.makeMCComponent("TTJets_LO_50ns", "/TTJets_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v1/MINIAODSIM", "CMS", ".*root", 809.1)
DYJetsToLL_M50_50ns = kreator.makeMCComponent("DYJetsToLL_M50_50ns", "/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2/MINIAODSIM", "CMS", ".*root")

###

mcSamples_Asymptotic25ns = [TTJets, TTJets_LO, WJetsToLNu, DYJetsToLL_M50] + WJetsToLNuHT + QCDPt
mcSamples_Asymptotic50ns = [ TTJets_50ns, TTJets_LO_50ns, DYJetsToLL_M50_50ns ]

mcSamples = RelVals740 + RelVals741


#-----------DATA--------------- 
dataDir = "$CMSSW_BASE/src/CMGTools/MonoXAnalysis/data"
#json=dataDir+'/json/Cert_246908-247381_13TeV_PromptReco_Collisions15_ZeroTesla_JSON.txt'
json="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions15/13TeV/DCSOnly/json_DCSONLY.txt"

privEGamma2015Afiles = [ f.strip() for f in open("%s/src/CMGTools/MonoXAnalysis/python/samples/privEGamma_2015A_MINIAOD.txt"  % os.environ['CMSSW_BASE'], "r") ]
privDoubleEG2015Afiles = [ f.strip() for f in open("%s/src/CMGTools/MonoXAnalysis/python/samples/privDoubleEG_2015A_MINIAOD.txt"  % os.environ['CMSSW_BASE'], "r") ]
privHLTPhysics2015Afiles = [ f.strip() for f in open("%s/src/CMGTools/MonoXAnalysis/python/samples/privHLTPhysics_2015A_MINIAOD.txt"  % os.environ['CMSSW_BASE'], "r") ]
def _grep(x,l): return [ i for i in l if x in i ]
privEGamma2015A = kreator.makePrivateDataComponent('EGamma2015A', '/store/group/dpg_ecal/comm_ecal/data13TeV/EGamma/MINIAOD', _grep('2015A', privEGamma2015Afiles), json )
privDoubleEG2015A = kreator.makePrivateDataComponent('DoubleEG', '/store/group/dpg_ecal/comm_ecal/data13TeV/DoubleEG/MINIAOD', _grep('PAT', privDoubleEG2015Afiles), json )
privHLTPhysics2015A = kreator.makePrivateDataComponent('HLTPhysics2015A', '/store/group/dpg_ecal/comm_ecal/data13TeV/HLTPhysics/MINIAOD', _grep('HLTPhysics', privHLTPhysics2015Afiles), json )

privDataSamples = [ privEGamma2015A, privDoubleEG2015A, privHLTPhysics2015A ]

dataSamples = privDataSamples


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
       from CMGTools.TTHAnalysis.samples.ComponentCreator import testSamples
       testSamples(mcSamples)
