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

mcSamples = RelVals740

from CMGTools.TTHAnalysis.setup.Efficiencies import *


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
