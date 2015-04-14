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

ZEE_bx50 = kreator.makeMCComponent("ZEE_50", "/RelValZEE_13/CMSSW_7_4_0-PU50ns_MCRUN2_74_V6_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root")
ZEE_bx25 = kreator.makeMCComponent("ZEE_25", "/RelValZEE_13/CMSSW_7_4_0-PU25ns_MCRUN2_74_V7_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root")
ZMM_bx25 = kreator.makeMCComponent("ZMM_25", "/RelValZMM_13/CMSSW_7_4_0-PU25ns_MCRUN2_74_V7_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root")
ZMM_bx50 = kreator.makeMCComponent("ZMM_50", "/RelValZMM_13/CMSSW_7_4_0-PU50ns_MCRUN2_74_V6_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root")
ZTT_bx25 = kreator.makeMCComponent("ZTT_25", "/RelValZTT_13/CMSSW_7_4_0-PU25ns_MCRUN2_74_V7_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root")
ZTT_bx50 = kreator.makeMCComponent("ZTT_50", "/RelValZTT_13/CMSSW_7_4_0-PU50ns_MCRUN2_74_V6_gs7115_puProd-v1/MINIAODSIM", "CMS", ".*root")

RelVals740 = [ TT_NoPU, TT_bx25, TT_bx50, TTLep_NoPU, ZEE_bx50, ZEE_bx25, ZMM_bx25, ZMM_bx50, ZTT_bx25, ZTT_bx50 ]

mcSamples = RelVals740

#-----------DATA---------------

dataDir = os.environ['CMSSW_BASE']+"/src/CMGTools/TTHAnalysis/data"
#lumi: 12.21+7.27+0.134 = 19.62 /fb @ 8TeV

json=dataDir+'/json/Cert_Run2012ABCD_22Jan2013ReReco.json'

SingleMu = cfg.DataComponent(
    name = 'SingleMu',
    files = kreator.getFiles('/SingleMu/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_sm2012D-v10/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
SingleMuZ = cfg.DataComponent(
    name = 'SingleMuZ',
    files = kreator.getFiles('/SingleMu/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_zMu2012D-v2/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
DoubleMu = cfg.DataComponent(
    name = 'DoubleMu',
    files = kreator.getFiles('/DoubleMuParked/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_dm2012D-v2/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
DoubleElectronZ = cfg.DataComponent(
    name = 'DoubleElectronZ',
    files = kreator.getFiles('/DoubleElectron/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_zEl2012D-v1/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
MuEG = cfg.DataComponent(
      name = 'MuEG',
      files = kreator.getFiles('/MuEG/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_meg2012D-v5/MINIAOD', 'CMS', '.*root'),
      intLumi = 1, triggers = [], json = json
      )
JetHT = cfg.DataComponent(
        name = 'JetHT',
        files = kreator.getFiles('/JetHT/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_jht2012D-v1/MINIAOD', 'CMS', '.*root'),
        intLumi = 1, triggers = [], json = json
        )
ZeroBias = cfg.DataComponent(
          name = 'ZeroBias',
          files = kreator.getFiles('/ZeroBias/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_zb2012C-v5/MINIAOD', 'CMS', '.*root'),
          intLumi = 1, triggers = [], json = json
          )


dataSamplesMu=[DoubleMu]
dataSamplesE=[DoubleElectronZ]
dataSamplesMuE=[MuEG]
dataSamples1Mu=[SingleMu,SingleMuZ]
dataSamplesJet=[JetHT]
dataSamplesOther=[ZeroBias]
dataSamplesAll = dataSamplesMu+dataSamplesE+dataSamplesMuE+dataSamples1Mu+dataSamplesJet+dataSamplesOther
dataSamples740p9 = dataSamplesAll

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
       from CMGTools.TTHAnalysis.samples.ComponentCreator import testSamples
       testSamples(mcSamples)
