import PhysicsTools.HeppyCore.framework.config as cfg
import os

#####COMPONENT CREATOR

from CMGTools.RootTools.samples.ComponentCreator import ComponentCreator
kreator = ComponentCreator()

#-----------DATA---------------

dataDir = "$CMSSW_BASE/src/CMGTools/TTHAnalysis/data"
#lumi: 12.21+7.27+0.134 = 19.62 /fb @ 8TeV

json=dataDir+'/json/Cert_Run2012ABCD_22Jan2013ReReco.json'

#------------- 740p9

SingleMu_740p9 = cfg.DataComponent(
    name = 'SingleMu_740p9',
    files = kreator.getFiles('/SingleMu/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_sm2012D-v10/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
SingleMuZ_740p9 = cfg.DataComponent(
    name = 'SingleMuZ_740p9',
    files = kreator.getFiles('/SingleMu/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_zMu2012D-v2/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
DoubleMu_740p9 = cfg.DataComponent(
    name = 'DoubleMu_740p9',
    files = kreator.getFiles('/DoubleMuParked/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_dm2012D-v2/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
DoubleElectronZ_740p9 = cfg.DataComponent(
    name = 'DoubleElectronZ',
    files = kreator.getFiles('/DoubleElectron/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_zEl2012D-v1/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
MuEG_740p9 = cfg.DataComponent(
    name = 'MuEG_740p9',
    files = kreator.getFiles('/MuEG/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_meg2012D-v5/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
JetHT_740p9 = cfg.DataComponent(
    name = 'JetHT_740p9',
    files = kreator.getFiles('/JetHT/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_jht2012D-v1/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
ZeroBias_740p9 = cfg.DataComponent(
    name = 'ZeroBias_740p9',
    files = kreator.getFiles('/ZeroBias/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_zb2012C-v5/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )


dataSamplesMu_740p9=[DoubleMu_740p9]
dataSamplesE_740p9=[DoubleElectronZ_740p9]
dataSamplesMuE_740p9=[MuEG_740p9]
dataSamples1Mu_740p9=[SingleMu_740p9,SingleMuZ_740p9]
dataSamplesJet_740p9=[JetHT_740p9]
dataSamplesOther_740p9=[ZeroBias_740p9]
dataSamples740p9 = dataSamplesMu_740p9+dataSamplesE_740p9+dataSamplesMuE_740p9+dataSamples1Mu_740p9+dataSamplesJet_740p9+dataSamplesOther_740p9 



#------------- 742 ----------------------------------------------

SingleMu_742 = cfg.DataComponent(
    name = 'SingleMu_742',
    files = kreator.getFiles('/SingleMu/CMSSW_7_4_2-GR_R_74_V12_19May_RelVal_sm2012D-v1/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
SingleMuZ_742 = cfg.DataComponent(
    name = 'SingleMuZ_742',
    files = kreator.getFiles('/SingleMu/CMSSW_7_4_2-GR_R_74_V12_19May_RelVal_zMu2012D-v1/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
DoubleMu_742 = cfg.DataComponent(
    name = 'DoubleMu_742',
    files = kreator.getFiles('/DoubleMuParked/CMSSW_7_4_2-GR_R_74_V12_19May_RelVal_dm2012D-v1/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
DoubleElectronZ_742 = cfg.DataComponent(
    name = 'DoubleElectronZ_742',
    files = kreator.getFiles('/DoubleElectron/CMSSW_7_4_2-GR_R_74_V12_19May_RelVal_zEl2012D-v1/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
MuEG_742 = cfg.DataComponent(
    name = 'MuEG_742',
    files = kreator.getFiles('/MuEG/CMSSW_7_4_2-GR_R_74_V12_19May_RelVal_meg2012D-v1/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
JetHT_742 = cfg.DataComponent(
    name = 'JetHT_742',
    files = kreator.getFiles('/JetHT/CMSSW_7_4_2-GR_R_74_V12_19May_RelVal_jht2012D-v1/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )

JetHT25_742 = cfg.DataComponent(
    name = 'JetHT25_742',
    files = kreator.getFiles('/JetHT25ns/CMSSW_7_4_2-GR_R_74_V12_19May_RelVal_jht25ns2012D-v1/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )
ZeroBias_742 = cfg.DataComponent(
    name = 'ZeroBias_742',
    files = kreator.getFiles('/ZeroBias/CMSSW_7_4_2-GR_R_74_V12_19May_RelVal_zb2012C-v5/MINIAOD', 'CMS', '.*root'),
    intLumi = 1, triggers = [], json = json
    )


dataSamplesMu_742=[DoubleMu_742]
dataSamplesE_742=[DoubleElectronZ_742]
dataSamplesMuE_742=[MuEG_742]
dataSamples1Mu_742=[SingleMu_742,SingleMuZ_742]
dataSamplesJet_742=[JetHT_742,JetHT25_742]
dataSamplesOther_742=[ZeroBias_742]
dataSamples742 = dataSamplesMu_742+dataSamplesE_742+dataSamplesMuE_742+dataSamples1Mu_742+dataSamplesJet_742+dataSamplesOther_742

#-------------------------------------

for comp in dataSamples742  :
    comp.splitFactor = 1000
    comp.isMC = False
    comp.isData = True

for comp in dataSamples740p9  :
    comp.splitFactor = 1000
    comp.isMC = False
    comp.isData = True

if __name__ == "__main__":
   import sys
   if "test" in sys.argv:
       from CMGTools.RootTools.samples.ComponentCreator import testSamples
       testSamples(mcSamples)
