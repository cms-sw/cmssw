import PhysicsTools.HeppyCore.framework.config as cfg

import os

#####COMPONENT CREATOR

from CMGTools.RootTools.samples.ComponentCreator import ComponentCreator
kreator = ComponentCreator()

dataDir = os.environ['CMSSW_BASE']+"/src/CMGTools/TTHAnalysis/data"
dataPrivDir = os.environ['CMSSW_BASE']+"/src/CMGTools/ObjectsStudies/data"

json=dataDir+'/json/Cert_Run2012ABCD_22Jan2013ReReco.json'

### samples for HCAL reco validation
JetHT_HcalExtValid_jet2012D_v1 = cfg.DataComponent(
    name = 'JetHT_HcalExtValid_jet2012D_v1',
    files = kreator.getFilesFromEOS('JetHT_HcalExtValid_jet2012D_v1', '/JetHT/CMSSW_7_3_2_patch1-GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v1/MINIAOD', '/store/relval/CMSSW_7_3_2_patch1/JetHT/MINIAOD/GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v1/00000/'),
    intLumi = 1,
    triggers = [],
    json = json
    )
JetHT_HcalExtValid_jet2012D_v2 = cfg.DataComponent(
    name = 'JetHT_HcalExtValid_jet2012D_v2',
    files = kreator.getFilesFromEOS('JetHT_HcalExtValid_jet2012D_v2', '/JetHT/CMSSW_7_3_2_patch1-GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v2/MINIAOD', '/store/relval/CMSSW_7_3_2_patch1/JetHT/MINIAOD/GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v2/00000/'),
    intLumi = 1,
    triggers = [],
    json = json
    )

DoubleMuparked_HcalExtValid_zMu2012D_v1 = cfg.DataComponent(
    name = 'DoubleMuparked_HcalExtValid_zMu2012D_v1',
    files = kreator.getFilesFromEOS('DoubleMuparked_HcalExtValid_zMu2012D_v1', '/DoubleMuParked/CMSSW_7_3_2_patch1-GR_R_73_V0_HcalExtValid_RelVal_zMu2012D-v1/MINIAOD', '/store/relval/CMSSW_7_3_2_patch1/DoubleMuParked/MINIAOD/GR_R_73_V0_HcalExtValid_RelVal_zMu2012D-v1/00000/'),
    intLumi = 1,
    triggers = [],
#    json = json
# below the json that correspond the the 740_pre9 re-reco of the same runD     
    json = json
    )


#----------- 74X re-reco ---------------

DoubleMuparked_1Apr_RelVal_dm2012D_v2 = cfg.DataComponent(
    name = 'DoubleMuparked_1Apr_RelVal_dm2012D_v2',
    files = kreator.getFilesFromEOS('DoubleMuparked_1Apr_RelVal_dm2012D_v2', '/DoubleMuParked/CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_dm2012D-v2/MINIAOD', '/store/relval/CMSSW_7_4_0_pre9_ROOT6/DoubleMuParked/MINIAOD/GR_R_74_V8_1Apr_RelVal_dm2012D-v2/00000/'),
    intLumi = 1,
    triggers = [],
#    json = json
# below the json that correspond the the 740_pre9 re-reco of the same runD     
    json = dataPrivDir+'/json/diMu_740pre9_miniAOD.json' 
    )


#------  PRIVATE reco - DOUBLE MU - 74X - new Egamma + new Hadron

DoubleMuParked_740_fixes = kreator.makeMCComponentFromEOS('DoubleMuParked_740_newEGHCalib','DoubleMuParked_CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_dm2012D-v2_ecalCalibNewPFHadCalib_mAOD/','/store/group/phys_jetmet/schoef/740_ecalCalibNewPFHadCalib/%s',".*root",1.)
#/store/group/phys_jetmet/schoef/740_ecalCalibNewPFHadCalib/DoubleMuParked_CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_dm2012D-v2_ecalCalibNewPFHadCalib_mAOD


dataSamplesAll = [JetHT_HcalExtValid_jet2012D_v1, JetHT_HcalExtValid_jet2012D_v2, DoubleMuparked_HcalExtValid_zMu2012D_v1, DoubleMuparked_1Apr_RelVal_dm2012D_v2,DoubleMuParked_740_fixes]


#----------- 2011 re-reco ---------------

DoubleMu_zMu2011A_CMSSW_7_0_6 = cfg.DataComponent(
    name = 'DoubleMu_zMu2011A_7_0_6',
    files = kreator.getFilesFromEOS('DoubleMu_zMu2011A_7_0_6', '/DoubleMu/MINIAOD/GR_70_V2_AN1_RelVal_zMu2011A-v1/MINIAOD', '/store/relval/CMSSW_7_0_6/DoubleMu/MINIAOD/GR_70_V2_AN1_RelVal_zMu2011A-v1/00000/'),
    )

DoubleMu_zMu2011A_CMSSW_7_3_0 = cfg.DataComponent(
    name = 'DoubleMu_zMu2011A_CMSSW_7_3_0',
    files = kreator.getFilesFromEOS('DoubleMu_zMu2011A_7_3_0', '/DoubleMu/MINIAOD/GR_R_73_V0A_RelVal_zMu2011A-v1/MINIAOD', '/eos/cms/store/relval/CMSSW_7_3_0/DoubleMu/MINIAOD/GR_R_73_V0A_RelVal_zMu2011A-v1/00000/'),
    )

DoubleMu_zMu2011A_CMSSW_7_3_1_patch1 = cfg.DataComponent(
    name = 'DoubleMu_zMu2011A_CMSSW_7_3_1_patch1',
    files = kreator.getFilesFromEOS('DoubleMu_zMu2011A_7_3_1_patch1', '/DoubleMu/MINIAOD/GR_R_73_V0A_RelVal_zMu2011A-v1/MINIAOD', '/store/relval/CMSSW_7_3_1_patch1/DoubleMu/MINIAOD/GR_R_73_V0A_RelVal_zMu2011A-v1/00000/'),
    )

DoubleMu_zMu2011A_CMSSW_7_3_3 = cfg.DataComponent(
    name = 'DoubleMu_zMu2011A_CMSSW_7_3_3',
    files = kreator.getFilesFromEOS('DoubleMu_zMu2011A_7_3_3', '/DoubleMu/MINIAOD/GR_R_73_V2A_RelVal_zMu2011A-v1/MINIAOD', '/store/relval/CMSSW_7_3_3/DoubleMu/MINIAOD/GR_R_73_V2A_RelVal_zMu2011A-v1/00000/'),
    )

DoubleMu_zMu2011A_7_4_0_pre9 = cfg.DataComponent(
    name = 'DoubleMu_zMu2011A_7_4_0_pre9',
    files = kreator.getFilesFromEOS('DoubleMu_zMu2011A_7_4_0_pre9', '/DoubleMu/MINIAOD/GR_R_74_V8A_RelVal_zMu2011A-v1/MINIAOD', '/store/relval/CMSSW_7_4_0_pre9/DoubleMu/MINIAOD/GR_R_74_V8A_RelVal_zMu2011A-v1/00000/'),
    )

data2011All = [ DoubleMu_zMu2011A_CMSSW_7_0_6 , DoubleMu_zMu2011A_CMSSW_7_3_0, DoubleMu_zMu2011A_CMSSW_7_3_1_patch1, DoubleMu_zMu2011A_CMSSW_7_3_3, DoubleMu_zMu2011A_7_4_0_pre9 ]

#----------- MC relVal --------------

RelValZMM_25ns_7_3_1_patch1 = cfg.DataComponent(
    name = 'RelValZMM_25ns_7_3_1_patch1',
    files = kreator.getFilesFromEOS('RelValZMM_25ns_7_3_1_patch1', '/RelValZMM_13/MINIAOD/PU25ns_MCRUN2_73_V9-v1/MINIAOD', '/store/relval/CMSSW_7_3_1_patch1/RelValZMM_13/MINIAODSIM/PU25ns_MCRUN2_73_V9-v1/00000/'),
    )

RelValZMM_25ns_7_3_3 = cfg.DataComponent(
    name = 'RelValZMM_25ns_7_3_3',
    files = kreator.getFilesFromEOS('RelValZMM_25ns_7_3_3', '/RelValZMM_13/MINIAOD/PU25ns_MCRUN2_73_V11-v1/MINIAOD', '/eos/cms/store/relval/CMSSW_7_3_3/RelValZMM_13/MINIAODSIM/PU25ns_MCRUN2_73_V11-v1/00000/'),
    )

RelValZMM_25ns_7_4_0_pre9 = cfg.DataComponent(
    name = 'RelValZMM_25ns_7_4_0_pre9',
    files = kreator.getFilesFromEOS('RelValZMM_25ns_7_4_0_pre9', '/RelValZMM_7_4_0_pre9/MINIAOD/PU25ns_MCRUN2_74_V7-v1/MINIAOD', '/store/relval/CMSSW_7_4_0_pre9/RelValZMM_13/MINIAODSIM/PU25ns_MCRUN2_74_V7-v1/00000/'),
    )

#------

RelValZMM_50ns_7_3_1_patch1 = cfg.DataComponent(
    name = 'RelValZMM_50ns_7_3_1_patch1',
    files = kreator.getFilesFromEOS('RelValZMM_50ns_7_3_1_patch1', '/RelValZMM_13/MINIAOD/PU50ns_MCRUN2_73_V9-v1/MINIAOD', '/store/relval/CMSSW_7_3_1_patch1/RelValZMM_13/MINIAODSIM/PU50ns_MCRUN2_73_V9-v1/00000/'),
    )

RelValZMM_50ns_7_3_3 = cfg.DataComponent(
    name = 'RelValZMM_50ns_7_3_3',
    files = kreator.getFilesFromEOS('RelValZMM_50ns_7_3_3', '/RelValZMM_13/MINIAOD/PU50ns_MCRUN2_73_V10-v1/MINIAOD', '/eos/cms/store/relval/CMSSW_7_3_3/RelValZMM_13/MINIAODSIM/PU50ns_MCRUN2_73_V10-v1/00000/'),
    )

RelValZMM_50ns_7_4_0_pre9 = cfg.DataComponent(
    name = 'RelValZMM_50ns_7_4_0_pre9',
    files = kreator.getFilesFromEOS('RelValZMM_50ns_7_4_0_pre9', '/RelValZMM_7_4_0_pre9/MINIAOD/PU50ns_MCRUN2_74_V6-v1/MINIAOD', '/store/relval/CMSSW_7_4_0_pre9/RelValZMM_13/MINIAODSIM/PU50ns_MCRUN2_74_V6-v1/00000/'),
    )


relValMCstrange = [ RelValZMM_25ns_7_3_1_patch1, RelValZMM_25ns_7_3_3, RelValZMM_25ns_7_4_0_pre9, RelValZMM_50ns_7_3_1_patch1, RelValZMM_50ns_7_3_3, RelValZMM_50ns_7_4_0_pre9 ]

#------  PRIVATE reco - RelVal MU - 741 - PhilFixes

RelVal_741_Philfixes = kreator.makeMCComponentFromEOS('RelValZMM_741_PhilFixes','RelValZMM_13_CMSSW_7_4_1-MCRUN2_74_V9_extended-v2_PhilFix_mAOD/','/store/group/phys_jetmet/schoef/741_relval_rereco/%s',".*root",1.)


#------  PRIVATE reco - RelVal MU - 741 - Katarina Fixes

#RelVal_741_Method0_Katefixes = kreator.makeMCComponentFromEOS('RelValZMM_741_Method0_KateFixes','crab_ZMM_Method0/150515_162358/0000','/store/user/kfiekas/HCAL/JetResponse/RelValZMM_13/%s',".*root",1.)
#RelVal_741_Method2Default_Katefixes = kreator.makeMCComponentFromEOS('RelValZMM_741_Method2Default_KateFixes','crab_ZMM_Method2Default/150515_161935/0000','/store/user/kfiekas/HCAL/JetResponse/RelValZMM_13/%s',".*root",1.)
#RelVal_741_Method2NoThreshold_Katefixes = kreator.makeMCComponentFromEOS('RelValZMM_741_Method2NoThreshold_KateFixes','crab_ZMM_Method2NoThreshold/150515_161712/0000','/store/user/kfiekas/HCAL/JetResponse/RelValZMM_13/%s',".*root",1.)
#RelVal_741_Method2Thr100fCNoM0NewTiming_Katefixes = kreator.makeMCComponentFromEOS('RelValZMM_741_Method2Thr100fCNoM0NewTiming_KateFixes','crab_ZMM_Method2Thr100fCNoM0NewTiming/150515_154343/0000','/store/user/kfiekas/HCAL/JetResponse/RelValZMM_13/%s',".*root",1.)
#RelVal_741_Method3_Katefixes = kreator.makeMCComponentFromEOS('RelValZMM_741_Method3_KateFixes','crab_ZMM_Method3/150515_162551/0000','/store/user/kfiekas/HCAL/JetResponse/RelValZMM_13/%s',".*root",1.)

ZMM_mAOD_Method2Default = kreator.makeMCComponentFromEOS('RelValZMM_741_Method2Default','ZMM_mAOD_Method2Default','/eos/cms/store/group/phys_jetmet/schoef/741_relval_rereco/katharina/%s',".*root",1.) 
ZMM_mAOD_Method2Thr100fCNoM0NewTimingNegEnergyFix = kreator.makeMCComponentFromEOS('RelValZMM_Method2Thr100fCNoM0NewTimingNegEnergyFix','ZMM_mAOD_Method2Thr100fCNoM0NewTimingNegEnergyFix','/eos/cms/store/group/phys_jetmet/schoef/741_relval_rereco/katharina/%s',".*root",1.)

#relValkate = [ RelVal_741_Method0_Katefixes, RelVal_741_Method2Default_Katefixes , RelVal_741_Method2NoThreshold_Katefixes, RelVal_741_Method2Thr100fCNoM0NewTiming_Katefixes, RelVal_741_Method3_Katefixes ]

relValkate = [ ZMM_mAOD_Method2Default, ZMM_mAOD_Method2Thr100fCNoM0NewTimingNegEnergyFix ]

#------  PRIVATE reco

MVAegamma_OFF_25ns = kreator.makeMCComponentFromEOS('MVAegamma_OFF_25ns','RelValZMM_13_CMSSW_7_4_0_pre9-PU25ns_MCRUN2_74_V7-v1_MVA_OFF_mAOD','/store/group/phys_jetmet/schoef/740pre9_relVal_rereco/%s',".*root",1.)

MVAegamma_ON_25ns = kreator.makeMCComponentFromEOS('MVAegamma_ON_25ns','RelValZMM_13_CMSSW_7_4_0_pre9-PU25ns_MCRUN2_74_V7-v1_MVA_ON_mAOD','/store/group/phys_jetmet/schoef/740pre9_relVal_rereco/%s',".*root",1.)

MVAegamma_OFF_50ns = kreator.makeMCComponentFromEOS('MVAegamma_OFF_50ns','RelValZMM_13_CMSSW_7_4_0_pre9-PU50ns_MCRUN2_74_V6-v1_MVA_OFF_mAOD','/store/group/phys_jetmet/schoef/740pre9_relVal_rereco/%s',".*root",1.)

MVAegamma_ON_50ns = kreator.makeMCComponentFromEOS('MVAegamma_ON_50ns','RelValZMM_13_CMSSW_7_4_0_pre9-PU50ns_MCRUN2_74_V6-v1_MVA_ON_mAOD','/store/group/phys_jetmet/schoef/740pre9_relVal_rereco/%s',".*root",1.)

MVAegammaMC = [ MVAegamma_OFF_25ns, MVAegamma_ON_25ns, MVAegamma_OFF_50ns, MVAegamma_ON_50ns ]



#------  PRIVATE reco - DATA Matthieu's fix

DoubleMuParked_1Apr_RelVal_dm2012D_v2_newPFHCalib = kreator.makeMCComponentFromEOS('DoubleMuParked_1Apr_RelVal_dm2012D_v2_newPFHCalib','DoubleMuParked_CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_dm2012D-v2_newCalib_mAOD/','/store/group/phys_jetmet/schoef/740pre9_data_rereco/%s',".*root",1.)   

DoubleMuParked_1Apr_RelVal_dm2012D_v2_oldPFHCalib = kreator.makeMCComponentFromEOS('DoubleMuParked_1Apr_RelVal_dm2012D_v2_oldPFHCalib','DoubleMuParked_CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_dm2012D-v2_oldCalib_mAOD/','/store/group/phys_jetmet/schoef/740pre9_data_rereco/%s',".*root",1.)   

JetHT_GR_R_74_V8_1Apr_v1_newPFHCalib = kreator.makeMCComponentFromEOS('JetHT_GR_R_74_V8_1Apr_v1_newPFHCalib','JetHT_CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_jht2012D-v1_newCalib_mAOD/','/store/group/phys_jetmet/schoef/740pre9_data_rereco/%s',".*root",1.)

JetHT_GR_R_74_V8_1Apr_v1_oldPFHCalib = kreator.makeMCComponentFromEOS('JetHT_GR_R_74_V8_1Apr_v1_oldPFHCalib','JetHT_CMSSW_7_4_0_pre9_ROOT6-GR_R_74_V8_1Apr_RelVal_jht2012D-v1_oldCalib_mAOD/','/store/group/phys_jetmet/schoef/740pre9_data_rereco/%s',".*root",1.)

HCALcalibDATA = [ DoubleMuParked_1Apr_RelVal_dm2012D_v2_newPFHCalib , DoubleMuParked_1Apr_RelVal_dm2012D_v2_oldPFHCalib , JetHT_GR_R_74_V8_1Apr_v1_newPFHCalib , JetHT_GR_R_74_V8_1Apr_v1_oldPFHCalib ]


#------ Official relVal

RelValZMM_7_4_1 = cfg.DataComponent(
    name = 'RelValZMM_7_4_1',
    files = kreator.getFilesFromEOS('RelValZMM_7_4_1', '/RelValZMM_7_4_1/MINIAOD/MCRUN2_74_V9_extended-v2/MINIAOD', '/store/relval/CMSSW_7_4_1//RelValZMM_13/MINIAODSIM/MCRUN2_74_V9_extended-v2/00000/'),
    )

RelValZMM_7_4_0_pre9 = cfg.DataComponent(
    name = 'RelValZMM_7_4_0_pre9',
    files = kreator.getFilesFromEOS('RelValZMM_7_4_0_pre9', '/RelValZMM_7_4_0_pre9/MINIAOD/MCRUN2_74_V7_extended-v2/MINIAOD', '/store/relval/CMSSW_7_4_0_pre9/RelValZMM_13/MINIAODSIM/MCRUN2_74_V7_extended-v2/00000/'),
    )

relValMCofficial = [ RelValZMM_7_4_1, RelValZMM_7_4_0_pre9 ]

#-----------DATA---------------

for comp in dataSamplesAll:
    comp.splitFactor = 1
    comp.triggers = []
    comp.isMC = False
    comp.isData = True

for comp in data2011All:
    comp.splitFactor = 1
    comp.triggers = []
    comp.intLumi = 1
    comp.json = dataDir+'/json/Cert_160404-180252_7TeV_ReRecoNov08_Collisions11.json'
#    comp.isMC = False
#    comp.isData = True

for comp in HCALcalibDATA:
    comp.isMC = False
    comp.isData = True
    comp.splitFactor = 1
    comp.triggers = []
    comp.intLumi = 1
    comp.json = dataPrivDir+'/json/diMu_740pre9_miniAOD.json'

for comp in relValMCstrange:
    comp.splitFactor = 1
    comp.triggers = []
    comp.isMC = True
    comp.isData = False

for comp in [RelVal_741_Philfixes]+relValkate:
    comp.splitFactor = 1
    comp.triggers = []
    comp.isMC = True
    comp.isData = False

for comp in MVAegammaMC:
    comp.splitFactor = 1
    comp.triggers = []
    comp.isMC = True
    comp.isData = False

for comp in relValMCofficial :
    comp.splitFactor = 1
    comp.triggers = []
    comp.isMC = True
    comp.isData = False

#if __name__ == "__main__":
#   import sys
#   if "test" in sys.argv:
#       from CMGTools.TTHAnalysis.samples.ComponentCreator import testSamples
#       testSamples(mcSamples)
