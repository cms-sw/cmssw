from  Configuration.PyReleaseValidation.relval_steps import *

workflows = Matrix()

def WFN(index): 
    offset=2500
    return offset+index

_runOnly20events={'-n':'20'}
_run10kevents={'-n':'10000'}


steps['NANO_data'] = merge([{'-s':'NANO,DQM:@nanoAODDQM',
                             '--process':'NANO',
                             '--data':'',
                             '--eventcontent':'NANOAOD,DQM',
                             '--datatier':'NANOAOD,DQMIO'
                         }])

steps['NANO_mc']= merge([{'-s':'NANO,DQM:@nanoAODDQM',
                          '--process':'NANO',
                          '--mc':'',
                          '--eventcontent':'NANOAODSIM,DQM',
                          '--datatier':'NANOAODSIM,DQMIO'
                      }])

##8.0 INPUT and workflows

##9.4 INPUT and workflows
steps['TTbarMINIAOD9.4v0'] = {'INPUT':InputInfo(
    dataSet='/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/RunIISummer16MiniAODv3-PUMoriond17_94X_mcRun2_asymptotic_v3-v1/MINIAODSIM',location='STD')}
steps['NANO_mc9.4v0']=merge([{'--era':'Run2_2016,run2_nanoAOD_94X2016','--conditions':'auto:run2_mc'},steps['NANO_mc']])

steps['TTbarMINIAOD9.4v1'] = {'INPUT':InputInfo(
    dataSet='/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/RunIIFall17MiniAOD-94X_mc2017_realistic_v10-v1/MINIAODSIM',location='STD')}
steps['NANO_mc9.4v1']=merge([{'--era':'Run2_2017,run2_nanoAOD_94XMiniAODv1','--conditions':'auto:phase1_2017_realistic'},steps['NANO_mc']])
                            
steps['TTbarMINIAOD9.4v2'] = {'INPUT':InputInfo(
    dataSet='/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/MINIAODSIM',location='STD')}
steps['NANO_mc9.4v2']=merge([{'--era':'Run2_2017,run2_nanoAOD_94XMiniAODv2','--conditions':'auto:phase1_2017_realistic'},steps['NANO_mc']])

##10.2 INPUT and worflows
steps['TTbarMINIAOD10.2'] = {'INPUT':InputInfo(
    dataSet='/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM',location='STD')}
steps['NANO_mc10.2']= merge([{'--era':'Run2_2018,run2_nanoAOD_102Xv1','--conditions':'auto:phase1_2018_realistic'},steps['NANO_mc']])

##10.6 INPUT and worflows

##12.4 INPUT

##12.6 workflows


## replicating the simple tests
workflows[WFN(1)] = ['', ['TTbarMINIAOD9.4v0','NANO_mc9.4v0']]
##input dataset is marked as DELETED workflows[WFN(2)] = ['', ['TTbarMINIAOD9.4v1','NANO_mc9.4v1']]
workflows[WFN(3)] = ['', ['TTbarMINIAOD9.4v2','NANO_mc9.4v2']]

workflows[WFN(10)] = ['', ['TTbarMINIAOD10.2','NANO_mc10.2']]

## replicating the longer tests
