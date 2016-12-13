
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used



#just define all of them

#2017 WFs to run in IB (TenMuE_0_200, TTbar, ZEE, MinBias, TTbar PU, ZEE PU, TTbar design)
numWFIB = [10021.0,10024.0,10025.0,10026.0,10023.0,10224.0,10225.0,10424.0]
for i,key in enumerate(upgradeKeys[2017]):
    numWF=numWFAll[2017][i]
    for frag in upgradeFragments:
        k=frag[:-4]+'_'+key
        stepList=[]
        for step in upgradeProperties[2017][key]['ScenToRun']:
            if 'Sim' in step:
                stepList.append(k+'_'+step)
            else:
                stepList.append(step+'_'+key)
        if numWF in numWFIB:
            workflows[numWF] = [ upgradeDatasetFromFragment[frag], stepList]
        numWF+=1

# Tracking-specific special workflows
def _trackingOnly(stepList):
    res = []
    for step in stepList:
        s = step
        if 'RecoFull' in step or 'HARVESTFull' in step:
            s = s.replace('Full', 'Full_trackingOnly')
        if 'ALCA' in s:
            continue
        res.append(s)
    return res
def _trackingRun2(stepList):
    res = []
    for step in stepList:
        s = step
        if 'RecoFull' in step:
            if 'trackingOnly' in step:
                s = s.replace('Only', 'OnlyRun2')
            else:
                s = s.replace('Full', 'Full_trackingRun2')
        if 'ALCA' in s:
            continue
        res.append(s)
    return res
def _trackingPhase1CA(stepList):
    res = []
    for step in stepList:
        s = step
        if 'RecoFull' in step:
            if 'trackingOnly' in step:
                s = s.replace('Only', 'OnlyPhase1CA')
            else:
                s = s.replace('Full', 'Full_trackingPhase1CA')
        if 'ALCA' in s:
            continue
        res.append(s)
    return res

# compose and adding tracking specific workflows in the IB test. 
# NB. those workflows are expected to be only used forIB test.
#  if you really want to run them locally, do runTheMatrix.py --what 2017  -l workflow numbers
workflows[10024.1] = [ workflows[10024.0][0], _trackingOnly(workflows[10024.0][1]) ]
workflows[10024.2] = [ workflows[10024.0][0], _trackingRun2(workflows[10024.0][1]) ]
workflows[10024.3] = [ workflows[10024.1][0], _trackingRun2(workflows[10024.1][1]) ]
workflows[10024.4] = [ workflows[10024.0][0], _trackingPhase1CA(workflows[10024.0][1]) ]
workflows[10024.5] = [ workflows[10024.1][0], _trackingPhase1CA(workflows[10024.1][1]) ]
