
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

#just define all of them

#WFs to run in IB:
# mc 2018   (Patatrack pixel-only: ZMM - on GPU, both CPU and GPU, auto)
#           (Patatrack pixel-only: TTbar - on GPU, both CPU and GPU, auto)
#           (Patatrack ECAL-only: TTbar - on GPU, both CPU and GPU, auto)
#           (Patatrack HCAL-only: TTbar - on GPU, both CPU and GPU, auto)
# mc 2021   (Patatrack pixel-only: ZMM - on GPU, both CPU and GPU, auto)
#           (Patatrack pixel-only: TTbar - on GPU, both CPU and GPU, auto)
#           (Patatrack ECAL-only: TTbar - on GPU, both CPU and GPU, auto)
#           (Patatrack HCAL-only: TTbar - on GPU, both CPU and GPU, auto)
numWFIB = [
           10842.502, # 10842.503,10842.504,
           10824.502, # 10824.503,10824.504,
           10824.512, # 10824.513,10824.514,
           10824.522, # 10824.523,10824.524,
           11650.502, # 11650.503,11650.504,
           11634.502, # 11634.503,11634.504,
           11634.512, # 11634.513,11634.514,
           11634.522, # 11634.523,11634.524
        ]
for numWF in numWFIB:
    if not numWF in _upgrade_workflows: continue
    workflows[numWF] = _upgrade_workflows[numWF]

# data 2018 (Patatrack pixel-only: RunHLTPhy2018D, RunJetHT2018D on GPU)
#           (Patatrack ECAL-only:  RunHLTPhy2018D, RunJetHT2018D on GPU)
#           (Patatrack HCAL-only:  RunHLTPhy2018D, RunJetHT2018D on GPU)

workflows[136.885502] = ['',['RunHLTPhy2018D','HLTDR2_2018','RECODR2_2018reHLT_Patatrack_PixelOnlyGPU','HARVEST2018_pixelTrackingOnly']]
workflows[136.888502] = ['',['RunJetHT2018D','HLTDR2_2018','RECODR2_2018reHLT_Patatrack_PixelOnlyGPU','HARVEST2018_pixelTrackingOnly']]
workflows[136.885512] = ['',['RunHLTPhy2018D','HLTDR2_2018','RECODR2_2018reHLT_ECALOnlyGPU','HARVEST2018_ECALOnly']]
workflows[136.888512] = ['',['RunJetHT2018D','HLTDR2_2018','RECODR2_2018reHLT_ECALOnlyGPU','HARVEST2018_ECALOnly']]
workflows[136.885522] = ['',['RunHLTPhy2018D','HLTDR2_2018','RECODR2_2018reHLT_HCALOnlyGPU','HARVEST2018_HCALOnly']]
workflows[136.888522] = ['',['RunJetHT2018D','HLTDR2_2018','RECODR2_2018reHLT_HCALOnlyGPU','HARVEST2018_HCALOnly']]
