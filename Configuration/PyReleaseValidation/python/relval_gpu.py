
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

# mc WFs to run in IB:

# mc 2023   Patatrack pixel-only quadruplets:                   ZMM - on GPU (optional), GPU-vs-CPU validation, profiling
#           Patatrack pixel-only triplets:                      ZMM - on GPU (optional), GPU-vs-CPU validation, profiling
#           Patatrack pixel-only quadruplets:                   TTbar - on GPU (optional), GPU-vs-CPU validation, profiling
#           Patatrack pixel-only triplets:                      TTbar - on GPU (optional), GPU-vs-CPU validation, profiling
#           Patatrack ECAL-only:                                TTbar - on GPU (optional), GPU-vs-CPU validation, profiling
#           Patatrack HCAL-only:                                TTbar - on GPU (optional), GPU-vs-CPU validation, profiling
#           Patatrack pixel-only quadruplets, ECAL, HCAL:       TTbar - on GPU (optional), GPU-vs-CPU validation, profiling (to be implemented)
#           Patatrack pixel-only triplets, ECAL, HCAL:          TTbar - on GPU (optional), GPU-vs-CPU validation, profiling (to be implemented)
#           full reco with Patatrack pixel-only quadruplets:    TTbar - on GPU (optional), GPU-vs-CPU validation
#           full reco with Patatrack pixel-only triplets:       TTbar - on GPU (optional), GPU-vs-CPU validation
numWFIB = [
           # 2023
           12450.502, 12450.503, 12450.504,
           12450.506, 12450.507, 12450.508,
           12434.502, 12434.503, 12434.504,
           12434.506, 12434.507, 12434.508,
           12434.512, 12434.513, 12434.514,
           12434.522, 12434.523, 12434.524,
           12434.582, 12434.583, # 12434.584,
           12434.586, 12434.587, # 12434.588,
           12434.592, 12434.593,
           12434.596, 12434.597,
        ]

for numWF in numWFIB:
    if not numWF in _upgrade_workflows:
        continue
    workflows[numWF] = _upgrade_workflows[numWF]

# data WFs to run in IB:

# data 2023 Patatrack pixel-only triplets:   RunJetMET2022D on GPU (optional), RunJetMET2022D GPU-vs-CPU validation, RunJetMET2022D profiling
#           Patatrack ECAL-only:             RunJetMET2022D on GPU (optional), RunJetMET2022D GPU-vs-CPU validation, RunJetMET2022D profiling
#           Patatrack HCAL-only:             RunJetMET2022D on GPU (optional), RunJetMET2022D GPU-vs-CPU validation, RunJetMET2022D profiling
workflows[141.008506] = ['Run3-2023_JetMET2023B_RecoPixelOnlyTripletsGPU',[
                            'RunJetMET2023B',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_Patatrack_PixelOnlyTripletsGPU',
                            'HARVESTRUN3_pixelTrackingOnly'
                        ]]
workflows[141.008507] = ['Run3-2023_JetMET2023B_RecoPixelOnlyTripletsGPU_Validation',[
                            'RunJetMET2023B',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_Patatrack_PixelOnlyTripletsGPUValidation',
                            'HARVESTRUN3_pixelTrackingOnlyGPUValidation'
                        ]]
workflows[141.008508] = ['Run3-2023_JetMET2023B_RecoPixelOnlyTripletsGPU_Profiling',[
                            'RunJetMET2023B',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_Patatrack_PixelOnlyTripletsGPUProfiling'
                        ]]

workflows[141.008512] = ['Run3-2023_JetMET2023B_RecoECALOnlyGPU',[
                            'RunJetMET2023B',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_ECALOnlyGPU',
                            'HARVESTRUN3_ECALOnly'
                        ]]
workflows[141.008513] = ['Run3-2023_JetMET2023B_RecoECALOnlyGPU_Validation',[
                            'RunJetMET2023B',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_ECALOnlyGPUValidation',
                            'HARVESTRUN3_ECALOnlyGPUValidation'
                        ]]
workflows[141.008514] = ['Run3-2023_JetMET2023B_RecoECALOnlyGPU_Profiling',[
                            'RunJetMET2023B',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_ECALOnlyGPUProfiling'
                        ]]

workflows[141.008522] = ['Run3-2023_JetMET2023B_RecoHCALOnlyGPU',[
                            'RunJetMET2023B',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_HCALOnlyGPU',
                            'HARVESTRUN3_HCALOnly'
                        ]]
workflows[141.008523] = ['Run3-2023_JetMET2023B_RecoHCALOnlyGPU_Validation',[
                            'RunJetMET2023B',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_HCALOnlyGPUValidation',
                            'HARVESTRUN3_HCALOnlyGPUValidation'
                        ]]
workflows[141.008524] = ['Run3-2023_JetMET2023B_RecoHCALOnlyGPU_Profiling',[
                            'RunJetMET2023B',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_HCALOnlyGPUProfiling'
                        ]]

workflows[141.008583] = ['Run3-2023_JetMET2023B_GPUValidation',[
                            'RunJetMET2023B',
                            'RecoData_Patatrack_AllGPU_Validation_2023',
                            'HARVESTData_Patatrack_AllGPU_Validation_2023'
                        ]]

# 2023 HIon MC Patatrack pixel-only quadruplets on HydjetQ_MinBias_5362GeV_2023_ppReco on GPU (optional)
workflows[160.03502] = ['',['HydjetQ_MinBias_5362GeV_2023_ppReco','DIGIHI2023PPRECO','RAWPRIMESIMHI18','RECOHI2023PPRECOMB_PatatrackGPU','MINIHI2023PROD']]
