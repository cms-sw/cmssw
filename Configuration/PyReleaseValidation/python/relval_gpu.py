
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done.
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

# mc WFs to run in IB:

# mc 2023   Alpaka pixel-only quadruplets:                      TTbar: any backend, any backend vs cpu validation, profiling
#           Alpaka ECAL-only:                                   TTbar: any backend
#           Alpaka HCAL-only:                                   TTbar: any backend, any backend vs cpu validation, profiling
#           Alpaka with full reco and pixel-only quadruplets:   TTbar: any backend
#           Alpaka pixel-only quadruplets:                      ZMM: any backend, any backend vs cpu validation, profiling
#           Alpaka pixel-only quadruplets:                      Single Nu E10: any backend
#           Patatrack pixel-only quadruplets:                   ZMM - on GPU (optional), GPU-vs-CPU validation, profiling
#           Patatrack pixel-only triplets:                      ZMM - on GPU (optional), GPU-vs-CPU validation, profiling
#           Patatrack pixel-only quadruplets:                   TTbar - on GPU (optional), GPU-vs-CPU validation, profiling
#           Patatrack pixel-only triplets:                      TTbar - on GPU (optional), GPU-vs-CPU validation, profiling
#           Patatrack ECAL-only:                                TTbar - on GPU (optional), GPU-vs-CPU validation, profiling
#           Patatrack HCAL-only:                                TTbar - on GPU (optional), GPU-vs-CPU validation, profiling
#           Patatrack pixel-only quadruplets, ECAL, HCAL:       TTbar - on GPU (optional), GPU-vs-CPU validation
#           Patatrack pixel-only triplets, ECAL, HCAL:          TTbar - on GPU (optional), GPU-vs-CPU validation
#           full reco with Patatrack pixel-only quadruplets:    TTbar - on GPU (optional), GPU-vs-CPU validation
#           full reco with Patatrack pixel-only triplets:       TTbar - on GPU (optional), GPU-vs-CPU validation
#           Patatrack pixel-only quadruplets:                   Single Nu E10 on GPU (optional)
#           Alpaka pixel-only quadruplets:                      TTbar with PU: any backend, any backend vs cpu validation, profiling
#           Alpaka ECAL-only:                                   TTbar with PU: any backend
#           Alpaka HCAL-only:                                   TTbar with PU: any backend, any backend vs cpu validation, profiling
#           Alpaka with full reco and pixel-only quadruplets:   TTbar with PU: any backend
#           Alpaka pixel-only quadruplets:                      ZMM with PU: any backend, any backend vs cpu validation, profiling
#           Alpaka pixel-only quadruplets:                      Single Nu E10 with PU: any backend
# mc 2026   Patatrack pixel-only quadruplets:                   Single Nu E10: on GPU (optional)
numWFIB = [
           # 2023, Alpaka-based
           12434.402, 12434.403, 12434.404,
          #12434.406, 12434.407, 12434.408,
           12434.412,#12434.413, 12434.414,
           12434.422, 12434.423, 12434.424,
          #12434.482, 12434.483, 12434.484
          #12434.486, 12434.487, 12434.488
           12434.492,#12434.493
           12450.402, 12450.403, 12450.404,
          #12450.406, 12450.407, 12450.408,
           12461.402,

           # 2023, CUDA-based
           12450.502, 12450.503, 12450.504,
           12450.506, 12450.507, 12450.508,
           12434.502, 12434.503, 12434.504,
           12434.506, 12434.507, 12434.508,
           12434.512, 12434.513, 12434.514,
           12434.522, 12434.523, 12434.524,
           12434.582, 12434.583,#12434.584,
           12434.586, 12434.587,#12434.588,
           12434.592, 12434.593,
           12434.596, 12434.597,
           12461.502,

           # 2023 with PU, Alpaka-based
           12634.402, 12634.403, 12634.404,
          #12634.406, 12634.407, 12634.408
           12634.412,#12634.413, 12634.414
           12634.422, 12634.423, 12634.424,
          #12634.482, 12634.483, 12634.484
          #12634.486, 12634.487, 12634.488
           12634.492,#12634.493
           12650.402, 12650.403, 12650.404,
          #12650.406, 12650.407, 12650.408
           12661.402,

           # 2026, CUDA-based
           24861.502
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
