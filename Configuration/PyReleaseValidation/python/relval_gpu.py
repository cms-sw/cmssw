
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done.
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

# mc WFs to run in IB:

# mc 2023   #FIXME to be removed as soon as cms-bot is updated 
# no PU     Alpaka pixel-only                                   TTbar: quadruplets any backend and profiling; ECAL-only any backend; HCAL-only any backend and profiling
# mc 2024
# no PU  
#           Alpaka pixel-only quadruplets:                      TTbar: any backend, any backend vs cpu validation, profiling
#           Alpaka pixel-only triplets:                         TTbar: any backend, any backend vs cpu validation, profiling
#           Alpaka ECAL-only:                                   TTbar: any backend
#           Alpaka HCAL-only:                                   TTbar: any backend, any backend vs cpu validation, profiling
#           Alpaka with full reco and pixel-only:               TTbar: any backend quadruplets, any backend triplets
#           Alpaka pixel-only quadruplets:                      ZMM: any backend, any backend vs cpu validation, profiling
#           Alpaka pixel-only triplets:                         ZMM: any backend, any backend vs cpu validation, profiling
#           Alpaka pixel-only quadruplets:                      Single Nu E10: any backend
# with PU
#           Alpaka pixel-only quadruplets:                      TTbar with PU: any backend, any backend vs cpu validation, profiling
#           Alpaka pixel-only triplets:                         TTbar with PU: any backend, any backend vs cpu validation, profiling
#           Alpaka ECAL-only:                                   TTbar with PU: any backend
#           Alpaka HCAL-only:                                   TTbar with PU: any backend, any backend vs cpu validation, profiling
#           Alpaka with full reco and pixel-only:               TTbar with PU: any backend quadruplets, any backend triplets
#           Alpaka pixel-only quadruplets:                      ZMM with PU: any backend, any backend vs cpu validation, profiling
#           Alpaka pixel-only triplets:                         ZMM with PU: any backend, any backend vs cpu validation, profiling
#           Alpaka pixel-only quadruplets:                      Single Nu E10 with PU: any backend
# mc Run4   
# no PU
#           Alpaka pixel-only:                                  TTbar: quadruplets any backend, any backend vs cpu validation, profiling, triplets      
#           Alpaka pixel-only:                                  Single Nu E10: any backend
# with PU
#           Alpaka pixel-only:                                  TTbar with PU: quadruplets any backend, any backend vs cpu validation, profiling 

numWFIB = [
           # 2023, Alpaka-based noPU
           12434.402,12434.403,12434.412,12434.422,12434.423,

           # 2024, Alpaka-based noPU
           12834.402, 12834.403, 12834.404,
           12834.406, 12834.407, 12834.408,
           12834.412, 12834.413,#12834.414,
           12834.422, 12834.423, 12834.424,
           #12834.482, 12834.483, 12834.484
           #12834.486, 12834.487, 12834.488
           12834.492, 12834.493,
           12850.402, 12850.403, 12850.404,
           12450.406, 12450.407, 12450.408,
           12861.402,

           # 2024 with PU, Alpaka-based
           13034.402, 13034.403, 13034.404,
           13034.406, 13034.407, 13034.408,
           13034.412, 13034.413, #13034.414
           13034.422, 13034.423, 13034.424,
           #13034.482, 13034.483, 13034.484
           #13034.486, 13034.487, 13034.488
           13034.492, 13034.493,
           13050.402, 13050.403, 13050.404,
           13050.406, 13050.407, 13050.408,
           13061.402,

           # Run4, Alpaka-based noPU
           29634.402, 29634.403, 29634.404, 29634.406, 29634.704,
           29661.402,

           # Run4, Alpaka-based PU
           29834.402, 29834.403, 29834.404, 29834.704
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
