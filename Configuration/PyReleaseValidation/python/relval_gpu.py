
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done.
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

# mc WFs to run in IB:

# no PU     Alpaka pixel-only                                   TTbar: quadruplets any backend and profiling; ECAL-only any backend; HCAL-only any backend and profiling
# mc 2025
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

           # 2024, Alpaka-based noPU
           16834.402, 16834.403, 16834.404,
           16834.406, 16834.407, 16834.408,
           16834.412, 16834.413,#16834.414,
           16834.422, 16834.423, 16834.424,
           #16834.482, 16834.483, 16834.484
           #16834.486, 16834.487, 16834.488
           16834.492, 16834.493,
           16850.402, 16850.403, 16850.404,
           16850.406, 16850.407, 16850.408,
           16861.402,

           # 2024 with PU, Alpaka-based
           17034.402, 17034.403, 17034.404,
           17034.406, 17034.407, 17034.408,
           17034.409,
           17034.412, 17034.413, #17034.414
           17034.422, 17034.423, 17034.424,
           #17034.482, 17034.483, 17034.484
           #17034.486, 17034.487, 17034.488
           17034.492, 17034.493, 17034.409,
           17050.402, 17050.403, 17050.404,
           17050.406, 17050.407, 17050.408,
           17061.402,

           # Run4, Alpaka-based noPU
           29634.402, 29634.403, 29634.404, 29634.406, 29634.704,
           29661.402,

           # Run4, Alpaka-based PU
           29834.402, 29834.403, 29834.404, 29834.704,

           #FIXME 2024 wfs, to be removed when the bot wfs are migrated
           12834.402, 12834.403, 12834.406, 
           12834.412, 12834.422, 12834.423
        ]

for numWF in numWFIB:
    if not numWF in _upgrade_workflows:
        continue
    workflows[numWF] = _upgrade_workflows[numWF]

# data WFs to run in IB:

# data 2023 Alpaka pixel-only triplets:   RunJetMET2023D on GPU (optional), RunJetMET2023D GPU-vs-CPU validation, RunJetMET2023D profiling
#           Alpaka ECAL-only:             RunJetMET2023D on GPU (optional), RunJetMET2023D GPU-vs-CPU validation, RunJetMET2023D profiling
#           Alpaka HCAL-only:             RunJetMET2023D on GPU (optional), RunJetMET2023D GPU-vs-CPU validation, RunJetMET2023D profiling

workflows[141.044406] = ['Run3-2023_JetMET2023D_RecoPixelOnlyTripletsGPU',[
                            'RunJetMET2023D',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_Alpaka_PixelOnlyTripletsGPU',
                            'HARVESTRUN3_pixelTrackingOnly'
                        ]]
workflows[141.044407] = ['Run3-2023_JetMET2023D_RecoPixelOnlyTripletsGPU_Validation',[
                            'RunJetMET2023D',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_Alpaka_PixelOnlyTripletsGPUValidation',
                            'HARVESTRUN3_pixelTrackingOnlyGPUValidation'
                        ]]
workflows[141.044408] = ['Run3-2023_JetMET2023D_RecoPixelOnlyTripletsGPU_Profiling',[
                            'RunJetMET2023D',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_Alpaka_PixelOnlyTripletsGPUProfiling'
                        ]]

workflows[141.044412] = ['Run3-2023_JetMET2023D_RecoECALOnlyGPU',[
                            'RunJetMET2023D',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_Alpaka_ECALOnlyGPU',
                            'HARVESTRUN3_ECALOnly'
                        ]]
workflows[141.044413] = ['Run3-2023_JetMET2023D_RecoECALOnlyGPU_Validation',[
                            'RunJetMET2023D',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_Alpaka_ECALOnlyGPUValidation',
                            'HARVESTRUN3_ECALOnlyGPUValidation'
                        ]]
workflows[141.044414] = ['Run3-2023_JetMET2023D_RecoECALOnlyGPU_Profiling',[
                            'RunJetMET2023D',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_Alpaka_ECALOnlyGPUProfiling'
                        ]]

workflows[141.044422] = ['Run3-2023_JetMET2023D_RecoHCALOnlyGPU',[
                            'RunJetMET2023D',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_Alpaka_HCALOnlyGPU',
                            'HARVESTRUN3_HCALOnly'
                        ]]
workflows[141.044423] = ['Run3-2023_JetMET2023D_RecoHCALOnlyGPU_Validation',[
                            'RunJetMET2023D',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_Alpaka_HCALOnlyGPUValidation',
                            'HARVESTRUN3_HCALOnlyGPUValidation'
                        ]]
workflows[141.044424] = ['Run3-2023_JetMET2023D_RecoHCALOnlyGPU_Profiling',[
                            'RunJetMET2023D',
                            'HLTDR3_2023',
                            'RECODR3_reHLT_Alpaka_HCALOnlyGPUProfiling'
                        ]]

workflows[141.044483] = ['Run3-2023_JetMET2023D_GPUValidation',[
                            'RunJetMET2023D',
                            'RecoData_Alpaka_AllGPU_Validation_2023',
                            'HARVESTData_Alpaka_AllGPU_Validation_2023'
                        ]]

# 2023 HIon MC Alpaka pixel-only quadruplets on HydjetQ_MinBias_5362GeV_2023_ppReco on GPU (optional)
workflows[160.03502] = ['',['HydjetQ_MinBias_5362GeV_2023_ppReco','DIGIHI2023PPRECO','RAWPRIMESIMHI18','RECOHI2023PPRECOMB_AlpakaGPU','MINIHI2023PROD']]
