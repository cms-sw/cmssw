
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *
from .MatrixUtil import Matrix

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done.
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

# to get the default upgrade geometry
from Configuration.PyReleaseValidation.relval_Run4 import prefixDet

# mc WFs to run in IB:

# no PU     Alpaka pixel-only                                   TTbar: quadruplets any backend and profiling; ECAL-only any backend; HCAL-only any backend and profiling
# mc 2025 
#           (Alpaka wfs to be removed, kept here to be used by the bot in the transition)
# mc 2026
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
#           Alpaka pixel-only:                                  TTbar: quadruplets any backend, CA Extension any backend, any backend vs cpu validation, profiling, triplets
#           Alpaka ECAL-only development:                       TTbar: any backend
#           Alpaka pixel-only:                                  Single Nu E10: any backend
#           Alpaka LST-only:                                    TTbar: TRK-only w/ 2 iterations and LST building on any backend
#           Alpaka HLTTiming75e33:                              TTbar: any backend
#           Alpaka HLTTiming75e33:                              Single Nu E10: any backend
# with PU
#           Alpaka pixel-only:                                  TTbar: quadruplets any backend, CA Extension any backend, any backend vs cpu validation, profiling
#           Alpaka LST-only:                                    TTbar: TRK-only w/ 2 iterations and LST building on any backend
#           Alpaka HLTTiming75e33:                              TTbar: any backend
numWFIB = [
           # 2025, Alpaka-based (to be removed, used by the bot)
           17034.422, 17034.403, 17034.406, 17034.412, 17034.402, 17034.423,
           # 2026, Alpaka-based noPU
           18434.402, 18434.403, 18434.404,
           18434.406, 18434.407, 18434.408,
           18434.412, 18434.413,#18434.414,
           18434.422, 18434.423, 18434.424,
           #18434.482, 18434.483, 18434.484
           #18434.486, 18434.487, 18434.488
           18434.492, 18434.493,
           18450.402, 18450.403, 18450.404,
           18450.406, 18450.407, 18450.408,
           18461.402,

           # 2026 with PU, Alpaka-based
           18634.402, 18634.403, 18634.404,
           18634.406, 18634.407, 18634.408,
           18634.412, 18634.413, #18634.414
           18634.422, 18634.423, 18634.424,
           #18634.482, 18634.483, 18634.484
           #18634.486, 18634.487, 18634.488
           18634.492, 18634.493,
           18650.402, 18650.403, 18650.404,
           18650.406, 18650.407, 18650.408,
           18661.402,

           # Run4, Alpaka-based noPU
           prefixDet+34.402, prefixDet+34.4021, prefixDet+34.403, prefixDet+34.404, prefixDet+34.406,
           prefixDet+34.612,
           prefixDet+61.402,
           prefixDet+34.704,
           prefixDet+34.751,
           prefixDet+61.751,

           # Run4, Alpaka-based PU
           prefixDet+234.402, prefixDet+234.4021, prefixDet+234.403, prefixDet+234.404,
           prefixDet+234.704,
           prefixDet+234.751,
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
