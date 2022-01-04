import FWCore.ParameterSet.Config as cms

from CalibTracker.SiPixelLorentzAngle.siPixelLorentzAnglePCLWorker_cfi import siPixelLorentzAnglePCLWorker as _defaultWorker
SiPixelLorentzAnglePCLWorker = _defaultWorker.clone()

## modules replaced over 2017/2018 EOY shutdown, "new" in 2018
from Configuration.Eras.Modifier_run2_SiPixel_2018_cff import run2_SiPixel_2018
run2_SiPixel_2018.toModify(SiPixelLorentzAnglePCLWorker, newmodulelist = ["BPix_BpO_SEC1_LYR1_LDR1F_MOD3", # 303054876
                                                                          "BPix_BpO_SEC4_LYR1_LDR3F_MOD3", # 303063068
                                                                          "BPix_BmO_SEC2_LYR1_LDR1F_MOD1", # 303054864
                                                                          "BPix_BmO_SEC1_LYR1_LDR1F_MOD3", # 303054856
                                                                          "BPix_BmO_SEC4_LYR1_LDR3F_MOD1", # 303063056
                                                                          "BPix_BmO_SEC7_LYR1_LDR5F_MOD1"  # 303071248
                                                                          ])

## modules replaced over LS2 ("new from 2021 onwards)
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(SiPixelLorentzAnglePCLWorker, newmodulelist = ["BPix_BmI_SEC7_LYR2_LDR12F_MOD1", # 304185360
                                                                    "BPix_BmI_SEC8_LYR2_LDR14F_MOD1", # 304177168
                                                                    "BPix_BmO_SEC3_LYR2_LDR5F_MOD1",  # 304136208
                                                                    "BPix_BmO_SEC3_LYR2_LDR5F_MOD2",  # 304136204
                                                                    "BPix_BmO_SEC3_LYR2_LDR5F_MOD3",  # 304136200
                                                                    "BPix_BpO_SEC1_LYR2_LDR1F_MOD1",  # 304119828
                                                                    "BPix_BpO_SEC1_LYR2_LDR1F_MOD2",  # 304119832
                                                                    "BPix_BpO_SEC1_LYR2_LDR1F_MOD3"   # 304119836
                                                                    ])
