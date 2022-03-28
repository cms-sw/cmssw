from ROOT import *
gROOT.SetBatch(True)

def main():

  run_num = 349840
#  out_dir = "gem_collision_plots/"
  out_dir = "gem_cosmic_plots/"
#  in_file = TFile.Open('RelVal_ZMM_sum.root', 'OPEN')
#  in_file = TFile.Open('DQM_V0001_L1T_R000286520.root', 'OPEN')

  in_file = TFile.Open('upload/DQM_V0001_L1T_R000%d.root'%run_num, 'OPEN')
  plots = ['gemHitBX',
           'gemHitOccupancy', 
#           'cscLCTOccupancy',
#           'cscDQMOccupancy',

#           'hitTypeBX',
#           'hitTypeSector',
#           'hitTypeNumber',
#           'hitTypeNumSecGE11Pos',
#           'hitTypeNumSecGE11Neg',
#           'hitCoincideME11',
#           'hitCoincideGE11',
#
#           'SameSectorTimingCSCGEM',
#           'SameSectorChamberCSCGEM',
#           'SameSectorGEMPadPartition',
#           'SameSectorGEMminusCSCfpThetaPhi',
#           'gemNegBXAddress0134',
#
#           'GEMInput/gemChamberAddressGENeg11',
#           'GEMInput/gemChamberAddressGEPos11',
#           'GEMInput/gemChamberPadGENeg11',
#           'GEMInput/gemChamberPadGEPos11',
#           'GEMInput/gemChamberPartitionGENeg11',
#           'GEMInput/gemChamberPartitionGEPos11', 
#
          'GEMInput/gemChamberVFATGENeg11',
          'GEMInput/gemChamberVFATGEPos11',
          #'GEMInput/gemBXVFATPerChamber'
          #'GEMInput/gemChamberVFATBX'
#           'GEMInput/gemBXVFATGENeg11',
#           'GEMInput/gemBXVFATGEPos11',
#           'GEMInput/gemBXVFATC91011GENeg11',
#           'GEMInput/gemBXVFATC91011GEPos11',

# Uncommmeted for new plots 06-09-22
          'GEMInput/gemBXVFATChamber9Layer1',
          'GEMInput/gemBXVFATChamber9Layer2',
          'GEMInput/gemBXVFATChamber10Layer1',
          'GEMInput/gemBXVFATChamber10Layer2',
          'GEMInput/gemBXVFATChamber11Layer1',
          'GEMInput/gemBXVFATChamber11Layer2',
          'GEMInput/gemChamberVFATBXGENeg11',
          'GEMInput/gemChamberVFATBXGEPos11',
# Commented out for now
#           'GEMInput/gemBXVFATPerChamber',
#           'GEMInput/gemChamberVFATBX',

           'GEMInput/gemBXVFATPerChamber_0_0_0',
           'GEMInput/gemBXVFATPerChamber_1_0_0',
           'GEMInput/gemBXVFATPerChamber_2_0_0',
           'GEMInput/gemBXVFATPerChamber_3_0_0',
           'GEMInput/gemBXVFATPerChamber_4_0_0',
           'GEMInput/gemBXVFATPerChamber_5_0_0',
           'GEMInput/gemBXVFATPerChamber_6_0_0',
           'GEMInput/gemBXVFATPerChamber_7_0_0',
           'GEMInput/gemBXVFATPerChamber_8_0_0',
           'GEMInput/gemBXVFATPerChamber_9_0_0',
           'GEMInput/gemBXVFATPerChamber_10_0_0',
           'GEMInput/gemBXVFATPerChamber_11_0_0',
           'GEMInput/gemBXVFATPerChamber_12_0_0',
           'GEMInput/gemBXVFATPerChamber_13_0_0',
           'GEMInput/gemBXVFATPerChamber_14_0_0',
           'GEMInput/gemBXVFATPerChamber_15_0_0',
           'GEMInput/gemBXVFATPerChamber_16_0_0',
           'GEMInput/gemBXVFATPerChamber_17_0_0',
           'GEMInput/gemBXVFATPerChamber_18_0_0',
           'GEMInput/gemBXVFATPerChamber_19_0_0',
           'GEMInput/gemBXVFATPerChamber_20_0_0',
           'GEMInput/gemBXVFATPerChamber_21_0_0',
           'GEMInput/gemBXVFATPerChamber_22_0_0',
           'GEMInput/gemBXVFATPerChamber_23_0_0',
           'GEMInput/gemBXVFATPerChamber_24_0_0',
           'GEMInput/gemBXVFATPerChamber_25_0_0',
           'GEMInput/gemBXVFATPerChamber_26_0_0',
           'GEMInput/gemBXVFATPerChamber_27_0_0',
           'GEMInput/gemBXVFATPerChamber_28_0_0',
           'GEMInput/gemBXVFATPerChamber_29_0_0',
           'GEMInput/gemBXVFATPerChamber_30_0_0',
           'GEMInput/gemBXVFATPerChamber_31_0_0',
           'GEMInput/gemBXVFATPerChamber_32_0_0',
           'GEMInput/gemBXVFATPerChamber_33_0_0',
           'GEMInput/gemBXVFATPerChamber_34_0_0',
           'GEMInput/gemBXVFATPerChamber_35_0_0',

           'GEMInput/gemBXVFATPerChamber_0_0_1',
           'GEMInput/gemBXVFATPerChamber_1_0_1',
           'GEMInput/gemBXVFATPerChamber_2_0_1',
           'GEMInput/gemBXVFATPerChamber_3_0_1',
           'GEMInput/gemBXVFATPerChamber_4_0_1',
           'GEMInput/gemBXVFATPerChamber_5_0_1',
           'GEMInput/gemBXVFATPerChamber_6_0_1',
           'GEMInput/gemBXVFATPerChamber_7_0_1',
           'GEMInput/gemBXVFATPerChamber_8_0_1',
           'GEMInput/gemBXVFATPerChamber_9_0_1',
           'GEMInput/gemBXVFATPerChamber_10_0_1',
           'GEMInput/gemBXVFATPerChamber_11_0_1',
           'GEMInput/gemBXVFATPerChamber_12_0_1',
           'GEMInput/gemBXVFATPerChamber_13_0_1',
           'GEMInput/gemBXVFATPerChamber_14_0_1',
           'GEMInput/gemBXVFATPerChamber_15_0_1',
           'GEMInput/gemBXVFATPerChamber_16_0_1',
           'GEMInput/gemBXVFATPerChamber_17_0_1',
           'GEMInput/gemBXVFATPerChamber_18_0_1',
           'GEMInput/gemBXVFATPerChamber_19_0_1',
           'GEMInput/gemBXVFATPerChamber_20_0_1',
           'GEMInput/gemBXVFATPerChamber_21_0_1',
           'GEMInput/gemBXVFATPerChamber_22_0_1',
           'GEMInput/gemBXVFATPerChamber_23_0_1',
           'GEMInput/gemBXVFATPerChamber_24_0_1',
           'GEMInput/gemBXVFATPerChamber_25_0_1',
           'GEMInput/gemBXVFATPerChamber_26_0_1',
           'GEMInput/gemBXVFATPerChamber_27_0_1',
           'GEMInput/gemBXVFATPerChamber_28_0_1',
           'GEMInput/gemBXVFATPerChamber_29_0_1',
           'GEMInput/gemBXVFATPerChamber_30_0_1',
           'GEMInput/gemBXVFATPerChamber_31_0_1',
           'GEMInput/gemBXVFATPerChamber_32_0_1',
           'GEMInput/gemBXVFATPerChamber_33_0_1',
           'GEMInput/gemBXVFATPerChamber_34_0_1',
           'GEMInput/gemBXVFATPerChamber_35_0_1',

           'GEMInput/gemBXVFATPerChamber_0_1_0',
           'GEMInput/gemBXVFATPerChamber_1_1_0',
           'GEMInput/gemBXVFATPerChamber_2_1_0',
           'GEMInput/gemBXVFATPerChamber_3_1_0',
           'GEMInput/gemBXVFATPerChamber_4_1_0',
           'GEMInput/gemBXVFATPerChamber_5_1_0',
           'GEMInput/gemBXVFATPerChamber_6_1_0',
           'GEMInput/gemBXVFATPerChamber_7_1_0',
           'GEMInput/gemBXVFATPerChamber_8_1_0',
           'GEMInput/gemBXVFATPerChamber_9_1_0',
           'GEMInput/gemBXVFATPerChamber_10_1_0',
           'GEMInput/gemBXVFATPerChamber_11_1_0',
           'GEMInput/gemBXVFATPerChamber_12_1_0',
           'GEMInput/gemBXVFATPerChamber_13_1_0',
           'GEMInput/gemBXVFATPerChamber_14_1_0',
           'GEMInput/gemBXVFATPerChamber_15_1_0',
           'GEMInput/gemBXVFATPerChamber_16_1_0',
           'GEMInput/gemBXVFATPerChamber_17_1_0',
           'GEMInput/gemBXVFATPerChamber_18_1_0',
           'GEMInput/gemBXVFATPerChamber_19_1_0',
           'GEMInput/gemBXVFATPerChamber_20_1_0',
           'GEMInput/gemBXVFATPerChamber_21_1_0',
           'GEMInput/gemBXVFATPerChamber_22_1_0',
           'GEMInput/gemBXVFATPerChamber_23_1_0',
           'GEMInput/gemBXVFATPerChamber_24_1_0',
           'GEMInput/gemBXVFATPerChamber_25_1_0',
           'GEMInput/gemBXVFATPerChamber_26_1_0',
           'GEMInput/gemBXVFATPerChamber_27_1_0',
           'GEMInput/gemBXVFATPerChamber_28_1_0',
           'GEMInput/gemBXVFATPerChamber_29_1_0',
           'GEMInput/gemBXVFATPerChamber_30_1_0',
           'GEMInput/gemBXVFATPerChamber_31_1_0',
           'GEMInput/gemBXVFATPerChamber_32_1_0',
           'GEMInput/gemBXVFATPerChamber_33_1_0',
           'GEMInput/gemBXVFATPerChamber_34_1_0',
           'GEMInput/gemBXVFATPerChamber_35_1_0',

           'GEMInput/gemBXVFATPerChamber_0_1_1',
           'GEMInput/gemBXVFATPerChamber_1_1_1',
           'GEMInput/gemBXVFATPerChamber_2_1_1',
           'GEMInput/gemBXVFATPerChamber_3_1_1',
           'GEMInput/gemBXVFATPerChamber_4_1_1',
           'GEMInput/gemBXVFATPerChamber_5_1_1',
           'GEMInput/gemBXVFATPerChamber_6_1_1',
           'GEMInput/gemBXVFATPerChamber_7_1_1',
           'GEMInput/gemBXVFATPerChamber_8_1_1',
           'GEMInput/gemBXVFATPerChamber_9_1_1',
           'GEMInput/gemBXVFATPerChamber_10_1_1',
           'GEMInput/gemBXVFATPerChamber_11_1_1',
           'GEMInput/gemBXVFATPerChamber_12_1_1',
           'GEMInput/gemBXVFATPerChamber_13_1_1',
           'GEMInput/gemBXVFATPerChamber_14_1_1',
           'GEMInput/gemBXVFATPerChamber_15_1_1',
           'GEMInput/gemBXVFATPerChamber_16_1_1',
           'GEMInput/gemBXVFATPerChamber_17_1_1',
           'GEMInput/gemBXVFATPerChamber_18_1_1',
           'GEMInput/gemBXVFATPerChamber_19_1_1',
           'GEMInput/gemBXVFATPerChamber_20_1_1',
           'GEMInput/gemBXVFATPerChamber_21_1_1',
           'GEMInput/gemBXVFATPerChamber_22_1_1',
           'GEMInput/gemBXVFATPerChamber_23_1_1',
           'GEMInput/gemBXVFATPerChamber_24_1_1',
           'GEMInput/gemBXVFATPerChamber_25_1_1',
           'GEMInput/gemBXVFATPerChamber_26_1_1',
           'GEMInput/gemBXVFATPerChamber_27_1_1',
           'GEMInput/gemBXVFATPerChamber_28_1_1',
           'GEMInput/gemBXVFATPerChamber_29_1_1',
           'GEMInput/gemBXVFATPerChamber_30_1_1',
           'GEMInput/gemBXVFATPerChamber_31_1_1',
           'GEMInput/gemBXVFATPerChamber_32_1_1',
           'GEMInput/gemBXVFATPerChamber_33_1_1',
           'GEMInput/gemBXVFATPerChamber_34_1_1',
           'GEMInput/gemBXVFATPerChamber_35_1_1',

#           GEM-CSC chamber coincidence

           'GEMInput/gemBXVFATPerChamberCoincidence_0_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_1_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_2_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_3_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_4_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_5_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_6_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_7_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_8_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_9_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_10_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_11_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_12_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_13_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_14_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_15_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_16_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_17_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_18_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_19_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_20_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_21_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_22_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_23_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_24_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_25_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_26_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_27_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_28_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_29_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_30_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_31_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_32_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_33_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_34_0_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_35_0_0',

           'GEMInput/gemBXVFATPerChamberCoincidence_0_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_1_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_2_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_3_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_4_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_5_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_6_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_7_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_8_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_9_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_10_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_11_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_12_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_13_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_14_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_15_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_16_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_17_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_18_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_19_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_20_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_21_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_22_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_23_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_24_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_25_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_26_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_27_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_28_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_29_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_30_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_31_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_32_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_33_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_34_0_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_35_0_1',

           'GEMInput/gemBXVFATPerChamberCoincidence_0_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_1_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_2_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_3_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_4_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_5_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_6_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_7_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_8_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_9_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_10_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_11_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_12_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_13_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_14_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_15_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_16_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_17_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_18_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_19_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_20_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_21_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_22_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_23_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_24_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_25_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_26_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_27_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_28_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_29_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_30_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_31_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_32_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_33_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_34_1_0',
           'GEMInput/gemBXVFATPerChamberCoincidence_35_1_0',

           'GEMInput/gemBXVFATPerChamberCoincidence_0_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_1_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_2_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_3_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_4_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_5_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_6_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_7_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_8_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_9_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_10_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_11_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_12_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_13_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_14_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_15_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_16_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_17_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_18_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_19_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_20_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_21_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_22_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_23_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_24_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_25_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_26_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_27_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_28_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_29_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_30_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_31_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_32_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_33_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_34_1_1',
           'GEMInput/gemBXVFATPerChamberCoincidence_35_1_1',


#           'GEMInput/gemPosCham32S5NPadPart',
#           'GEMInput/gemPosCham02S6NPadPart',
#           'GEMInput/gemNegCham08S1NPadPart',
#           'GEMInput/gemNegCham20S3NPadPart',
#           'GEMInput/gemNegCham12PadPart',


#           'Timing/gemHitTimingBX0',
#           'Timing/gemHitTimingBXNeg1',
#           'Timing/gemHitTimingBXNeg2',
#           'Timing/gemHitTimingBXPos1', 
#           'Timing/gemHitTimingBXPos2',
#           'Timing/gemHitTimingTot',
#
#           'Timing/gemHitTimingFracBX0',
#           'Timing/gemHitTimingFracBXNeg1',
#           'Timing/gemHitTimingFracBXNeg2',
#           'Timing/gemHitTimingFracBXPos1',
#           'Timing/gemHitTimingFracBXPos2',
#
#           'Timing/emtfTrackBXVsGEMHit2Station',
#           'Timing/emtfTrackBXVsGEMHit3Station',
#           'Timing/emtfTrackBXVsGEMHit4Station',
#
#           'Timing/emtfTrackModeVsCSCBXDiffMENeg4',
#           'Timing/emtfTrackModeVsCSCBXDiffMENeg3',
#           'Timing/emtfTrackModeVsCSCBXDiffMENeg2',
#           'Timing/emtfTrackModeVsCSCBXDiffMENeg1',
#           'Timing/emtfTrackModeVsCSCBXDiffMEPos4',
#           'Timing/emtfTrackModeVsCSCBXDiffMEPos3',
#           'Timing/emtfTrackModeVsCSCBXDiffMEPos2',
#           'Timing/emtfTrackModeVsCSCBXDiffMEPos1',
#
#           'Timing/emtfTrackModeVsRPCBXDiffRENeg4',
#           'Timing/emtfTrackModeVsRPCBXDiffRENeg3',
#           'Timing/emtfTrackModeVsRPCBXDiffRENeg2',
#           'Timing/emtfTrackModeVsRPCBXDiffREPos4',
#           'Timing/emtfTrackModeVsRPCBXDiffREPos3',
#           'Timing/emtfTrackModeVsRPCBXDiffREPos2',           
#
#           'Timing/emtfTrackModeVsGEMBXDiffGENeg11',
#           'Timing/emtfTrackModeVsGEMBXDiffGEPos11',
#
#           'GEMVsCSC/gemHitPhiGENeg11',
#           'GEMVsCSC/gemHitPhiGEPos11',
#           'GEMVsCSC/gemHitThetaGENeg11',
#           'GEMVsCSC/gemHitThetaGEPos11',
#             
#           'GEMVsCSC/gemHitVScscLCTPhiGENeg11',
#           'GEMVsCSC/gemHitVScscLCTPhiGEPos11',
#           'GEMVsCSC/gemHitVScscLCTThetaGENeg11',
#           'GEMVsCSC/gemHitVScscLCTThetaGEPos11',
          ]


  gStyle.SetOptStat(0000)
  gStyle.SetPadTopMargin(0.07);
  gStyle.SetPadBottomMargin(0.10);
  gStyle.SetPadLeftMargin(0.13);
  gStyle.SetPadRightMargin(0.14);
  gStyle.SetPalette(kLightTemperature);
#  gStyle.SetPaintTextFormat(".1e");
  for plot in plots:
    hist = False
    # hist = in_file.Get('DQMData/Run %d/L1T/Run summary/L1TStage2EMTF/%s' % (run_num,plot)).Clone(plot.replace('/','_'))
    hist = in_file.Get('DQMData/Run %d/L1T/Run summary/L1TStage2EMTF/%s' % (run_num, plot)).Clone(plot.replace('/','_'))
    if 'VFAT' in plot:
      hist.GetYaxis().SetRange(1,24);
    if 'Chamber' in plot or 'Occupancy' in plot:
#      gStyle.SetPaintTextFormat(".0e");
      canv = TCanvas(plot.replace('/','_'), plot.replace('/','_'), 2000, 500)
    else:
#      gStyle.SetPaintTextFormat("5.1e");
      canv = TCanvas(plot.replace('/','_'), plot.replace('/','_'), 800, 600) # 2000,500

#    if 'Same' in plot or plot == 'hitCoincideME11' or plot == 'hitCoincideGE11': gStyle.SetPaintTextFormat(".0f");
    if 'hit' in plot and 'Num' in plot: canv.SetLogy()
    canv.cd()

#    if plot == 'GEMInput/gemChamberPadGENeg11': hist.Draw('colz')
#    if plot == 'GEMInput/gemChamberAddressGENeg11':
#      hist_big = hist.Clone(plot + '_more1000')
#      hist_small = hist.Clone(plot + '_less1000')
#      for xbin in range(1, hist.GetNbinsX()+1):
#        for ybin in range(1, hist.GetNbinsY()+1):
#          if hist_big.GetBinContent(xbin, ybin) < 1000: hist_big.SetBinContent(xbin, ybin, 0)
#          if hist_small.GetBinContent(xbin, ybin) > 1000: hist_small.SetBinContent(xbin, ybin, 0)
#
#      print (hist_small.Integral(1, hist.GetNbinsX(), 1, 384))
#      print (hist_small.Integral(1, hist.GetNbinsX(), 577, 960)) 

#      gStyle.SetPaintTextFormat(".0f")
#      hist_big.Draw('colztext')
#      canv.SaveAs(out_dir + plot.replace('/','_') + '_more1000' + '.png')
#      canv.Clear()
#      hist_small.Draw('colztext')
#      canv.SaveAs(out_dir + plot.replace('/','_') + '_less1000' + '.png')
#    else: 
      
    hist.Draw('colztext')    
    canv.SaveAs(out_dir + plot.replace('/','_') + '.png')


main()
