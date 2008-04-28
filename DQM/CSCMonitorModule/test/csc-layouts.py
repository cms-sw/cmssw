
def csclayout(i, p, *rows): i["Layouts/CSC Layouts/" + p] = DQMItem(layout=rows)

<<<<<<< csc-layouts.py
csclayout(dqmitems,"EMU Test00 - Readout Buffer Errors",

  ["EMU/All_Readout_Errors"]

)
=======
>>>>>>> 1.3

csclayout(dqmitems,"EMU Test01 - DDUs in Readout",

  ["EMU/All_DDUs_in_Readout"]

 ,["EMU/All_DDUs_L1A_Increment"]

)

csclayout(dqmitems,"EMU Test02 - DDU Event Size",
<<<<<<< csc-layouts.py

  ["EMU/All_DDUs_Event_Size"]

 ,["EMU/All_DDUs_Average_Event_Size"]

=======

  ["EMU/All_DDUs_Event_Size"]

>>>>>>> 1.3
)

csclayout(dqmitems,"EMU Test03 - DDU Reported Errors",

  ["EMU/All_DDUs_Trailer_Errors"]

)

csclayout(dqmitems,"EMU Test04 - DDU Format Errors",

  ["EMU/All_DDUs_Format_Errors"]

)

<<<<<<< csc-layouts.py
csclayout(dqmitems,"EMU Test05 - DDU Live Inputs",

  ["EMU/All_DDUs_Live_Inputs"]

 ,["EMU/All_DDUs_Average_Live_Inputs"]

=======
csclayout(dqmitems,"EMU Test05 - DDU Inputs Status",

  ["EMU/All_DDUs_Live_Inputs"]

 ,["EMU/All_DDUs_Inputs_with_Data"]

>>>>>>> 1.3
)

csclayout(dqmitems,"EMU Test06 - DDU Inputs in ERROR/WARNING State",

  ["EMU/All_DDUs_Inputs_Errors"]

 ,["EMU/All_DDUs_Inputs_Warnings"]

)

<<<<<<< csc-layouts.py
csclayout(dqmitems,"EMU Test07 - DDU Inputs with Data",

  ["EMU/All_DDUs_Inputs_with_Data"]

 ,["EMU/All_DDUs_Average_Inputs_with_Data"]

)

csclayout(dqmitems,"EMU Test08a - DMBs Reporting Data",

  ["EMU/DMB_Reporting"]

)

csclayout(dqmitems,"EMU Test08b - CSCs Reporting Data",

  ["EMU/DrawChamberMap(CSC_Reporting)"]

)

csclayout(dqmitems,"EMU Test09a - Unpacked DMBs",

  ["EMU/DMB_Unpacked"]

)

csclayout(dqmitems,"EMU Test09b - Unpacked CSCs",

  ["EMU/DrawChamberMap(CSC_Unpacked)"]

)

csclayout(dqmitems,"EMU Test09c - Unpacked DMBs (Fractions)",

  ["EMU/DMB_Unpacked_Fract"]

)

csclayout(dqmitems,"EMU Test09d - Unpacked CSCs (Fractions)",

  ["EMU/DrawChamberMap(CSC_Unpacked_Fract)"]

)

csclayout(dqmitems,"EMU Test10a - DMBs with Format Errors",

  ["EMU/DMB_Format_Errors"]

)

csclayout(dqmitems,"EMU Test10b - CSCs with Format Errors",

  ["EMU/DrawChamberMap(CSC_Format_Errors)"]

)

csclayout(dqmitems,"EMU Test10c - DMBs with Format Errors (Fractions)",

  ["EMU/DMB_Format_Errors_Fract"]

)

csclayout(dqmitems,"EMU Test10d - CSCs with Format Errors (Fractions)",

  ["EMU/DrawChamberMap(CSC_Format_Errors_Fract)"]

)

csclayout(dqmitems,"EMU Test11a - DMBs without ALCT Data",

  ["EMU/DMB_wo_ALCT"]

)

csclayout(dqmitems,"EMU Test11b - CSCs without ALCT Data",

  ["EMU/DrawChamberMap(CSC_wo_ALCT)"]

)

csclayout(dqmitems,"EMU Test11c - DMBs without ALCT Data (Fractions)",

  ["EMU/DMB_wo_ALCT_Fract"]

)

csclayout(dqmitems,"EMU Test11d - CSCs without ALCT Data (Fractions)",

  ["EMU/DrawChamberMap(CSC_wo_ALCT_Fract)"]

)

csclayout(dqmitems,"EMU Test12a - DMBs without CLCT Data",

  ["EMU/DMB_wo_CLCT"]

)

csclayout(dqmitems,"EMU Test12b - CSCs without CLCT Data",

  ["EMU/DrawChamberMap(CSC_wo_CLCT)"]

)

csclayout(dqmitems,"EMU Test12c - DMBs without CLCT Data (Fractions)",

  ["EMU/DMB_wo_CLCT_Fract"]

)

csclayout(dqmitems,"EMU Test12d - CSCs without CLCT Data (Fractions)",

  ["EMU/DrawChamberMap(CSC_wo_CLCT_Fract)"]

)

csclayout(dqmitems,"EMU Test13a - DMBs without CFEB Data",

  ["EMU/DMB_wo_CFEB"]

)

csclayout(dqmitems,"EMU Test13b - CSCs without CFEB Data",

  ["EMU/DrawChamberMap(CSC_wo_CFEB)"]

)

csclayout(dqmitems,"EMU Test13c - DMBs without CFEB Data (Fractions)",

  ["EMU/DMB_wo_CFEB_Fract"]

)

csclayout(dqmitems,"EMU Test13d - CSCs without CFEB Data (Fractions)",

  ["EMU/DrawChamberMap(CSC_wo_CFEB_Fract)"]

)

csclayout(dqmitems,"EMU Test14a - DMBs with CFEB B-Words",

  ["EMU/DMB_Format_Warnings"]

)

csclayout(dqmitems,"EMU Test14b - CSCs with CFEB B-Words",

  ["EMU/DrawChamberMap(CSC_Format_Warnings)"]

)

csclayout(dqmitems,"EMU Test14c - DMBs with CFEB B-Words (Fraction)",

  ["EMU/DMB_Format_Warnings_Fract"]

)

csclayout(dqmitems,"EMU Test14d - CSCs with CFEB B-Words (Fraction)",

  ["EMU/DrawChamberMap(CSC_Format_Warnings_Fract)"]

)

csclayout(dqmitems,"EMU Test15a - DMBs with DMB-INPUT-FIFO FULL Status",

  ["EMU/DMB_input_fifo_full"]

)

csclayout(dqmitems,"EMU Test15b - CSCs with DMB-INPUT-FIFO FULL Status",

  ["EMU/DrawChamberMap(CSC_DMB_input_fifo_full)"]

)

csclayout(dqmitems,"EMU Test15c - DMBs with DMB-INPUT-FIFO FULL Status (Fraction)",

  ["EMU/DMB_input_fifo_full_Fract"]

)

csclayout(dqmitems,"EMU Test15d - CSCs with DMB-INPUT-FIFO FULL Status (Fraction)",

  ["EMU/DrawChamberMap(CSC_DMB_input_fifo_full_Fract)"]

)

csclayout(dqmitems,"EMU Test16a - DMBs with DMB-INPUT TIMEOUT Status",

  ["EMU/DMB_input_timeout"]

)

csclayout(dqmitems,"EMU Test16b - CSCs with DMB-INPUT TIMEOUT Status",

  ["EMU/DrawChamberMap(CSC_DMB_input_timeout)"]

)

csclayout(dqmitems,"EMU Test16c - DMBs with DMB-INPUT TIMEOUT Status (Fraction)",

  ["EMU/DMB_input_timeout_Fract"]

)

csclayout(dqmitems,"EMU Test16d - CSCs with DMB-INPUT TIMEOUT Status (Fraction)",

  ["EMU/DrawChamberMap(CSC_DMB_input_timeout_Fract)"]

)

csclayout(dqmitems,"DMBs DAV and Unpacked vs DMBs Active",

  ["DDU/DMB_Active_Header_Count"]

 ,["DDU/DMB_DAV_Header_Count_vs_DMB_Active_Header_Count"]

 ,["DDU/DMB_unpacked_vs_DAV"]

)

csclayout(dqmitems,"Error Status from DDU Trailer",

  ["DDU/Trailer_ErrorStat_Table"]

 ,["DDU/Trailer_ErrorStat_Frequency"]

)

csclayout(dqmitems,"Connected and Active Inputs",

  ["DDU/DMB_Connected_Inputs"]

 ,["DDU/DMB_DAV_Header_Occupancy"]

)

csclayout(dqmitems,"Event Buffer Size and DDU Word Count",

  ["DDU/Buffer_Size"]

 ,["DDU/Word_Count"]

)

csclayout(dqmitems,"L1A and BXN Counters",

  ["DDU/BXN"]

 ,["DDU/L1A_Increment"]

)

csclayout(dqmitems,"State of CSCs",

  ["DDU/CSC_Errors"]

 ,["DDU/CSC_Warnings"]

)

csclayout(dqmitems,"ALCT: ALCT0 Key Wiregroups, Patterns and Quality",

  ["CSC/ALCT0_KeyWG"]

 ,["CSC/ALCT0_Quality"]

 ,["CSC/ALCT0_Pattern"]

 ,["CSC/ALCT0_Quality_Profile"]

)

csclayout(dqmitems,"ALCT: ALCT0_BXN and ALCT_L1A_BXN Synchronization",

  ["CSC/ALCT0_BXN"]

 ,["CSC/ALCT0_dTime_vs_KeyWG"]

 ,["CSC/ALCT0_dTime"]

 ,["CSC/ALCT0_dTime_Profile"]

)

csclayout(dqmitems,"ALCT: ALCT1 Key Wiregroups, Patterns and Quality",

  ["CSC/ALCT1_KeyWG"]

 ,["CSC/ALCT1_Quality"]

 ,["CSC/ALCT1_Pattern"]

 ,["CSC/ALCT1_Quality_Profile"]

=======
csclayout(dqmitems,"EMU Test08 - CSCs Reporting Data and Unpacked",

  ["EMU/CSC_Reporting"]

 ,["EMU/CSC_Unpacked_Fract"]

>>>>>>> 1.3
)

<<<<<<< csc-layouts.py
csclayout(dqmitems,"ALCT: ALCT1_BXN and ALCT_L1A_BXN Synchronization",

  ["CSC/ALCT1_BXN"]

 ,["CSC/ALCT1_dTime_vs_KeyWG"]

 ,["CSC/ALCT1_dTime"]

 ,["CSC/ALCT1_dTime_Profile"]

)
=======
csclayout(dqmitems,"EMU Test10 - CSCs with Errors and Warnings (Fractions)",

  ["EMU/CSC_Format_Errors_Fract"]

 ,["EMU/CSC_Format_Warnings_Fract"]

 ,["EMU/CSC_DMB_input_fifo_full_Fract"]
>>>>>>> 1.3

<<<<<<< csc-layouts.py
csclayout(dqmitems,"ALCT: ALCTs Found",

  ["CSC/ALCT_Number_Efficiency"]

 ,["CSC/ALCT1_vs_ALCT0_KeyWG"]

)
=======
 ,["EMU/CSC_DMB_input_timeout_Fract"]
>>>>>>> 1.3

<<<<<<< csc-layouts.py
csclayout(dqmitems,"ALCT: Anode Hit Occupancy per Chamber",

  ["CSC/ALCT_Number_Of_Layers_With_Hits"]

 ,["CSC/ALCT_Number_Of_WireGroups_With_Hits"]

=======
>>>>>>> 1.3
)

<<<<<<< csc-layouts.py
csclayout(dqmitems,"ALCT: Anode Hit Occupancy per Wire Group",

  ["CSC/ALCT_Ly1_Efficiency"]

 ,["CSC/ALCT_Ly2_Efficiency"]

 ,["CSC/ALCT_Ly3_Efficiency"]

 ,["CSC/ALCT_Ly4_Efficiency"]

 ,["CSC/ALCT_Ly5_Efficiency"]

 ,["CSC/ALCT_Ly6_Efficiency"]

)
=======
csclayout(dqmitems,"EMU Test11 - CSCs without Data Blocks",
>>>>>>> 1.3

<<<<<<< csc-layouts.py
csclayout(dqmitems,"ALCT: Number of Words in ALCT",

  ["CSC/ALCT_Word_Count"]

)
=======
  ["EMU/CSC_wo_ALCT_Fract"]
>>>>>>> 1.3

<<<<<<< csc-layouts.py
csclayout(dqmitems,"ALCT: Raw Hit Time Bin Average Occupancy",

  ["CSC/ALCTTime_Ly1_Profile"]

 ,["CSC/ALCTTime_Ly2_Profile"]

 ,["CSC/ALCTTime_Ly3_Profile"]

 ,["CSC/ALCTTime_Ly4_Profile"]

 ,["CSC/ALCTTime_Ly5_Profile"]

 ,["CSC/ALCTTime_Ly6_Profile"]

)
=======
 ,["EMU/CSC_wo_CLCT_Fract"]
>>>>>>> 1.3

<<<<<<< csc-layouts.py
csclayout(dqmitems,"ALCT: Raw Hit Time Bin Occupancy",

  ["CSC/ALCTTime_Ly1"]

 ,["CSC/ALCTTime_Ly2"]

 ,["CSC/ALCTTime_Ly3"]

 ,["CSC/ALCTTime_Ly4"]

 ,["CSC/ALCTTime_Ly5"]

 ,["CSC/ALCTTime_Ly6"]

)
=======
 ,["EMU/CSC_wo_CFEB_Fract"]
>>>>>>> 1.3

<<<<<<< csc-layouts.py
csclayout(dqmitems,"CFEB: Cluster Duration",

  ["CSC/CFEB_Cluster_Duration_Ly_1"]

 ,["CSC/CFEB_Cluster_Duration_Ly_2"]

 ,["CSC/CFEB_Cluster_Duration_Ly_3"]

 ,["CSC/CFEB_Cluster_Duration_Ly_4"]

 ,["CSC/CFEB_Cluster_Duration_Ly_5"]

 ,["CSC/CFEB_Cluster_Duration_Ly_6"]

=======
>>>>>>> 1.3
)

csclayout(dqmitems,"CFEB: Clusters Charge",

  ["CSC/CFEB_Clusters_Charge_Ly_1"]

 ,["CSC/CFEB_Clusters_Charge_Ly_2"]

 ,["CSC/CFEB_Clusters_Charge_Ly_3"]

 ,["CSC/CFEB_Clusters_Charge_Ly_4"]

 ,["CSC/CFEB_Clusters_Charge_Ly_5"]

 ,["CSC/CFEB_Clusters_Charge_Ly_6"]

)

csclayout(dqmitems,"CFEB: Clusters Width",

  ["CSC/CFEB_Width_of_Clusters_Ly_1"]

 ,["CSC/CFEB_Width_of_Clusters_Ly_2"]

 ,["CSC/CFEB_Width_of_Clusters_Ly_3"]

 ,["CSC/CFEB_Width_of_Clusters_Ly_4"]

 ,["CSC/CFEB_Width_of_Clusters_Ly_5"]

 ,["CSC/CFEB_Width_of_Clusters_Ly_6"]

)

csclayout(dqmitems,"CFEB: Free SCA Cells",

  ["CSC/CFEB0_Free_SCA_Cells"]

 ,["CSC/CFEB1_Free_SCA_Cells"]

 ,["CSC/CFEB2_Free_SCA_Cells"]

 ,["CSC/CFEB3_Free_SCA_Cells"]

 ,["CSC/CFEB4_Free_SCA_Cells"]

)

csclayout(dqmitems,"CFEB: Number of Clusters",

  ["CSC/CFEB_Number_of_Clusters_Ly_1"]

 ,["CSC/CFEB_Number_of_Clusters_Ly_2"]

 ,["CSC/CFEB_Number_of_Clusters_Ly_3"]

 ,["CSC/CFEB_Number_of_Clusters_Ly_4"]

 ,["CSC/CFEB_Number_of_Clusters_Ly_5"]

 ,["CSC/CFEB_Number_of_Clusters_Ly_6"]

)

csclayout(dqmitems,"CFEB: Number of SCA blocks locked by LCTs",

  ["CSC/CFEB0_SCA_Blocks_Locked_by_LCTs"]

 ,["CSC/CFEB1_SCA_Blocks_Locked_by_LCTs"]

 ,["CSC/CFEB2_SCA_Blocks_Locked_by_LCTs"]

 ,["CSC/CFEB3_SCA_Blocks_Locked_by_LCTs"]

 ,["CSC/CFEB4_SCA_Blocks_Locked_by_LCTs"]

)

csclayout(dqmitems,"CFEB: Number of SCA blocks locked by LCTxL1",

  ["CSC/CFEB0_SCA_Blocks_Locked_by_LCTxL1"]

 ,["CSC/CFEB1_SCA_Blocks_Locked_by_LCTxL1"]

 ,["CSC/CFEB2_SCA_Blocks_Locked_by_LCTxL1"]

 ,["CSC/CFEB3_SCA_Blocks_Locked_by_LCTxL1"]

 ,["CSC/CFEB4_SCA_Blocks_Locked_by_LCTxL1"]

)

csclayout(dqmitems,"CFEB: Out of ADC Range Strips",

  ["CSC/CFEB_Out_Off_Range_Strips_Ly1"]

 ,["CSC/CFEB_Out_Off_Range_Strips_Ly2"]

 ,["CSC/CFEB_Out_Off_Range_Strips_Ly3"]

 ,["CSC/CFEB_Out_Off_Range_Strips_Ly4"]

 ,["CSC/CFEB_Out_Off_Range_Strips_Ly5"]

 ,["CSC/CFEB_Out_Off_Range_Strips_Ly6"]

)

csclayout(dqmitems,"CFEB: Pedestals (First Sample)",

  ["CSC/CFEB_Pedestal_withEMV_Sample_01_Ly1"]

 ,["CSC/CFEB_Pedestal_withEMV_Sample_01_Ly2"]

 ,["CSC/CFEB_Pedestal_withEMV_Sample_01_Ly3"]

 ,["CSC/CFEB_Pedestal_withEMV_Sample_01_Ly4"]

 ,["CSC/CFEB_Pedestal_withEMV_Sample_01_Ly5"]

 ,["CSC/CFEB_Pedestal_withEMV_Sample_01_Ly6"]

)

csclayout(dqmitems,"CFEB: Pedestals RMS",

  ["CSC/CFEB_PedestalRMS_Sample_01_Ly1"]

 ,["CSC/CFEB_PedestalRMS_Sample_01_Ly2"]

 ,["CSC/CFEB_PedestalRMS_Sample_01_Ly3"]

 ,["CSC/CFEB_PedestalRMS_Sample_01_Ly4"]

 ,["CSC/CFEB_PedestalRMS_Sample_01_Ly5"]

 ,["CSC/CFEB_PedestalRMS_Sample_01_Ly6"]

)

csclayout(dqmitems,"CFEB: SCA Active Strips Occupancy",

  ["CSC/CFEB_ActiveStrips_Ly1"]

 ,["CSC/CFEB_ActiveStrips_Ly2"]

 ,["CSC/CFEB_ActiveStrips_Ly3"]

 ,["CSC/CFEB_ActiveStrips_Ly4"]

 ,["CSC/CFEB_ActiveStrips_Ly5"]

 ,["CSC/CFEB_ActiveStrips_Ly6"]

)

csclayout(dqmitems,"CFEB: SCA Active Time Samples vs Strip Numbers",

  ["CSC/CFEB_Active_Samples_vs_Strip_Ly1"]

 ,["CSC/CFEB_Active_Samples_vs_Strip_Ly2"]

 ,["CSC/CFEB_Active_Samples_vs_Strip_Ly3"]

 ,["CSC/CFEB_Active_Samples_vs_Strip_Ly4"]

 ,["CSC/CFEB_Active_Samples_vs_Strip_Ly5"]

 ,["CSC/CFEB_Active_Samples_vs_Strip_Ly6"]

)

csclayout(dqmitems,"CFEB: SCA Active Time Samples vs Strip Numbers Profile",

  ["CSC/CFEB_Active_Samples_vs_Strip_Ly1_Profile"]

 ,["CSC/CFEB_Active_Samples_vs_Strip_Ly2_Profile"]

 ,["CSC/CFEB_Active_Samples_vs_Strip_Ly3_Profile"]

 ,["CSC/CFEB_Active_Samples_vs_Strip_Ly4_Profile"]

 ,["CSC/CFEB_Active_Samples_vs_Strip_Ly5_Profile"]

 ,["CSC/CFEB_Active_Samples_vs_Strip_Ly6_Profile"]

)

csclayout(dqmitems,"CFEB: SCA Block Occupancy",

  ["CSC/CFEB0_SCA_Block_Occupancy"]

 ,["CSC/CFEB1_SCA_Block_Occupancy"]

 ,["CSC/CFEB2_SCA_Block_Occupancy"]

 ,["CSC/CFEB3_SCA_Block_Occupancy"]

 ,["CSC/CFEB4_SCA_Block_Occupancy"]

)

csclayout(dqmitems,"CFEB: SCA Cell Peak",

  ["CSC/CFEB_SCA_Cell_Peak_Ly_1"]

 ,["CSC/CFEB_SCA_Cell_Peak_Ly_2"]

 ,["CSC/CFEB_SCA_Cell_Peak_Ly_3"]

 ,["CSC/CFEB_SCA_Cell_Peak_Ly_4"]

 ,["CSC/CFEB_SCA_Cell_Peak_Ly_5"]

 ,["CSC/CFEB_SCA_Cell_Peak_Ly_6"]

)

csclayout(dqmitems,"CSC: Data Block Finding Efficiency",

  ["CSC/CSC_Efficiency"]

)

csclayout(dqmitems,"CSC: Data Format Errors and Warnings",

  ["CSC/BinCheck_ErrorStat_Table"]

 ,["CSC/BinCheck_ErrorStat_Frequency"]

 ,["CSC/BinCheck_WarningStat_Table"]

 ,["CSC/BinCheck_WarningStat_Frequency"]

)

csclayout(dqmitems,"DMB: CFEB Multiple Overlaps",

  ["CSC/DMB_CFEB_MOVLP"]

)

csclayout(dqmitems,"DMB: CFEBs DAV and Active",

  ["CSC/DMB_CFEB_Active_vs_DAV"]

 ,["CSC/DMB_CFEB_Active"]

 ,["CSC/DMB_CFEB_DAV"]

 ,["CSC/DMB_CFEB_DAV_multiplicity"]

)

csclayout(dqmitems,"DMB: DMB-CFEB-SYNC BXN Counter",

  ["CSC/DMB_CFEB_Sync"]

)

csclayout(dqmitems,"DMB: FEB Status (Timeouts, FIFO, L1 pipe)",

  ["CSC/DMB_FEB_Timeouts"]

 ,["CSC/DMB_L1_Pipe"]

 ,["CSC/DMB_FIFO_stats"]

)

csclayout(dqmitems,"DMB: FEBs DAV and Unpacked",

  ["CSC/DMB_FEB_DAV_Efficiency"]

 ,["CSC/DMB_FEB_Combinations_DAV_Efficiency"]

 ,["CSC/DMB_FEB_Unpacked_vs_DAV"]

 ,["CSC/DMB_FEB_Combinations_Unpacked_vs_DAV"]

)

csclayout(dqmitems,"SYNC: ALCT - DMB Synchronization",

  ["CSC/ALCT_L1A"]

 ,["CSC/ALCT_DMB_L1A_diff"]

 ,["CSC/DMB_L1A_vs_ALCT_L1A"]

 ,["CSC/ALCT_BXN"]

 ,["CSC/ALCT_DMB_BXN_diff"]

 ,["CSC/ALCT_BXN_vs_DMB_BXN"]

)

csclayout(dqmitems,"SYNC: CFEB - DMB Synchronization",

  ["CSC/CFEB0_L1A_Sync_Time"]

 ,["CSC/CFEB1_L1A_Sync_Time"]

 ,["CSC/CFEB2_L1A_Sync_Time"]

 ,["CSC/CFEB3_L1A_Sync_Time"]

 ,["CSC/CFEB4_L1A_Sync_Time"]

 ,["CSC/CFEB0_L1A_Sync_Time_vs_DMB"]

 ,["CSC/CFEB1_L1A_Sync_Time_vs_DMB"]

 ,["CSC/CFEB2_L1A_Sync_Time_vs_DMB"]

 ,["CSC/CFEB3_L1A_Sync_Time_vs_DMB"]

 ,["CSC/CFEB4_L1A_Sync_Time_vs_DMB"]

 ,["CSC/CFEB0_L1A_Sync_Time_DMB_diff"]

 ,["CSC/CFEB1_L1A_Sync_Time_DMB_diff"]

 ,["CSC/CFEB2_L1A_Sync_Time_DMB_diff"]

 ,["CSC/CFEB3_L1A_Sync_Time_DMB_diff"]

 ,["CSC/CFEB4_L1A_Sync_Time_DMB_diff"]

)

csclayout(dqmitems,"SYNC: DMB - DDU Synchronization",

  ["CSC/DMB_L1A_Distrib"]

 ,["CSC/DMB_DDU_L1A_diff"]

 ,["CSC/DMB_L1A_vs_DDU_L1A"]

 ,["CSC/DMB_BXN_Distrib"]

 ,["CSC/DMB_DDU_BXN_diff"]

 ,["CSC/DMB_BXN_vs_DDU_BXN"]

)

csclayout(dqmitems,"SYNC: TMB - ALCT Syncronization",

  ["CSC/TMB_L1A_vs_ALCT_L1A"]

 ,["CSC/TMB_ALCT_L1A_diff"]

 ,["CSC/TMB_BXN_vs_ALCT_BXN"]

 ,["CSC/TMB_ALCT_BXN_diff"]

)

csclayout(dqmitems,"SYNC: TMB - DMB Synchronization",

  ["CSC/CLCT_L1A"]

 ,["CSC/CLCT_DMB_L1A_diff"]

 ,["CSC/DMB_L1A_vs_CLCT_L1A"]

 ,["CSC/CLCT_BXN"]

 ,["CSC/CLCT_DMB_BXN_diff"]

 ,["CSC/CLCT_BXN_vs_DMB_BXN"]

)

csclayout(dqmitems,"TMB-CLCT: CLCT0 Key HalfStrips, Patterns and Quality",

  ["CSC/CLCT0_KeyHalfStrip"]

 ,["CSC/CLCT0_Half_Strip_Quality"]

 ,["CSC/CLCT0_Half_Strip_Pattern"]

 ,["CSC/CLCT0_Half_Strip_Quality_Profile"]

)

csclayout(dqmitems,"TMB-CLCT: CLCT0_BXN and TMB_L1A_BXN Synchronization",

  ["CSC/CLCT0_BXN"]

 ,["CSC/CLCT0_dTime_vs_Half_Strip"]

 ,["CSC/CLCT0_dTime"]

)

csclayout(dqmitems,"TMB-CLCT: CLCT1 Key HalfStrips, Patterns and Quality",

  ["CSC/CLCT1_KeyHalfStrip"]

 ,["CSC/CLCT1_Half_Strip_Quality"]

 ,["CSC/CLCT1_Half_Strip_Pattern"]

 ,["CSC/CLCT1_Half_Strip_Quality_Profile"]

)

csclayout(dqmitems,"TMB-CLCT: CLCT1_BXN and TMB_L1A_BXN Synchronization",

  ["CSC/CLCT1_BXN"]

 ,["CSC/CLCT1_dTime_vs_Half_Strip"]

 ,["CSC/CLCT1_dTime"]

)

csclayout(dqmitems,"TMB-CLCT: CLCTs Found",

  ["CSC/CLCT_Number"]

 ,["CSC/CLCT1_vs_CLCT0_Key_Strip"]

)

csclayout(dqmitems,"TMB-CLCT: Cathode Comparator Hit Occupancy per Chamber",

  ["CSC/CLCT_Number_Of_Layers_With_Hits"]

 ,["CSC/CLCT_Number_Of_HalfStrips_With_Hits"]

)

csclayout(dqmitems,"TMB-CLCT: Cathode Comparator Hit Occupancy per Half Strip",

  ["CSC/CLCT_Ly1_Efficiency"]

 ,["CSC/CLCT_Ly2_Efficiency"]

 ,["CSC/CLCT_Ly3_Efficiency"]

 ,["CSC/CLCT_Ly4_Efficiency"]

 ,["CSC/CLCT_Ly5_Efficiency"]

 ,["CSC/CLCT_Ly6_Efficiency"]

)

csclayout(dqmitems,"TMB-CLCT: Comparator Raw Hit Time Bin Average Occupancy",

  ["CSC/CLCTTime_Ly1_Profile"]

 ,["CSC/CLCTTime_Ly2_Profile"]

 ,["CSC/CLCTTime_Ly3_Profile"]

 ,["CSC/CLCTTime_Ly4_Profile"]

 ,["CSC/CLCTTime_Ly5_Profile"]

 ,["CSC/CLCTTime_Ly6_Profile"]

)

csclayout(dqmitems,"TMB-CLCT: Comparator Raw Hit Time Bin Occupancy",

  ["CSC/CLCTTime_Ly1"]

 ,["CSC/CLCTTime_Ly2"]

 ,["CSC/CLCTTime_Ly3"]

 ,["CSC/CLCTTime_Ly4"]

 ,["CSC/CLCTTime_Ly5"]

 ,["CSC/CLCTTime_Ly6"]

)

csclayout(dqmitems,"TMB: ALCT - CLCT Time MatchingSynchronization",

  ["CSC/ALCT_Match_Time"]

 ,["CSC/LCT_Match_Status"]

 ,["CSC/LCT0_Match_BXN_Difference"]

 ,["CSC/LCT1_Match_BXN_Difference"]

)

csclayout(dqmitems,"TMB: Number of Words in TMB",

  ["CSC/TMB_Word_Count"]

)

csclayout(dqmitems,"TMB: ALCT0 KeyWiregroup vs CLCT0 Key DiStrip",

  ["CSC/CLCT0_KeyDiStrip_vs_ALCT0_KeyWiregroup"]

)
