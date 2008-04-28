
def csclayout(i, p, *rows): i["Layouts/CSC Layouts/" + p] = DQMItem(layout=rows)


csclayout(dqmitems,"EMU Test01 - DDUs in Readout",

  ["CSC/All_DDUs_in_Readout"]

 ,["CSC/All_DDUs_L1A_Increment"]

)

csclayout(dqmitems,"EMU Test02 - DDU Event Size",

  ["CSC/All_DDUs_Event_Size"]

)

csclayout(dqmitems,"EMU Test03 - DDU Reported Errors",

  ["CSC/All_DDUs_Trailer_Errors"]

)

csclayout(dqmitems,"EMU Test04 - DDU Format Errors",

  ["CSC/All_DDUs_Format_Errors"]

)

csclayout(dqmitems,"EMU Test05 - DDU Inputs Status",

  ["CSC/All_DDUs_Live_Inputs"]

 ,["CSC/All_DDUs_Inputs_with_Data"]

)

csclayout(dqmitems,"EMU Test06 - DDU Inputs in ERROR/WARNING State",

  ["CSC/All_DDUs_Inputs_Errors"]

 ,["CSC/All_DDUs_Inputs_Warnings"]

)

csclayout(dqmitems,"EMU Test08 - CSCs Reporting Data and Unpacked",

  ["CSC/CSC_Reporting"]

 ,["CSC/CSC_Unpacked_Fract"]

)

csclayout(dqmitems,"EMU Test10 - CSCs with Errors and Warnings (Fractions)",

  ["CSC/CSC_Format_Errors_Fract"]

 ,["CSC/CSC_Format_Warnings_Fract"]

 ,["CSC/CSC_DMB_input_fifo_full_Fract"]

 ,["CSC/CSC_DMB_input_timeout_Fract"]

)

csclayout(dqmitems,"EMU Test11 - CSCs without Data Blocks",

  ["CSC/CSC_wo_ALCT_Fract"]

 ,["CSC/CSC_wo_CLCT_Fract"]

 ,["CSC/CSC_wo_CFEB_Fract"]

)

