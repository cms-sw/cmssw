def csclayout(i, p, *rows): i["CSC/Layouts/" + p] = DQMItem(layout=rows)
  
  csclayout(dqmitems,"EMU00 Summary/EMU Test02 - DDU Event Size",
    ["CSC/All_DDUs_Event_Size"],
    ["CSC/All_DDUs_Average_Event_Size"]
  )
  csclayout(dqmitems,"EMU00 Summary/EMU Test03 - DDU Reported Errors",
    ["CSC/All_DDUs_Trailer_Errors"]
  )
  csclayout(dqmitems,"EMU00 Summary/EMU Test04 - DDU Format Errors",
    ["CSC/All_DDUs_Format_Errors"]
  )
  csclayout(dqmitems,"EMU00 Summary/EMU Test05 - DDU Inputs Status",
    ["CSC/All_DDUs_Live_Inputs"],
    ["CSC/All_DDUs_Inputs_with_Data"]
  )
  csclayout(dqmitems,"EMU00 Summary/EMU Test06 - DDU Inputs in ERROR/WARNING State",
    ["CSC/All_DDUs_Inputs_Errors"],
    ["CSC/All_DDUs_Inputs_Warnings"]
  )
  csclayout(dqmitems,"EMU00 Summary/EMU Test08 - CSCs Reporting Data and Unpacked",
    ["CSC/CSC_Reporting"],
    ["CSC/CSC_Unpacked_Fract"]
  )
  csclayout(dqmitems,"EMU00 Summary/EMU Test10 - CSCs with Errors and Warnings (Fractions)",
    ["CSC/CSC_Format_Errors_Fract",
     "CSC/CSC_Format_Warnings_Fract"],
    ["CSC/CSC_DMB_input_fifo_full_Fract",
     "CSC/CSC_DMB_input_timeout_Fract"]
  )
  csclayout(dqmitems,"EMU00 Summary/EMU Test11 - CSCs without Data Blocks",
    ["CSC/CSC_wo_ALCT_Fract",
     "CSC/CSC_wo_CLCT_Fract"],
    ["CSC/CSC_wo_CFEB_Fract"]
  )
  
