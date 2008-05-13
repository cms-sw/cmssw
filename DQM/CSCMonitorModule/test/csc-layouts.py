def csclayout(i, p, *rows): i["CSC/Summary/Layouts/" + p] = DQMItem(layout=rows)

csclayout(dqmitems,"EMU Summary/EMU Test01 - DDUs in Readout",
        ["CSC/Summary/All_DDUs_in_Readout"],
        ["CSC/Summary/All_DDUs_L1A_Increment"])

csclayout(dqmitems,"EMU Summary/EMU Test03 - DDU Reported Errors",
  	["CSC/Summary/All_DDUs_Trailer_Errors"])

csclayout(dqmitems,"EMU Summary/EMU Test04 - DDU Format Errors",
  	["CSC/Summary/All_DDUs_Format_Errors"])

csclayout(dqmitems,"EMU Summary/EMU Test05 - DDU Inputs Status",
  	["CSC/Summary/All_DDUs_Live_Inputs"],
	["CSC/Summary/All_DDUs_Inputs_with_Data"])

csclayout(dqmitems,"EMU Summary/EMU Test06 - DDU Inputs in ERROR-WARNING State",
  	["CSC/Summary/All_DDUs_Inputs_Errors"],
	["CSC/Summary/All_DDUs_Inputs_Warnings"])

csclayout(dqmitems,"EMU Summary/EMU Test08 - CSCs Reporting Data and Unpacked",
  	["CSC/Summary/CSC_Reporting"],
	["CSC/Summary/CSC_Unpacked_Fract"])

csclayout(dqmitems,"EMU Summary/EMU Test10 - CSCs with Errors and Warnings (Fractions)",
  	["CSC/Summary/CSC_Format_Errors_Fract",
	 "CSC/Summary/CSC_Format_Warnings_Fract"],
	["CSC/Summary/CSC_DMB_input_fifo_full_Fract",
	 "CSC/Summary/CSC_DMB_input_timeout_Fract"])

csclayout(dqmitems,"EMU Summary/EMU Test11 - CSCs without Data Blocks",
  	["CSC/Summary/CSC_wo_ALCT_Fract",
	 "CSC/Summary/CSC_wo_CLCT_Fract"],
	["CSC/Summary/CSC_wo_CFEB_Fract",
         ""])

