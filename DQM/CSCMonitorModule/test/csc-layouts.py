def csclayout(i, p, *rows): i["Layouts/CSC Layouts/" + p] = DQMItem(layout=rows)

csclayout(dqmitems,"EMU Test00 - Readout Buffer Errors",
["EMU/Common/h2_EMU_All_Readout_Errors"]
)

csclayout(dqmitems,"EMU Test01 - DDUs in Readout",
["EMU/Common/h1_EMU_All_DDUs_in_Readout"],
["EMU/Common/h2_EMU_All_DDUs_L1A_Increment"]
)

csclayout(dqmitems,"EMU Test02 - DDU Event Size",
["EMU/Common/h2_EMU_All_DDUs_Event_Size"],
["EMU/Common/hp_EMU_All_DDUs_Average_Event_Size"]
)

csclayout(dqmitems,"EMU Test03 - DDU Reported Errors",
["EMU/Common/h2_EMU_All_DDUs_Trailer_Errors"]
)

csclayout(dqmitems,"EMU Test04 - DDU Format Errors",
["EMU/Common/h2_EMU_All_DDUs_Format_Errors"]
)

csclayout(dqmitems,"EMU Test05 - DDU Live Inputs",
["EMU/Common/h2_EMU_All_DDUs_Live_Inputs"],
["EMU/Common/hp_EMU_All_DDUs_Average_Live_Inputs"]
)

csclayout(dqmitems,"EMU Test06 - DDU Inputs in ERROR/WARNING State",
["EMU/Common/h2_EMU_All_DDUs_Inputs_Errors"],
["EMU/Common/h2_EMU_All_DDUs_Inputs_Warnings"]
)

csclayout(dqmitems,"EMU Test07 - DDU Inputs with Data",
["EMU/Common/h2_EMU_All_DDUs_Inputs_with_Data"],
["EMU/Common/hp_EMU_All_DDUs_Average_Inputs_with_Data"]
)

csclayout(dqmitems,"EMU Test08a - Unpacked DMBs",
["EMU/Common/h2_EMU_DMB_Unpacked"]
)

csclayout(dqmitems,"EMU Test08b - Unpacked CSCs",
["EMU/Common/h2_EMU_CSC_Unpacked"]
)

csclayout(dqmitems,"EMU Test09a - DMBs with Format Errors",
["EMU/Common/h2_EMU_DMB_Format_Errors"]
)

csclayout(dqmitems,"EMU Test09b - CSCs with Format Errors",
["EMU/Common/h2_EMU_CSC_Format_Errors"]
)

csclayout(dqmitems,"EMU Test10a - DMBs with Format Errors (Fractions)",
["EMU/Common/h2_EMU_DMB_Format_Errors_Fract"]
)

csclayout(dqmitems,"EMU Test10b - CSCs with Format Errors (Fractions)",
["EMU/Common/h2_EMU_CSC_Format_Errors_Fract"]
)

csclayout(dqmitems,"EMU Test11 - DMBs with CFEB B-Words",
["EMU/Common/h2_EMU_DMB_Format_Warnings"]
)

csclayout(dqmitems,"EMU Test12 - CSCs with CFEB B-Words (Fraction)",
["EMU/Common/h2_EMU_CSC_Format_Warnings_Fract"]
)

