def csclayout(i, p, *rows): i["CSC/Layouts/" + p] = DQMItem(layout=rows)
  
csclayout(dqmitems,"00 Top Physics Efficiency",
  	[{'path': "CSC/Summary/Physics_EMU", 'description': "CSC Efficiency for Physics. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#reportSummaryMap\">here</a>."}])

csclayout(dqmitems,"01 Station Physics Efficiency",
  	[{'path': "CSC/Summary/Physics_ME1", 'description': "EMU station ME1 status: physics efficiency by reporting area and hardware efficiency based on reporting number of hardware elements. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#Summary_ME\">here</a>."},
  	 {'path': "CSC/Summary/Physics_ME2", 'description': "EMU station ME2 status: physics efficiency by reporting area and hardware efficiency based on reporting number of hardware elements. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#Summary_ME\">here</a>."}],
  	[{'path': "CSC/Summary/Physics_ME3", 'description': "EMU station ME3 status: physics efficiency by reporting area and hardware efficiency based on reporting number of hardware elements. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#Summary_ME\">here</a>."},
  	 {'path': "CSC/Summary/Physics_ME4", 'description': "EMU station ME4 status: physics efficiency by reporting area and hardware efficiency based on reporting number of hardware elements. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#Summary_ME\">here</a>."}])

csclayout(dqmitems,"02 EMU Summary/EMU Test01 - DDUs in Readout",
  	[{'path': "CSC/Summary/All_DDUs_in_Readout", 'description': "Number of Events in DDU. If Readout and Trigger Enable were started in a correct sequence (first, Readout Enable and, then, Trigger Enable) and the rate of events with CSC payload present is not too high (<100 CSCs with data per second per DDU) then all DDUs should give the exact same number of events. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#All_DDUs_in_Readout\">here</a>."}],
	[{'path': "CSC/Summary/All_DDUs_L1A_Increment", 'description': "L1A increment from event for each DDU. If Readout and Trigger Enable were started in a correct sequence (first, Readout Enable and, then, Trigger Enable) and the rate of events with CSC payload present is not too high (<100 CSCs with data per second per DDU) L1A increment from event to event must be 1. However, when the rate goes up, DAQ-DQM may not be able to keep up with the data and are designed to skip events. Occasional skips (L1A>1) will inevitably happen at low rates due to Poisson nature of cosmic ray rates. Under no circumstances, the incremental L1A can be zero. There should be no entries in the bottom row of bins of the bottom histogram. One may also want to flag runs with extremely non-uniform response DDUs. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#All_DDUs_L1A_Increment\">here</a>."}])

csclayout(dqmitems,"02 EMU Summary/EMU Test03 - DDU Reported Errors",
  	[{'path': "CSC/Summary/All_DDUs_Trailer_Errors", 'description': "This histogram shows errors identified by DDU firmware and reported in the DDU trailer. Ideally, all entries should be in the bottom row of bins No Errors. However, this is not likely what one would typically see. Note that some of the errors are rather benign (e.g. #13 DAQ FIFO Near Full), while others are certainly very bad and may even require re-synchronization. The OR of all particularly bad errors in collected in bit #16 DDU Critical Error. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#All_DDUs_Trailer_Errors\">here</a>."}])

csclayout(dqmitems,"02 EMU Summary/EMU Test04 - DDU Format Errors",
  	[{'path': "CSC/Summary/All_DDUs_Format_Errors", 'description': " For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#All_DDUs_Format_Errors\">here</a>."}])

csclayout(dqmitems,"02 EMU Summary/EMU Test05 - DDU Inputs Status",
  	[{'path': "CSC/Summary/All_DDUs_Live_Inputs", 'description': "Inputs receiving a handshake from DMB (the handshake is obligatory, regardless of whether DMB has any data to report). If Readout and Trigger Enable were started in a correct sequence (first, Readout Enable and, then, Trigger Enable) and the rate of events with CSC payload present is not too high (<100 CSCs with data per second per DDU), the histogram should be absolutely flat for all DDUs and their enabled inputs. Otherwise histogram can show uneven response of DDUs when Readout was enabled after enabling the trigger (Test01 is even better suited to see this). This is not a normal sequence of run initialization. Under no circumstances, DDU inputs within one DDU may have different number of entries. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#All_DDUs_Live_Inputs\">here</a>."}],
	[{'path': "CSC/Summary/All_DDUs_Inputs_with_Data", 'description': "Inputs receiving DMB data. Is typically very uneven due to varying event rates from different chambers. The top example is typical of a single-CSC trigger. Particularly bad cases of hot chamber (chambers reporting data too frequently) can be easily spotted here - the bottom example illustrates such a problem. However, Test08 is much more suited for this. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#All_DDUs_Inputs_with_Data\">here</a>."}])

csclayout(dqmitems,"02 EMU Summary/EMU Test06 - DDU Inputs in ERROR-WARNING State",
  	[{'path': "CSC/Summary/All_DDUs_Inputs_Errors", 'description': "Inputs in ERROR State. Normally all entries should appear in the bottom raw. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#All_DDUs_Inputs_Errors\">here</a>."}],
	[{'path': "CSC/Summary/All_DDUs_Inputs_Warnings", 'description': "Inputs in WARNING State. Normally all entries should appear in the bottom raw. This histogram can be ignored by a shifter. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#All_DDUs_Inputs_Warnings\">here</a>."}])

csclayout(dqmitems,"02 EMU Summary/EMU Test08 - CSCs Reporting Data and Unpacked",
  	[{'path': "CSC/Summary/CSC_Reporting", 'description': "The occupancy histogram shows CSCs reporting data, regardless of whether data format was intact or broken. Chambers within one raw belong to one 360 - ring of chamber. Note that ME +/- 2/1, 3/1, and 4/1 rings have only 18 20 -chambers, while all others have 36 10 -chambers. The rainbow pattern of color within one row is due to directionality of cosmic rays. At LHC, all chambers within one row should have equal occupancy. One should check that there are no new empty cells (check for the list of currently disabled CSCs) there are no hot CSCs. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_Reporting\">here</a>."}],
	[{'path': "CSC/Summary/CSC_Unpacked_Fract", 'description': "Histogram shows unpacking efficiency. Gross problems at a scale of >10% inefficiency can be easily seen as deviations from the flat red color corresponding to 100% efficiency. Smaller scale problems can be chased using the EMU Test10 canvas. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_Unpacked_Fract\">here</a>."}])

csclayout(dqmitems,"02 EMU Summary/EMU Test10 - CSCs with Errors and Warnings (Fractions)",
  	[{'path': "CSC/Summary/CSC_Format_Errors_Fract", 'description': "Histogram shows frequency of format errors per CSC record. Pay attention to the temperature scale (it changes from run to run and during a run according to the histogram content). CSCs reporting format errors in more than 1% of events should be flagged. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_Format_Errors_Fract\">here</a>."},
	 {'path': "CSC/Summary/CSC_L1A_out_of_sync_Fract", 'description': "(no description). For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_L1A_out_of_sync\">here</a>."}],
	[{'path': "CSC/Summary/CSC_DMB_input_fifo_full_Fract", 'description': "Shows a frequency of FIFO-FULL condition on DMB inputs (OR of 7 FIFOs: 5 CFEBs, ALCT, TMB). Appearance of entries in this histogram is very bad and would typically imply a loss of synchronization, even if FIFO-FULL condition clears away. To dig out which of the 7 boards is actually responsible for the trouble, one needs to refer the FEB Status (Timeouts, FIFO, L1 pipe) canvas for the offensive chamber (this canvas can be found in the DMB group) For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_DMB_input_fifo_full_Fract\">here</a>."},
	 {'path': "CSC/Summary/CSC_DMB_input_timeout_Fract", 'description': "Shows a frequency of a TIMEOUT condition on DMB inputs (OR start/stop timeouts for 5 CFEBs, ALCT, TMB). Appearance of entries in this histogram is very bad and typically implies badly timed-in CSCs. To dig out which of the 7 boards is actually responsible for the trouble, one needs to refer the FEB Status (Timeouts, FIFO, L1 pipe) canvas for the offensive chamber (this canvas can be found in the DMB group) For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_DMB_input_timeout_Fract\">here</a>."}])

csclayout(dqmitems,"02 EMU Summary/EMU Test11 - CSCs without Data Blocks",
  	[{'path': "CSC/Summary/CSC_wo_ALCT_Fract", 'description': "Histogram shows how often CSC events come without ALCT data. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_wo_ALCT_Fract\">here</a>."},
	 {'path': "CSC/Summary/CSC_wo_CLCT_Fract", 'description': "Histogram shows how often CSC events come without CLCT data. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_wo_CLCT_Fract\">here</a>."}],
	[{'path': "CSC/Summary/CSC_wo_CFEB_Fract", 'description': "Histogram shows how often CSC events come without CFEB data. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_wo_CFEB_Fract\">here</a>."},
	 {'path': "CSC/Summary/CSC_Format_Warnings_Fract", 'description': "Histogram shows occurrences when SCA cells were filled due to too-high rate of LCTs and/or LCT-L1A coincidences. In conditions of cosmic ray runs, appearance of entries is indicative of hardware problems, or more specifically hot CFEBs. This typically happens due to a loose CFEB-TMB cable generating a flood of CLCT pre-triggers (CBEBs are the only board that are readout on coincidence between pre-CLCT and L1A). For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_Format_Warnings_Fract\">here</a>."}])

csclayout(dqmitems,"03 Shifter/Chamber Errors and Warnings (Statistically Significant)",
  	[{'path': "CSC/Summary/CSC_STATS_format_err", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_format_err\">here</a>."},
  	 {'path': "CSC/Summary/CSC_STATS_l1sync_err", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_l1sync_err\">here</a>."}],
  	[{'path': "CSC/Summary/CSC_STATS_fifofull_err", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_fifofull_err\">here</a>."},
  	 {'path': "CSC/Summary/CSC_STATS_inputto_err", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_inputto_err\">here</a>."}])

csclayout(dqmitems,"03 Shifter/Chamber Occupancy Exceptions (Statistically Significant)",
  	[{'path': "CSC/Summary/CSC_STATS_occupancy", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_occupancy\">here</a>."}])

csclayout(dqmitems,"03 Shifter/Chambers without Data (Statistically Significant)",
  	[{'path': "CSC/Summary/CSC_STATS_wo_alct", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_wo_alct\">here</a>."},
  	 {'path': "CSC/Summary/CSC_STATS_wo_clct", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_wo_clct\">here</a>."}],
  	[{'path': "CSC/Summary/CSC_STATS_wo_cfeb", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_wo_cfeb\">here</a>."},
  	 {'path': "CSC/Summary/CSC_STATS_cfeb_bwords", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_cfeb_bwords\">here</a>."}])

csclayout(dqmitems,"04 Timing/00 ALCT Timing",
        [{'path': "CSC/Summary/CSC_ALCT0_BXN_mean", 'description': "ALCT0 BXN Mean"},
         {'path': "CSC/Summary/Plus_endcap_ALCT0_dTime", 'description': "Plus Endcap ALCT0 BXN - ALCT_L1A BXN Difference"}],
        [{'path': "CSC/Summary/CSC_ALCT0_BXN_rms", 'description': "ALCT0 BXN RMS"},
         {'path': "CSC/Summary/Minus_endcap_ALCT0_dTime", 'description': "Minus Endcap ALCT0 BXN - ALCT_L1A BXN Difference"}])

csclayout(dqmitems,"04 Timing/01 CLCT Timing",
        [{'path': "CSC/Summary/CSC_CLCT0_BXN_mean", 'description': "CLCT0 BXN Mean"},
         {'path': "CSC/Summary/Plus_endcap_CLCT0_dTime", 'description': "Plus Endcap CLCT0 BXN - CLCT_L1A BXN Difference"}],
        [{'path': "CSC/Summary/CSC_CLCT0_BXN_rms", 'description': "CLCT0 BXN RMS"},
         {'path': "CSC/Summary/Minus_endcap_CLCT0_dTime", 'description': "Minus Endcap CLCT0 BXN - CLCT_L1A BXN Difference"}])
  
csclayout(dqmitems,"04 Timing/02 AFEB RawHits Timing",
        [{'path': "CSC/Summary/CSC_AFEB_RawHits_Time_mean", 'description': "AFEB RawHits Time Mean"},
         {'path': "CSC/Summary/Plus_endcap_AFEB_RawHits_Time", 'description': "Plus Endcap AFEB RawHits Time Bins Distribution"}],
        [{'path': "CSC/Summary/CSC_AFEB_RawHits_Time_rms", 'description': "AFEB RawHits Time RMS"},
         {'path': "CSC/Summary/Minus_endcap_AFEB_RawHits_Time", 'description': "Minus Endcap AFEB RawHits Time Bins Distribution"}])
  
csclayout(dqmitems,"04 Timing/03 CFEB Comparator Hits Timing",
        [{'path': "CSC/Summary/CSC_CFEB_Comparators_Time_mean", 'description': "CFEB Comparator Hits Time Mean"},
         {'path': "CSC/Summary/Plus_endcap_CFEB_Comparators_Time", 'description': "Plus Endcap CFEB Comparator Hits Time Bin Distribution"}],
        [{'path': "CSC/Summary/CSC_CFEB_Comparators_Time_rms", 'description': "CFEB Comparator Hits Time RMS"},
         {'path': "CSC/Summary/Minus_endcap_CFEB_Comparators_Time", 'description': "Minus Endcap CFEB Comparator Hits Time Bin Distribution"}])
  
csclayout(dqmitems,"04 Timing/04 CFEB SCA Cell Peak Timing",
        [{'path': "CSC/Summary/CSC_CFEB_SCA_CellPeak_Time_mean", 'description': "CFEB SCA Cell Peak Time Mean"},
         {'path': "CSC/Summary/Plus_endcap_CFEB_SCA_CellPeak_Time", 'description': "Plus Endcap CFEB SCA Cell Peak Time Bin Distribution"}],
        [{'path': "CSC/Summary/CSC_CFEB_SCA_CellPeak_Time_rms", 'description': "CFEB SCA Cell Peak Time RMS"},
         {'path': "CSC/Summary/Minus_endcap_CFEB_SCA_CellPeak_Time", 'description': "Minus Endcap CFEB SCA Cell Peak Time Bin Distribution"}])
  
csclayout(dqmitems,"04 Timing/05 ALCT-CLCT Match Timing",
        [{'path': "CSC/Summary/CSC_ALCT_CLCT_Match_mean", 'description': "ALCT-CLCT Match Timing Mean"},
         {'path': "CSC/Summary/Plus_endcap_CFEB_SCA_CellPeak_Time", 'description': "Plus Endcap ALCT-CLCT Match Time Bin Distribution"}],
        [{'path': "CSC/Summary/CSC_ALCT_CLCT_Match_rms", 'description': "ALCT-CLCT Match Timing RMS"},
         {'path': "CSC/Summary/Minus_endcap_ALCT_CLCT_Match_Time", 'description': "Minus Endcap ALCT-CLCT Match Time Bin Distribution"}])
                                                                                                                                                                                                                     
csclayout(dqmitems,"05 EventDisplay/01 Event Display in Z-R projection",
        [{'path': "CSC/Summary/Event_Display_Anode", 'description': "Event Display in Z-R projection (wiregroups and half-strips)"}])

csclayout(dqmitems,"05 EventDisplay/02 Event Display in Z-Phi projection)",
        [{'path': "CSC/Summary/Event_Display_Cathode", 'description': "Event Display in Z-Phi projection (strips)"}])

csclayout(dqmitems,"05 EventDisplay/03 Event Display in X-Y projection",
        [{'path': "CSC/Summary/Event_Display_XY", 'description': "Event Display in X-Y projection (wiregroups and half-strips)"}])

csclayout(dqmitems,"06 Physics Efficiency - RecHits Minus",
        [{'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalm1", 'description': "Histogram shows 2D RecHits distribution in ME-1. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."},
        {'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalm2", 'description': "Histogram shows 2D RecHits distribution in ME-2. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauh/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."}],
        [{'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalm3", 'description': "Histogram shows 2D RecHits distribution in ME-3. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."},
        {'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalm4", 'description': "Histogram shows 2D RecHits distribution in ME-4. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauh/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."}
        ]
        )

csclayout(dqmitems,"07 Physics Efficiency - RecHits Plus",
        [{'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalp1", 'description': "Histogram shows 2D RecHits distribution in ME+1. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."},
        {'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalp2", 'description': "Histogram shows 2D RecHits distribution in ME+2. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."}],
        [{'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalp3", 'description': "Histogram shows 2D RecHits distribution in ME+3. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewath/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."},
        {'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalp4", 'description': "Histogram shows 2D RecHits distribution in ME+4. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."}
        ]
        )
