# Dummy check of python syntax within file when run stand-alone
if __name__=="__main__":
    class DQMItem:
        def __init__(self,layout):
            print layout

    dqmitems={}

def hcallayout(i, p, *rows): i["Hcal/Layouts/" + p] = DQMItem(layout=rows)

hcallayout(dqmitems, "01 HCAL Summaries",
           [{ 'path':"Hcal/EventInfo/reportSummaryMapShift",
             'description':"This shows the fraction of bad cells in each subdetector.  All subdetectors should appear green.  Values should all be above 98%."}]
           )

hcallayout(dqmitems, "02 HCAL Digi Problems",
          [{ 'path': "Hcal/DigiMonitor_Hcal/problem_digis/HB HE HF Depth 1  Problem Digi Rate",
             'description': "A digi cell is considered bad if the capid rotation for that digi was incorrect, if the digi's data valid/error flags are incorrect, or if there is an IDLE-BCN mismatch.  Currently, only digis with IDLE-BCN mismatches are displayed.  This plot is over HB HE HF depth 1. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },
           { 'path': "Hcal/DigiMonitor_Hcal/problem_digis/HB HE HF Depth 2  Problem Digi Rate",
             'description': "A digi cell is considered bad if the capid rotation for that digi was incorrect, if the digi's data valid/error flags are incorrect, or if there is an IDLE-BCN mismatch.  Currently, only digis with IDLE-BCN mismatches are displayed.  This plot is over HB HE HF depth 2. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }],
          [{ 'path': "Hcal/DigiMonitor_Hcal/problem_digis/HE Depth 3  Problem Digi Rate",
             'description': "A digi cell is considered bad if the capid rotation for that digi was incorrect, if the digi's data valid/error flags are incorrect, or if there is an IDLE-BCN mismatch.  Currently, only digis with IDLE-BCN mismatches are displayed.  This plot is over HE depth 3. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },
           { 'path': "Hcal/DigiMonitor_Hcal/problem_digis/HO Depth 4  Problem Digi Rate",
             'description': "A digi cell is considered bad if the capid rotation for that digi was incorrect, if the digi's data valid/error flags are incorrect, or if there is an IDLE-BCN mismatch.  Currently, only digis with IDLE-BCN mismatches are displayed.  This plot is over HO depth 4. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }],
           [{ 'path': "Hcal/DigiMonitor_Hcal/bad_digis/1D_digi_plots/HBHEHF Bad Quality Digis vs LB",
              'description': "Total number of bad digis found in HBHEHF vs luminosity section"},
            { 'path': "Hcal/DigiMonitor_Hcal/bad_digis/1D_digi_plots/HO Bad Quality Digis vs LB",
              'description': "Total number of bad digis found in HO vs luminosity section"}]
           )

hcallayout(dqmitems, "03 HCAL Dead Cell Check",
          [{ 'path': "Hcal/DeadCellMonitor_Hcal/problem_deadcells/HB HE HF Depth 1  Problem Dead Cell Rate",
             'description': "Potential dead cell candidates in HB HE HF depth 1. Seriously dead if dead for >5% of a full run. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },
           { 'path': "Hcal/DeadCellMonitor_Hcal/problem_deadcells/HB HE HF Depth 2  Problem Dead Cell Rate",
             'description': "Potential dead cell candidates in HB HE HF depth 2. Seriously dead if dead for >5% of a full run. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }],
          [{ 'path': "Hcal/DeadCellMonitor_Hcal/problem_deadcells/HE Depth 3  Problem Dead Cell Rate",
             'description': "Potential dead cell candidates in HE depth 3. Seriously dead if dead for >5% of a full run. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },
           { 'path': "Hcal/DeadCellMonitor_Hcal/problem_deadcells/HO Depth 4  Problem Dead Cell Rate",
             'description': "Potential dead cell candidates in HO depth 4. Seriously dead if dead for >5% of a full run. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }],
           [{ 'path': "Hcal/DeadCellMonitor_Hcal/TotalDeadCells_HBHEHF_vs_LS",
             'description': "Total number of dead cells found in HBHEHF vs luminosity section"},
            { 'path': "Hcal/DeadCellMonitor_Hcal/TotalDeadCells_HO_vs_LS",
             'description': "Total number of dead cells found in HO vs luminosity section"}]
           )

hcallayout(dqmitems, "04 HCAL Hot Cell Check",
          [{ 'path': "Hcal/HotCellMonitor_Hcal/problem_hotcells/HB HE HF Depth 1  Problem Hot Cell Rate",
                   'description': "A cell is considered potentially hot if it is above some threshold energy, or if it is persistently below some (lower) threshold enery for a number of consecutive events. Seriously hot if hot for >5% of a full run. All depths. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },
           { 'path': "Hcal/HotCellMonitor_Hcal/problem_hotcells/HB HE HF Depth 2  Problem Hot Cell Rate",
                   'description': "A cell is considered potentially hot if it is above some threshold energy, or if it is persistently below some (lower) threshold enery for a number of consecutive events. Seriously hot if hot for >5% of a full run. All depths. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }],
          [{ 'path': "Hcal/HotCellMonitor_Hcal/problem_hotcells/HE Depth 3  Problem Hot Cell Rate",
                   'description': "A cell is considered potentially hot if it is above some threshold energy, or if it is persistently below some (lower) threshold enery for a number of consecutive events. Seriously hot if hot for >5% of a full run. All depths. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },
           { 'path': "Hcal/HotCellMonitor_Hcal/problem_hotcells/HO Depth 4  Problem Hot Cell Rate",
                   'description': "A cell is considered potentially hot if it is above some threshold energy, or if it is persistently below some (lower) threshold enery for a number of consecutive events. Seriously hot if hot for >5% of a full run. All depths. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }],
           [{ 'path': "Hcal/HotCellMonitor_Hcal/TotalHotCells_HBHEHF_vs_LS",
              'description': "Total number of hot cells found in HBHEHF vs luminosity section"},
            { 'path': "Hcal/HotCellMonitor_Hcal/TotalHotCells_HO_vs_LS",
              'description': "Total number of hot cells found in HO vs luminosity section"}]
           )

hcallayout(dqmitems, "05 HCAL Raw Data",
           [{ 'path': "Hcal/RawDataMonitor_Hcal/problem_rawdata/HB HE HF Depth 1  Problem Raw Data Rate",
              'description': "A Raw Data error indicates that the data received from this channel was somehow corrupted or compromised.  This plot is over HB HE HF depth 1. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },
            { 'path': "Hcal/RawDataMonitor_Hcal/problem_rawdata/HB HE HF Depth 2  Problem Raw Data Rate",
              'description': "A Raw Data error indicates that the data received from this channel was somehow corrupted or compromised.  This plot is over HB HE HF depth 2. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }],
           [{ 'path': "Hcal/RawDataMonitor_Hcal/problem_rawdata/HE Depth 3  Problem Raw Data Rate",
              'description': "A Raw Data error indicates that the data received from this channel was somehow corrupted or compromised.  This plot is over HE depth 3. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },
            { 'path': "Hcal/RawDataMonitor_Hcal/problem_rawdata/HO Depth 4  Problem Raw Data Rate",
              'description': "A Raw Data error indicates that the data received from this channel was somehow corrupted or compromised.  This plot is over HO depth 4. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }],
           [{ 'path': "Hcal/RawDataMonitor_Hcal/Total_RAW_Problems_HBHEHF_vs_LS",
              'description': "Total number of raw data errors in HBHEHF vs luminosity section"}
            ]
           )

hcallayout(dqmitems, "06 HCAL Trigger Primitives",
           [{ 'path': "Hcal/TrigPrimMonitor_Hcal/ ProblemTriggerPrimitives",
              'description': "See details at: <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }]
           )

hcallayout(dqmitems, "07 HCAL Pedestal Problems",
           [{'path':"Hcal/CoarsePedestalMonitor_Hcal/ ProblemCoarsePedestals",
             'description': "See details at: <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }]
           )

hcallayout(dqmitems, "08 HCAL Lumi Problems",
           [{'path': "Hcal/BeamMonitor_Hcal/ Problem BeamMonitor",
             'description':"This shows problems only in the sections of HF used for luminosity monitoring.  Channels that are hot or dead are considered as problems, where the definitions of 'hot' and 'dead' are slightly different than in the normal HCAL monitors.  More details at  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }]
           )

hcallayout(dqmitems, "09 HCAL Calibration Type",
           [{'path':"Hcal/HcalInfo/CalibrationType",
             'description':"This shows the distribution of HCAL event types received by DQM.  Calibration events (pedestal, laser, etc.) are used for additional monitoring and diagnostics."}])

hcallayout(dqmitems, "10 HCAL Error Thresholds",
           [{'path':"Hcal/HcalInfo/SummaryClientPlots/MinErrorRate",
             'description':"This shows the fraction of events that must be bad in each task to be counted as a problem by reportSummary."}
            ])

hcallayout(dqmitems, "11 ZDC Rechit Energies",
           [{'path':"Hcal/ZDCMonitor/ZDCMonitor_Hcal/2D_RecHitEnergy",
             'description':"This shows the map of rechit mean energy depositions in ZDC. Should show a uniform distribution of energy in the EM sections of ZDC (bottom five rows), followed by a peak in the first 2 rows of HAD sections. No empty sections in the detector"}
            ])

hcallayout(dqmitems, "12 ZDC Rechit Timing",
           [{'path':"Hcal/ZDCMonitor/ZDCMonitor_Hcal/2D_RecHitTime",
             'description':"This shows the map of  mean rechit time in ZDC. The scale on the histogram should be in the range between 19-24 nanoseconds (5 ns)"}
            ])
