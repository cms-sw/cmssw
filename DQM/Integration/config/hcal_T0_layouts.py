# Dummy check of python syntax within file when run stand-alone
if __name__=="__main__":
    class DQMItem:
        def __init__(self,layout):
            print layout

    dqmitems={}



def hcallayout(i, p, *rows): i["Hcal/Layouts/" + p] = DQMItem(layout=rows)

hcallayout(dqmitems, "01 HCAL Summaries",
           [{ 'path':"Hcal/EventInfo/reportSummaryMap",
             'description':"This shows the fraction of bad cells in each subdetector.  All subdetectors should appear green"}]
           )

hcallayout(dqmitems, "02 HCAL Events Processed",
          [{ 'path': "Hcal/HcalInfo/EventsInHcalMonitorModule",
             'description': "This histogram counts the total events seen by this process." }]
           )

hcallayout(dqmitems, "03 HCAL Sufficient Events",
          [{ 'path': "Hcal/HcalInfo/SummaryClientPlots/EnoughEvents",
             'description': "This histogram indicates whether the individual tasks have produced enough events to make a proper evaluation of reportSummary values.  All individual tasks should have 'true' values (1).  HcalMonitorModule may be either 1 or 0. " }]
           )

hcallayout(dqmitems, "04 HCAL Raw Data",
            [{ 'path': "Hcal/RawDataMonitor_Hcal/problem_rawdata/HB HE HF Depth 1  Problem Raw Data Rate",
              'description': "A Raw Data error indicates that the data received from this channel was somehow corrupted or compromised.  This plot is over HB HE HF depth 1. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },
            { 'path': "Hcal/RawDataMonitor_Hcal/problem_rawdata/HB HE HF Depth 2  Problem Raw Data Rate",
              'description': "A Raw Data error indicates that the data received from this channel was somehow corrupted or compromised.  This plot is over HB HE HF depth 2. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }],
           [{ 'path': "Hcal/RawDataMonitor_Hcal/problem_rawdata/HE Depth 3  Problem Raw Data Rate",
              'description': "A Raw Data error indicates that the data received from this channel was somehow corrupted or compromised.  This plot is over HE depth 3. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" },
            { 'path': "Hcal/RawDataMonitor_Hcal/problem_rawdata/HO Depth 4  Problem Raw Data Rate",
              'description': "A Raw Data error indicates that the data received from this channel was somehow corrupted or compromised.  This plot is over HO depth 4. iPhi (1 to 72) by iEta (-41 to 41) More at <a href=https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMHistograms>HcalDQMHistograms</a>" }]
           )

hcallayout(dqmitems, "05 HCAL Digi Problems",
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

hcallayout(dqmitems, "06 HCAL Dead Cell Check",
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

hcallayout(dqmitems, "07 HCAL Hot Cell Check",
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
