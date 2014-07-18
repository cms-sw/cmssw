def shifthlxlayout(i, p, *rows): i["00 Shift/HLX/" + p] = DQMItem(layout=rows)

shifthlxlayout(dqmitems, "Shifter HLX Lumi Summary",
  [ {'path':"HLX/Luminosity/LumiIntegratedEtSum", 'description':"<a href=https://twiki.cern.ch/twiki/bin/view/CMS/HLXLumiDQMLuminosityMonitor>Integrated Luminosity ET Sum.</a>"},
    {'path':"HLX/Luminosity/LumiIntegratedOccSet1",'description':"<a href=https://twiki.cern.ch/twiki/bin/view/CMS/HLXLumiDQMLuminosityMonitor>Integrated Luminosity Tower Occ Set 1.</a>"},
    {'path':"HLX/Luminosity/LumiIntegratedOccSet2",'description':"<a href=https://twiki.cern.ch/twiki/bin/view/CMS/HLXLumiDQMLuminosityMonitor>Integrated Luminosity Tower Occ Set 2.</a>"} ])

shifthlxlayout(dqmitems, "Shifter HLX Lumi - HF Lumi Comparison Summary",
  [ {'path':"HLX/HistoryLumi/HistInstantLumiEtSum",'description':"<a href=https://twiki.cern.ch/twiki/bin/view/CMS/HLXLumiDQMLumiHistoryMonitor>Et Sum luminosity history - integrated luminosity for each LS in the run to date.</a>"}],
  [ {'path':"HLX/HistoryLumi/HistInstantLumiOccSet1",'description':"<a href=https://twiki.cern.ch/twiki/bin/view/CMS/HLXLumiDQMLumiHistoryMonitor>Tower occupancy set 1 luminosity history - integrated luminosity for each LS in the run to date.</a>"},
    {'path':"HLX/HistoryLumi/HistInstantLumiOccSet2",'description':"<a href=https://twiki.cern.ch/twiki/bin/view/CMS/HLXLumiDQMLumiHistoryMonitor>Tower occupancy set 2 luminosity history - integrated luminosity for each LS in the run to date.</a>"}])

shifthlxlayout(dqmitems, "Shifter HLX Lumi Errors Summary",
  [ {'path':"HLX/HistoryLumi/HistInstantLumiEtSumError",'description':"<a href=https://twiki.cern.ch/twiki/bin/view/CMS/HLXLumiDQMLumiHistoryMonitor>Et Sum luminosity error history - error on the integrated luminosity for each LS in the run to date.</a>"}],
  [ {'path':"HLX/HistoryLumi/HistInstantLumiOccSet1Error",'description':"<a href=https://twiki.cern.ch/twiki/bin/view/CMS/HLXLumiDQMLumiHistoryMonitor>Tower occupancy set 1 luminosity error history - error on the integrated luminosity for each LS in the run to date.</a>"},
    {'path':"HLX/HistoryLumi/HistInstantLumiOccSet2Error",'description':"<a href=https://twiki.cern.ch/twiki/bin/view/CMS/HLXLumiDQMLumiHistoryMonitor>Tower occupancy set 2 luminosity error history - error on the integrated luminosity for each LS in the run to date.</a>"}])

shifthlxlayout(dqmitems, "Shifter HLX CheckSum Summary",
  [ {'path':"HLX/CheckSums/SumAllOccSet1",'description':"<a href=https://twiki.cern.ch/twiki/bin/view/CMS/HLXLumiDQMCellSumMonitor>Cell's alive per wedge, occupancy set 1.</a>"},
    {'path':"HLX/CheckSums/SumAllOccSet2",'description':"<a href=https://twiki.cern.ch/twiki/bin/view/CMS/HLXLumiDQMCellSumMonitor>Cell's alive per wedge, occupancy set 1.</a>"} ])


