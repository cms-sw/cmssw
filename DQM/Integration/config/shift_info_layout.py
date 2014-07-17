def shiftinfolayout(i, p, *rows): i["00 Shift/Info/" + p] = DQMItem(layout=rows)

shiftinfolayout(dqmitems, "00-InfoReportSummary",
  [{ 'path': "Info/EventInfo/reportSummaryMap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftInfo>Description and Instructions</a>" }])

shiftinfolayout(dqmitems, "01-InfoBeamMode",
  [{ 'path': "Info/LhcInfo/beamMode", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftInfo>Description and Instructions</a>" }])


