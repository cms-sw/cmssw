def shiftbeamlayout(i, p, *rows): i["00 Shift/BeamMonitor/" + p] = DQMItem(layout=rows)

shiftbeamlayout(dqmitems, "00 - BeamMonitor ReportSummary",
 [{ 'path': "BeamMonitor/EventInfo/reportSummaryMap",
    'description': "BeamSpot summary map"}])
shiftbeamlayout(dqmitems, "01 - d0-phi0 of selected tracks",
 [{ 'path': "BeamMonitor/Fit/d0_phi0",
    'description': "d0-phi0 correlation of selected tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/BeamMonitorOnlineDQMInstructions>BeamMonitorOnlineDQMInstructions</a> "}])
shiftbeamlayout(dqmitems, "02 - z0 of selected tracks",
 [{ 'path': "BeamMonitor/Fit/trk_z0",
    'description': "Z0 distribution of selected tracks  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/BeamMonitorOnlineDQMInstructions>BeamMonitorOnlineDQMInstructions</a> "}])
shiftbeamlayout(dqmitems, "03 - fit results beam spot",
 [{ 'path': "BeamMonitor/Fit/fitResults",
    'description': "d_{0}-#phi correlation fit results of beam spot"}])
shiftbeamlayout(dqmitems, "04 - fitted x0, sigma(x0) vs LS",
 [{ 'path': "BeamMonitor/Fit/x0_lumi",'description': "x coordinate of beamspot vs LS"}],
 [{ 'path': "BeamMonitor/Fit/sigmaX0_lumi",'description': "sigma X of beamspot vs LS"}])
shiftbeamlayout(dqmitems, "05 - fitted y0, sigma(y0) vs LS",
 [{ 'path': "BeamMonitor/Fit/y0_lumi",'description': "y coordinate of beamspot vs LS"}],
 [{ 'path': "BeamMonitor/Fit/sigmaY0_lumi",'description': "sigma Y of beamspot vs LS"}])
shiftbeamlayout(dqmitems, "06 - fitted z0, sigma(z0) vs LS",
 [{ 'path': "BeamMonitor/Fit/z0_lumi",'description': "z coordinate of beamspot vs LS"}],
 [{ 'path': "BeamMonitor/Fit/sigmaZ0_lumi",'description': "sigma Z of beamspot vs LS"}])

