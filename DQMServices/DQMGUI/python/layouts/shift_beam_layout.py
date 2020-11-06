from ..layouts.layout_manager import register_layout

register_layout(source='BeamMonitor/EventInfo/reportSummaryMap', destination='00 Shift/BeamMonitor/reportSummaryMap', name='00 - BeamMonitor ReportSummary', description='BeamSpot summary map', overlay='')
register_layout(source='BeamMonitor/Fit/d0_phi0', destination='00 Shift/BeamMonitor/d0_phi0', name='01 - d0-phi0 of selected tracks', description='d0-phi0 correlation of selected tracks -  href=https://twiki.cern.ch/twiki/bin/view/CMS/BeamMonitorOnlineDQMInstructions>BeamMonitorOnlineDQMInstructions ', overlay='')
register_layout(source='BeamMonitor/Fit/trk_z0', destination='00 Shift/BeamMonitor/trk_z0', name='02 - z0 of selected tracks', description='Z0 distribution of selected tracks  -  href=https://twiki.cern.ch/twiki/bin/view/CMS/BeamMonitorOnlineDQMInstructions>BeamMonitorOnlineDQMInstructions ', overlay='')
register_layout(source='BeamMonitor/Fit/fitResults', destination='00 Shift/BeamMonitor/fitResults', name='03 - fit results beam spot', description='d_{0}-#phi correlation fit results of beam spot', overlay='')
register_layout(source='BeamMonitor/Fit/x0_lumi', destination='00 Shift/BeamMonitor/x0_lumi', name='04 - fitted x0, sigma(x0) vs LS', description='x coordinate of beamspot vs LS', overlay='')
register_layout(source='BeamMonitor/Fit/sigmaX0_lumi', destination='00 Shift/BeamMonitor/sigmaX0_lumi', name='04 - fitted x0, sigma(x0) vs LS', description='sigma X of beamspot vs LS', overlay='')
register_layout(source='BeamMonitor/Fit/y0_lumi', destination='00 Shift/BeamMonitor/y0_lumi', name='05 - fitted y0, sigma(y0) vs LS', description='y coordinate of beamspot vs LS', overlay='')
register_layout(source='BeamMonitor/Fit/sigmaY0_lumi', destination='00 Shift/BeamMonitor/sigmaY0_lumi', name='05 - fitted y0, sigma(y0) vs LS', description='sigma Y of beamspot vs LS', overlay='')
register_layout(source='BeamMonitor/Fit/z0_lumi', destination='00 Shift/BeamMonitor/z0_lumi', name='06 - fitted z0, sigma(z0) vs LS', description='z coordinate of beamspot vs LS', overlay='')
register_layout(source='BeamMonitor/Fit/sigmaZ0_lumi', destination='00 Shift/BeamMonitor/sigmaZ0_lumi', name='06 - fitted z0, sigma(z0) vs LS', description='sigma Z of beamspot vs LS', overlay='')
