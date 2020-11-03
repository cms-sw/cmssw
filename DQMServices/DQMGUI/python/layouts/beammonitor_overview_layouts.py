from ..layouts.layout_manager import register_layout

register_layout(source='BeamMonitor/Fit/d0_phi0', destination='Collisions/BeamMonitorFeedBack/Collisions/BeamMonitorFeedBack/d0_phi0', name='00 - d0-phi0 of selected tracks')
register_layout(source='BeamMonitor/Fit/trk_z0', destination='Collisions/BeamMonitorFeedBack/Collisions/BeamMonitorFeedBack/trk_z0', name='01 - z0 of selected tracks')
register_layout(source='BeamMonitor/Fit/BeamMonitorFeedBack_x0', destination='Collisions/BeamMonitorFeedBack/Collisions/BeamMonitorFeedBack/BeamMonitorFeedBack_x0', name='02 - x position of beam spot')
register_layout(source='BeamMonitor/Fit/BeamMonitorFeedBack_y0', destination='Collisions/BeamMonitorFeedBack/Collisions/BeamMonitorFeedBack/BeamMonitorFeedBack_y0', name='03 - y position of beam spot')
register_layout(source='BeamMonitor/Fit/BeamMonitorFeedBack_z0', destination='Collisions/BeamMonitorFeedBack/Collisions/BeamMonitorFeedBack/BeamMonitorFeedBack_z0', name='04 - z position of beam spot')
register_layout(source='BeamMonitor/Fit/BeamMonitorFeedBack_sigmaZ0', destination='Collisions/BeamMonitorFeedBack/Collisions/BeamMonitorFeedBack/BeamMonitorFeedBack_sigmaZ0', name='05 - sigma z of beam spot')
register_layout(source='BeamMonitor/Fit/fitResults', destination='Collisions/BeamMonitorFeedBack/Collisions/BeamMonitorFeedBack/fitResults', name='06 - fit results beam spot')
register_layout(source='BeamPixel/fit results', destination='Collisions/BeamMonitorFeedBack/Collisions/BeamMonitorFeedBack/fit results', name='07 - Pixel-Vertices: Results of Beam Spot Fit')
register_layout(source='BeamPixel/muX vs lumi', destination='Collisions/BeamMonitorFeedBack/Collisions/BeamMonitorFeedBack/muX vs lumi', name='08 - Pixel-Vertices: X0 vs. Lumisection')
register_layout(source='BeamPixel/muY vs lumi', destination='Collisions/BeamMonitorFeedBack/Collisions/BeamMonitorFeedBack/muY vs lumi', name='09 - Pixel-Vertices: Y0 vs. Lumisection')
register_layout(source='BeamPixel/muZ vs lumi', destination='Collisions/BeamMonitorFeedBack/Collisions/BeamMonitorFeedBack/muZ vs lumi', name='10 - Pixel-Vertices: Z0 vs. Lumisection')
