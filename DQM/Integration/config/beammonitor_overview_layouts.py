def bmoverviewlayout(i, p, *rows): i["Collisions/BeamMonitorFeedBack/" + p] = DQMItem(layout=rows)

bmoverviewlayout(dqmitems, "00 - d0-phi0 of selected tracks",
                 [{ 'path': "BeamMonitor/Fit/d0_phi0",
                    'description': "d0-phi0 correlation of selected tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/BeamMonitorOnlineDQMInstructions>BeamMonitorOnlineDQMInstructions</a> "}])
bmoverviewlayout(dqmitems, "01 - z0 of selected tracks",
                 [{ 'path': "BeamMonitor/Fit/trk_z0",
                    'description': "Z0 distribution of selected tracks  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/BeamMonitorOnlineDQMInstructions>BeamMonitorOnlineDQMInstructions</a> "}])
bmoverviewlayout(dqmitems, "02 - x position of beam spot",
                 [{ 'path': "BeamMonitor/Fit/BeamMonitorFeedBack_x0",
                    'description': "x coordinate of fitted beam spot"}])
bmoverviewlayout(dqmitems, "03 - y position of beam spot",
                 [{ 'path': "BeamMonitor/Fit/BeamMonitorFeedBack_y0",
                    'description': "y coordinate of fitted beam spot"}])
bmoverviewlayout(dqmitems, "04 - z position of beam spot",
                 [{ 'path': "BeamMonitor/Fit/BeamMonitorFeedBack_z0",
                    'description': "z coordinate of fitted beam spot"}])
bmoverviewlayout(dqmitems, "05 - sigma z of beam spot",
                 [{ 'path': "BeamMonitor/Fit/BeamMonitorFeedBack_sigmaZ0",
                    'description': "sigma z of fitted beam spot"}])
bmoverviewlayout(dqmitems, "06 - fit results beam spot",
                 [{ 'path': "BeamMonitor/Fit/fitResults",
                    'description': "d_{0}-#phi correlation fit results of beam spot"}])
bmoverviewlayout(dqmitems, "07 - Pixel-Vertices: Results of Beam Spot Fit",
                 [{ 'path': "BeamPixel/fit results",
                    'description': "Beam spot parameters from pixel-vertices"}])
bmoverviewlayout(dqmitems, "08 - Pixel-Vertices: X0 vs. Lumisection",
                 [{ 'path': "BeamPixel/muX vs lumi",
                    'description': "Beam spot X0 from pixel-vertices"}])
bmoverviewlayout(dqmitems, "09 - Pixel-Vertices: Y0 vs. Lumisection",
                 [{ 'path': "BeamPixel/muY vs lumi",
                    'description': "Beam spot Y0 from pixel-vertices"}])
bmoverviewlayout(dqmitems, "10 - Pixel-Vertices: Z0 vs. Lumisection",
                 [{ 'path': "BeamPixel/muZ vs lumi",
                    'description': "Beam spot Z0 from pixel-vertices"}])
