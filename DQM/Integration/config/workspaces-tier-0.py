server.workspace('DQMQuality', 0, 'Summaries', 'Summary')
server.workspace('DQMSummary', 1, 'Summaries', 'Reports')
server.workspace('DQMShift',   2, 'Summaries', 'Shift')
server.workspace('DQMContent', 3, 'Summaries', 'Everything', '^')

server.workspace('DQMContent', 10, 'Tracker/Muons', 'Pixel', '^Pixel/',
                 'Pixel/Layouts/00b - Pixel_Error_Summary',
                 'Pixel/Layouts/01 - Pixel_FEDOccupancy_Summary',
                 'Pixel/Layouts/02 - Pixel_Cluster_Summary',
                 'Pixel/Layouts/03 - Pixel_Track_Summary',
                 'Pixel/Layouts/05 - Barrel OnTrack cluster positions',
                 'Pixel/Layouts/06 - Endcap OnTrack cluster positions',
                 'Pixel/Layouts/07 - Pixel_Digi_Summary',)

server.workspace('DQMContent', 11, 'Tracker/Muons', 'SiStrip', '^SiStrip/',
                                  'SiStrip/Layouts/00 - SiStrip ReportSummary',
                                  'SiStrip/Layouts/01 - FED-Detected Errors Summary',
                                  'SiStrip/Layouts/02 - FED-Detected Errors',
                                  'SiStrip/Layouts/03 - # of Cluster Trend',
                                  'SiStrip/Layouts/04 - OnTrackCluster (StoN)',
                                  'SiStrip/Layouts/05 - OffTrackCluster (Total Number)'
                                  )

server.workspace('DQMContent', 12, 'Tracker/Muons', 'CSC', '^CSC/',
	         'CSC/Layouts/00 Data Integrity/Physics Efficiency 01',
		 'CSC/Layouts/00 Data Integrity/Physics Efficiency 02',
	         'CSC/Layouts/00 Data Integrity/Physics Efficiency 04 - CSCs Reporting Data and Unpacked',
		 'CSC/Layouts/00 Data Integrity/Physics Efficiency 08 - CSCs Occupancy Overal',
	         'CSC/Layouts/00 Data Integrity/Physics Efficiency 07 - CSCs Occupancy 2D',
	         'CSC/Layouts/00 Data Integrity/Physics Efficiency 09 - RecHits Minus',
		 'CSC/Layouts/00 Data Integrity/Physics Efficiency 10 - RecHits Plus'
	         )

server.workspace('DQMContent', 13, 'Tracker/Muons', 'DT', '^DT/')
server.workspace('DQMContent', 14, 'Tracker/Muons', 'RPC', '^RPC/') 

server.workspace('DQMContent', 21, 'Calorimeter', 'EcalBarrel', '^EcalBarrel/',
                 'EcalBarrel/Layouts/00 Global Summary EcalBarrel',
                 'EcalBarrel/Layouts/01 Occupancy Summary EcalBarrel',
                 'EcalBarrel/Layouts/02 Cluster Summary EcalBarrel')

server.workspace('DQMContent', 22, 'Calorimeter', 'EcalEndcap', '^EcalEndcap/',
                 'EcalEndcap/Layouts/00 Global Summary EcalEndcap',
                 'EcalEndcap/Layouts/01 Occupancy Summary EcalEndcap',
                 'EcalEndcap/Layouts/02 Cluster Summary EcalEndcap')

server.workspace('DQMContent', 23, 'Calorimeter', 'EcalPreshower', '^EcalPreshower/',
                'EcalPreshower/Layouts/01-IntegritySummary-EcalPreshower',
                'EcalPreshower/Layouts/02-OccupancySummary-EcalPreshower',
                'EcalPreshower/Layouts/03-RechitEnergySummary-EcalPreshower')

server.workspace('DQMContent', 24, 'Calorimeter', 'HCAL', '^Hcal/',
                 'Hcal/Layouts/01 HCAL Summaries',
                 'Hcal/Layouts/02 HCAL Events Processed',
                 'Hcal/Layouts/03 HCAL Sufficient Events',
                 'Hcal/Layouts/04 HCAL Raw Data',
                 'Hcal/Layouts/05 HCAL Digi Problems',
                 'Hcal/Layouts/06 HCAL Dead Cell Check',
                 'Hcal/Layouts/07 HCAL Hot Cell Check'
                 )

server.workspace('DQMContent', 20, 'Calorimeter', 'CASTOR', '^Castor/',
                 'Castor/Layouts/CASTOR Channel Status',
                 'Castor/Layouts/CASTOR RecHit Energies',
		 'Castor/Layouts/CASTOR RecHit Energy in modules',
		 'Castor/Layouts/CASTOR RecHit Energy in sectors',
		 'Castor/Layouts/CASTOR RecHitEnergy 2D Map',
                 'Castor/Layouts/CASTOR All Digi Values',
                 'Castor/Layouts/CASTOR average pulse in bunch crossings',
                 'Castor/Layouts/CASTOR hits 3D- cumulative',
                 'Castor/Layouts/CASTOR hits 3D- event with the largest deposited E'
                )

server.workspace('DQMContent', 31, 'Trigger/Lumi', 'L1T', '^L1T/')
server.workspace('DQMContent', 32, 'Trigger/Lumi', 'L1TEMU', '^L1TEMU/')
server.workspace('DQMContent', 33, 'Trigger/Lumi', 'HLT', '^HLT/')
server.workspace('DQMContent', 34, 'Trigger/Lumi', 'HLX', '^HLX')

server.workspace('DQMContent', 41, 'POG', 'Muons', '^Muons/')
server.workspace('DQMContent', 42, 'POG', 'JetMet', '^JetMET/')
server.workspace('DQMContent', 43, 'POG', 'EGamma', '^Egamma/')
server.workspace('DQMContent', 44, 'POG', 'Btag', '^Btag/',
                 'Btag/Layouts/00 - Jet Property',
                 'Btag/Layouts/01 - Tracks in Jet',
                 'Btag/Layouts/02 - Vertex Property',
                 'Btag/Layouts/03 - Flight Distance Summary',
                 'Btag/Layouts/04 - Discriminator Summary',
                 'Btag/Layouts/05 - 2D-Impact Parameter',
                 'Btag/Layouts/06 - 3D-Impact Parameter'
)

server.workspace('DQMContent', 45, 'POG', 'Tracking', '^(Tracking|AlcaBeamMonitor|OfflinePV)/',
                                  'Tracking/Layouts/01 - Tracking ReportSummary',
                                  'Tracking/Layouts/02 - Tracks (pp collisions)',
                                  'Tracking/Layouts/03 - Tracks (HI collisions)',
                                  'Tracking/Layouts/04 - Tracks (Cosmic Tracking)'
                                  )

server.workspace('DQMContent', 51,'FeedBack for Collisions', 'Tracking FeedBack', '^(Collisions|SiStrip|Tracking|Pixel|AlcaBeamMonitor|OfflinePV)/',
                 'Collisions/TrackingFeedBack/00 - Number Of Tracks',
                 'Collisions/TrackingFeedBack/01 - Track Pt',
                 'Collisions/TrackingFeedBack/02 - Track Phi',
                 'Collisions/TrackingFeedBack/03 - Track Eta',
                 'Collisions/TrackingFeedBack/04 - X-Position Of Closest Approach',
                 'Collisions/TrackingFeedBack/05 - Y-Position Of Closest Approach',
                 'Collisions/TrackingFeedBack/06 - Z-Position Of Closest Approach',
                 'Collisions/TrackingFeedBack/07 - Cluster y width vs. cluster eta'
)
server.workspace('DQMContent', 52,'FeedBack for Collisions', 'Ecal FeedBack', '^(Collisions|EcalBarrel|EcalEndcap|EcalPreshower)/',
                 "Collisions/EcalFeedBack/00 Single Event Timing EE",
                 "Collisions/EcalFeedBack/01 Timing Mean EE",
                 "Collisions/EcalFeedBack/02 Timing Map EE -",
                 "Collisions/EcalFeedBack/02 Timing Map EE +",
                 "Collisions/EcalFeedBack/03 Occupancy EE -",
                 "Collisions/EcalFeedBack/03 Occupancy EE +",
                 "Collisions/EcalFeedBack/04 Single Event Timing EB",
                 "Collisions/EcalFeedBack/05 Timing Mean EB",
                 "Collisions/EcalFeedBack/06 Timing Map EB",
                 "Collisions/EcalFeedBack/07 Occupancy EB",
                 "Collisions/EcalFeedBack/08 ES Occupancy",
                 "Collisions/EcalFeedBack/09 ES Energy Map",
                 "Collisions/EcalFeedBack/10 ES Timing Plot"
                 )
server.workspace('DQMContent', 53,'FeedBack for Collisions', 'Hcal FeedBack', '^(Collisions|Hcal)/',
                 "Collisions/HcalFeedBack/01 - HF+,HF- coincidences (with BPTX)",
                 "Collisions/HcalFeedBack/02 - HF+,HF- coincidences (without BPTX)",
                 "Collisions/HcalFeedBack/03 - Digi Shapes for Total Digi Signals > N counts",
                 "Collisions/HcalFeedBack/04 - Lumi Bunch Crossing Checks",
                 "Collisions/HcalFeedBack/05 - Events Per Lumi Section",
                 "Collisions/HcalFeedBack/06 - Lumi Distributions",
                 )

server.workspace('DQMContent', 54,'FeedBack for Collisions', 'L1T FeedBack','^(Collisions|L1T)/',
                "Collisions/L1TFeedBack/00 Rate BSCL.BSCR",
                "Collisions/L1TFeedBack/01 Rate BSC splash right",
                "Collisions/L1TFeedBack/02 Rate BSC splash left",
                "Collisions/L1TFeedBack/03 Integ BSCL*BSCR Triggers vs LS",
                "Collisions/L1TFeedBack/04 Integ BSCL or BSCR Triggers vs LS",
                "Collisions/L1TFeedBack/05 Integ HF Triggers vs LS"
                )

server.workspace('DQMContent', 55,'FeedBack for Collisions', 'HLT FeedBack','^(Collisions|HLT)/',
                "Collisions/HLTFeedBack/00 HLT_Egamma_Pass_Any",
                "Collisions/HLTFeedBack/01 HLT_JetMet_Pass_Any",
                "Collisions/HLTFeedBack/02 HLT_Muon_Pass_Any",
                "Collisions/HLTFeedBack/03 HLT_Rest_Pass_Any",
                "Collisions/HLTFeedBack/04 HLT_Special_Pass_Any"
                )

server.workspace('DQMContent', 56, 'FeedBack for Collisions', 'CSC FeedBack', '^(Collisions|CSC)/',
                'CSC/Layouts/04 Timing/00 ALCT Timing',
                'CSC/Layouts/04 Timing/01 CLCT Timing'
                )

