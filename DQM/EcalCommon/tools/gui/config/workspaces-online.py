server.workspace('DQMQuality', 0, 'Summaries', 'Summary')
server.workspace('DQMSummary', 1, 'Summaries', 'Report')
server.workspace('DQMShift',   2, 'Summaries', 'Shift')

server.workspace('DQMContent', 3, 'Summaries', 'Everything', '^', '^')

server.workspace('DQMContent', 20, 'Calorimeter', 'Ecal', '^Ecal(|Barrel|Endcap|Calibration)/', 'Ecal/Layouts',
                 'Ecal/Layouts/00 Summary',
                 'Ecal/Layouts/01 Occupancy Summary',
                 'Ecal/Layouts/02 Calibration Summary')

server.workspace('DQMContent', 50,'Collisions', 'Ecal FeedBack', '^(Collisions|Ecal[^/]*)/', 'Collisions/EcalFeedBack')
