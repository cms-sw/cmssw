server.workspace('DQMQuality', 0, 'Summaries', 'Summary')
server.workspace('DQMSummary', 1, 'Summaries', 'Report')
server.workspace('DQMShift',   2, 'Summaries', 'Shift')

server.workspace('DQMContent', 3, 'Summaries', 'Everything', '^', '^', '')

server.workspace('DQMContent', 20, 'Calorimeter', 'EcalBarrel', '^EcalBarrel/', '',
                 'EcalBarrel/Layouts/00 Global Summary EcalBarrel',
                 'EcalBarrel/Layouts/01 Occupancy Summary EcalBarrel',
                 'EcalBarrel/Layouts/02 Cluster Summary EcalBarrel')

server.workspace('DQMContent', 20, 'Calorimeter', 'EcalEndcap', '^EcalEndcap/', '',
                 'EcalEndcap/Layouts/00 Global Summary EcalEndcap',
                 'EcalEndcap/Layouts/01 Occupancy Summary EcalEndcap',
                 'EcalEndcap/Layouts/02 Cluster Summary EcalEndcap')

server.workspace('DQMContent', 20, 'Calorimeter', 'EcalCalibration', '^(EcalCalibration|EcalBarrel|EcalEndcap)/', '')

server.workspace('DQMContent', 20, 'Calorimeter', 'Ecal', '^Ecal/', '',
#                 'Ecal/Summary/SummaryClient global quality EB',
#                 'Ecal/Summary/SummaryClient global quality EE',
                 'Ecal/Layouts/02 Occupancy/01 Digi EB',
                 'Ecal/Layouts/02 Occupancy/02 Digi EE-',
                 'Ecal/Layouts/02 Occupancy/03 Digi EE+')

server.workspace('DQMContent', 50,'Collisions', 'Ecal FeedBack', '^(Collisions|EcalBarrel|EcalEndcap|EcalCalibration)/', '',
                 "Collisions/EcalFeedBack/00 Single Event Timing EE",
                 "Collisions/EcalFeedBack/01 Timing Mean EE",
                 "Collisions/EcalFeedBack/02 Timing Map EE -",
                 "Collisions/EcalFeedBack/02 Timing Map EE +",
                 "Collisions/EcalFeedBack/03 Occupancy EE -",
                 "Collisions/EcalFeedBack/03 Occupancy EE +",
                 "Collisions/EcalFeedBack/04 Single Event Timing EB",
                 "Collisions/EcalFeedBack/05 Timing Mean EB",
                 "Collisions/EcalFeedBack/06 Timing Map EB",
                 "Collisions/EcalFeedBack/07 Occupancy EB")

