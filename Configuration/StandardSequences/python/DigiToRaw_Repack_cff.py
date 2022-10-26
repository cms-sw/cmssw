import FWCore.ParameterSet.Config as cms

##
## (1) Remake RAW from ZS tracker digis
##

import EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi
SiStripDigiToZSRaw = EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi.SiStripDigiToRaw.clone(
    InputDigis = cms.InputTag('siStripZeroSuppression', 'VirginRaw'),
    FedReadoutMode = cms.string('ZERO_SUPPRESSED'),
    PacketCode = cms.string('ZERO_SUPPRESSED'),
    CopyBufferHeader = cms.bool(True),
    RawDataTag = cms.InputTag('rawDataCollector')
    )

SiStripDigiToHybridRaw = SiStripDigiToZSRaw.clone(
    PacketCode = cms.string('ZERO_SUPPRESSED10'),
    )

SiStripRawDigiToVirginRaw = SiStripDigiToZSRaw.clone(
    FedReadoutMode = cms.string('VIRGIN_RAW'),
    PacketCode = cms.string('VIRGIN_RAW')
)

##
## (2) Combine new ZS RAW from tracker with existing RAW for other FEDs
##

from EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi import rawDataCollector

rawDataRepacker = rawDataCollector.clone(
    verbose = cms.untracked.int32(0),
    RawCollectionList = cms.VInputTag( cms.InputTag('SiStripDigiToZSRaw'),
                                       cms.InputTag('source'),
                                       cms.InputTag('rawDataCollector'))
    )
hybridRawDataRepacker = rawDataRepacker.clone(
    RawCollectionList = cms.VInputTag( cms.InputTag('SiStripDigiToHybridRaw'),
                                       cms.InputTag('source'),
                                       cms.InputTag('rawDataCollector'))
    )

virginRawDataRepacker = rawDataRepacker.clone(
	RawCollectionList = cms.VInputTag( cms.InputTag('SiStripRawDigiToVirginRaw'))
)

##
## Repacked DigiToRaw Sequence
##

DigiToRawRepackTask = cms.Task(SiStripDigiToZSRaw, rawDataRepacker)
DigiToHybridRawRepackTask = cms.Task(SiStripDigiToHybridRaw, hybridRawDataRepacker)
DigiToVirginRawRepackTask = cms.Task(SiStripRawDigiToVirginRaw, virginRawDataRepacker)

DigiToRawRepack = cms.Sequence( DigiToRawRepackTask )
DigiToHybridRawRepack = cms.Sequence( DigiToHybridRawRepackTask )
DigiToVirginRawRepack = cms.Sequence( DigiToVirginRawRepackTask )
DigiToSplitRawRepack = cms.Sequence( DigiToRawRepackTask, DigiToVirginRawRepackTask )

from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import siStripDigis
siStripDigisHLT = siStripDigis.clone(ProductLabel = "rawDataRepacker")

from RecoLocalTracker.Configuration.RecoLocalTracker_cff import siStripZeroSuppressionHLT

from RecoLocalTracker.SiStripClusterizer.DefaultClusterizer_cff import *
siStripClustersHLT = cms.EDProducer("SiStripClusterizer",
                                    Clusterizer = DefaultClusterizer,
                                    DigiProducersList = cms.VInputTag(
                                        cms.InputTag('siStripDigisHLT','ZeroSuppressed'),
                                        cms.InputTag('siStripZeroSuppressionHLT','VirginRaw'),
                                        cms.InputTag('siStripZeroSuppressionHLT','ProcessedRaw'),
                                        cms.InputTag('siStripZeroSuppressionHLT','ScopeMode')),
                                )

from RecoLocalTracker.SiStripClusterizer.SiStripClusters2ApproxClusters_cff import hltSiStripClusters2ApproxClusters


from EventFilter.Utilities.EvFFEDSelector_cfi import *
rawPrimeDataRepacker = cms.EDProducer( "EvFFEDSelector",
  fedList = cms.vuint32( [ 520, 522, 523, 524, 525, 528, 529, 530, 531, 532, 534, 535, 537, 539, 540, 541, 542, 545, 546, 547, 548, 549, 551, 553, 554, 555, 556, 557, 560, 561, 563, 564, 565, 566, 568, 570, 571, 572, 573, 574, 577, 578, 580, 582, 583, 584, 585, 586, 587, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 661, 662, 663, 664, 690, 691, 692, 693, 724, 725, 726, 727, 728, 729, 730, 731, 735, 790, 791, 792, 793, 814, 816, 817, 818, 819, 820, 821, 822, 823, 824, 831, 832, 833, 834, 835, 836, 837, 838, 839, 841, 842, 843, 844, 845, 846, 847, 848, 849, 851, 852, 853, 854, 855, 856, 857, 858, 859, 861, 862, 863, 864, 865, 866, 867, 868, 869, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1134, 1135, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1354, 1356, 1358, 1360, 1368, 1369, 1370, 1371, 1376, 1377, 1380, 1381, 1384, 1385, 1386, 1390, 1391, 1392, 1393, 1394, 1395, 1402, 1404, 1405, 1462, 1463, 1467 ] ),
  inputTag = cms.InputTag( "rawDataCollector" )
)

DigiToApproxClusterRawTask = cms.Task(siStripDigisHLT,siStripZeroSuppressionHLT,siStripClustersHLT,hltSiStripClusters2ApproxClusters,rawPrimeDataRepacker)
DigiToApproxClusterRaw = cms.Sequence(DigiToApproxClusterRawTask)
