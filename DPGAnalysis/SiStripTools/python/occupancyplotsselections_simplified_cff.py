import FWCore.ParameterSet.Config as cms

OccupancyPlotsPixelWantedLayers = cms.VPSet (
    cms.PSet(detSelection=cms.uint32(110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f0000-0x12010000")),      # BPix L1
    cms.PSet(detSelection=cms.uint32(120),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f0000-0x12020000")),      # BPix L2
    cms.PSet(detSelection=cms.uint32(130),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f0000-0x12030000")),      # BPix L3
    )

OccupancyPlotsPixelWantedSubDets = cms.VPSet (
    cms.PSet(detSelection=cms.uint32(111),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12010004")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(112),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12010008")),      # BPix L1 mod 2
    cms.PSet(detSelection=cms.uint32(113),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x1201000c")),      # BPix L1 mod 3
    cms.PSet(detSelection=cms.uint32(114),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12010010")),      # BPix L1 mod 4
    cms.PSet(detSelection=cms.uint32(115),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12010014")),      # BPix L1 mod 5
    cms.PSet(detSelection=cms.uint32(116),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12010018")),      # BPix L1 mod 6
    cms.PSet(detSelection=cms.uint32(117),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x1201001c")),      # BPix L1 mod 7
    cms.PSet(detSelection=cms.uint32(118),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12010020")),      # BPix L1 mod 8

    cms.PSet(detSelection=cms.uint32(121),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12020004")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(122),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12020008")),      # BPix L1 mod 2
    cms.PSet(detSelection=cms.uint32(123),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x1202000c")),      # BPix L1 mod 3
    cms.PSet(detSelection=cms.uint32(124),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12020010")),      # BPix L1 mod 4
    cms.PSet(detSelection=cms.uint32(125),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12020014")),      # BPix L1 mod 5
    cms.PSet(detSelection=cms.uint32(126),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12020018")),      # BPix L1 mod 6
    cms.PSet(detSelection=cms.uint32(127),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x1202001c")),      # BPix L1 mod 7
    cms.PSet(detSelection=cms.uint32(128),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12020020")),      # BPix L1 mod 8

    cms.PSet(detSelection=cms.uint32(131),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12030004")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(132),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12030008")),      # BPix L1 mod 2
    cms.PSet(detSelection=cms.uint32(133),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x1203000c")),      # BPix L1 mod 3
    cms.PSet(detSelection=cms.uint32(134),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12030010")),      # BPix L1 mod 4
    cms.PSet(detSelection=cms.uint32(135),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12030014")),      # BPix L1 mod 5
    cms.PSet(detSelection=cms.uint32(136),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12030018")),      # BPix L1 mod 6
    cms.PSet(detSelection=cms.uint32(137),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x1203001c")),      # BPix L1 mod 7
    cms.PSet(detSelection=cms.uint32(138),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12030020")),      # BPix L1 mod 8

    cms.PSet(detSelection=cms.uint32(211),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14810104")),      # FPix- D1 pan1 mod1
    cms.PSet(detSelection=cms.uint32(212),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14810204")),      # FPix- D1 pan2 mod1
    cms.PSet(detSelection=cms.uint32(213),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14810108")),      # FPix- D1 pan1 mod2
    cms.PSet(detSelection=cms.uint32(214),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14810208")),      # FPix- D1 pan2 mod2
    cms.PSet(detSelection=cms.uint32(215),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1481010c")),      # FPix- D1 pan1 mod3
    cms.PSet(detSelection=cms.uint32(216),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1481020c")),      # FPix- D1 pan2 mod3
    cms.PSet(detSelection=cms.uint32(217),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14810110")),      # FPix- D1 pan1 mod4
#    cms.PSet(detSelection=cms.uint32(),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14810210"))      # FPix- D1
    
    cms.PSet(detSelection=cms.uint32(221),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14820104")),      # FPix- D2 pan1 mod1
    cms.PSet(detSelection=cms.uint32(222),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14820204")),      # FPix- D2 pan2 mod1
    cms.PSet(detSelection=cms.uint32(223),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14820108")),      # FPix- D2 pan1 mod2
    cms.PSet(detSelection=cms.uint32(224),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14820208")),      # FPix- D2 pan2 mod2
    cms.PSet(detSelection=cms.uint32(225),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1482010c")),      # FPix- D2 pan1 mod3
    cms.PSet(detSelection=cms.uint32(226),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1482020c")),      # FPix- D2 pan2 mod3
    cms.PSet(detSelection=cms.uint32(227),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14820110")),      # FPix- D2 pan1 mod4
#    cms.PSet(detSelection=cms.uint32(),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14820210"))      # FPix- D2
    
    cms.PSet(detSelection=cms.uint32(231),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15010104")),      # FPix+ D1 pan1 mod1
    cms.PSet(detSelection=cms.uint32(232),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15010204")),      # FPix+ D1 pan2 mod1
    cms.PSet(detSelection=cms.uint32(233),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15010108")),      # FPix+ D1 pan1 mod2
    cms.PSet(detSelection=cms.uint32(234),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15010208")),      # FPix+ D1 pan2 mod2
    cms.PSet(detSelection=cms.uint32(235),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1501010c")),      # FPix+ D1 pan1 mod3
    cms.PSet(detSelection=cms.uint32(236),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1501020c")),      # FPix+ D1 pan2 mod3
    cms.PSet(detSelection=cms.uint32(237),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15010110")),      # FPix+ D1 pan1 mod4
#    cms.PSet(detSelection=cms.uint32(),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15010210"))      # FPix+ D1
    
    cms.PSet(detSelection=cms.uint32(241),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15020104")),      # FPix+ D2 pan1 mod1
    cms.PSet(detSelection=cms.uint32(242),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15020204")),      # FPix+ D2 pan2 mod1
    cms.PSet(detSelection=cms.uint32(243),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15020108")),      # FPix+ D2 pan1 mod2
    cms.PSet(detSelection=cms.uint32(244),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15020208")),      # FPix+ D2 pan2 mod2
    cms.PSet(detSelection=cms.uint32(245),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1502010c")),      # FPix+ D2 pan1 mod3
    cms.PSet(detSelection=cms.uint32(246),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1502020c")),      # FPix+ D2 pan2 mod3
    cms.PSet(detSelection=cms.uint32(247),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15020110"))       # FPix+ D2 pan1 mod4
#    cms.PSet(detSelection=cms.uint32(),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15020210"))      # FPix+ D2

    )

OccupancyPlotsStripWantedLayers = cms.VPSet (
    cms.PSet(detSelection=cms.uint32(1100),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01c000-0x16004000")),     # TIB L1
    cms.PSet(detSelection=cms.uint32(1200),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01c000-0x16008000")),     # TIB L2 
    cms.PSet(detSelection=cms.uint32(1300),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01c000-0x1600c000")),     # TIB L3
    cms.PSet(detSelection=cms.uint32(1400),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01c000-0x16010000")),     # TIB L4
    cms.PSet(detSelection=cms.uint32(3100),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01c000-0x1a004000")),     # TOB L1
    cms.PSet(detSelection=cms.uint32(3200),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01c000-0x1a008000")),     # TOB L2
    cms.PSet(detSelection=cms.uint32(3300),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01c000-0x1a00c000")),     # TOB L3
    cms.PSet(detSelection=cms.uint32(3400),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01c000-0x1a010000")),     # TOB L4
    cms.PSet(detSelection=cms.uint32(3500),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01c000-0x1a014000")),     # TOB L5
    cms.PSet(detSelection=cms.uint32(3600),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01c000-0x1a018000")),     # TOB L6
    )

OccupancyPlotsStripWantedSubDets = cms.VPSet (
     cms.PSet(detSelection=cms.uint32(1101),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600640c")),     # TIB+ L1 int m3
     cms.PSet(detSelection=cms.uint32(1102),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600680c")),     # TIB+ L1 ext m3
     cms.PSet(detSelection=cms.uint32(1103),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16006408")),     # TIB+ L1 int m2
     cms.PSet(detSelection=cms.uint32(1104),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16006808")),     # TIB+ L1 ext m2
     cms.PSet(detSelection=cms.uint32(1105),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16006404")),     # TIB+ L1 int m1
     cms.PSet(detSelection=cms.uint32(1106),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16006804")),     # TIB+ L1 ext m1
     cms.PSet(detSelection=cms.uint32(1107),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16005404")),     # TIB- L1 int m1
     cms.PSet(detSelection=cms.uint32(1108),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16005804")),     # TIB- L1 ext m1
     cms.PSet(detSelection=cms.uint32(1109),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16005408")),     # TIB- L1 int m2
     cms.PSet(detSelection=cms.uint32(1110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16005808")),     # TIB- L1 ext m2
     cms.PSet(detSelection=cms.uint32(1111),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600540c")),     # TIB- L1 int m3
     cms.PSet(detSelection=cms.uint32(1112),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600580c")),     # TIB- L1 ext m3
     cms.PSet(detSelection=cms.uint32(1201),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600a80c")),     # TIB+ L2 ext m3
     cms.PSet(detSelection=cms.uint32(1202),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600a40c")),     # TIB+ L2 int m3
     cms.PSet(detSelection=cms.uint32(1203),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600a808")),     # TIB+ L2 ext m2
     cms.PSet(detSelection=cms.uint32(1204),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600a408")),     # TIB+ L2 int m2
     cms.PSet(detSelection=cms.uint32(1205),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600a804")),     # TIB+ L2 ext m1
     cms.PSet(detSelection=cms.uint32(1206),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600a404")),     # TIB+ L2 int m1
     cms.PSet(detSelection=cms.uint32(1207),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16009804")),     # TIB- L2 ext m1
     cms.PSet(detSelection=cms.uint32(1208),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16009404")),     # TIB- L2 int m1
     cms.PSet(detSelection=cms.uint32(1209),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16009808")),     # TIB- L2 ext m2
     cms.PSet(detSelection=cms.uint32(1210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16009408")),     # TIB- L2 int m2
     cms.PSet(detSelection=cms.uint32(1211),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600980c")),     # TIB- L2 ext m3
     cms.PSet(detSelection=cms.uint32(1212),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600940c")),     # TIB- L2 int m3
     cms.PSet(detSelection=cms.uint32(1301),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600e40c")),     # TIB+ L3 int m3
     cms.PSet(detSelection=cms.uint32(1302),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600e80c")),     # TIB+ L3 ext m3
     cms.PSet(detSelection=cms.uint32(1303),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600e408")),     # TIB+ L3 int m2
     cms.PSet(detSelection=cms.uint32(1304),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600e808")),     # TIB+ L3 ext m2
     cms.PSet(detSelection=cms.uint32(1305),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600e404")),     # TIB+ L3 int m1
     cms.PSet(detSelection=cms.uint32(1306),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600e804")),     # TIB+ L3 ext m1
     cms.PSet(detSelection=cms.uint32(1307),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600d404")),     # TIB- L3 int m1
     cms.PSet(detSelection=cms.uint32(1308),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600d804")),     # TIB- L3 ext m1
     cms.PSet(detSelection=cms.uint32(1309),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600d408")),     # TIB- L3 int m2
     cms.PSet(detSelection=cms.uint32(1310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600d808")),     # TIB- L3 ext m2
     cms.PSet(detSelection=cms.uint32(1311),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600d40c")),     # TIB- L3 int m3
     cms.PSet(detSelection=cms.uint32(1312),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1600d80c")),     # TIB- L3 ext m3
     cms.PSet(detSelection=cms.uint32(1401),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1601280c")),     # TIB+ L4 ext m3
     cms.PSet(detSelection=cms.uint32(1402),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1601240c")),     # TIB+ L4 int m3
     cms.PSet(detSelection=cms.uint32(1403),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16012808")),     # TIB+ L4 ext m2
     cms.PSet(detSelection=cms.uint32(1404),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16012408")),     # TIB+ L4 int m2
     cms.PSet(detSelection=cms.uint32(1405),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16012804")),     # TIB+ L4 ext m1
     cms.PSet(detSelection=cms.uint32(1406),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16012404")),     # TIB+ L4 int m1
     cms.PSet(detSelection=cms.uint32(1407),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16011804")),     # TIB- L4 ext m1
     cms.PSet(detSelection=cms.uint32(1408),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16011404")),     # TIB- L4 int m1
     cms.PSet(detSelection=cms.uint32(1409),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16011808")),     # TIB- L4 ext m2
     cms.PSet(detSelection=cms.uint32(1410),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x16011408")),     # TIB- L4 int m2
     cms.PSet(detSelection=cms.uint32(1411),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1601180c")),     # TIB- L4 ext m3
     cms.PSet(detSelection=cms.uint32(1412),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01fc0c-0x1601140c")),     # TIB- L4 int m3

     cms.PSet(detSelection=cms.uint32(2110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18002a00")),     # TID- D1 R1 
     cms.PSet(detSelection=cms.uint32(2120),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003200")),     # TID- D2 R1 
     cms.PSet(detSelection=cms.uint32(2130),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003a00")),     # TID- D3 R1 
     cms.PSet(detSelection=cms.uint32(2140),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18004a00")),     # TID+ D1 R1 
     cms.PSet(detSelection=cms.uint32(2150),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005200")),     # TID+ D2 R1 
     cms.PSet(detSelection=cms.uint32(2160),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005a00")),     # TID+ D3 R1 

     cms.PSet(detSelection=cms.uint32(2210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18002c00")),     # TID- D1 R2 
     cms.PSet(detSelection=cms.uint32(2220),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003400")),     # TID- D2 R2 
     cms.PSet(detSelection=cms.uint32(2230),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003c00")),     # TID- D3 R2 
     cms.PSet(detSelection=cms.uint32(2240),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18004c00")),     # TID+ D1 R2 
     cms.PSet(detSelection=cms.uint32(2250),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005400")),     # TID+ D2 R2 
     cms.PSet(detSelection=cms.uint32(2260),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005c00")),     # TID+ D3 R2 

     cms.PSet(detSelection=cms.uint32(2310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18002e00")),     # TID- D1 R3 
     cms.PSet(detSelection=cms.uint32(2320),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003600")),     # TID- D2 R3 
     cms.PSet(detSelection=cms.uint32(2330),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003e00")),     # TID- D3 R3 
     cms.PSet(detSelection=cms.uint32(2340),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18004e00")),     # TID+ D1 R3 
     cms.PSet(detSelection=cms.uint32(2350),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005600")),     # TID+ D2 R3 
     cms.PSet(detSelection=cms.uint32(2360),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005e00")),     # TID+ D3 R3 

    cms.PSet(detSelection=cms.uint32(3101),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a006018")),     # TOB+ L1 m6
    cms.PSet(detSelection=cms.uint32(3102),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a006014")),     # TOB+ L1 m5
    cms.PSet(detSelection=cms.uint32(3103),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a006010")),     # TOB+ L1 m4
    cms.PSet(detSelection=cms.uint32(3104),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00600c")),     # TOB+ L1 m3
    cms.PSet(detSelection=cms.uint32(3105),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a006008")),     # TOB+ L1 m2
    cms.PSet(detSelection=cms.uint32(3106),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a006004")),     # TOB+ L1 m1
    cms.PSet(detSelection=cms.uint32(3107),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a005004")),     # TOB- L1 m1
    cms.PSet(detSelection=cms.uint32(3108),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a005008")),     # TOB- L1 m2
    cms.PSet(detSelection=cms.uint32(3109),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00500c")),     # TOB- L1 m3
    cms.PSet(detSelection=cms.uint32(3110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a005010")),     # TOB- L1 m4
    cms.PSet(detSelection=cms.uint32(3111),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a005014")),     # TOB- L1 m5
    cms.PSet(detSelection=cms.uint32(3112),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a005018")),     # TOB- L1 m6

    cms.PSet(detSelection=cms.uint32(3201),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00a018")),     # TOB+ L2 m6
    cms.PSet(detSelection=cms.uint32(3202),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00a014")),     # TOB+ L2 m5
    cms.PSet(detSelection=cms.uint32(3203),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00a010")),     # TOB+ L2 m4
    cms.PSet(detSelection=cms.uint32(3204),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00a00c")),     # TOB+ L2 m3
    cms.PSet(detSelection=cms.uint32(3205),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00a008")),     # TOB+ L2 m2
    cms.PSet(detSelection=cms.uint32(3206),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00a004")),     # TOB+ L2 m1
    cms.PSet(detSelection=cms.uint32(3207),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a009004")),     # TOB- L2 m1
    cms.PSet(detSelection=cms.uint32(3208),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a009008")),     # TOB- L2 m2
    cms.PSet(detSelection=cms.uint32(3209),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00900c")),     # TOB- L2 m3
    cms.PSet(detSelection=cms.uint32(3210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a009010")),     # TOB- L2 m4
    cms.PSet(detSelection=cms.uint32(3211),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a009014")),     # TOB- L2 m5
    cms.PSet(detSelection=cms.uint32(3212),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a009018")),     # TOB- L2 m6

    cms.PSet(detSelection=cms.uint32(3301),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00e018")),     # TOB+ L3 m6
    cms.PSet(detSelection=cms.uint32(3302),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00e014")),     # TOB+ L3 m5
    cms.PSet(detSelection=cms.uint32(3303),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00e010")),     # TOB+ L3 m4
    cms.PSet(detSelection=cms.uint32(3304),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00e00c")),     # TOB+ L3 m3
    cms.PSet(detSelection=cms.uint32(3305),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00e008")),     # TOB+ L3 m2
    cms.PSet(detSelection=cms.uint32(3306),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00e004")),     # TOB+ L3 m1
    cms.PSet(detSelection=cms.uint32(3307),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00d004")),     # TOB- L3 m1
    cms.PSet(detSelection=cms.uint32(3308),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00d008")),     # TOB- L3 m2
    cms.PSet(detSelection=cms.uint32(3309),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00d00c")),     # TOB- L3 m3
    cms.PSet(detSelection=cms.uint32(3310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00d010")),     # TOB- L3 m4
    cms.PSet(detSelection=cms.uint32(3311),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00d014")),     # TOB- L3 m5
    cms.PSet(detSelection=cms.uint32(3312),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a00d018")),     # TOB- L3 m6

    cms.PSet(detSelection=cms.uint32(3401),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a012018")),     # TOB+ L4 m6
    cms.PSet(detSelection=cms.uint32(3402),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a012014")),     # TOB+ L4 m5
    cms.PSet(detSelection=cms.uint32(3403),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a012010")),     # TOB+ L4 m4
    cms.PSet(detSelection=cms.uint32(3404),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01200c")),     # TOB+ L4 m3
    cms.PSet(detSelection=cms.uint32(3405),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a012008")),     # TOB+ L4 m2
    cms.PSet(detSelection=cms.uint32(3406),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a012004")),     # TOB+ L4 m1
    cms.PSet(detSelection=cms.uint32(3407),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a011004")),     # TOB- L4 m1
    cms.PSet(detSelection=cms.uint32(3408),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a011008")),     # TOB- L4 m2
    cms.PSet(detSelection=cms.uint32(3409),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01100c")),     # TOB- L4 m3
    cms.PSet(detSelection=cms.uint32(3410),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a011010")),     # TOB- L4 m4
    cms.PSet(detSelection=cms.uint32(3411),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a011014")),     # TOB- L4 m5
    cms.PSet(detSelection=cms.uint32(3412),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a011018")),     # TOB- L4 m6

    cms.PSet(detSelection=cms.uint32(3501),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a016018")),     # TOB+ L5 m6
    cms.PSet(detSelection=cms.uint32(3502),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a016014")),     # TOB+ L5 m5
    cms.PSet(detSelection=cms.uint32(3503),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a016010")),     # TOB+ L5 m4
    cms.PSet(detSelection=cms.uint32(3504),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01600c")),     # TOB+ L5 m3
    cms.PSet(detSelection=cms.uint32(3505),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a016008")),     # TOB+ L5 m2
    cms.PSet(detSelection=cms.uint32(3506),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a016004")),     # TOB+ L5 m1
    cms.PSet(detSelection=cms.uint32(3507),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a015004")),     # TOB- L5 m1
    cms.PSet(detSelection=cms.uint32(3508),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a015008")),     # TOB- L5 m2
    cms.PSet(detSelection=cms.uint32(3509),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01500c")),     # TOB- L5 m3
    cms.PSet(detSelection=cms.uint32(3510),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a015010")),     # TOB- L5 m4
    cms.PSet(detSelection=cms.uint32(3511),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a015014")),     # TOB- L5 m5
    cms.PSet(detSelection=cms.uint32(3512),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a015018")),     # TOB- L5 m6

    cms.PSet(detSelection=cms.uint32(3601),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01a018")),     # TOB+ L6 m6
    cms.PSet(detSelection=cms.uint32(3602),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01a014")),     # TOB+ L6 m5
    cms.PSet(detSelection=cms.uint32(3603),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01a010")),     # TOB+ L6 m4
    cms.PSet(detSelection=cms.uint32(3604),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01a00c")),     # TOB+ L6 m3
    cms.PSet(detSelection=cms.uint32(3605),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01a008")),     # TOB+ L6 m2
    cms.PSet(detSelection=cms.uint32(3606),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01a004")),     # TOB+ L6 m1
    cms.PSet(detSelection=cms.uint32(3607),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a019004")),     # TOB- L6 m1
    cms.PSet(detSelection=cms.uint32(3608),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a019008")),     # TOB- L6 m2
    cms.PSet(detSelection=cms.uint32(3609),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a01900c")),     # TOB- L6 m3
    cms.PSet(detSelection=cms.uint32(3610),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a019010")),     # TOB- L6 m4
    cms.PSet(detSelection=cms.uint32(3611),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a019014")),     # TOB- L6 m5
    cms.PSet(detSelection=cms.uint32(3612),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e01f01c-0x1a019018"))     # TOB- L6 m6
    )

OccupancyPlotsStripWantedSubDets.extend(
    cms.VPSet(
    cms.PSet(detSelection=cms.uint32(4110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044020")),    # TEC- D1 R1 
    cms.PSet(detSelection=cms.uint32(4120),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048020")),    # TEC- D2 R1 
    cms.PSet(detSelection=cms.uint32(4130),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c020")),    # TEC- D3 R1 
#    cms.PSet(detSelection=cms.uint32(4140),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050020")),    # TEC- D4 R1 
#    cms.PSet(detSelection=cms.uint32(4150),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054020")),    # TEC- D5 R1 
#    cms.PSet(detSelection=cms.uint32(4160),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058020")),    # TEC- D6 R1 
#    cms.PSet(detSelection=cms.uint32(4170),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c020")),    # TEC- D7 R1 
#    cms.PSet(detSelection=cms.uint32(4180),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060020")),    # TEC- D8 R1 
#    cms.PSet(detSelection=cms.uint32(4190),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064020")),    # TEC- D9 R1 

    cms.PSet(detSelection=cms.uint32(4210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044040")),    # TEC- D1 R2 
    cms.PSet(detSelection=cms.uint32(4220),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048040")),    # TEC- D2 R2 
    cms.PSet(detSelection=cms.uint32(4230),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c040")),    # TEC- D3 R2 
    cms.PSet(detSelection=cms.uint32(4240),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050040")),    # TEC- D4 R2 
    cms.PSet(detSelection=cms.uint32(4250),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054040")),    # TEC- D5 R2 
    cms.PSet(detSelection=cms.uint32(4260),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058040")),    # TEC- D6 R2 
#    cms.PSet(detSelection=cms.uint32(4270),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c040")),    # TEC- D7 R2 
#    cms.PSet(detSelection=cms.uint32(4280),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060040")),    # TEC- D8 R2 
#    cms.PSet(detSelection=cms.uint32(4290),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064040")),    # TEC- D9 R2 

    cms.PSet(detSelection=cms.uint32(4310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044060")),    # TEC- D1 R3 
    cms.PSet(detSelection=cms.uint32(4320),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048060")),    # TEC- D2 R3 
    cms.PSet(detSelection=cms.uint32(4330),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c060")),    # TEC- D3 R3 
    cms.PSet(detSelection=cms.uint32(4340),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050060")),    # TEC- D4 R3 
    cms.PSet(detSelection=cms.uint32(4350),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054060")),    # TEC- D5 R3 
    cms.PSet(detSelection=cms.uint32(4360),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058060")),    # TEC- D6 R3 
    cms.PSet(detSelection=cms.uint32(4370),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c060")),    # TEC- D7 R3 
    cms.PSet(detSelection=cms.uint32(4380),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060060")),    # TEC- D8 R3 
#    cms.PSet(detSelection=cms.uint32(4390),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064060")),    # TEC- D9 R3 

    cms.PSet(detSelection=cms.uint32(4410),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044080")),    # TEC- D1 R4 
    cms.PSet(detSelection=cms.uint32(4420),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048080")),    # TEC- D2 R4 
    cms.PSet(detSelection=cms.uint32(4430),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c080")),    # TEC- D3 R4 
    cms.PSet(detSelection=cms.uint32(4440),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050080")),    # TEC- D4 R4 
    cms.PSet(detSelection=cms.uint32(4450),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054080")),    # TEC- D5 R4 
    cms.PSet(detSelection=cms.uint32(4460),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058080")),    # TEC- D6 R4 
    cms.PSet(detSelection=cms.uint32(4470),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c080")),    # TEC- D7 R4 
    cms.PSet(detSelection=cms.uint32(4480),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060080")),    # TEC- D8 R4 
    cms.PSet(detSelection=cms.uint32(4490),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064080")),    # TEC- D9 R4 
    
    cms.PSet(detSelection=cms.uint32(4510),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0440a0")),    # TEC- D1 R5 
    cms.PSet(detSelection=cms.uint32(4520),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0480a0")),    # TEC- D2 R5 
    cms.PSet(detSelection=cms.uint32(4530),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c0a0")),    # TEC- D3 R5 
    cms.PSet(detSelection=cms.uint32(4540),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0500a0")),    # TEC- D4 R5 
    cms.PSet(detSelection=cms.uint32(4550),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0540a0")),    # TEC- D5 R5 
    cms.PSet(detSelection=cms.uint32(4560),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0580a0")),    # TEC- D6 R5 
    cms.PSet(detSelection=cms.uint32(4570),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c0a0")),    # TEC- D7 R5 
    cms.PSet(detSelection=cms.uint32(4580),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0600a0")),    # TEC- D8 R5 
    cms.PSet(detSelection=cms.uint32(4590),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0640a0")),    # TEC- D9 R5 

    cms.PSet(detSelection=cms.uint32(4610),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0440c0")),    # TEC- D1 R6 
    cms.PSet(detSelection=cms.uint32(4620),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0480c0")),    # TEC- D2 R6 
    cms.PSet(detSelection=cms.uint32(4630),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c0c0")),    # TEC- D3 R6 
    cms.PSet(detSelection=cms.uint32(4640),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0500c0")),    # TEC- D4 R6 
    cms.PSet(detSelection=cms.uint32(4650),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0540c0")),    # TEC- D5 R6 
    cms.PSet(detSelection=cms.uint32(4660),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0580c0")),    # TEC- D6 R6 
    cms.PSet(detSelection=cms.uint32(4670),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c0c0")),    # TEC- D7 R6 
    cms.PSet(detSelection=cms.uint32(4680),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0600c0")),    # TEC- D8 R6 
    cms.PSet(detSelection=cms.uint32(4690),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0640c0")),    # TEC- D9 R6 

    cms.PSet(detSelection=cms.uint32(4710),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0440e0")),    # TEC- D1 R7 
    cms.PSet(detSelection=cms.uint32(4720),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0480e0")),    # TEC- D2 R7 
    cms.PSet(detSelection=cms.uint32(4730),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c0e0")),    # TEC- D3 R7 
    cms.PSet(detSelection=cms.uint32(4740),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0500e0")),    # TEC- D4 R7 
    cms.PSet(detSelection=cms.uint32(4750),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0540e0")),    # TEC- D5 R7 
    cms.PSet(detSelection=cms.uint32(4760),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0580e0")),    # TEC- D6 R7 
    cms.PSet(detSelection=cms.uint32(4770),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c0e0")),    # TEC- D7 R7 
    cms.PSet(detSelection=cms.uint32(4780),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0600e0")),    # TEC- D8 R7 
    cms.PSet(detSelection=cms.uint32(4790),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0640e0")),    # TEC- D9 R7 



    cms.PSet(detSelection=cms.uint32(5110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084020")),    # TEC+ D1 R1 
    cms.PSet(detSelection=cms.uint32(5120),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088020")),    # TEC+ D2 R1 
    cms.PSet(detSelection=cms.uint32(5130),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c020")),    # TEC+ D3 R1 
#    cms.PSet(detSelection=cms.uint32(5140),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090020")),    # TEC+ D4 R1 
#    cms.PSet(detSelection=cms.uint32(5150),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094020")),    # TEC+ D5 R1 
#    cms.PSet(detSelection=cms.uint32(5160),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098020")),    # TEC+ D6 R1 
#    cms.PSet(detSelection=cms.uint32(5170),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c020")),    # TEC+ D7 R1 
#    cms.PSet(detSelection=cms.uint32(5180),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0020")),    # TEC+ D8 R1 
#    cms.PSet(detSelection=cms.uint32(5190),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4020")),    # TEC+ D9 R1 


    cms.PSet(detSelection=cms.uint32(5210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084040")),    # TEC+ D1 R2 
    cms.PSet(detSelection=cms.uint32(5220),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088040")),    # TEC+ D2 R2 
    cms.PSet(detSelection=cms.uint32(5230),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c040")),    # TEC+ D3 R2 
    cms.PSet(detSelection=cms.uint32(5240),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090040")),    # TEC+ D4 R2 
    cms.PSet(detSelection=cms.uint32(5250),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094040")),    # TEC+ D5 R2 
    cms.PSet(detSelection=cms.uint32(5260),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098040")),    # TEC+ D6 R2 
#    cms.PSet(detSelection=cms.uint32(5270),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c040")),    # TEC+ D7 R2 
#    cms.PSet(detSelection=cms.uint32(5280),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0040")),    # TEC+ D8 R2 
#    cms.PSet(detSelection=cms.uint32(5290),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4040")),    # TEC+ D9 R2 

    cms.PSet(detSelection=cms.uint32(5310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084060")),    # TEC+ D1 R3 
    cms.PSet(detSelection=cms.uint32(5320),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088060")),    # TEC+ D2 R3 
    cms.PSet(detSelection=cms.uint32(5330),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c060")),    # TEC+ D3 R3 
    cms.PSet(detSelection=cms.uint32(5340),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090060")),    # TEC+ D4 R3 
    cms.PSet(detSelection=cms.uint32(5350),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094060")),    # TEC+ D5 R3 
    cms.PSet(detSelection=cms.uint32(5360),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098060")),    # TEC+ D6 R3 
    cms.PSet(detSelection=cms.uint32(5370),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c060")),    # TEC+ D7 R3 
    cms.PSet(detSelection=cms.uint32(5380),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0060")),    # TEC+ D8 R3 
#    cms.PSet(detSelection=cms.uint32(5390),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4060")),    # TEC+ D9 R3 

    cms.PSet(detSelection=cms.uint32(5410),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084080")),    # TEC+ D1 R4 
    cms.PSet(detSelection=cms.uint32(5420),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088080")),    # TEC+ D2 R4 
    cms.PSet(detSelection=cms.uint32(5430),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c080")),    # TEC+ D3 R4 
    cms.PSet(detSelection=cms.uint32(5440),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090080")),    # TEC+ D4 R4 
    cms.PSet(detSelection=cms.uint32(5450),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094080")),    # TEC+ D5 R4 
    cms.PSet(detSelection=cms.uint32(5460),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098080")),    # TEC+ D6 R4 
    cms.PSet(detSelection=cms.uint32(5470),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c080")),    # TEC+ D7 R4 
    cms.PSet(detSelection=cms.uint32(5480),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0080")),    # TEC+ D8 R4 
    cms.PSet(detSelection=cms.uint32(5490),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4080")),    # TEC+ D9 R4 

    cms.PSet(detSelection=cms.uint32(5510),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0840a0")),    # TEC+ D1 R5 
    cms.PSet(detSelection=cms.uint32(5520),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0880a0")),    # TEC+ D2 R5 
    cms.PSet(detSelection=cms.uint32(5530),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c0a0")),    # TEC+ D3 R5 
    cms.PSet(detSelection=cms.uint32(5540),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0900a0")),    # TEC+ D4 R5 
    cms.PSet(detSelection=cms.uint32(5550),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0940a0")),    # TEC+ D5 R5 
    cms.PSet(detSelection=cms.uint32(5560),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0980a0")),    # TEC+ D6 R5 
    cms.PSet(detSelection=cms.uint32(5570),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c0a0")),    # TEC+ D7 R5 
    cms.PSet(detSelection=cms.uint32(5580),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a00a0")),    # TEC+ D8 R5 
    cms.PSet(detSelection=cms.uint32(5590),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a40a0")),    # TEC+ D9 R5 

    cms.PSet(detSelection=cms.uint32(5610),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0840c0")),    # TEC+ D1 R6 
    cms.PSet(detSelection=cms.uint32(5620),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0880c0")),    # TEC+ D2 R6 
    cms.PSet(detSelection=cms.uint32(5630),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c0c0")),    # TEC+ D3 R6 
    cms.PSet(detSelection=cms.uint32(5640),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0900c0")),    # TEC+ D4 R6 
    cms.PSet(detSelection=cms.uint32(5650),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0940c0")),    # TEC+ D5 R6 
    cms.PSet(detSelection=cms.uint32(5660),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0980c0")),    # TEC+ D6 R6 
    cms.PSet(detSelection=cms.uint32(5670),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c0c0")),    # TEC+ D7 R6 
    cms.PSet(detSelection=cms.uint32(5680),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a00c0")),    # TEC+ D8 R6 
    cms.PSet(detSelection=cms.uint32(5690),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a40c0")),    # TEC+ D9 R6 

    cms.PSet(detSelection=cms.uint32(5710),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0840e0")),    # TEC+ D1 R7 
    cms.PSet(detSelection=cms.uint32(5720),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0880e0")),    # TEC+ D2 R7 
    cms.PSet(detSelection=cms.uint32(5730),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c0e0")),    # TEC+ D3 R7 
    cms.PSet(detSelection=cms.uint32(5740),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0900e0")),    # TEC+ D4 R7 
    cms.PSet(detSelection=cms.uint32(5750),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0940e0")),    # TEC+ D5 R7 
    cms.PSet(detSelection=cms.uint32(5760),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0980e0")),    # TEC+ D6 R7 
    cms.PSet(detSelection=cms.uint32(5770),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c0e0")),    # TEC+ D7 R7 
    cms.PSet(detSelection=cms.uint32(5780),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a00e0")),    # TEC+ D8 R7 
    cms.PSet(detSelection=cms.uint32(5790),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a40e0"))    # TEC+ D9 R7 



    )
    )
