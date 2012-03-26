import FWCore.ParameterSet.Config as cms

OccupancyPlotsPixelWantedSubDets = cms.VPSet (
    cms.PSet(detSelection=cms.uint32(111),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12010004")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(112),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12010008")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(113),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x1201000c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(114),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12010010")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(115),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12010014")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(116),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12010018")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(117),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x1201001c")),      # BPix L1 mod 1

    cms.PSet(detSelection=cms.uint32(118),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12010020")),      # BPix L1 mod 1#
    cms.PSet(detSelection=cms.uint32(121),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12020004")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(122),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12020008")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(123),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x1202000c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(124),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12020010")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(125),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12020014")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(126),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12020018")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(127),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x1202001c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(128),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12020020")),      # BPix L1 mod 1

    cms.PSet(detSelection=cms.uint32(131),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12030004")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(132),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12030008")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(133),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x1203000c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(134),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12030010")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(135),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12030014")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(136),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12030018")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(137),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x1203001c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(138),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0f00fc-0x12030020")),      # BPix L1 mod 1

    cms.PSet(detSelection=cms.uint32(211),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14810104")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(212),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14810204")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(213),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14810108")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(214),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14810208")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(215),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1481010c")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(216),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1481020c")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(217),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14810110")),      # FPix minus
#    cms.PSet(detSelection=cms.uint32(),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14810210"))      # FPix minus
    
    cms.PSet(detSelection=cms.uint32(221),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14820104")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(222),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14820204")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(223),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14820108")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(224),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14820208")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(225),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1482010c")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(226),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1482020c")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(227),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14820110")),      # FPix minus
#    cms.PSet(detSelection=cms.uint32(),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x14820210"))      # FPix minus
    
    cms.PSet(detSelection=cms.uint32(231),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15010104")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(232),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15010204")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(233),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15010108")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(234),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15010208")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(235),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1501010c")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(236),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1501020c")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(237),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15010110")),      # FPix minus
#    cms.PSet(detSelection=cms.uint32(),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15010210"))      # FPix minus
    
    cms.PSet(detSelection=cms.uint32(241),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15020104")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(242),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15020204")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(243),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15020108")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(244),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15020208")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(245),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1502010c")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(246),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x1502020c")),      # FPix minus
    cms.PSet(detSelection=cms.uint32(247),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15020110"))      # FPix minus
#    cms.PSet(detSelection=cms.uint32(),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1f8f03fc-0x15020210"))      # FPix minus

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

     cms.PSet(detSelection=cms.uint32(2111),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18002b00")),     # TID- D1 R1 Front
     cms.PSet(detSelection=cms.uint32(2112),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18002a80")),     # TID- D1 R1 Back
     cms.PSet(detSelection=cms.uint32(2121),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18003300")),     # TID- D2 R1 Front
     cms.PSet(detSelection=cms.uint32(2122),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18003280")),     # TID- D2 R1 Back
     cms.PSet(detSelection=cms.uint32(2131),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18003b00")),     # TID- D3 R1 Front
     cms.PSet(detSelection=cms.uint32(2132),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18003a80")),     # TID- D3 R1 Back

     cms.PSet(detSelection=cms.uint32(2211),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18002d00")),     # TID- D1 R2 Front
     cms.PSet(detSelection=cms.uint32(2212),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18002c80")),     # TID- D1 R2 Back
     cms.PSet(detSelection=cms.uint32(2221),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18003500")),     # TID- D2 R2 Front
     cms.PSet(detSelection=cms.uint32(2222),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18003480")),     # TID- D2 R2 Back
     cms.PSet(detSelection=cms.uint32(2231),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18003d00")),     # TID- D3 R2 Front
     cms.PSet(detSelection=cms.uint32(2232),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18003c80")),     # TID- D3 R2 Back

     cms.PSet(detSelection=cms.uint32(2311),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18002f00")),     # TID- D1 R3 Front
     cms.PSet(detSelection=cms.uint32(2312),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18002e80")),     # TID- D1 R3 Back
     cms.PSet(detSelection=cms.uint32(2321),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18003700")),     # TID- D2 R3 Front
     cms.PSet(detSelection=cms.uint32(2322),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18003680")),     # TID- D2 R3 Back
     cms.PSet(detSelection=cms.uint32(2331),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18003f00")),     # TID- D3 R3 Front
     cms.PSet(detSelection=cms.uint32(2332),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18003e80")),     # TID- D3 R3 Back

     cms.PSet(detSelection=cms.uint32(2141),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18004b00")),     # TID+ D1 R1 Front
     cms.PSet(detSelection=cms.uint32(2142),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18004a80")),     # TID+ D1 R1 Back
     cms.PSet(detSelection=cms.uint32(2151),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18005300")),     # TID+ D2 R1 Front
     cms.PSet(detSelection=cms.uint32(2152),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18005280")),     # TID+ D2 R1 Back
     cms.PSet(detSelection=cms.uint32(2161),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18005b00")),     # TID+ D3 R1 Front
     cms.PSet(detSelection=cms.uint32(2162),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18005a80")),     # TID+ D3 R1 Back

     cms.PSet(detSelection=cms.uint32(2241),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18004d00")),     # TID+ D1 R2 Front
     cms.PSet(detSelection=cms.uint32(2242),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18004c80")),     # TID+ D1 R2 Back
     cms.PSet(detSelection=cms.uint32(2251),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18005500")),     # TID+ D2 R2 Front
     cms.PSet(detSelection=cms.uint32(2252),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18005480")),     # TID+ D2 R2 Back
     cms.PSet(detSelection=cms.uint32(2261),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18005d00")),     # TID+ D3 R2 Front
     cms.PSet(detSelection=cms.uint32(2262),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18005c80")),     # TID+ D3 R2 Back

     cms.PSet(detSelection=cms.uint32(2341),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18004f00")),     # TID+ D1 R3 Front
     cms.PSet(detSelection=cms.uint32(2342),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18004e80")),     # TID+ D1 R3 Back
     cms.PSet(detSelection=cms.uint32(2351),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18005700")),     # TID+ D2 R3 Front
     cms.PSet(detSelection=cms.uint32(2352),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18005680")),     # TID+ D2 R3 Back
     cms.PSet(detSelection=cms.uint32(2361),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18005f00")),     # TID+ D3 R3 Front
     cms.PSet(detSelection=cms.uint32(2362),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007f80-0x18005e80")),     # TID+ D3 R3 Back

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
    cms.PSet(detSelection=cms.uint32(4111),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c045020")),    # TEC- D1 R1 back
    cms.PSet(detSelection=cms.uint32(4112),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c046020")),    # TEC- D1 R1 front
    cms.PSet(detSelection=cms.uint32(4121),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c049020")),    # TEC- D2 R1 back
    cms.PSet(detSelection=cms.uint32(4122),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04a020")),    # TEC- D2 R1 front
    cms.PSet(detSelection=cms.uint32(4131),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04d020")),    # TEC- D3 R1 back
    cms.PSet(detSelection=cms.uint32(4132),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04e020")),    # TEC- D3 R1 front
#    cms.PSet(detSelection=cms.uint32(4141),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c051020")),    # TEC- D4 R1 back
#    cms.PSet(detSelection=cms.uint32(4142),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c052020")),    # TEC- D4 R1 front
#    cms.PSet(detSelection=cms.uint32(4151),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c055020")),    # TEC- D5 R1 back
#    cms.PSet(detSelection=cms.uint32(4152),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c056020")),    # TEC- D5 R1 front
#    cms.PSet(detSelection=cms.uint32(4161),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c059020")),    # TEC- D6 R1 back
#    cms.PSet(detSelection=cms.uint32(4162),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05a020")),    # TEC- D6 R1 front
#    cms.PSet(detSelection=cms.uint32(4171),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05d020")),    # TEC- D7 R1 back
#    cms.PSet(detSelection=cms.uint32(4172),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05e020")),    # TEC- D7 R1 front
#    cms.PSet(detSelection=cms.uint32(4181),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c061020")),    # TEC- D8 R1 back
#    cms.PSet(detSelection=cms.uint32(4182),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c062020")),    # TEC- D8 R1 front
#    cms.PSet(detSelection=cms.uint32(4191),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c065020")),    # TEC- D9 R1 back
#    cms.PSet(detSelection=cms.uint32(4192),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c066020")),    # TEC- D9 R1 front

    cms.PSet(detSelection=cms.uint32(4211),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c045040")),    # TEC- D1 R2 back
    cms.PSet(detSelection=cms.uint32(4212),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c046040")),    # TEC- D1 R2 front
    cms.PSet(detSelection=cms.uint32(4221),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c049040")),    # TEC- D2 R2 back
    cms.PSet(detSelection=cms.uint32(4222),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04a040")),    # TEC- D2 R2 front
    cms.PSet(detSelection=cms.uint32(4231),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04d040")),    # TEC- D3 R2 back
    cms.PSet(detSelection=cms.uint32(4232),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04e040")),    # TEC- D3 R2 front
    cms.PSet(detSelection=cms.uint32(4241),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c051040")),    # TEC- D4 R2 back
    cms.PSet(detSelection=cms.uint32(4242),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c052040")),    # TEC- D4 R2 front
    cms.PSet(detSelection=cms.uint32(4251),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c055040")),    # TEC- D5 R2 back
    cms.PSet(detSelection=cms.uint32(4252),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c056040")),    # TEC- D5 R2 front
    cms.PSet(detSelection=cms.uint32(4261),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c059040")),    # TEC- D6 R2 back
    cms.PSet(detSelection=cms.uint32(4262),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05a040")),    # TEC- D6 R2 front
#    cms.PSet(detSelection=cms.uint32(4271),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05d040")),    # TEC- D7 R2 back
#    cms.PSet(detSelection=cms.uint32(4272),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05e040")),    # TEC- D7 R2 front
#    cms.PSet(detSelection=cms.uint32(4281),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c061040")),    # TEC- D8 R2 back
#    cms.PSet(detSelection=cms.uint32(4282),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c062040")),    # TEC- D8 R2 front
#    cms.PSet(detSelection=cms.uint32(4291),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c065040")),    # TEC- D9 R2 back
#    cms.PSet(detSelection=cms.uint32(4292),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c066040")),    # TEC- D9 R2 front

    cms.PSet(detSelection=cms.uint32(4311),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c045060")),    # TEC- D1 R3 back
    cms.PSet(detSelection=cms.uint32(4312),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c046060")),    # TEC- D1 R3 front
    cms.PSet(detSelection=cms.uint32(4321),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c049060")),    # TEC- D2 R3 back
    cms.PSet(detSelection=cms.uint32(4322),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04a060")),    # TEC- D2 R3 front
    cms.PSet(detSelection=cms.uint32(4331),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04d060")),    # TEC- D3 R3 back
    cms.PSet(detSelection=cms.uint32(4332),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04e060")),    # TEC- D3 R3 front
    cms.PSet(detSelection=cms.uint32(4341),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c051060")),    # TEC- D4 R3 back
    cms.PSet(detSelection=cms.uint32(4342),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c052060")),    # TEC- D4 R3 front
    cms.PSet(detSelection=cms.uint32(4351),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c055060")),    # TEC- D5 R3 back
    cms.PSet(detSelection=cms.uint32(4352),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c056060")),    # TEC- D5 R3 front
    cms.PSet(detSelection=cms.uint32(4361),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c059060")),    # TEC- D6 R3 back
    cms.PSet(detSelection=cms.uint32(4362),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05a060")),    # TEC- D6 R3 front
    cms.PSet(detSelection=cms.uint32(4371),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05d060")),    # TEC- D7 R3 back
    cms.PSet(detSelection=cms.uint32(4372),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05e060")),    # TEC- D7 R3 front
    cms.PSet(detSelection=cms.uint32(4381),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c061060")),    # TEC- D8 R3 back
    cms.PSet(detSelection=cms.uint32(4382),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c062060")),    # TEC- D8 R3 front
#    cms.PSet(detSelection=cms.uint32(4391),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c065060")),    # TEC- D9 R3 back
#    cms.PSet(detSelection=cms.uint32(4392),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c066060")),    # TEC- D9 R3 front

    cms.PSet(detSelection=cms.uint32(4411),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c045080")),    # TEC- D1 R4 back
    cms.PSet(detSelection=cms.uint32(4412),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c046080")),    # TEC- D1 R4 front
    cms.PSet(detSelection=cms.uint32(4421),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c049080")),    # TEC- D2 R4 back
    cms.PSet(detSelection=cms.uint32(4422),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04a080")),    # TEC- D2 R4 front
    cms.PSet(detSelection=cms.uint32(4431),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04d080")),    # TEC- D3 R4 back
    cms.PSet(detSelection=cms.uint32(4432),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04e080")),    # TEC- D3 R4 front
    cms.PSet(detSelection=cms.uint32(4441),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c051080")),    # TEC- D4 R4 back
    cms.PSet(detSelection=cms.uint32(4442),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c052080")),    # TEC- D4 R4 front
    cms.PSet(detSelection=cms.uint32(4451),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c055080")),    # TEC- D5 R4 back
    cms.PSet(detSelection=cms.uint32(4452),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c056080")),    # TEC- D5 R4 front
    cms.PSet(detSelection=cms.uint32(4461),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c059080")),    # TEC- D6 R4 back
    cms.PSet(detSelection=cms.uint32(4462),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05a080")),    # TEC- D6 R4 front
    cms.PSet(detSelection=cms.uint32(4471),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05d080")),    # TEC- D7 R4 back
    cms.PSet(detSelection=cms.uint32(4472),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05e080")),    # TEC- D7 R4 front
    cms.PSet(detSelection=cms.uint32(4481),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c061080")),    # TEC- D8 R4 back
    cms.PSet(detSelection=cms.uint32(4482),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c062080")),    # TEC- D8 R4 front
    cms.PSet(detSelection=cms.uint32(4491),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c065080")),    # TEC- D9 R4 back
    cms.PSet(detSelection=cms.uint32(4492),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c066080")),    # TEC- D9 R4 front

    cms.PSet(detSelection=cms.uint32(4511),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0450a0")),    # TEC- D1 R5 back
    cms.PSet(detSelection=cms.uint32(4512),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0460a0")),    # TEC- D1 R5 front
    cms.PSet(detSelection=cms.uint32(4521),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0490a0")),    # TEC- D2 R5 back
    cms.PSet(detSelection=cms.uint32(4522),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04a0a0")),    # TEC- D2 R5 front
    cms.PSet(detSelection=cms.uint32(4531),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04d0a0")),    # TEC- D3 R5 back
    cms.PSet(detSelection=cms.uint32(4532),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04e0a0")),    # TEC- D3 R5 front
    cms.PSet(detSelection=cms.uint32(4541),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0510a0")),    # TEC- D4 R5 back
    cms.PSet(detSelection=cms.uint32(4542),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0520a0")),    # TEC- D4 R5 front
    cms.PSet(detSelection=cms.uint32(4551),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0550a0")),    # TEC- D5 R5 back
    cms.PSet(detSelection=cms.uint32(4552),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0560a0")),    # TEC- D5 R5 front
    cms.PSet(detSelection=cms.uint32(4561),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0590a0")),    # TEC- D6 R5 back
    cms.PSet(detSelection=cms.uint32(4562),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05a0a0")),    # TEC- D6 R5 front
    cms.PSet(detSelection=cms.uint32(4571),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05d0a0")),    # TEC- D7 R5 back
    cms.PSet(detSelection=cms.uint32(4572),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05e0a0")),    # TEC- D7 R5 front
    cms.PSet(detSelection=cms.uint32(4581),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0610a0")),    # TEC- D8 R5 back
    cms.PSet(detSelection=cms.uint32(4582),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0620a0")),    # TEC- D8 R5 front
    cms.PSet(detSelection=cms.uint32(4591),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0650a0")),    # TEC- D9 R5 back
    cms.PSet(detSelection=cms.uint32(4592),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0660a0")),    # TEC- D9 R5 front

    cms.PSet(detSelection=cms.uint32(4611),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0450c0")),    # TEC- D1 R6 back
    cms.PSet(detSelection=cms.uint32(4612),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0460c0")),    # TEC- D1 R6 front
    cms.PSet(detSelection=cms.uint32(4621),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0490c0")),    # TEC- D2 R6 back
    cms.PSet(detSelection=cms.uint32(4622),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04a0c0")),    # TEC- D2 R6 front
    cms.PSet(detSelection=cms.uint32(4631),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04d0c0")),    # TEC- D3 R6 back
    cms.PSet(detSelection=cms.uint32(4632),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04e0c0")),    # TEC- D3 R6 front
    cms.PSet(detSelection=cms.uint32(4641),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0510c0")),    # TEC- D4 R6 back
    cms.PSet(detSelection=cms.uint32(4642),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0520c0")),    # TEC- D4 R6 front
    cms.PSet(detSelection=cms.uint32(4651),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0550c0")),    # TEC- D5 R6 back
    cms.PSet(detSelection=cms.uint32(4652),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0560c0")),    # TEC- D5 R6 front
    cms.PSet(detSelection=cms.uint32(4661),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0590c0")),    # TEC- D6 R6 back
    cms.PSet(detSelection=cms.uint32(4662),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05a0c0")),    # TEC- D6 R6 front
    cms.PSet(detSelection=cms.uint32(4671),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05d0c0")),    # TEC- D7 R6 back
    cms.PSet(detSelection=cms.uint32(4672),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05e0c0")),    # TEC- D7 R6 front
    cms.PSet(detSelection=cms.uint32(4681),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0610c0")),    # TEC- D8 R6 back
    cms.PSet(detSelection=cms.uint32(4682),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0620c0")),    # TEC- D8 R6 front
    cms.PSet(detSelection=cms.uint32(4691),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0650c0")),    # TEC- D9 R6 back
    cms.PSet(detSelection=cms.uint32(4692),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0660c0")),    # TEC- D9 R6 front

    cms.PSet(detSelection=cms.uint32(4711),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0450e0")),    # TEC- D1 R7 back
    cms.PSet(detSelection=cms.uint32(4712),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0460e0")),    # TEC- D1 R7 front
    cms.PSet(detSelection=cms.uint32(4721),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0490e0")),    # TEC- D2 R7 back
    cms.PSet(detSelection=cms.uint32(4722),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04a0e0")),    # TEC- D2 R7 front
    cms.PSet(detSelection=cms.uint32(4731),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04d0e0")),    # TEC- D3 R7 back
    cms.PSet(detSelection=cms.uint32(4732),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c04e0e0")),    # TEC- D3 R7 front
    cms.PSet(detSelection=cms.uint32(4741),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0510e0")),    # TEC- D4 R7 back
    cms.PSet(detSelection=cms.uint32(4742),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0520e0")),    # TEC- D4 R7 front
    cms.PSet(detSelection=cms.uint32(4751),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0550e0")),    # TEC- D5 R7 back
    cms.PSet(detSelection=cms.uint32(4752),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0560e0")),    # TEC- D5 R7 front
    cms.PSet(detSelection=cms.uint32(4761),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0590e0")),    # TEC- D6 R7 back
    cms.PSet(detSelection=cms.uint32(4762),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05a0e0")),    # TEC- D6 R7 front
    cms.PSet(detSelection=cms.uint32(4771),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05d0e0")),    # TEC- D7 R7 back
    cms.PSet(detSelection=cms.uint32(4772),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c05e0e0")),    # TEC- D7 R7 front
    cms.PSet(detSelection=cms.uint32(4781),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0610e0")),    # TEC- D8 R7 back
    cms.PSet(detSelection=cms.uint32(4782),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0620e0")),    # TEC- D8 R7 front
    cms.PSet(detSelection=cms.uint32(4791),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0650e0")),    # TEC- D9 R7 back
    cms.PSet(detSelection=cms.uint32(4792),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0660e0")),    # TEC- D9 R7 front

    cms.PSet(detSelection=cms.uint32(5111),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c085020")),    # TEC+ D1 R1 back
    cms.PSet(detSelection=cms.uint32(5112),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c086020")),    # TEC+ D1 R1 front
    cms.PSet(detSelection=cms.uint32(5121),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c089020")),    # TEC+ D2 R1 back
    cms.PSet(detSelection=cms.uint32(5122),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08a020")),    # TEC+ D2 R1 front
    cms.PSet(detSelection=cms.uint32(5131),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08d020")),    # TEC+ D3 R1 back
    cms.PSet(detSelection=cms.uint32(5132),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08e020")),    # TEC+ D3 R1 front
#    cms.PSet(detSelection=cms.uint32(51411),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c091020")),    # TEC+ D4 R1 back
#    cms.PSet(detSelection=cms.uint32(5142),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c092020")),    # TEC+ D4 R1 front
#    cms.PSet(detSelection=cms.uint32(5151),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c095020")),    # TEC+ D5 R1 back
#    cms.PSet(detSelection=cms.uint32(5152),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c096020")),    # TEC+ D5 R1 front
#    cms.PSet(detSelection=cms.uint32(5161),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c099020")),    # TEC+ D6 R1 back
#    cms.PSet(detSelection=cms.uint32(5162),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09a020")),    # TEC+ D6 R1 front
#    cms.PSet(detSelection=cms.uint32(5171),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09d020")),    # TEC+ D7 R1 back
#    cms.PSet(detSelection=cms.uint32(5172),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09e020")),    # TEC+ D7 R1 front
#    cms.PSet(detSelection=cms.uint32(5181),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a1020")),    # TEC+ D8 R1 back
#    cms.PSet(detSelection=cms.uint32(5182),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a2020")),    # TEC+ D8 R1 front
#    cms.PSet(detSelection=cms.uint32(5191),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a5020")),    # TEC+ D9 R1 back
#    cms.PSet(detSelection=cms.uint32(5192),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a6020")),    # TEC+ D9 R1 front

    cms.PSet(detSelection=cms.uint32(5211),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c085040")),    # TEC+ D1 R2 back
    cms.PSet(detSelection=cms.uint32(5212),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c086040")),    # TEC+ D1 R2 front
    cms.PSet(detSelection=cms.uint32(5221),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c089040")),    # TEC+ D2 R2 back
    cms.PSet(detSelection=cms.uint32(5222),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08a040")),    # TEC+ D2 R2 front
    cms.PSet(detSelection=cms.uint32(5231),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08d040")),    # TEC+ D3 R2 back
    cms.PSet(detSelection=cms.uint32(5232),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08e040")),    # TEC+ D3 R2 front
    cms.PSet(detSelection=cms.uint32(5241),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c091040")),    # TEC+ D4 R2 back
    cms.PSet(detSelection=cms.uint32(5242),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c092040")),    # TEC+ D4 R2 front
    cms.PSet(detSelection=cms.uint32(5251),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c095040")),    # TEC+ D5 R2 back
    cms.PSet(detSelection=cms.uint32(5252),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c096040")),    # TEC+ D5 R2 front
    cms.PSet(detSelection=cms.uint32(5261),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c099040")),    # TEC+ D6 R2 back
    cms.PSet(detSelection=cms.uint32(5262),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09a040")),    # TEC+ D6 R2 front
#    cms.PSet(detSelection=cms.uint32(5271),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09d040")),    # TEC+ D7 R2 back
#    cms.PSet(detSelection=cms.uint32(5272),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09e040")),    # TEC+ D7 R2 front
#    cms.PSet(detSelection=cms.uint32(5281),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a1040")),    # TEC+ D8 R2 back
#    cms.PSet(detSelection=cms.uint32(5282),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a2040")),    # TEC+ D8 R2 front
#    cms.PSet(detSelection=cms.uint32(5291),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a5040")),    # TEC+ D9 R2 back
#    cms.PSet(detSelection=cms.uint32(5292),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a6040")),    # TEC+ D9 R2 front

    cms.PSet(detSelection=cms.uint32(5311),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c085060")),    # TEC+ D1 R3 back
    cms.PSet(detSelection=cms.uint32(5312),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c086060")),    # TEC+ D1 R3 front
    cms.PSet(detSelection=cms.uint32(5321),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c089060")),    # TEC+ D2 R3 back
    cms.PSet(detSelection=cms.uint32(5322),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08a060")),    # TEC+ D2 R3 front
    cms.PSet(detSelection=cms.uint32(5331),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08d060")),    # TEC+ D3 R3 back
    cms.PSet(detSelection=cms.uint32(5332),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08e060")),    # TEC+ D3 R3 front
    cms.PSet(detSelection=cms.uint32(5341),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c091060")),    # TEC+ D4 R3 back
    cms.PSet(detSelection=cms.uint32(5342),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c092060")),    # TEC+ D4 R3 front
    cms.PSet(detSelection=cms.uint32(5351),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c095060")),    # TEC+ D5 R3 back
    cms.PSet(detSelection=cms.uint32(5352),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c096060")),    # TEC+ D5 R3 front
    cms.PSet(detSelection=cms.uint32(5361),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c099060")),    # TEC+ D6 R3 back
    cms.PSet(detSelection=cms.uint32(5362),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09a060")),    # TEC+ D6 R3 front
    cms.PSet(detSelection=cms.uint32(5371),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09d060")),    # TEC+ D7 R3 back
    cms.PSet(detSelection=cms.uint32(5372),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09e060")),    # TEC+ D7 R3 front
    cms.PSet(detSelection=cms.uint32(5381),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a1060")),    # TEC+ D8 R3 back
    cms.PSet(detSelection=cms.uint32(5382),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a2060")),    # TEC+ D8 R3 front
#    cms.PSet(detSelection=cms.uint32(5391),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a5060")),    # TEC+ D9 R3 back
#    cms.PSet(detSelection=cms.uint32(5392),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a6060")),    # TEC+ D9 R3 front

    cms.PSet(detSelection=cms.uint32(5411),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c085080")),    # TEC+ D1 R4 back
    cms.PSet(detSelection=cms.uint32(5412),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c086080")),    # TEC+ D1 R4 front
    cms.PSet(detSelection=cms.uint32(5421),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c089080")),    # TEC+ D2 R4 back
    cms.PSet(detSelection=cms.uint32(5422),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08a080")),    # TEC+ D2 R4 front
    cms.PSet(detSelection=cms.uint32(5431),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08d080")),    # TEC+ D3 R4 back
    cms.PSet(detSelection=cms.uint32(5432),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08e080")),    # TEC+ D3 R4 front
    cms.PSet(detSelection=cms.uint32(5441),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c091080")),    # TEC+ D4 R4 back
    cms.PSet(detSelection=cms.uint32(5442),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c092080")),    # TEC+ D4 R4 front
    cms.PSet(detSelection=cms.uint32(5451),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c095080")),    # TEC+ D5 R4 back
    cms.PSet(detSelection=cms.uint32(5452),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c096080")),    # TEC+ D5 R4 front
    cms.PSet(detSelection=cms.uint32(5461),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c099080")),    # TEC+ D6 R4 back
    cms.PSet(detSelection=cms.uint32(5462),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09a080")),    # TEC+ D6 R4 front
    cms.PSet(detSelection=cms.uint32(5471),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09d080")),    # TEC+ D7 R4 back
    cms.PSet(detSelection=cms.uint32(5472),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09e080")),    # TEC+ D7 R4 front
    cms.PSet(detSelection=cms.uint32(5481),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a1080")),    # TEC+ D8 R4 back
    cms.PSet(detSelection=cms.uint32(5482),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a2080")),    # TEC+ D8 R4 front
    cms.PSet(detSelection=cms.uint32(5491),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a5080")),    # TEC+ D9 R4 back
    cms.PSet(detSelection=cms.uint32(5492),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a6080")),    # TEC+ D9 R4 front

    cms.PSet(detSelection=cms.uint32(5511),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0850a0")),    # TEC+ D1 R5 back
    cms.PSet(detSelection=cms.uint32(5512),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0860a0")),    # TEC+ D1 R5 front
    cms.PSet(detSelection=cms.uint32(5521),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0890a0")),    # TEC+ D2 R5 back
    cms.PSet(detSelection=cms.uint32(5522),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08a0a0")),    # TEC+ D2 R5 front
    cms.PSet(detSelection=cms.uint32(5531),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08d0a0")),    # TEC+ D3 R5 back
    cms.PSet(detSelection=cms.uint32(5532),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08e0a0")),    # TEC+ D3 R5 front
    cms.PSet(detSelection=cms.uint32(5541),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0910a0")),    # TEC+ D4 R5 back
    cms.PSet(detSelection=cms.uint32(5542),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0920a0")),    # TEC+ D4 R5 front
    cms.PSet(detSelection=cms.uint32(5551),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0950a0")),    # TEC+ D5 R5 back
    cms.PSet(detSelection=cms.uint32(5552),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0960a0")),    # TEC+ D5 R5 front
    cms.PSet(detSelection=cms.uint32(5561),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0990a0")),    # TEC+ D6 R5 back
    cms.PSet(detSelection=cms.uint32(5562),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09a0a0")),    # TEC+ D6 R5 front
    cms.PSet(detSelection=cms.uint32(5571),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09d0a0")),    # TEC+ D7 R5 back
    cms.PSet(detSelection=cms.uint32(5572),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09e0a0")),    # TEC+ D7 R5 front
    cms.PSet(detSelection=cms.uint32(5571),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09d0a0")),    # TEC+ D7 R5 back
    cms.PSet(detSelection=cms.uint32(5572),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09e0a0")),    # TEC+ D7 R5 front
    cms.PSet(detSelection=cms.uint32(5581),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a10a0")),    # TEC+ D8 R5 back
    cms.PSet(detSelection=cms.uint32(5582),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a20a0")),    # TEC+ D8 R5 front
    cms.PSet(detSelection=cms.uint32(5591),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a50a0")),    # TEC+ D9 R5 back
    cms.PSet(detSelection=cms.uint32(5592),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a60a0")),    # TEC+ D9 R5 front

    cms.PSet(detSelection=cms.uint32(5611),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0850c0")),    # TEC+ D1 R6 back
    cms.PSet(detSelection=cms.uint32(5612),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0860c0")),    # TEC+ D1 R6 front
    cms.PSet(detSelection=cms.uint32(5621),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0890c0")),    # TEC+ D2 R6 back
    cms.PSet(detSelection=cms.uint32(5622),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08a0c0")),    # TEC+ D2 R6 front
    cms.PSet(detSelection=cms.uint32(5631),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08d0c0")),    # TEC+ D3 R6 back
    cms.PSet(detSelection=cms.uint32(5632),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08e0c0")),    # TEC+ D3 R6 front
    cms.PSet(detSelection=cms.uint32(5641),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0910c0")),    # TEC+ D4 R6 back
    cms.PSet(detSelection=cms.uint32(5642),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0920c0")),    # TEC+ D4 R6 front
    cms.PSet(detSelection=cms.uint32(5651),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0950c0")),    # TEC+ D5 R6 back
    cms.PSet(detSelection=cms.uint32(5652),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0960c0")),    # TEC+ D5 R6 front
    cms.PSet(detSelection=cms.uint32(5661),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0990c0")),    # TEC+ D6 R6 back
    cms.PSet(detSelection=cms.uint32(5662),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09a0c0")),    # TEC+ D6 R6 front
    cms.PSet(detSelection=cms.uint32(5671),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09d0c0")),    # TEC+ D7 R6 back
    cms.PSet(detSelection=cms.uint32(5672),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09e0c0")),    # TEC+ D7 R6 front
    cms.PSet(detSelection=cms.uint32(5681),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a10c0")),    # TEC+ D8 R6 back
    cms.PSet(detSelection=cms.uint32(5682),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a20c0")),    # TEC+ D8 R6 front
    cms.PSet(detSelection=cms.uint32(5691),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a50c0")),    # TEC+ D9 R6 back
    cms.PSet(detSelection=cms.uint32(5692),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a60c0")),    # TEC+ D9 R6 front

    cms.PSet(detSelection=cms.uint32(5711),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0850e0")),    # TEC+ D1 R7 back
    cms.PSet(detSelection=cms.uint32(5712),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0860e0")),    # TEC+ D1 R7 front
    cms.PSet(detSelection=cms.uint32(5721),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0890e0")),    # TEC+ D2 R7 back
    cms.PSet(detSelection=cms.uint32(5722),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08a0e0")),    # TEC+ D2 R7 front
    cms.PSet(detSelection=cms.uint32(5731),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08d0e0")),    # TEC+ D3 R7 back
    cms.PSet(detSelection=cms.uint32(5732),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c08e0e0")),    # TEC+ D3 R7 front
    cms.PSet(detSelection=cms.uint32(5741),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0910e0")),    # TEC+ D4 R7 back
    cms.PSet(detSelection=cms.uint32(5742),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0920e0")),    # TEC+ D4 R7 front
    cms.PSet(detSelection=cms.uint32(5751),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0950e0")),    # TEC+ D5 R7 back
    cms.PSet(detSelection=cms.uint32(5752),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0960e0")),    # TEC+ D5 R7 front
    cms.PSet(detSelection=cms.uint32(5761),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0990e0")),    # TEC+ D6 R7 back
    cms.PSet(detSelection=cms.uint32(5762),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09a0e0")),    # TEC+ D6 R7 front
    cms.PSet(detSelection=cms.uint32(5771),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09d0e0")),    # TEC+ D7 R7 back
    cms.PSet(detSelection=cms.uint32(5772),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c09e0e0")),    # TEC+ D7 R7 front
    cms.PSet(detSelection=cms.uint32(5781),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a10e0")),    # TEC+ D8 R7 back
    cms.PSet(detSelection=cms.uint32(5782),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a20e0")),    # TEC+ D8 R7 front
    cms.PSet(detSelection=cms.uint32(5791),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a50e0")),    # TEC+ D9 R7 back
    cms.PSet(detSelection=cms.uint32(5792),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0ff0e0-0x1c0a60e0"))    # TEC+ D9 R7 front


    )
    )
