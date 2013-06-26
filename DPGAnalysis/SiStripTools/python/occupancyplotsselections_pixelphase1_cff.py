import FWCore.ParameterSet.Config as cms

#OccupancyPlotsPixelWantedSubDets = cms.VPSet (
#    cms.PSet(detSelection = cms.uint32(111),detLabel = cms.string("FPIXmD1pan1"),selection=cms.untracked.vstring("0x1f8f0300-0x14810100")),
#    cms.PSet(detSelection = cms.uint32(121),detLabel = cms.string("FPIXmD2pan1"),selection=cms.untracked.vstring("0x1f8f0300-0x14820100")),
#    cms.PSet(detSelection = cms.uint32(131),detLabel = cms.string("FPIXmD3pan1"),selection=cms.untracked.vstring("0x1f8f0300-0x14830100")),
#    cms.PSet(detSelection = cms.uint32(211),detLabel = cms.string("FPIXpD1pan1"),selection=cms.untracked.vstring("0x1f8f0300-0x15010100")),
#    cms.PSet(detSelection = cms.uint32(221),detLabel = cms.string("FPIXpD2pan1"),selection=cms.untracked.vstring("0x1f8f0300-0x15020100")),
#    cms.PSet(detSelection = cms.uint32(231),detLabel = cms.string("FPIXpD3pan1"),selection=cms.untracked.vstring("0x1f8f0300-0x15030100")),
#    cms.PSet(detSelection = cms.uint32(112),detLabel = cms.string("FPIXmD1pan2"),selection=cms.untracked.vstring("0x1f8f0300-0x14810200")),
#    cms.PSet(detSelection = cms.uint32(122),detLabel = cms.string("FPIXmD2pan2"),selection=cms.untracked.vstring("0x1f8f0300-0x14820200")),
#    cms.PSet(detSelection = cms.uint32(132),detLabel = cms.string("FPIXmD3pan2"),selection=cms.untracked.vstring("0x1f8f0300-0x14830200")),
#    cms.PSet(detSelection = cms.uint32(212),detLabel = cms.string("FPIXpD1pan2"),selection=cms.untracked.vstring("0x1f8f0300-0x15010200")),
#    cms.PSet(detSelection = cms.uint32(222),detLabel = cms.string("FPIXpD2pan2"),selection=cms.untracked.vstring("0x1f8f0300-0x15020200")),
#    cms.PSet(detSelection = cms.uint32(232),detLabel = cms.string("FPIXpD3pan2"),selection=cms.untracked.vstring("0x1f8f0300-0x15030200"))
#    )

OccupancyPlotsPixelWantedSubDets = cms.VPSet (
    cms.PSet(detSelection=cms.uint32(111),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12040004")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(112),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12040008")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(113),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x1204000c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(114),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12040010")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(115),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12040014")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(116),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12040018")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(117),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x1204001c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(118),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12040020")),      # BPix L1 mod 1

    cms.PSet(detSelection=cms.uint32(121),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12080004")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(122),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12080008")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(123),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x1208000c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(124),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12080010")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(125),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12080014")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(126),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12080018")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(127),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x1208001c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(128),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12080020")),      # BPix L1 mod 1

    cms.PSet(detSelection=cms.uint32(131),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x120c0004")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(132),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x120c0008")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(133),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x120c000c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(134),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x120c0010")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(135),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x120c0014")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(136),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x120c0018")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(137),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x120c001c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(138),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x120c0020")),      # BPix L1 mod 1

    cms.PSet(detSelection=cms.uint32(141),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12100004")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(142),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12100008")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(143),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x1210000c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(144),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12100010")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(145),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12100014")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(146),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12100018")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(147),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x1210001c")),      # BPix L1 mod 1
    cms.PSet(detSelection=cms.uint32(148),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e3c00fc-0x12100020")),      # BPix L1 mod 1



    cms.PSet(detSelection = cms.uint32(211),detLabel = cms.string("FPIXmD1R1pan1"),selection=cms.untracked.vstring("0x1f8fff00-0x14810500",
                                                                                                                   "0x1f8fff00-0x14810900",
                                                                                                                   "0x1f8fff00-0x14810d00",
                                                                                                                   "0x1f8fff00-0x14811100",
                                                                                                                   "0x1f8fff00-0x14811500",
                                                                                                                   "0x1f8fff00-0x14811900",
                                                                                                                   "0x1f8fff00-0x14811d00",
                                                                                                                   "0x1f8fff00-0x14812100",
                                                                                                                   "0x1f8fff00-0x14812500",
                                                                                                                   "0x1f8fff00-0x14812900",
                                                                                                                   "0x1f8fff00-0x14812d00",
                                                                                                                   "0x1f8fff00-0x14813100",
                                                                                                                   "0x1f8fff00-0x14813500",
                                                                                                                   "0x1f8fff00-0x14813900",
                                                                                                                   "0x1f8fff00-0x14813d00",
                                                                                                                   "0x1f8fff00-0x14814100",
                                                                                                                   "0x1f8fff00-0x14814500",
                                                                                                                   "0x1f8fff00-0x14814900",
                                                                                                                   "0x1f8fff00-0x14814d00",
                                                                                                                   "0x1f8fff00-0x14815100",
                                                                                                                   "0x1f8fff00-0x14815500",
                                                                                                                   "0x1f8fff00-0x14815900")),
    cms.PSet(detSelection = cms.uint32(221),detLabel = cms.string("FPIXmD2R1pan1"),selection=cms.untracked.vstring("0x1f8fff00-0x14820500",
                                                                                                                   "0x1f8fff00-0x14820900",
                                                                                                                   "0x1f8fff00-0x14820d00",
                                                                                                                   "0x1f8fff00-0x14821100",
                                                                                                                   "0x1f8fff00-0x14821500",
                                                                                                                   "0x1f8fff00-0x14821900",
                                                                                                                   "0x1f8fff00-0x14821d00",
                                                                                                                   "0x1f8fff00-0x14822100",
                                                                                                                   "0x1f8fff00-0x14822500",
                                                                                                                   "0x1f8fff00-0x14822900",
                                                                                                                   "0x1f8fff00-0x14822d00",
                                                                                                                   "0x1f8fff00-0x14823100",
                                                                                                                   "0x1f8fff00-0x14823500",
                                                                                                                   "0x1f8fff00-0x14823900",
                                                                                                                   "0x1f8fff00-0x14823d00",
                                                                                                                   "0x1f8fff00-0x14824100",
                                                                                                                   "0x1f8fff00-0x14824500",
                                                                                                                   "0x1f8fff00-0x14824900",
                                                                                                                   "0x1f8fff00-0x14824d00",
                                                                                                                   "0x1f8fff00-0x14825100",
                                                                                                                   "0x1f8fff00-0x14825500",
                                                                                                                   "0x1f8fff00-0x14825900")),
    cms.PSet(detSelection = cms.uint32(231),detLabel = cms.string("FPIXmD3R1pan1"),selection=cms.untracked.vstring("0x1f8fff00-0x14830500",
                                                                                                                   "0x1f8fff00-0x14830900",
                                                                                                                   "0x1f8fff00-0x14830d00",
                                                                                                                   "0x1f8fff00-0x14831100",
                                                                                                                   "0x1f8fff00-0x14831500",
                                                                                                                   "0x1f8fff00-0x14831900",
                                                                                                                   "0x1f8fff00-0x14831d00",
                                                                                                                   "0x1f8fff00-0x14832100",
                                                                                                                   "0x1f8fff00-0x14832500",
                                                                                                                   "0x1f8fff00-0x14832900",
                                                                                                                   "0x1f8fff00-0x14832d00",
                                                                                                                   "0x1f8fff00-0x14833100",
                                                                                                                   "0x1f8fff00-0x14833500",
                                                                                                                   "0x1f8fff00-0x14833900",
                                                                                                                   "0x1f8fff00-0x14833d00",
                                                                                                                   "0x1f8fff00-0x14834100",
                                                                                                                   "0x1f8fff00-0x14834500",
                                                                                                                   "0x1f8fff00-0x14834900",
                                                                                                                   "0x1f8fff00-0x14834d00",
                                                                                                                   "0x1f8fff00-0x14835100",
                                                                                                                   "0x1f8fff00-0x14835500",
                                                                                                                   "0x1f8fff00-0x14835900")),
    cms.PSet(detSelection = cms.uint32(241),detLabel = cms.string("FPIXpD1R1pan1"),selection=cms.untracked.vstring("0x1f8fff00-0x15010500",
                                                                                                                   "0x1f8fff00-0x15010900",
                                                                                                                   "0x1f8fff00-0x15010d00",
                                                                                                                   "0x1f8fff00-0x15011100",
                                                                                                                   "0x1f8fff00-0x15011500",
                                                                                                                   "0x1f8fff00-0x15011900",
                                                                                                                   "0x1f8fff00-0x15011d00",
                                                                                                                   "0x1f8fff00-0x15012100",
                                                                                                                   "0x1f8fff00-0x15012500",
                                                                                                                   "0x1f8fff00-0x15012900",
                                                                                                                   "0x1f8fff00-0x15012d00",
                                                                                                                   "0x1f8fff00-0x15013100",
                                                                                                                   "0x1f8fff00-0x15013500",
                                                                                                                   "0x1f8fff00-0x15013900",
                                                                                                                   "0x1f8fff00-0x15013d00",
                                                                                                                   "0x1f8fff00-0x15014100",
                                                                                                                   "0x1f8fff00-0x15014500",
                                                                                                                   "0x1f8fff00-0x15014900",
                                                                                                                   "0x1f8fff00-0x15014d00",
                                                                                                                   "0x1f8fff00-0x15015100",
                                                                                                                   "0x1f8fff00-0x15015500",
                                                                                                                   "0x1f8fff00-0x15015900")),
    cms.PSet(detSelection = cms.uint32(251),detLabel = cms.string("FPIXpD2R1pan1"),selection=cms.untracked.vstring("0x1f8fff00-0x15020500",
                                                                                                                   "0x1f8fff00-0x15020900",
                                                                                                                   "0x1f8fff00-0x15020d00",
                                                                                                                   "0x1f8fff00-0x15021100",
                                                                                                                   "0x1f8fff00-0x15021500",
                                                                                                                   "0x1f8fff00-0x15021900",
                                                                                                                   "0x1f8fff00-0x15021d00",
                                                                                                                   "0x1f8fff00-0x15022100",
                                                                                                                   "0x1f8fff00-0x15022500",
                                                                                                                   "0x1f8fff00-0x15022900",
                                                                                                                   "0x1f8fff00-0x15022d00",
                                                                                                                   "0x1f8fff00-0x15023100",
                                                                                                                   "0x1f8fff00-0x15023500",
                                                                                                                   "0x1f8fff00-0x15023900",
                                                                                                                   "0x1f8fff00-0x15023d00",
                                                                                                                   "0x1f8fff00-0x15024100",
                                                                                                                   "0x1f8fff00-0x15024500",
                                                                                                                   "0x1f8fff00-0x15024900",
                                                                                                                   "0x1f8fff00-0x15024d00",
                                                                                                                   "0x1f8fff00-0x15025100",
                                                                                                                   "0x1f8fff00-0x15025500",
                                                                                                                   "0x1f8fff00-0x15025900")),
    cms.PSet(detSelection = cms.uint32(261),detLabel = cms.string("FPIXpD3R1pan1"),selection=cms.untracked.vstring("0x1f8fff00-0x15030500",
                                                                                                                   "0x1f8fff00-0x15030900",
                                                                                                                   "0x1f8fff00-0x15030d00",
                                                                                                                   "0x1f8fff00-0x15031100",
                                                                                                                   "0x1f8fff00-0x15031500",
                                                                                                                   "0x1f8fff00-0x15031900",
                                                                                                                   "0x1f8fff00-0x15031d00",
                                                                                                                   "0x1f8fff00-0x15032100",
                                                                                                                   "0x1f8fff00-0x15032500",
                                                                                                                   "0x1f8fff00-0x15032900",
                                                                                                                   "0x1f8fff00-0x15032d00",
                                                                                                                   "0x1f8fff00-0x15033100",
                                                                                                                   "0x1f8fff00-0x15033500",
                                                                                                                   "0x1f8fff00-0x15033900",
                                                                                                                   "0x1f8fff00-0x15033d00",
                                                                                                                   "0x1f8fff00-0x15034100",
                                                                                                                   "0x1f8fff00-0x15034500",
                                                                                                                   "0x1f8fff00-0x15034900",
                                                                                                                   "0x1f8fff00-0x15034d00",
                                                                                                                   "0x1f8fff00-0x15035100",
                                                                                                                   "0x1f8fff00-0x15035500",
                                                                                                                   "0x1f8fff00-0x15035900")),
    cms.PSet(detSelection = cms.uint32(212),detLabel = cms.string("FPIXmD1R1pan2"),selection=cms.untracked.vstring("0x1f8fff00-0x14810600",
                                                                                                                   "0x1f8fff00-0x14810a00",
                                                                                                                   "0x1f8fff00-0x14810e00",
                                                                                                                   "0x1f8fff00-0x14811200",
                                                                                                                   "0x1f8fff00-0x14811600",
                                                                                                                   "0x1f8fff00-0x14811a00",
                                                                                                                   "0x1f8fff00-0x14811e00",
                                                                                                                   "0x1f8fff00-0x14812200",
                                                                                                                   "0x1f8fff00-0x14812600",
                                                                                                                   "0x1f8fff00-0x14812a00",
                                                                                                                   "0x1f8fff00-0x14812e00",
                                                                                                                   "0x1f8fff00-0x14813200",
                                                                                                                   "0x1f8fff00-0x14813600",
                                                                                                                   "0x1f8fff00-0x14813a00",
                                                                                                                   "0x1f8fff00-0x14813e00",
                                                                                                                   "0x1f8fff00-0x14814200",
                                                                                                                   "0x1f8fff00-0x14814600",
                                                                                                                   "0x1f8fff00-0x14814a00",
                                                                                                                   "0x1f8fff00-0x14814e00",
                                                                                                                   "0x1f8fff00-0x14815200",
                                                                                                                   "0x1f8fff00-0x14815600",
                                                                                                                   "0x1f8fff00-0x14815a00")),
    cms.PSet(detSelection = cms.uint32(222),detLabel = cms.string("FPIXmD2R1pan2"),selection=cms.untracked.vstring("0x1f8fff00-0x14820600",
                                                                                                                   "0x1f8fff00-0x14820a00",
                                                                                                                   "0x1f8fff00-0x14820e00",
                                                                                                                   "0x1f8fff00-0x14821200",
                                                                                                                   "0x1f8fff00-0x14821600",
                                                                                                                   "0x1f8fff00-0x14821a00",
                                                                                                                   "0x1f8fff00-0x14821e00",
                                                                                                                   "0x1f8fff00-0x14822200",
                                                                                                                   "0x1f8fff00-0x14822600",
                                                                                                                   "0x1f8fff00-0x14822a00",
                                                                                                                   "0x1f8fff00-0x14822e00",
                                                                                                                   "0x1f8fff00-0x14823200",
                                                                                                                   "0x1f8fff00-0x14823600",
                                                                                                                   "0x1f8fff00-0x14823a00",
                                                                                                                   "0x1f8fff00-0x14823e00",
                                                                                                                   "0x1f8fff00-0x14824200",
                                                                                                                   "0x1f8fff00-0x14824600",
                                                                                                                   "0x1f8fff00-0x14824a00",
                                                                                                                   "0x1f8fff00-0x14824e00",
                                                                                                                   "0x1f8fff00-0x14825200",
                                                                                                                   "0x1f8fff00-0x14825600",
                                                                                                                   "0x1f8fff00-0x14825a00")),
    cms.PSet(detSelection = cms.uint32(232),detLabel = cms.string("FPIXmD3R1pan2"),selection=cms.untracked.vstring("0x1f8fff00-0x14830600",
                                                                                                                   "0x1f8fff00-0x14830a00",
                                                                                                                   "0x1f8fff00-0x14830e00",
                                                                                                                   "0x1f8fff00-0x14831200",
                                                                                                                   "0x1f8fff00-0x14831600",
                                                                                                                   "0x1f8fff00-0x14831a00",
                                                                                                                   "0x1f8fff00-0x14831e00",
                                                                                                                   "0x1f8fff00-0x14832200",
                                                                                                                   "0x1f8fff00-0x14832600",
                                                                                                                   "0x1f8fff00-0x14832a00",
                                                                                                                   "0x1f8fff00-0x14832e00",
                                                                                                                   "0x1f8fff00-0x14833200",
                                                                                                                   "0x1f8fff00-0x14833600",
                                                                                                                   "0x1f8fff00-0x14833a00",
                                                                                                                   "0x1f8fff00-0x14833e00",
                                                                                                                   "0x1f8fff00-0x14834200",
                                                                                                                   "0x1f8fff00-0x14834600",
                                                                                                                   "0x1f8fff00-0x14834a00",
                                                                                                                   "0x1f8fff00-0x14834e00",
                                                                                                                   "0x1f8fff00-0x14835200",
                                                                                                                   "0x1f8fff00-0x14835600",
                                                                                                                   "0x1f8fff00-0x14835a00")),
    cms.PSet(detSelection = cms.uint32(242),detLabel = cms.string("FPIXpD1R1pan2"),selection=cms.untracked.vstring("0x1f8fff00-0x15010600",
                                                                                                                   "0x1f8fff00-0x15010a00",
                                                                                                                   "0x1f8fff00-0x15010e00",
                                                                                                                   "0x1f8fff00-0x15011200",
                                                                                                                   "0x1f8fff00-0x15011600",
                                                                                                                   "0x1f8fff00-0x15011a00",
                                                                                                                   "0x1f8fff00-0x15011e00",
                                                                                                                   "0x1f8fff00-0x15012200",
                                                                                                                   "0x1f8fff00-0x15012600",
                                                                                                                   "0x1f8fff00-0x15012a00",
                                                                                                                   "0x1f8fff00-0x15012e00",
                                                                                                                   "0x1f8fff00-0x15013200",
                                                                                                                   "0x1f8fff00-0x15013600",
                                                                                                                   "0x1f8fff00-0x15013a00",
                                                                                                                   "0x1f8fff00-0x15013e00",
                                                                                                                   "0x1f8fff00-0x15014200",
                                                                                                                   "0x1f8fff00-0x15014600",
                                                                                                                   "0x1f8fff00-0x15014a00",
                                                                                                                   "0x1f8fff00-0x15014e00",
                                                                                                                   "0x1f8fff00-0x15015200",
                                                                                                                   "0x1f8fff00-0x15015600",
                                                                                                                   "0x1f8fff00-0x15015a00")),
    cms.PSet(detSelection = cms.uint32(252),detLabel = cms.string("FPIXpD2R1pan2"),selection=cms.untracked.vstring("0x1f8fff00-0x15020600",
                                                                                                                   "0x1f8fff00-0x15020a00",
                                                                                                                   "0x1f8fff00-0x15020e00",
                                                                                                                   "0x1f8fff00-0x15021200",
                                                                                                                   "0x1f8fff00-0x15021600",
                                                                                                                   "0x1f8fff00-0x15021a00",
                                                                                                                   "0x1f8fff00-0x15021e00",
                                                                                                                   "0x1f8fff00-0x15022200",
                                                                                                                   "0x1f8fff00-0x15022600",
                                                                                                                   "0x1f8fff00-0x15022a00",
                                                                                                                   "0x1f8fff00-0x15022e00",
                                                                                                                   "0x1f8fff00-0x15023200",
                                                                                                                   "0x1f8fff00-0x15023600",
                                                                                                                   "0x1f8fff00-0x15023a00",
                                                                                                                   "0x1f8fff00-0x15023e00",
                                                                                                                   "0x1f8fff00-0x15024200",
                                                                                                                   "0x1f8fff00-0x15024600",
                                                                                                                   "0x1f8fff00-0x15024a00",
                                                                                                                   "0x1f8fff00-0x15024e00",
                                                                                                                   "0x1f8fff00-0x15025200",
                                                                                                                   "0x1f8fff00-0x15025600",
                                                                                                                   "0x1f8fff00-0x15025a00")),
    cms.PSet(detSelection = cms.uint32(262),detLabel = cms.string("FPIXpD3R1pan2"),selection=cms.untracked.vstring("0x1f8fff00-0x15030600",
                                                                                                                   "0x1f8fff00-0x15030a00",
                                                                                                                   "0x1f8fff00-0x15030e00",
                                                                                                                   "0x1f8fff00-0x15031200",
                                                                                                                   "0x1f8fff00-0x15031600",
                                                                                                                   "0x1f8fff00-0x15031a00",
                                                                                                                   "0x1f8fff00-0x15031e00",
                                                                                                                   "0x1f8fff00-0x15032200",
                                                                                                                   "0x1f8fff00-0x15032600",
                                                                                                                   "0x1f8fff00-0x15032a00",
                                                                                                                   "0x1f8fff00-0x15032e00",
                                                                                                                   "0x1f8fff00-0x15033200",
                                                                                                                   "0x1f8fff00-0x15033600",
                                                                                                                   "0x1f8fff00-0x15033a00",
                                                                                                                   "0x1f8fff00-0x15033e00",
                                                                                                                   "0x1f8fff00-0x15034200",
                                                                                                                   "0x1f8fff00-0x15034600",
                                                                                                                   "0x1f8fff00-0x15034a00",
                                                                                                                   "0x1f8fff00-0x15034e00",
                                                                                                                   "0x1f8fff00-0x15035200",
                                                                                                                   "0x1f8fff00-0x15035600",
                                                                                                                   "0x1f8fff00-0x15035a00")),
    
    cms.PSet(detSelection = cms.uint32(213),detLabel = cms.string("FPIXmD1R2pan1"),selection=cms.untracked.vstring("0x1f8fff00-0x14815d00",
                                                                                                                   "0x1f8fff00-0x14816100",
                                                                                                                   "0x1f8fff00-0x14816500",
                                                                                                                   "0x1f8fff00-0x14816900",
                                                                                                                   "0x1f8fff00-0x14816d00",
                                                                                                                   "0x1f8fff00-0x14817100",
                                                                                                                   "0x1f8fff00-0x14817500",
                                                                                                                   "0x1f8fff00-0x14817900",
                                                                                                                   "0x1f8fff00-0x14817d00",
                                                                                                                   "0x1f8fff00-0x14818100",
                                                                                                                   "0x1f8fff00-0x14818500",
                                                                                                                   "0x1f8fff00-0x14818900",
                                                                                                                   "0x1f8fff00-0x14818d00",
                                                                                                                   "0x1f8fff00-0x14819100",
                                                                                                                   "0x1f8fff00-0x14819500",
                                                                                                                   "0x1f8fff00-0x14819900",
                                                                                                                   "0x1f8fff00-0x14819d00",
                                                                                                                   "0x1f8fff00-0x1481a100",
                                                                                                                   "0x1f8fff00-0x1481a500",
                                                                                                                   "0x1f8fff00-0x1481a900",
                                                                                                                   "0x1f8fff00-0x1481ad00",
                                                                                                                   "0x1f8fff00-0x1481b100",
                                                                                                                   "0x1f8fff00-0x1481b500",
                                                                                                                   "0x1f8fff00-0x1481b900",
                                                                                                                   "0x1f8fff00-0x1481bd00",
                                                                                                                   "0x1f8fff00-0x1481c100",
                                                                                                                   "0x1f8fff00-0x1481c500",
                                                                                                                   "0x1f8fff00-0x1481c900",
                                                                                                                   "0x1f8fff00-0x1481cd00",
                                                                                                                   "0x1f8fff00-0x1481d100",
                                                                                                                   "0x1f8fff00-0x1481d500",
                                                                                                                   "0x1f8fff00-0x1481d900",
                                                                                                                   "0x1f8fff00-0x1481dd00",
                                                                                                                   "0x1f8fff00-0x1481e100")),
    cms.PSet(detSelection = cms.uint32(223),detLabel = cms.string("FPIXmD2R2pan1"),selection=cms.untracked.vstring("0x1f8fff00-0x14825d00",
                                                                                                                   "0x1f8fff00-0x14826100",
                                                                                                                   "0x1f8fff00-0x14826500",
                                                                                                                   "0x1f8fff00-0x14826900",
                                                                                                                   "0x1f8fff00-0x14826d00",
                                                                                                                   "0x1f8fff00-0x14827100",
                                                                                                                   "0x1f8fff00-0x14827500",
                                                                                                                   "0x1f8fff00-0x14827900",
                                                                                                                   "0x1f8fff00-0x14827d00",
                                                                                                                   "0x1f8fff00-0x14828100",
                                                                                                                   "0x1f8fff00-0x14828500",
                                                                                                                   "0x1f8fff00-0x14828900",
                                                                                                                   "0x1f8fff00-0x14828d00",
                                                                                                                   "0x1f8fff00-0x14829100",
                                                                                                                   "0x1f8fff00-0x14829500",
                                                                                                                   "0x1f8fff00-0x14829900",
                                                                                                                   "0x1f8fff00-0x14829d00",
                                                                                                                   "0x1f8fff00-0x1482a100",
                                                                                                                   "0x1f8fff00-0x1482a500",
                                                                                                                   "0x1f8fff00-0x1482a900",
                                                                                                                   "0x1f8fff00-0x1482ad00",
                                                                                                                   "0x1f8fff00-0x1482b100",
                                                                                                                   "0x1f8fff00-0x1482b500",
                                                                                                                   "0x1f8fff00-0x1482b900",
                                                                                                                   "0x1f8fff00-0x1482bd00",
                                                                                                                   "0x1f8fff00-0x1482c100",
                                                                                                                   "0x1f8fff00-0x1482c500",
                                                                                                                   "0x1f8fff00-0x1482c900",
                                                                                                                   "0x1f8fff00-0x1482cd00",
                                                                                                                   "0x1f8fff00-0x1482d100",
                                                                                                                   "0x1f8fff00-0x1482d500",
                                                                                                                   "0x1f8fff00-0x1482d900",
                                                                                                                   "0x1f8fff00-0x1482dd00",
                                                                                                                   "0x1f8fff00-0x1482e100")),
    cms.PSet(detSelection = cms.uint32(233),detLabel = cms.string("FPIXmD3R2pan1"),selection=cms.untracked.vstring("0x1f8fff00-0x14835d00",
                                                                                                                   "0x1f8fff00-0x14836100",
                                                                                                                   "0x1f8fff00-0x14836500",
                                                                                                                   "0x1f8fff00-0x14836900",
                                                                                                                   "0x1f8fff00-0x14836d00",
                                                                                                                   "0x1f8fff00-0x14837100",
                                                                                                                   "0x1f8fff00-0x14837500",
                                                                                                                   "0x1f8fff00-0x14837900",
                                                                                                                   "0x1f8fff00-0x14837d00",
                                                                                                                   "0x1f8fff00-0x14838100",
                                                                                                                   "0x1f8fff00-0x14838500",
                                                                                                                   "0x1f8fff00-0x14838900",
                                                                                                                   "0x1f8fff00-0x14838d00",
                                                                                                                   "0x1f8fff00-0x14839100",
                                                                                                                   "0x1f8fff00-0x14839500",
                                                                                                                   "0x1f8fff00-0x14839900",
                                                                                                                   "0x1f8fff00-0x14839d00",
                                                                                                                   "0x1f8fff00-0x1483a100",
                                                                                                                   "0x1f8fff00-0x1483a500",
                                                                                                                   "0x1f8fff00-0x1483a900",
                                                                                                                   "0x1f8fff00-0x1483ad00",
                                                                                                                   "0x1f8fff00-0x1483b100",
                                                                                                                   "0x1f8fff00-0x1483b500",
                                                                                                                   "0x1f8fff00-0x1483b900",
                                                                                                                   "0x1f8fff00-0x1483bd00",
                                                                                                                   "0x1f8fff00-0x1483c100",
                                                                                                                   "0x1f8fff00-0x1483c500",
                                                                                                                   "0x1f8fff00-0x1483c900",
                                                                                                                   "0x1f8fff00-0x1483cd00",
                                                                                                                   "0x1f8fff00-0x1483d100",
                                                                                                                   "0x1f8fff00-0x1483d500",
                                                                                                                   "0x1f8fff00-0x1483d900",
                                                                                                                   "0x1f8fff00-0x1483dd00",
                                                                                                                   "0x1f8fff00-0x1483e100")),
    cms.PSet(detSelection = cms.uint32(243),detLabel = cms.string("FPIXpD1R2pan1"),selection=cms.untracked.vstring("0x1f8fff00-0x15015d00",
                                                                                                                   "0x1f8fff00-0x15016100",
                                                                                                                   "0x1f8fff00-0x15016500",
                                                                                                                   "0x1f8fff00-0x15016900",
                                                                                                                   "0x1f8fff00-0x15016d00",
                                                                                                                   "0x1f8fff00-0x15017100",
                                                                                                                   "0x1f8fff00-0x15017500",
                                                                                                                   "0x1f8fff00-0x15017900",
                                                                                                                   "0x1f8fff00-0x15017d00",
                                                                                                                   "0x1f8fff00-0x15018100",
                                                                                                                   "0x1f8fff00-0x15018500",
                                                                                                                   "0x1f8fff00-0x15018900",
                                                                                                                   "0x1f8fff00-0x15018d00",
                                                                                                                   "0x1f8fff00-0x15019100",
                                                                                                                   "0x1f8fff00-0x15019500",
                                                                                                                   "0x1f8fff00-0x15019900",
                                                                                                                   "0x1f8fff00-0x15019d00",
                                                                                                                   "0x1f8fff00-0x1501a100",
                                                                                                                   "0x1f8fff00-0x1501a500",
                                                                                                                   "0x1f8fff00-0x1501a900",
                                                                                                                   "0x1f8fff00-0x1501ad00",
                                                                                                                   "0x1f8fff00-0x1501b100",
                                                                                                                   "0x1f8fff00-0x1501b500",
                                                                                                                   "0x1f8fff00-0x1501b900",
                                                                                                                   "0x1f8fff00-0x1501bd00",
                                                                                                                   "0x1f8fff00-0x1501c100",
                                                                                                                   "0x1f8fff00-0x1501c500",
                                                                                                                   "0x1f8fff00-0x1501c900",
                                                                                                                   "0x1f8fff00-0x1501cd00",
                                                                                                                   "0x1f8fff00-0x1501d100",
                                                                                                                   "0x1f8fff00-0x1501d500",
                                                                                                                   "0x1f8fff00-0x1501d900",
                                                                                                                   "0x1f8fff00-0x1501dd00",
                                                                                                                   "0x1f8fff00-0x1501e100")),
    cms.PSet(detSelection = cms.uint32(253),detLabel = cms.string("FPIXpD2R2pan1"),selection=cms.untracked.vstring("0x1f8fff00-0x15025d00",
                                                                                                                   "0x1f8fff00-0x15026100",
                                                                                                                   "0x1f8fff00-0x15026500",
                                                                                                                   "0x1f8fff00-0x15026900",
                                                                                                                   "0x1f8fff00-0x15026d00",
                                                                                                                   "0x1f8fff00-0x15027100",
                                                                                                                   "0x1f8fff00-0x15027500",
                                                                                                                   "0x1f8fff00-0x15027900",
                                                                                                                   "0x1f8fff00-0x15027d00",
                                                                                                                   "0x1f8fff00-0x15028100",
                                                                                                                   "0x1f8fff00-0x15028500",
                                                                                                                   "0x1f8fff00-0x15028900",
                                                                                                                   "0x1f8fff00-0x15028d00",
                                                                                                                   "0x1f8fff00-0x15029100",
                                                                                                                   "0x1f8fff00-0x15029500",
                                                                                                                   "0x1f8fff00-0x15029900",
                                                                                                                   "0x1f8fff00-0x15029d00",
                                                                                                                   "0x1f8fff00-0x1502a100",
                                                                                                                   "0x1f8fff00-0x1502a500",
                                                                                                                   "0x1f8fff00-0x1502a900",
                                                                                                                   "0x1f8fff00-0x1502ad00",
                                                                                                                   "0x1f8fff00-0x1502b100",
                                                                                                                   "0x1f8fff00-0x1502b500",
                                                                                                                   "0x1f8fff00-0x1502b900",
                                                                                                                   "0x1f8fff00-0x1502bd00",
                                                                                                                   "0x1f8fff00-0x1502c100",
                                                                                                                   "0x1f8fff00-0x1502c500",
                                                                                                                   "0x1f8fff00-0x1502c900",
                                                                                                                   "0x1f8fff00-0x1502cd00",
                                                                                                                   "0x1f8fff00-0x1502d100",
                                                                                                                   "0x1f8fff00-0x1502d500",
                                                                                                                   "0x1f8fff00-0x1502d900",
                                                                                                                   "0x1f8fff00-0x1502dd00",
                                                                                                                   "0x1f8fff00-0x1502e100")),
    cms.PSet(detSelection = cms.uint32(263),detLabel = cms.string("FPIXpD3R2pan1"),selection=cms.untracked.vstring("0x1f8fff00-0x15035d00",
                                                                                                                   "0x1f8fff00-0x15036100",
                                                                                                                   "0x1f8fff00-0x15036500",
                                                                                                                   "0x1f8fff00-0x15036900",
                                                                                                                   "0x1f8fff00-0x15036d00",
                                                                                                                   "0x1f8fff00-0x15037100",
                                                                                                                   "0x1f8fff00-0x15037500",
                                                                                                                   "0x1f8fff00-0x15037900",
                                                                                                                   "0x1f8fff00-0x15037d00",
                                                                                                                   "0x1f8fff00-0x15038100",
                                                                                                                   "0x1f8fff00-0x15038500",
                                                                                                                   "0x1f8fff00-0x15038900",
                                                                                                                   "0x1f8fff00-0x15038d00",
                                                                                                                   "0x1f8fff00-0x15039100",
                                                                                                                   "0x1f8fff00-0x15039500",
                                                                                                                   "0x1f8fff00-0x15039900",
                                                                                                                   "0x1f8fff00-0x15039d00",
                                                                                                                   "0x1f8fff00-0x1503a100",
                                                                                                                   "0x1f8fff00-0x1503a500",
                                                                                                                   "0x1f8fff00-0x1503a900",
                                                                                                                   "0x1f8fff00-0x1503ad00",
                                                                                                                   "0x1f8fff00-0x1503b100",
                                                                                                                   "0x1f8fff00-0x1503b500",
                                                                                                                   "0x1f8fff00-0x1503b900",
                                                                                                                   "0x1f8fff00-0x1503bd00",
                                                                                                                   "0x1f8fff00-0x1503c100",
                                                                                                                   "0x1f8fff00-0x1503c500",
                                                                                                                   "0x1f8fff00-0x1503c900",
                                                                                                                   "0x1f8fff00-0x1503cd00",
                                                                                                                   "0x1f8fff00-0x1503d100",
                                                                                                                   "0x1f8fff00-0x1503d500",
                                                                                                                   "0x1f8fff00-0x1503d900",
                                                                                                                   "0x1f8fff00-0x1503dd00",
                                                                                                                   "0x1f8fff00-0x1503e100")),
    cms.PSet(detSelection = cms.uint32(214),detLabel = cms.string("FPIXmD1R2pan2"),selection=cms.untracked.vstring("0x1f8fff00-0x14815e00",
                                                                                                                   "0x1f8fff00-0x14816200",
                                                                                                                   "0x1f8fff00-0x14816600",
                                                                                                                   "0x1f8fff00-0x14816a00",
                                                                                                                   "0x1f8fff00-0x14816e00",
                                                                                                                   "0x1f8fff00-0x14817200",
                                                                                                                   "0x1f8fff00-0x14817600",
                                                                                                                   "0x1f8fff00-0x14817a00",
                                                                                                                   "0x1f8fff00-0x14817e00",
                                                                                                                   "0x1f8fff00-0x14818200",
                                                                                                                   "0x1f8fff00-0x14818600",
                                                                                                                   "0x1f8fff00-0x14818a00",
                                                                                                                   "0x1f8fff00-0x14818e00",
                                                                                                                   "0x1f8fff00-0x14819200",
                                                                                                                   "0x1f8fff00-0x14819600",
                                                                                                                   "0x1f8fff00-0x14819a00",
                                                                                                                   "0x1f8fff00-0x14819e00",
                                                                                                                   "0x1f8fff00-0x1481a200",
                                                                                                                   "0x1f8fff00-0x1481a600",
                                                                                                                   "0x1f8fff00-0x1481aa00",
                                                                                                                   "0x1f8fff00-0x1481ae00",
                                                                                                                   "0x1f8fff00-0x1481b200",
                                                                                                                   "0x1f8fff00-0x1481b600",
                                                                                                                   "0x1f8fff00-0x1481ba00",
                                                                                                                   "0x1f8fff00-0x1481be00",
                                                                                                                   "0x1f8fff00-0x1481c200",
                                                                                                                   "0x1f8fff00-0x1481c600",
                                                                                                                   "0x1f8fff00-0x1481ca00",
                                                                                                                   "0x1f8fff00-0x1481ce00",
                                                                                                                   "0x1f8fff00-0x1481d200",
                                                                                                                   "0x1f8fff00-0x1481d600",
                                                                                                                   "0x1f8fff00-0x1481da00",
                                                                                                                   "0x1f8fff00-0x1481de00",
                                                                                                                   "0x1f8fff00-0x1481e200")),
    cms.PSet(detSelection = cms.uint32(224),detLabel = cms.string("FPIXmD2R2pan2"),selection=cms.untracked.vstring("0x1f8fff00-0x14825e00",
                                                                                                                   "0x1f8fff00-0x14826200",
                                                                                                                   "0x1f8fff00-0x14826600",
                                                                                                                   "0x1f8fff00-0x14826a00",
                                                                                                                   "0x1f8fff00-0x14826e00",
                                                                                                                   "0x1f8fff00-0x14827200",
                                                                                                                   "0x1f8fff00-0x14827600",
                                                                                                                   "0x1f8fff00-0x14827a00",
                                                                                                                   "0x1f8fff00-0x14827e00",
                                                                                                                   "0x1f8fff00-0x14828200",
                                                                                                                   "0x1f8fff00-0x14828600",
                                                                                                                   "0x1f8fff00-0x14828a00",
                                                                                                                   "0x1f8fff00-0x14828e00",
                                                                                                                   "0x1f8fff00-0x14829200",
                                                                                                                   "0x1f8fff00-0x14829600",
                                                                                                                   "0x1f8fff00-0x14829a00",
                                                                                                                   "0x1f8fff00-0x14829e00",
                                                                                                                   "0x1f8fff00-0x1482a200",
                                                                                                                   "0x1f8fff00-0x1482a600",
                                                                                                                   "0x1f8fff00-0x1482aa00",
                                                                                                                   "0x1f8fff00-0x1482ae00",
                                                                                                                   "0x1f8fff00-0x1482b200",
                                                                                                                   "0x1f8fff00-0x1482b600",
                                                                                                                   "0x1f8fff00-0x1482ba00",
                                                                                                                   "0x1f8fff00-0x1482be00",
                                                                                                                   "0x1f8fff00-0x1482c200",
                                                                                                                   "0x1f8fff00-0x1482c600",
                                                                                                                   "0x1f8fff00-0x1482ca00",
                                                                                                                   "0x1f8fff00-0x1482ce00",
                                                                                                                   "0x1f8fff00-0x1482d200",
                                                                                                                   "0x1f8fff00-0x1482d600",
                                                                                                                   "0x1f8fff00-0x1482da00",
                                                                                                                   "0x1f8fff00-0x1482de00",
                                                                                                                   "0x1f8fff00-0x1482e200")),
    cms.PSet(detSelection = cms.uint32(234),detLabel = cms.string("FPIXmD3R2pan2"),selection=cms.untracked.vstring("0x1f8fff00-0x14835e00",
                                                                                                                   "0x1f8fff00-0x14836200",
                                                                                                                   "0x1f8fff00-0x14836600",
                                                                                                                   "0x1f8fff00-0x14836a00",
                                                                                                                   "0x1f8fff00-0x14836e00",
                                                                                                                   "0x1f8fff00-0x14837200",
                                                                                                                   "0x1f8fff00-0x14837600",
                                                                                                                   "0x1f8fff00-0x14837a00",
                                                                                                                   "0x1f8fff00-0x14837e00",
                                                                                                                   "0x1f8fff00-0x14838200",
                                                                                                                   "0x1f8fff00-0x14838600",
                                                                                                                   "0x1f8fff00-0x14838a00",
                                                                                                                   "0x1f8fff00-0x14838e00",
                                                                                                                   "0x1f8fff00-0x14839200",
                                                                                                                   "0x1f8fff00-0x14839600",
                                                                                                                   "0x1f8fff00-0x14839a00",
                                                                                                                   "0x1f8fff00-0x14839e00",
                                                                                                                   "0x1f8fff00-0x1483a200",
                                                                                                                   "0x1f8fff00-0x1483a600",
                                                                                                                   "0x1f8fff00-0x1483aa00",
                                                                                                                   "0x1f8fff00-0x1483ae00",
                                                                                                                   "0x1f8fff00-0x1483b200",
                                                                                                                   "0x1f8fff00-0x1483b600",
                                                                                                                   "0x1f8fff00-0x1483ba00",
                                                                                                                   "0x1f8fff00-0x1483be00",
                                                                                                                   "0x1f8fff00-0x1483c200",
                                                                                                                   "0x1f8fff00-0x1483c600",
                                                                                                                   "0x1f8fff00-0x1483ca00",
                                                                                                                   "0x1f8fff00-0x1483ce00",
                                                                                                                   "0x1f8fff00-0x1483d200",
                                                                                                                   "0x1f8fff00-0x1483d600",
                                                                                                                   "0x1f8fff00-0x1483da00",
                                                                                                                   "0x1f8fff00-0x1483de00",
                                                                                                                   "0x1f8fff00-0x1483e200")),
    cms.PSet(detSelection = cms.uint32(244),detLabel = cms.string("FPIXpD1R2pan2"),selection=cms.untracked.vstring("0x1f8fff00-0x15015e00",
                                                                                                                   "0x1f8fff00-0x15016200",
                                                                                                                   "0x1f8fff00-0x15016600",
                                                                                                                   "0x1f8fff00-0x15016a00",
                                                                                                                   "0x1f8fff00-0x15016e00",
                                                                                                                   "0x1f8fff00-0x15017200",
                                                                                                                   "0x1f8fff00-0x15017600",
                                                                                                                   "0x1f8fff00-0x15017a00",
                                                                                                                   "0x1f8fff00-0x15017e00",
                                                                                                                   "0x1f8fff00-0x15018200",
                                                                                                                   "0x1f8fff00-0x15018600",
                                                                                                                   "0x1f8fff00-0x15018a00",
                                                                                                                   "0x1f8fff00-0x15018e00",
                                                                                                                   "0x1f8fff00-0x15019200",
                                                                                                                   "0x1f8fff00-0x15019600",
                                                                                                                   "0x1f8fff00-0x15019a00",
                                                                                                                   "0x1f8fff00-0x15019e00",
                                                                                                                   "0x1f8fff00-0x1501a200",
                                                                                                                   "0x1f8fff00-0x1501a600",
                                                                                                                   "0x1f8fff00-0x1501aa00",
                                                                                                                   "0x1f8fff00-0x1501ae00",
                                                                                                                   "0x1f8fff00-0x1501b200",
                                                                                                                   "0x1f8fff00-0x1501b600",
                                                                                                                   "0x1f8fff00-0x1501ba00",
                                                                                                                   "0x1f8fff00-0x1501be00",
                                                                                                                   "0x1f8fff00-0x1501c200",
                                                                                                                   "0x1f8fff00-0x1501c600",
                                                                                                                   "0x1f8fff00-0x1501ca00",
                                                                                                                   "0x1f8fff00-0x1501ce00",
                                                                                                                   "0x1f8fff00-0x1501d200",
                                                                                                                   "0x1f8fff00-0x1501d600",
                                                                                                                   "0x1f8fff00-0x1501da00",
                                                                                                                   "0x1f8fff00-0x1501de00",
                                                                                                                   "0x1f8fff00-0x1501e200")),
    cms.PSet(detSelection = cms.uint32(254),detLabel = cms.string("FPIXpD2R2pan2"),selection=cms.untracked.vstring("0x1f8fff00-0x15025e00",
                                                                                                                   "0x1f8fff00-0x15026200",
                                                                                                                   "0x1f8fff00-0x15026600",
                                                                                                                   "0x1f8fff00-0x15026a00",
                                                                                                                   "0x1f8fff00-0x15026e00",
                                                                                                                   "0x1f8fff00-0x15027200",
                                                                                                                   "0x1f8fff00-0x15027600",
                                                                                                                   "0x1f8fff00-0x15027a00",
                                                                                                                   "0x1f8fff00-0x15027e00",
                                                                                                                   "0x1f8fff00-0x15028200",
                                                                                                                   "0x1f8fff00-0x15028600",
                                                                                                                   "0x1f8fff00-0x15028a00",
                                                                                                                   "0x1f8fff00-0x15028e00",
                                                                                                                   "0x1f8fff00-0x15029200",
                                                                                                                   "0x1f8fff00-0x15029600",
                                                                                                                   "0x1f8fff00-0x15029a00",
                                                                                                                   "0x1f8fff00-0x15029e00",
                                                                                                                   "0x1f8fff00-0x1502a200",
                                                                                                                   "0x1f8fff00-0x1502a600",
                                                                                                                   "0x1f8fff00-0x1502aa00",
                                                                                                                   "0x1f8fff00-0x1502ae00",
                                                                                                                   "0x1f8fff00-0x1502b200",
                                                                                                                   "0x1f8fff00-0x1502b600",
                                                                                                                   "0x1f8fff00-0x1502ba00",
                                                                                                                   "0x1f8fff00-0x1502be00",
                                                                                                                   "0x1f8fff00-0x1502c200",
                                                                                                                   "0x1f8fff00-0x1502c600",
                                                                                                                   "0x1f8fff00-0x1502ca00",
                                                                                                                   "0x1f8fff00-0x1502ce00",
                                                                                                                   "0x1f8fff00-0x1502d200",
                                                                                                                   "0x1f8fff00-0x1502d600",
                                                                                                                   "0x1f8fff00-0x1502da00",
                                                                                                                   "0x1f8fff00-0x1502de00",
                                                                                                                   "0x1f8fff00-0x1502e200")),
    cms.PSet(detSelection = cms.uint32(264),detLabel = cms.string("FPIXpD3R2pan2"),selection=cms.untracked.vstring("0x1f8fff00-0x15035e00",
                                                                                                                   "0x1f8fff00-0x15036200",
                                                                                                                   "0x1f8fff00-0x15036600",
                                                                                                                   "0x1f8fff00-0x15036a00",
                                                                                                                   "0x1f8fff00-0x15036e00",
                                                                                                                   "0x1f8fff00-0x15037200",
                                                                                                                   "0x1f8fff00-0x15037600",
                                                                                                                   "0x1f8fff00-0x15037a00",
                                                                                                                   "0x1f8fff00-0x15037e00",
                                                                                                                   "0x1f8fff00-0x15038200",
                                                                                                                   "0x1f8fff00-0x15038600",
                                                                                                                   "0x1f8fff00-0x15038a00",
                                                                                                                   "0x1f8fff00-0x15038e00",
                                                                                                                   "0x1f8fff00-0x15039200",
                                                                                                                   "0x1f8fff00-0x15039600",
                                                                                                                   "0x1f8fff00-0x15039a00",
                                                                                                                   "0x1f8fff00-0x15039e00",
                                                                                                                   "0x1f8fff00-0x1503a200",
                                                                                                                   "0x1f8fff00-0x1503a600",
                                                                                                                   "0x1f8fff00-0x1503aa00",
                                                                                                                   "0x1f8fff00-0x1503ae00",
                                                                                                                   "0x1f8fff00-0x1503b200",
                                                                                                                   "0x1f8fff00-0x1503b600",
                                                                                                                   "0x1f8fff00-0x1503ba00",
                                                                                                                   "0x1f8fff00-0x1503be00",
                                                                                                                   "0x1f8fff00-0x1503c200",
                                                                                                                   "0x1f8fff00-0x1503c600",
                                                                                                                   "0x1f8fff00-0x1503ca00",
                                                                                                                   "0x1f8fff00-0x1503ce00",
                                                                                                                   "0x1f8fff00-0x1503d200",
                                                                                                                   "0x1f8fff00-0x1503d600",
                                                                                                                   "0x1f8fff00-0x1503da00",
                                                                                                                   "0x1f8fff00-0x1503de00",
                                                                                                                   "0x1f8fff00-0x1503e200"))
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

     cms.PSet(detSelection=cms.uint32(2110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18002a00")),     # TID- D1 R1 Front
     cms.PSet(detSelection=cms.uint32(2120),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003200")),     # TID- D2 R1 Front
     cms.PSet(detSelection=cms.uint32(2130),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003a00")),     # TID- D3 R1 Front
     cms.PSet(detSelection=cms.uint32(2140),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18004a00")),     # TID+ D1 R1 Front
     cms.PSet(detSelection=cms.uint32(2150),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005200")),     # TID+ D2 R1 Front
     cms.PSet(detSelection=cms.uint32(2160),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005a00")),     # TID+ D3 R1 Front

     cms.PSet(detSelection=cms.uint32(2210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18002c00")),     # TID- D1 R2 Front
     cms.PSet(detSelection=cms.uint32(2220),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003400")),     # TID- D2 R2 Front
     cms.PSet(detSelection=cms.uint32(2230),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003c00")),     # TID- D3 R2 Front
     cms.PSet(detSelection=cms.uint32(2240),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18004c00")),     # TID+ D1 R2 Front
     cms.PSet(detSelection=cms.uint32(2250),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005400")),     # TID+ D2 R2 Front
     cms.PSet(detSelection=cms.uint32(2260),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005c00")),     # TID+ D3 R2 Front

     cms.PSet(detSelection=cms.uint32(2310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18002e00")),     # TID- D1 R3 Front
     cms.PSet(detSelection=cms.uint32(2320),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003600")),     # TID- D2 R3 Front
     cms.PSet(detSelection=cms.uint32(2330),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18003e00")),     # TID- D3 R3 Front
     cms.PSet(detSelection=cms.uint32(2340),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18004e00")),     # TID+ D1 R3 Front
     cms.PSet(detSelection=cms.uint32(2350),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005600")),     # TID+ D2 R3 Front
     cms.PSet(detSelection=cms.uint32(2360),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e007e00-0x18005e00")),     # TID+ D3 R3 Front

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
    cms.PSet(detSelection=cms.uint32(4110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044020")),    # TEC- D1 R1 back
    cms.PSet(detSelection=cms.uint32(4120),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048020")),    # TEC- D2 R1 back
    cms.PSet(detSelection=cms.uint32(4130),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c020")),    # TEC- D3 R1 back
#    cms.PSet(detSelection=cms.uint32(4140),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050020")),    # TEC- D4 R1 back
#    cms.PSet(detSelection=cms.uint32(4150),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054020")),    # TEC- D5 R1 back
#    cms.PSet(detSelection=cms.uint32(4160),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058020")),    # TEC- D6 R1 back
#    cms.PSet(detSelection=cms.uint32(4170),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c020")),    # TEC- D7 R1 back
#    cms.PSet(detSelection=cms.uint32(4180),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060020")),    # TEC- D8 R1 back
#    cms.PSet(detSelection=cms.uint32(4190),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064020")),    # TEC- D9 R1 back

    cms.PSet(detSelection=cms.uint32(4210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044040")),    # TEC- D1 R2 back
    cms.PSet(detSelection=cms.uint32(4220),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048040")),    # TEC- D2 R2 back
    cms.PSet(detSelection=cms.uint32(4230),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c040")),    # TEC- D3 R2 back
    cms.PSet(detSelection=cms.uint32(4240),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050040")),    # TEC- D4 R2 back
    cms.PSet(detSelection=cms.uint32(4250),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054040")),    # TEC- D5 R2 back
    cms.PSet(detSelection=cms.uint32(4260),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058040")),    # TEC- D6 R2 back
#    cms.PSet(detSelection=cms.uint32(4270),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c040")),    # TEC- D7 R2 back
#    cms.PSet(detSelection=cms.uint32(4280),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060040")),    # TEC- D8 R2 back
#    cms.PSet(detSelection=cms.uint32(4290),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064040")),    # TEC- D9 R2 back

    cms.PSet(detSelection=cms.uint32(4310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044060")),    # TEC- D1 R3 back
    cms.PSet(detSelection=cms.uint32(4320),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048060")),    # TEC- D2 R3 back
    cms.PSet(detSelection=cms.uint32(4330),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c060")),    # TEC- D3 R3 back
    cms.PSet(detSelection=cms.uint32(4340),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050060")),    # TEC- D4 R3 back
    cms.PSet(detSelection=cms.uint32(4350),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054060")),    # TEC- D5 R3 back
    cms.PSet(detSelection=cms.uint32(4360),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058060")),    # TEC- D6 R3 back
    cms.PSet(detSelection=cms.uint32(4370),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c060")),    # TEC- D7 R3 back
    cms.PSet(detSelection=cms.uint32(4380),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060060")),    # TEC- D8 R3 back
#    cms.PSet(detSelection=cms.uint32(4390),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064060")),    # TEC- D9 R3 back

    cms.PSet(detSelection=cms.uint32(4410),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044080")),    # TEC- D1 R4 back
    cms.PSet(detSelection=cms.uint32(4420),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048080")),    # TEC- D2 R4 back
    cms.PSet(detSelection=cms.uint32(4430),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c080")),    # TEC- D3 R4 back
    cms.PSet(detSelection=cms.uint32(4440),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050080")),    # TEC- D4 R4 back
    cms.PSet(detSelection=cms.uint32(4450),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054080")),    # TEC- D5 R4 back
    cms.PSet(detSelection=cms.uint32(4460),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058080")),    # TEC- D6 R4 back
    cms.PSet(detSelection=cms.uint32(4470),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c080")),    # TEC- D7 R4 back
    cms.PSet(detSelection=cms.uint32(4480),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060080")),    # TEC- D8 R4 back
    cms.PSet(detSelection=cms.uint32(4490),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064080")),    # TEC- D9 R4 back
    
    cms.PSet(detSelection=cms.uint32(4510),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0440a0")),    # TEC- D1 R5 back
    cms.PSet(detSelection=cms.uint32(4520),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0480a0")),    # TEC- D2 R5 back
    cms.PSet(detSelection=cms.uint32(4530),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c0a0")),    # TEC- D3 R5 back
    cms.PSet(detSelection=cms.uint32(4540),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0500a0")),    # TEC- D4 R5 back
    cms.PSet(detSelection=cms.uint32(4550),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0540a0")),    # TEC- D5 R5 back
    cms.PSet(detSelection=cms.uint32(4560),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0580a0")),    # TEC- D6 R5 back
    cms.PSet(detSelection=cms.uint32(4570),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c0a0")),    # TEC- D7 R5 back
    cms.PSet(detSelection=cms.uint32(4580),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0600a0")),    # TEC- D8 R5 back
    cms.PSet(detSelection=cms.uint32(4590),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0640a0")),    # TEC- D9 R5 back

    cms.PSet(detSelection=cms.uint32(4610),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0440c0")),    # TEC- D1 R6 back
    cms.PSet(detSelection=cms.uint32(4620),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0480c0")),    # TEC- D2 R6 back
    cms.PSet(detSelection=cms.uint32(4630),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c0c0")),    # TEC- D3 R6 back
    cms.PSet(detSelection=cms.uint32(4640),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0500c0")),    # TEC- D4 R6 back
    cms.PSet(detSelection=cms.uint32(4650),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0540c0")),    # TEC- D5 R6 back
    cms.PSet(detSelection=cms.uint32(4660),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0580c0")),    # TEC- D6 R6 back
    cms.PSet(detSelection=cms.uint32(4670),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c0c0")),    # TEC- D7 R6 back
    cms.PSet(detSelection=cms.uint32(4680),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0600c0")),    # TEC- D8 R6 back
    cms.PSet(detSelection=cms.uint32(4690),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0640c0")),    # TEC- D9 R6 back

    cms.PSet(detSelection=cms.uint32(4710),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0440e0")),    # TEC- D1 R7 back
    cms.PSet(detSelection=cms.uint32(4720),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0480e0")),    # TEC- D2 R7 back
    cms.PSet(detSelection=cms.uint32(4730),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c0e0")),    # TEC- D3 R7 back
    cms.PSet(detSelection=cms.uint32(4740),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0500e0")),    # TEC- D4 R7 back
    cms.PSet(detSelection=cms.uint32(4750),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0540e0")),    # TEC- D5 R7 back
    cms.PSet(detSelection=cms.uint32(4760),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0580e0")),    # TEC- D6 R7 back
    cms.PSet(detSelection=cms.uint32(4770),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c0e0")),    # TEC- D7 R7 back
    cms.PSet(detSelection=cms.uint32(4780),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0600e0")),    # TEC- D8 R7 back
    cms.PSet(detSelection=cms.uint32(4790),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0640e0")),    # TEC- D9 R7 back



    cms.PSet(detSelection=cms.uint32(5110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084020")),    # TEC+ D1 R1 back
    cms.PSet(detSelection=cms.uint32(5120),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088020")),    # TEC+ D2 R1 back
    cms.PSet(detSelection=cms.uint32(5130),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c020")),    # TEC+ D3 R1 back
#    cms.PSet(detSelection=cms.uint32(5140),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090020")),    # TEC+ D4 R1 back
#    cms.PSet(detSelection=cms.uint32(5150),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094020")),    # TEC+ D5 R1 back
#    cms.PSet(detSelection=cms.uint32(5160),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098020")),    # TEC+ D6 R1 back
#    cms.PSet(detSelection=cms.uint32(5170),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c020")),    # TEC+ D7 R1 back
#    cms.PSet(detSelection=cms.uint32(5180),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0020")),    # TEC+ D8 R1 back
#    cms.PSet(detSelection=cms.uint32(5190),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4020")),    # TEC+ D9 R1 back


    cms.PSet(detSelection=cms.uint32(5210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084040")),    # TEC+ D1 R2 back
    cms.PSet(detSelection=cms.uint32(5220),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088040")),    # TEC+ D2 R2 back
    cms.PSet(detSelection=cms.uint32(5230),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c040")),    # TEC+ D3 R2 back
    cms.PSet(detSelection=cms.uint32(5240),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090040")),    # TEC+ D4 R2 back
    cms.PSet(detSelection=cms.uint32(5250),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094040")),    # TEC+ D5 R2 back
    cms.PSet(detSelection=cms.uint32(5260),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098040")),    # TEC+ D6 R2 back
#    cms.PSet(detSelection=cms.uint32(5270),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c040")),    # TEC+ D7 R2 back
#    cms.PSet(detSelection=cms.uint32(5280),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0040")),    # TEC+ D8 R2 back
#    cms.PSet(detSelection=cms.uint32(5290),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4040")),    # TEC+ D9 R2 back

    cms.PSet(detSelection=cms.uint32(5310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084060")),    # TEC+ D1 R3 back
    cms.PSet(detSelection=cms.uint32(5320),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088060")),    # TEC+ D2 R3 back
    cms.PSet(detSelection=cms.uint32(5330),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c060")),    # TEC+ D3 R3 back
    cms.PSet(detSelection=cms.uint32(5340),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090060")),    # TEC+ D4 R3 back
    cms.PSet(detSelection=cms.uint32(5350),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094060")),    # TEC+ D5 R3 back
    cms.PSet(detSelection=cms.uint32(5360),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098060")),    # TEC+ D6 R3 back
    cms.PSet(detSelection=cms.uint32(5370),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c060")),    # TEC+ D7 R3 back
    cms.PSet(detSelection=cms.uint32(5380),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0060")),    # TEC+ D8 R3 back
#    cms.PSet(detSelection=cms.uint32(5390),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4060")),    # TEC+ D9 R3 back

    cms.PSet(detSelection=cms.uint32(5410),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084080")),    # TEC+ D1 R4 back
    cms.PSet(detSelection=cms.uint32(5420),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088080")),    # TEC+ D2 R4 back
    cms.PSet(detSelection=cms.uint32(5430),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c080")),    # TEC+ D3 R4 back
    cms.PSet(detSelection=cms.uint32(5440),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090080")),    # TEC+ D4 R4 back
    cms.PSet(detSelection=cms.uint32(5450),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094080")),    # TEC+ D5 R4 back
    cms.PSet(detSelection=cms.uint32(5460),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098080")),    # TEC+ D6 R4 back
    cms.PSet(detSelection=cms.uint32(5470),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c080")),    # TEC+ D7 R4 back
    cms.PSet(detSelection=cms.uint32(5480),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0080")),    # TEC+ D8 R4 back
    cms.PSet(detSelection=cms.uint32(5490),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4080")),    # TEC+ D9 R4 back

    cms.PSet(detSelection=cms.uint32(5510),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0840a0")),    # TEC+ D1 R5 back
    cms.PSet(detSelection=cms.uint32(5520),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0880a0")),    # TEC+ D2 R5 back
    cms.PSet(detSelection=cms.uint32(5530),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c0a0")),    # TEC+ D3 R5 back
    cms.PSet(detSelection=cms.uint32(5540),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0900a0")),    # TEC+ D4 R5 back
    cms.PSet(detSelection=cms.uint32(5550),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0940a0")),    # TEC+ D5 R5 back
    cms.PSet(detSelection=cms.uint32(5560),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0980a0")),    # TEC+ D6 R5 back
    cms.PSet(detSelection=cms.uint32(5570),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c0a0")),    # TEC+ D7 R5 back
    cms.PSet(detSelection=cms.uint32(5580),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a00a0")),    # TEC+ D8 R5 back
    cms.PSet(detSelection=cms.uint32(5590),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a40a0")),    # TEC+ D9 R5 back

    cms.PSet(detSelection=cms.uint32(5610),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0840c0")),    # TEC+ D1 R6 back
    cms.PSet(detSelection=cms.uint32(5620),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0880c0")),    # TEC+ D2 R6 back
    cms.PSet(detSelection=cms.uint32(5630),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c0c0")),    # TEC+ D3 R6 back
    cms.PSet(detSelection=cms.uint32(5640),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0900c0")),    # TEC+ D4 R6 back
    cms.PSet(detSelection=cms.uint32(5650),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0940c0")),    # TEC+ D5 R6 back
    cms.PSet(detSelection=cms.uint32(5660),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0980c0")),    # TEC+ D6 R6 back
    cms.PSet(detSelection=cms.uint32(5670),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c0c0")),    # TEC+ D7 R6 back
    cms.PSet(detSelection=cms.uint32(5680),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a00c0")),    # TEC+ D8 R6 back
    cms.PSet(detSelection=cms.uint32(5690),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a40c0")),    # TEC+ D9 R6 back

    cms.PSet(detSelection=cms.uint32(5710),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0840e0")),    # TEC+ D1 R7 back
    cms.PSet(detSelection=cms.uint32(5720),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0880e0")),    # TEC+ D2 R7 back
    cms.PSet(detSelection=cms.uint32(5730),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c0e0")),    # TEC+ D3 R7 back
    cms.PSet(detSelection=cms.uint32(5740),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0900e0")),    # TEC+ D4 R7 back
    cms.PSet(detSelection=cms.uint32(5750),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0940e0")),    # TEC+ D5 R7 back
    cms.PSet(detSelection=cms.uint32(5760),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0980e0")),    # TEC+ D6 R7 back
    cms.PSet(detSelection=cms.uint32(5770),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c0e0")),    # TEC+ D7 R7 back
    cms.PSet(detSelection=cms.uint32(5780),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a00e0")),    # TEC+ D8 R7 back
    cms.PSet(detSelection=cms.uint32(5790),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a40e0"))    # TEC+ D9 R7 back



    )
    )
