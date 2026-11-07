import FWCore.ParameterSet.Config as cms

# This file contains the mapping between offsets for a single core (in a single DTC) to GBT IDs for the CRACK geometry.
# Do not change unless you are an expert.

crackMapping = cms.VPSet(

    # CRACK Tray #1

    # offset 0 (EMP Channel 04) -> GBT 70
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(0), gbtID = cms.int32(70), coreID = cms.int32(0)),
    # offset 1 (EMP Channel 05) -> GBT 67
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(1), gbtID = cms.int32(67), coreID = cms.int32(0)),
    # offset 2 (EMP Channel 06) -> GBT 63
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(2), gbtID = cms.int32(63), coreID = cms.int32(0)),
    # offset 3 (EMP Channel 07) -> GBT 71
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(3), gbtID = cms.int32(71), coreID = cms.int32(0)),
    # offset 4 (EMP Channel 08) -> GBT 69
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(4), gbtID = cms.int32(69), coreID = cms.int32(0)),
    # offset 5 (EMP Channel 09) -> GBT 65
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(5), gbtID = cms.int32(65), coreID = cms.int32(0)),
    # offset 6 (EMP Channel 10) -> GBT 61
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(6), gbtID = cms.int32(61), coreID = cms.int32(0)),
    # offset 7 (EMP Channel 11) -> GBT 62
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(7), gbtID = cms.int32(62), coreID = cms.int32(0)),
    # offset 8 (EMP Channel 12) -> GBT 66
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(8), gbtID = cms.int32(66), coreID = cms.int32(0)),
    # offset 9 (EMP Channel 13) -> GBT 60
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(9), gbtID = cms.int32(60), coreID = cms.int32(0)),
    # offset 10 (EMP Channel 14) -> GBT 64
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(10), gbtID = cms.int32(64), coreID = cms.int32(0)),
    # offset 11 (EMP Channel 15) -> GBT 68
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(11), gbtID = cms.int32(68), coreID = cms.int32(0)),

    # CRACK Tray #2

    # offset 0 (EMP Channel 100) -> GBT 56
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(0), gbtID = cms.int32(56), coreID = cms.int32(1)),
    # offset 1 (EMP Channel 101) -> GBT 52
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(1), gbtID = cms.int32(52), coreID = cms.int32(1)),
    # offset 2 (EMP Channel 102) -> GBT 48
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(2), gbtID = cms.int32(48), coreID = cms.int32(1)),
    # offset 3 (EMP Channel 103) -> GBT 54
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(3), gbtID = cms.int32(54), coreID = cms.int32(1)),
    # offset 4 (EMP Channel 104) -> GBT 50
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(4), gbtID = cms.int32(50), coreID = cms.int32(1)),
    # offset 5 (EMP Channel 105) -> GBT 49
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(5), gbtID = cms.int32(49), coreID = cms.int32(1)),
    # offset 6 (EMP Channel 106) -> GBT 53
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(6), gbtID = cms.int32(53), coreID = cms.int32(1)),
    # offset 7 (EMP Channel 107) -> GBT 57
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(7), gbtID = cms.int32(57), coreID = cms.int32(1)),
    # offset 8 (EMP Channel 108) -> GBT 59
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(8), gbtID = cms.int32(59), coreID = cms.int32(1)),
    # offset 9 (EMP Channel 109) -> GBT 51
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(9), gbtID = cms.int32(51), coreID = cms.int32(1)),
    # offset 10 (EMP Channel 110) -> GBT 55
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(10), gbtID = cms.int32(55), coreID = cms.int32(1)),
    # offset 11 (EMP Channel 111) -> GBT 58
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(11), gbtID = cms.int32(58), coreID = cms.int32(1)),

    # CRACK Tray #3

    # offset 0 (EMP Channel 64) -> GBT 44
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(0), gbtID = cms.int32(44), coreID = cms.int32(2)),
    # offset 1 (EMP Channel 65) -> GBT 40
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(1), gbtID = cms.int32(40), coreID = cms.int32(2)),
    # offset 2 (EMP Channel 66) -> GBT 36
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(2), gbtID = cms.int32(36), coreID = cms.int32(2)),
    # offset 3 (EMP Channel 67) -> GBT 42
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(3), gbtID = cms.int32(42), coreID = cms.int32(2)),
    # offset 4 (EMP Channel 68) -> GBT 38
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(4), gbtID = cms.int32(38), coreID = cms.int32(2)),
    # offset 5 (EMP Channel 69) -> GBT 37
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(5), gbtID = cms.int32(37), coreID = cms.int32(2)),
    # offset 6 (EMP Channel 70) -> GBT 41
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(6), gbtID = cms.int32(41), coreID = cms.int32(2)),
    # offset 7 (EMP Channel 71) -> GBT 45
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(7), gbtID = cms.int32(45), coreID = cms.int32(2)),
    # offset 8 (EMP Channel 72) -> GBT 47
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(8), gbtID = cms.int32(47), coreID = cms.int32(2)),
    # offset 9 (EMP Channel 73) -> GBT 39
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(9), gbtID = cms.int32(39), coreID = cms.int32(2)),
    # offset 10 (EMP Channel 74) -> GBT 43
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(10), gbtID = cms.int32(43), coreID = cms.int32(2)),
    # offset 11 (EMP Channel 75) -> GBT 46
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(11), gbtID = cms.int32(46), coreID = cms.int32(2)),

    # CRACK Tray #4

    # offset 0 (EMP Channel 88) -> GBT 32
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(0), gbtID = cms.int32(32), coreID = cms.int32(3)),
    # offset 1 (EMP Channel 89) -> GBT 28
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(1), gbtID = cms.int32(28), coreID = cms.int32(3)),
    # offset 2 (EMP Channel 90) -> GBT 24
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(2), gbtID = cms.int32(24), coreID = cms.int32(3)),
    # offset 3 (EMP Channel 91) -> GBT 30
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(3), gbtID = cms.int32(30), coreID = cms.int32(3)),
    # offset 4 (EMP Channel 92) -> GBT 26
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(4), gbtID = cms.int32(26), coreID = cms.int32(3)),
    # offset 5 (EMP Channel 93) -> GBT 25
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(5), gbtID = cms.int32(25), coreID = cms.int32(3)),
    # offset 6 (EMP Channel 94) -> GBT 29
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(6), gbtID = cms.int32(29), coreID = cms.int32(3)),
    # offset 7 (EMP Channel 95) -> GBT 33
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(7), gbtID = cms.int32(33), coreID = cms.int32(3)),
    # offset 8 (EMP Channel 96) -> GBT 35
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(8), gbtID = cms.int32(35), coreID = cms.int32(3)),
    # offset 9 (EMP Channel 97) -> GBT 27
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(9), gbtID = cms.int32(27), coreID = cms.int32(3)),
    # offset 10 (EMP Channel 98) -> GBT 31
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(10), gbtID = cms.int32(31), coreID = cms.int32(3)),
    # offset 11 (EMP Channel 99) -> GBT 34
    cms.PSet(dtc = cms.int32(1), offset = cms.int32(11), gbtID = cms.int32(34), coreID = cms.int32(3)),

)