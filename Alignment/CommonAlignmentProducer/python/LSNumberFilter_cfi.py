import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.lsNumberFilter_cfi import lsNumberFilter
LSNumberFilter = lsNumberFilter.clone(
    minLS = 21,
    veto_HLT_Menu = [
        "LumiScan",
        "PPS",
        "ECALTiming",
        "ECAL"]
    )
-- dummy change --
