import FWCore.ParameterSet.Config as cms

L1MuGMTParameters = cms.ESProducer("L1MuGMTParametersProducer",
    EtaWeight_barrel = cms.double(0.028),
    PhiWeight_barrel = cms.double(1.0),
    EtaPhiThreshold_barrel = cms.double(0.062),
    EtaWeight_endcap = cms.double(0.13),
    PhiWeight_endcap = cms.double(1.0),
    EtaPhiThreshold_endcap = cms.double(0.062),
    EtaWeight_COU = cms.double(0.316),
    PhiWeight_COU = cms.double(1.0),
    EtaPhiThreshold_COU = cms.double(0.127),
    CaloTrigger = cms.bool(True),
    IsolationCellSizeEta = cms.int32(2),
    IsolationCellSizePhi = cms.int32(2),
    DoOvlRpcAnd = cms.bool(False),

    PropagatePhi = cms.bool(False),
    MergeMethodPhiBrl = cms.string('takeDT'),
    MergeMethodPhiFwd = cms.string('takeCSC'),
    MergeMethodEtaBrl = cms.string('Special'),
    MergeMethodEtaFwd = cms.string('Special'),
    MergeMethodPtBrl = cms.string('byMinPt'),
    MergeMethodPtFwd = cms.string('byMinPt'),
    MergeMethodChargeBrl = cms.string('takeDT'),
    MergeMethodChargeFwd = cms.string('takeCSC'),
    MergeMethodMIPBrl = cms.string('Special'),
    MergeMethodMIPFwd = cms.string('Special'),
    MergeMethodMIPSpecialUseANDBrl = cms.bool(False),
    MergeMethodMIPSpecialUseANDFwd = cms.bool(False),
    MergeMethodISOBrl = cms.string('Special'),
    MergeMethodISOFwd = cms.string('Special'),
    MergeMethodISOSpecialUseANDBrl = cms.bool(True),
    MergeMethodISOSpecialUseANDFwd = cms.bool(True),
    MergeMethodSRKBrl = cms.string('takeDT'),
    MergeMethodSRKFwd = cms.string('takeCSC'),
    HaloOverwritesMatchedBrl = cms.bool(True),
    HaloOverwritesMatchedFwd = cms.bool(True),
    SortRankOffsetBrl = cms.uint32(10),
    SortRankOffsetFwd = cms.uint32(10),

    CDLConfigWordDTCSC = cms.uint32(2),
    CDLConfigWordCSCDT = cms.uint32(3),
    CDLConfigWordbRPCCSC = cms.uint32(16),
    CDLConfigWordfRPCDT = cms.uint32(1),

#   VersionSortRankEtaQLUT - quality assign LUT
#   1 = full geometry tuned with ORCA
#   2 = staged RPC geometry - accept q=2 CSC candidates
#   275 = modification used since May 2011
    VersionSortRankEtaQLUT = cms.uint32(2),
#   General versioning of GMT LUTs introduced Feb 2012
#   0 = version until the end 2011
#   1 = version to be used in 2012
    VersionLUTs = cms.uint32(0),
#   Subsystem Mask:
#   4 bits: 1..off; 0..on
#   bit0:DTTF, bit1:RPCb, bit2:CSC, bit3:RPCf
    SubsystemMask = cms.uint32(0)
)


