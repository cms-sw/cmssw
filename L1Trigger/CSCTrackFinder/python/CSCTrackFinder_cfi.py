import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
CSCTrackFinderProducerNames = cms.PSet(
    #names of input sources
    CSCTrackProducer = cms.untracked.InputTag("l1csctracks"),
    SectorReceiverInput = cms.untracked.InputTag("lctproducer","MPCSORTED")
)
l1csctracks = cms.EDProducer("CSCTFTrackProducer",
    CSCTrackFinderProducerNames,
    isTMB07 = cms.bool(False),
    useDT = cms.bool(False),
    SectorProcessor = cms.PSet(
        CSCCommonTrigger,
        SRLUT = cms.PSet(
            Binary = cms.untracked.bool(False),
            ReadLUTs = cms.untracked.bool(False),
            LUTPath = cms.untracked.string('./')
        ),
        AllowALCTonly = cms.bool(True),
        PTLUT = cms.PSet(
            LowQualityFlag = cms.untracked.uint32(4),
            ReadPtLUT = cms.untracked.bool(False),
            PtMethod = cms.untracked.uint32(1)
        ),
        singlesTrackOutput = cms.uint32(1),
        singlesTrackPt = cms.uint32(255),
        trigger_on_MB1a = cms.bool(False),
        EtaMin = cms.vuint32(22, 22, 14, 14, 14, 10, 10, 10),
        trigger_on_ME1a = cms.bool(True),
        trigger_on_ME1b = cms.bool(True),
        EtaMax = cms.vuint32(127, 127, 127, 127, 127, 24, 24, 24),
        CoreLatency = cms.uint32(8),
        PreTrigger = cms.uint32(2),
        trigger_on_MB1d = cms.bool(False),
        run_core = cms.bool(True),
        mindeta_accp = cms.uint32(4),
        EtaWindows = cms.vuint32(2, 2, 2, 2, 2, 2),
        AllowCLCTonly = cms.bool(True),
        BXAdepth = cms.uint32(2),
        mindphip = cms.uint32(70),
        maxdeta_accp = cms.uint32(16),
        trigger_on_ME4 = cms.bool(True),
        maxdphi_accp = cms.uint32(128),
        trigger_on_ME3 = cms.bool(True),
        trigger_on_ME2 = cms.bool(True)
    )
)

csctfmuonsorter = cms.EDProducer("CSCTFCandidateProducer",
    CSCTrackFinderProducerNames,
    MuonSorter = cms.PSet(
        CSCCommonTrigger
    )
)

#the two possible execution sequences
csc_sp_tracks = cms.Sequence(l1csctracks)
csc_tf_cands = cms.Sequence(l1csctracks*csctfmuonsorter)

