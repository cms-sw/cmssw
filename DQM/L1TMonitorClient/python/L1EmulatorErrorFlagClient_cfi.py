import FWCore.ParameterSet.Config as cms

l1EmulatorErrorFlagClient = cms.EDAnalyzer("L1EmulatorErrorFlagClient",
    #
    # for each L1 system, give:
    #     - SystemLabel:  system label
    #     - HwValLabel:   system label as used in hardware validation package 
    #                     (the package producing the ErrorFlag histogram)
    #     - SystemMask:   system mask: if 1, the system is masked in the summary plot
    #     - SystemFolder: the folder where the ErrorFlag histogram is looked for
    #
    # the position in the parameter set gives, in reverse order, the position in the reportSummaryMap
    # in the emulator column (left column)
    L1Systems = cms.VPSet(
                    cms.PSet(
                        SystemLabel = cms.string("ECAL"),
                        HwValLabel = cms.string("ETP"),
                        SystemMask  = cms.uint32(1),
                        SystemFolder = cms.string("")
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("HCAL"),
                        HwValLabel = cms.string("HTP"),
                        SystemMask  = cms.uint32(1),
                        SystemFolder = cms.string("")
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("RCT"),
                        HwValLabel = cms.string("RCT"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string("")
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("Stage1Layer2"),
                        HwValLabel = cms.string("Stage1Layer2"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string("")
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("DTTF"),
                        HwValLabel = cms.string("DTF"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string("")
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("DTTPG"),
                        HwValLabel = cms.string("DTP"),
                        SystemMask  = cms.uint32(1),
                        SystemFolder = cms.string("")
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("CSCTF"),
                        HwValLabel = cms.string("CTF"),
                        SystemMask  = cms.uint32(1),
                        SystemFolder = cms.string("")
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("CSCTPG"),
                        HwValLabel = cms.string("CTP"),
                        SystemMask  = cms.uint32(1),
                        SystemFolder = cms.string("")
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("RPC"),
                        HwValLabel = cms.string("RPC"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string("")
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("GMT"),
                        HwValLabel = cms.string("GMT"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string("")
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("GT"),
                        HwValLabel = cms.string("GT"),
                        SystemMask  = cms.uint32(1),
                        SystemFolder = cms.string("L1TEMU/Stage1GTexpert")
                        )
                        
                     )
)
