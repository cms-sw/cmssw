import FWCore.ParameterSet.Config as cms

bxBins = [
    "1",
    "271",
    "541",
    "892",
    "1162",
    "1432",
    "1783",
    "2053",
    "2323",
    "2674",
    "2944",
    "3214",
    "3446",
    "3490",
    "3491",
    "3565"
]

ecalTrigPrimTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        #    HLTMuonPath = cms.untracked.string('HLT_Mu5_v*'),
        #    HLTCaloPath = cms.untracked.string('HLT_SingleJet*'),
        runOnEmul = cms.untracked.bool(True)
    ),
    MEs = cms.untracked.PSet(
        LowIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower low interest counter%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Tower occupancy of low interest flags.')
        ),
        FGEmulError = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/TriggerPrimitives/FGBEmulation/'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT EmulFineGrainVetoError %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('')
        ),
        EtMaxEmul = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/Emulated/%(prefix)sTTT Et spectrum Emulated Digis max%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(256.0),
                nbins = cms.untracked.int32(128),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP Et')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of the maximum Et value within one emulated TP')
        ),
        OccVsBx = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT TP occupancy vs bx Real Digis%(suffix)s'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(16.0),
                nbins = cms.untracked.int32(16),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('bunch crossing'),
                labels = cms.untracked.vstring(bxBins)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('TP occupancy in different bunch crossing intervals. This plot is filled by data from physics data stream. It is normal to have very little entries in BX >= 3490.')
        ),
        HighIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower high interest counter%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Tower occupancy of high interest flags.')
        ),
        EtVsBx = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et vs bx Real Digis%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('User'),
            kind = cms.untracked.string('TProfile'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(16.0),
                nbins = cms.untracked.int32(16),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('bunch crossing'),
                labels = cms.untracked.vstring(bxBins)
            ),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('TP Et')
            ),
            description = cms.untracked.string('Mean TP Et in different bunch crossing intervals. This plot is filled by data from physics data stream. It is normal to have very little entries in BX >= 3490.')
        ),
        EtEmulError = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/TriggerPrimitives/EtEmulation/'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT EmulError %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('')
        ),
        MatchedIndex = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT EmulMatch %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(6.0),
                nbins = cms.untracked.int32(6),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP index'),
                labels = cms.untracked.vstring(["no emul", "0", "1", "2", "3", "4"])
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Counter for TP "timing" (= index withing the emulated TP whose Et matched that of the real TP)')
        ),
        EmulMaxIndex = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT max TP matching index%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(5.5),
                nbins = cms.untracked.int32(6),
                low = cms.untracked.double(-0.5),
                title = cms.untracked.string('TP index'),
                labels = cms.untracked.vstring(["no maximum", "0", "1", "2", "3", "4"])
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of the index of emulated TP with the highest Et value.')
        ),
        MedIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower med interest counter%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Tower occupancy of medium interest flags.')
        ),
        TTFlags = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT TT Flags%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(7.5),
                nbins = cms.untracked.int32(8),
                low = cms.untracked.double(-0.5),
                title = cms.untracked.string('TT flag'),
                labels = cms.untracked.vstring(map(str, range(0, 8)))
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Distribution of the trigger tower flags.')
        ),
        TTFlags4 = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT TTF4 Occupancy%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Occupancy for TP digis with TTF=4.')
        ),
        TTMaskMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/TTStatus/%(prefix)sTTT TT Masking Status%(sm)s'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('PseudoStrip'),
            description = cms.untracked.string('Trigger tower and pseudo-strip masking status: a TT or strip is red if it is masked')
        ),
        TTMaskMapAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT TT Masking Status%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Trigger tower masking status: a TT is red if it is masked.')
        ),
        TTFMismatch = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/TriggerPrimitives/FlagMismatch/'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT TT flag mismatch%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('')
        ),
        EtSummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s Et trigger tower summary'),
            kind = cms.untracked.string('TProfile2D'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(256.0),
                nbins = cms.untracked.int32(128),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP Et')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('2D distribution of the trigger primitive Et.')
        ),
        EtRealMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et map Real Digis %(sm)s'),
            kind = cms.untracked.string('TProfile2D'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(256.0),
                nbins = cms.untracked.int32(128),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP Et')
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('2D distribution of the trigger primitive Et.')
        ),
        EtReal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et spectrum Real Digis%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(256.0),
                nbins = cms.untracked.int32(128),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP Et')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of the trigger primitive Et.')
        )
    )
)
