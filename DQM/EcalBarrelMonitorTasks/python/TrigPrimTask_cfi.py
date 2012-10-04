import FWCore.ParameterSet.Config as cms

ecalTrigPrimTask = cms.untracked.PSet(
#    HLTMuonPath = cms.untracked.string('HLT_Mu5_v*'),
#    HLTCaloPath = cms.untracked.string('HLT_SingleJet*'),
    runOnEmul = cms.untracked.bool(False),
    MEs = cms.untracked.PSet(
        LowIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower low interest counter%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Tower occupancy of low interest flags.')
        ),
        FGEmulError = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/TriggerPrimitives/FGBEmulation/'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Channel'),
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
                high = cms.untracked.double(15.0),
                nbins = cms.untracked.int32(15),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('bunch crossing')
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
            kind = cms.untracked.string('TProfile'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('TP Et')
            ),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.0),
                nbins = cms.untracked.int32(15),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('bunch crossing')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et vs bx Real Digis%(suffix)s'),
            description = cms.untracked.string('Mean TP Et in different bunch crossing intervals. This plot is filled by data from physics data stream. It is normal to have very little entries in BX >= 3490.')
        ),
        EtEmulError = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/TriggerPrimitives/EtEmulation/'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Channel'),
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
                title = cms.untracked.string('TP index')
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
                high = cms.untracked.double(6.0),
                nbins = cms.untracked.int32(6),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP index')
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
                high = cms.untracked.double(8.0),
                nbins = cms.untracked.int32(8),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TT flag')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Distribution of the trigger tower flags.')
        ),
        TTFMismatch = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/TriggerPrimitives/FlagMismatch/'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Channel'),
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
