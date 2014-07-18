import FWCore.ParameterSet.Config as cms

threshS9 = 0.125

ecalEnergyTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        #    threshS9 = cms.untracked.double(0.125),
        isPhysicsRun = cms.untracked.bool(True)
    ),
    MEs = cms.untracked.PSet(
        HitMapAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT%(suffix)s energy summary'), # In SummaryClient for historical reasons
            kind = cms.untracked.string('TProfile2D'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('2D distribution of the mean rec hit energy.')
        ),
        HitAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit spectrum%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Rec hit energy distribution.')
        ),
        Hit = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT energy spectrum %(sm)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Rec hit energy distribution.')
        ),
        HitMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit energy %(sm)s'),
            kind = cms.untracked.string('TProfile2D'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of the mean rec hit energy.')
        )
    )
)

