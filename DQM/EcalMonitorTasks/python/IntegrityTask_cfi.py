import FWCore.ParameterSet.Config as cms

ecalIntegrityTask = cms.untracked.PSet(
    MEs = cms.untracked.PSet(
        GainSwitch = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/Integrity/GainSwitch/'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/GainSwitch/%(prefix)sIT gain switch %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        BlockSize = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/Integrity/BlockSize/'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTBlockSize/%(prefix)sIT TTBlockSize %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('')
        ),
        ByLumi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/%(prefix)sIT weighted integrity errors by lumi'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            perLumi = cms.untracked.bool(True),
            description = cms.untracked.string('Total number of integrity errors for each FED in this lumi section.')
        ),
        Gain = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/Integrity/Gain/'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/Gain/%(prefix)sIT gain %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        Total = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT integrity quality errors summary'), # In SummaryClient for historical reasons
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Total number of integrity errors for each FED.')
        ),
        TrendNErrors = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/IntegrityTask number of integrity errors'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of the number of integrity errors.')
        ),
        ChId = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/Integrity/ChId/'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/ChId/%(prefix)sIT ChId %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        TowerId = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/Integrity/TowerId/'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTId/%(prefix)sIT TTId %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('')
        )
    )
)

