import FWCore.ParameterSet.Config as cms

ecalTowerStatusTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        doDAQInfo = cms.untracked.bool(True),
        doDCSInfo = cms.untracked.bool(True)
    ),
    MEs = cms.untracked.PSet(
        DAQSummary = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/DAQSummary'),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('Report'),
            kind = cms.untracked.string('REAL'),
            description = cms.untracked.string('')
        ),
        DAQSummaryMap = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/DAQSummaryMap'),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('DCC'),
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('')
        ),
        DAQContents = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/DAQContents/Ecal_%(sm)s'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Report'),
            kind = cms.untracked.string('REAL'),
            description = cms.untracked.string('')
        ),
        DCSSummary = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/DCSSummary'),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('Report'),
            kind = cms.untracked.string('REAL'),
            description = cms.untracked.string('')
        ),
        DCSSummaryMap = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/DCSSummaryMap'),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('DCC'),
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('')
        ),
        DCSContents = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/DCSContents/Ecal_%(sm)s'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Report'),
            kind = cms.untracked.string('REAL'),
            description = cms.untracked.string('')
        )
    )
)
