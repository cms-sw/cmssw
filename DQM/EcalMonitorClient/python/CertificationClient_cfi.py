import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.TowerStatusTask_cfi import ecalTowerStatusTask
from DQM.EcalMonitorClient.SummaryClient_cfi import ecalSummaryClient

ecalCertificationClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        DQM = ecalSummaryClient.MEs.ReportSummaryContents,
        DCS = ecalTowerStatusTask.MEs.DCSContents,
        DAQ = ecalTowerStatusTask.MEs.DAQContents
    ),
    MEs = cms.untracked.PSet(
        CertificationMap = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/CertificationSummaryMap'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('')
        ),
        CertificationContents = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/CertificationContents/Ecal_%(sm)s'),
            kind = cms.untracked.string('REAL'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Report'),
            description = cms.untracked.string('')            
        ),
        Certification = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/CertificationSummary'),
            kind = cms.untracked.string('REAL'),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('Report'),
            description = cms.untracked.string('')            
        )
    )
)
