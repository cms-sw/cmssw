import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask

#parameters derived from training
EBThreshold = 0.1761
EEpThreshold =  0.0003009
EEmThreshold =  0.0004360

EB_PUcorr_slope = 9087.286563128135
EB_PUcorr_intercept = 391987.0277612837

EEp_PUcorr_slope = 2.097273231210836457e+03 
EEp_PUcorr_intercept= 4.905224959496531665e+04

EEm_PUcorr_slope = 2.029645065864053095e+03
EEm_PUcorr_intercept= 4.874167219924630626e+04

ecalMLClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
        EBThreshold = cms.untracked.double(EBThreshold),
	EEpThreshold = cms.untracked.double(EEpThreshold),
	EEmThreshold = cms.untracked.double(EEmThreshold),
        EB_PUcorr_slope = cms.untracked.double(EB_PUcorr_slope),
        EB_PUcorr_intercept = cms.untracked.double(EB_PUcorr_intercept),
        EEp_PUcorr_slope = cms.untracked.double(EEp_PUcorr_slope),
        EEp_PUcorr_intercept = cms.untracked.double(EEp_PUcorr_intercept),
	EEm_PUcorr_slope = cms.untracked.double(EEm_PUcorr_slope),
        EEm_PUcorr_intercept = cms.untracked.double(EEm_PUcorr_intercept)
    ),
    sources = cms.untracked.PSet(
        DigiAllByLumi = ecalOccupancyTask.MEs.DigiAllByLumi,
        AELoss = ecalOccupancyTask.MEs.AELoss,
	AEReco = ecalOccupancyTask.MEs.AEReco,
        PU = ecalOccupancyTask.MEs.PU,
        NumEvents = ecalOccupancyTask.MEs.NEvents,
	BadTowerCount = ecalOccupancyTask.MEs.BadTowerCount,
	BadTowerCountNorm = ecalOccupancyTask.MEs.BadTowerCountNorm
    ),
    MEs = cms.untracked.PSet(
        MLQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT%(suffix)s ML quality summary'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Quality summary from the ML inference.')
        ),
       EventsperMLImage = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/Number of Events used per ML image'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of the number of events in an image fed into the ML model')
        ),
        TrendMLBadTower = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/Number of bad towers from MLDQM %(prefix)s'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of the number of bad towers flagged by the MLDQM model')
        )
    )
)
