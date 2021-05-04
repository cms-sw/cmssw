import FWCore.ParameterSet.Config as cms

#pieces needed to run the associator-related stuff
#from Geometry.CMSCommonData.cmsIdealGeometryXML_cff import *
#from Geometry.CaloEventSetup.CaloGeometry_cfi import *
#from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from PhysicsTools.RecoAlgos.highPtTracks_cfi import *
from RecoMuon.MuonIsolationProducers.isoDepositProducerIOBlocks_cff import *
from RecoMuon.MuonIsolationProducers.trackExtractorBlocks_cff import *
from RecoMuon.MuonIsolationProducers.caloExtractorByAssociatorBlocks_cff import *
from RecoMuon.MuonIsolationProducers.jetExtractorBlock_cff import *
from RecoMuon.MuonIsolationProducers.caloExtractorBlocks_cff import *
tkIsoDepositTk = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("highPtTracks"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('best'),
    ExtractorPSet = cms.PSet(
        MIsoTrackExtractorBlock
    )
)

tkIsoDepositCalByAssociatorTowers = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("highPtTracks"),
    MultipleDepositsFlag = cms.bool(True),
    trackType = cms.string('best'),
    ExtractorPSet = cms.PSet(
        MIsoCaloExtractorByAssociatorTowersBlock
    )
)

tkIsoDepositCalByAssociatorHits = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("highPtTracks"),
    MultipleDepositsFlag = cms.bool(True),
    trackType = cms.string('best'),
    ExtractorPSet = cms.PSet(
        MIsoCaloExtractorByAssociatorHitsBlock
    )
)

tkIsoDepositJets = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("highPtTracks"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('best'),
    ExtractorPSet = cms.PSet(
        MIsoJetExtractorBlock
    )
)

tkIsoDepositCalEcal = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("highPtTracks"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('best'),
    ExtractorPSet = cms.PSet(
        MIsoCaloExtractorEcalBlock
    )
)

tkIsoDepositCalHcal = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("highPtTracks"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('best'),
    ExtractorPSet = cms.PSet(
        MIsoCaloExtractorHcalBlock
    )
)

#
# and now sequences of the above
#
# "standard sequence"
tkIsoDeposits = cms.Sequence(highPtTracks+tkIsoDepositTk+tkIsoDepositCalByAssociatorTowers+tkIsoDepositJets)


