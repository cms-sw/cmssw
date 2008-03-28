import FWCore.ParameterSet.Config as cms

# electron tracking sequence after pixel seeding
# $Id: pixelMatchElectronTrackingSequenceForHLT.cff,v 1.3 2007/02/23 11:36:17 monicava Exp $
pixelMatchElectronTrackingSequenceForHLT = cms.Sequence(cms.SequencePlaceholder("ckfTrackCandidatesBarrel")+cms.SequencePlaceholder("ckfTrackCandidatesEndcap")+cms.SequencePlaceholder("ctfWithMaterialTracksBarrel")+cms.SequencePlaceholder("ctfWithMaterialTracksEndcap")+cms.SequencePlaceholder("pixelMatchElectronsForHLT"))

