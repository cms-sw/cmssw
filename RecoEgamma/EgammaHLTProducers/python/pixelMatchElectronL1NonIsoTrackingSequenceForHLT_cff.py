# The following comments couldn't be translated into the new config version:

#ckfL1NonIsoTrackCandidatesEndcap &

import FWCore.ParameterSet.Config as cms

# electron tracking sequence after pixel seeding
# $Id: pixelMatchElectronL1NonIsoTrackingSequenceForHLT.cff,v 1.2 2007/10/19 17:35:03 ghezzi Exp $
pixelMatchElectronL1NonIsoTrackingSequenceForHLT = cms.Sequence(cms.SequencePlaceholder("ckfL1NonIsoTrackCandidates")+cms.SequencePlaceholder("ctfL1NonIsoWithMaterialTracks")+cms.SequencePlaceholder("pixelMatchElectronsL1NonIsoForHLT"))

