# The following comments couldn't be translated into the new config version:

#ckfL1IsoTrackCandidatesEndcap &

import FWCore.ParameterSet.Config as cms

# electron tracking sequence after pixel seeding
# $Id: pixelMatchElectronL1IsoTrackingSequenceForHLT.cff,v 1.2 2007/10/19 17:35:02 ghezzi Exp $
pixelMatchElectronL1IsoTrackingSequenceForHLT = cms.Sequence(cms.SequencePlaceholder("ckfL1IsoTrackCandidates")+cms.SequencePlaceholder("ctfL1IsoWithMaterialTracks")+cms.SequencePlaceholder("pixelMatchElectronsL1IsoForHLT"))

