
import FWCore.ParameterSet.Config as cms

histDefinitions = cms.PSet(
    nVertex=cms.PSet(
        name=cms.untracked.string('nVertex'),
        title=cms.untracked.string('Number of event vertices in collection'),
        nbins=cms.untracked.int32(100),
        xmin=cms.untracked.double(-0.5),
        xmax=cms.untracked.double(99.5),
    ),
)
