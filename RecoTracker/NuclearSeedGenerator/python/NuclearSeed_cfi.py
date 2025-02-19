# The following comments couldn't be translated into the new config version:

# trajectory producer

# pt min of the seeds

# max number of hits of the primary tracks to be checked

# factor to rescale cov. matrix used to find compatible measurements

# if true check if nuclear interaction also on completed tracks

# exprimental seeding with 3 hits (does not work yet)

# NavigationSchool

import FWCore.ParameterSet.Config as cms

nuclearSeed = cms.EDProducer("NuclearSeedsEDProducer",
    producer = cms.string('TrackRefitter'),
    maxHits = cms.int32(5),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    # the measurements
    MeasurementTrackerName = cms.string(''),
    improveSeeds = cms.bool(False),
    rescaleErrorFactor = cms.double(1.5),
    ptMin = cms.double(0.3),
    checkCompletedTrack = cms.bool(True)
)


