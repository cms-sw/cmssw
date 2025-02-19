import FWCore.ParameterSet.Config as cms

hcallaserhbhefilter2012=cms.EDFilter("HcalLaserHBHEFilter2012",
                                     # If verbose==True, then events failing filter will be printed to cout (as run:LS:event)
                                     verbose = cms.untracked.bool(False),
                                     # String that will appear before any event printed to cout
                                     prefix  = cms.untracked.string(""),
                                     # If the number of HBHE calib channels in an event is greater than or equal to minCalibChannelsHBHELaser, then the event is considered to be a laser event 
                                     minCalibChannelsHBHEHELaser=cms.untracked.int32(20),
                                     # If the difference in good vs. bad frational occupancies is greater than minFracDiffHBHELaser, then the event is considered to be a laser event
                                     minFracDiffHBHELaser = cms.untracked.double(0.3),
                                     # Name of Hcal digi collection
                                     digilabel=cms.untracked.InputTag("hcalDigis"),
                                     # minimum charge threshold needed for a calib channel to count towards minCalibChannelsHBHEHELaser
                                     HBHEcalibThreshold = cms.untracked.double(15.),
                                     # Time slices used when computing total charge in a calib channel
                                     CalibTS = cms.untracked.vint32([3,4,5,6]),
                                     WriteBadToFile = cms.untracked.bool(False), # if set to 'True', then the list of events failing the filter cut will be written to a text file 'badHcalLaserList_hbhefilter.txt'.  Events in the file will not have any prefix added, but will be a simple list of run:ls:event.
                                     forceFilterTrue=cms.untracked.bool(False) # if specified, filter will always return 'True'.  You could use this along with the 'verbose' or 'WriteBadToFile' booleans in order to dump out bad event numbers without actually filtering them
                                     )
