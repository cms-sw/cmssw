import FWCore.ParameterSet.Config as cms

hcallaserhbhehffilter2012=cms.EDFilter("HcalLaserHBHEHFFilter2012",
                                     # flag to utilize  HBHE laser filter
                                     filterHBHE  = cms.bool(True),
                                     # If the number of HBHE calib channels in an event is greater than or equal to minCalibChannelsHBHELaser, then the event is considered to be a laser event 
                                     minCalibChannelsHBHELaser=cms.int32(20),
                                     # If the difference in good vs. bad frational occupancies is greater than minFracDiffHBHELaser, then the event is considered to be a laser event
                                     minFracDiffHBHELaser = cms.double(0.3),
                                     # minimum charge threshold needed for a calib channel to count towards minCalibChannelsHBHEHELaser
                                     HBHEcalibThreshold = cms.double(15.),
                                     # Time slices used when computing total charge in a calib channel
                                     CalibTS = cms.vint32([3,4,5,6]),

                                     # flag to utilize  HF laser filter
                                     filterHF    = cms.bool(True),
                                     # If the number of HF calib channels in an event is greater than or equal to minCalibChannelsHFLaser, then the event is considered to be a laser event 
                                     minCalibChannelsHFLaser=cms.int32(10),
                                     
                                     # Name of Hcal digi collection
                                     digiLabel=cms.InputTag("hcalDigis"),

                                     # If verbose==True, then events failing filter will be printed to cout (as run:LS:event)
                                     verbose = cms.untracked.bool(False),
                                     # String that will appear before any event printed to cout
                                     prefix  = cms.untracked.string(""),

                                     WriteBadToFile = cms.untracked.bool(False), # if set to 'True', then the list of events failing the filter cut will be written to a text file 'badHcalLaserList_hcalfilter.txt'.  Events in the file will not have any prefix added, but will be a simple list of run:ls:event.
                                     forceFilterTrue=cms.untracked.bool(False) # if specified, filter will always return 'True'.  You could use this along with the 'verbose' or 'WriteBadToFile' booleans in order to dump out bad event numbers without actually filtering them
                                     )
