import FWCore.ParameterSet.Config as cms

hcallaserhffilter2012=cms.EDFilter("HcalLaserHFFilter2012",
                                     # If verbose==True, then events failing filter will be printed to cout (as run:LS:event)
                                     verbose = cms.untracked.bool(False),
                                     # String that will appear before any event printed to cout
                                     prefix  = cms.untracked.string(""),
                                     # If the number of HF calib channels in an event is greater than or equal to minCalibChannelsHFLaser, then the event is considered to be a laser event 
                                     minCalibChannelsHFLaser=cms.untracked.int32(10),
                                     # Name of Hcal digi collection
                                     digilabel=cms.untracked.InputTag("hcalDigis"),

                                   WriteBadToFile = cms.untracked.bool(False), # if set to 'True', then the list of events failing the filter cut will be written to a text file 'badHcalLaserList_hffilter.txt'.  Events in the file will not have any prefix added, but will be a simple list of run:ls:event.       
                                   forceFilterTrue=cms.untracked.bool(False) # if specified, filter will always return 'True'.  You could use this along with the 'verbose' or 'WriteBadToFile' booleans in order to dump out bad event numbers without actually filtering them
                                     )
