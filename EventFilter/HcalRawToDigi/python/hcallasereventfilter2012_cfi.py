import FWCore.ParameterSet.Config as cms

hcallasereventfilter2012=cms.EDFilter("HcalLaserEventFilter2012",
                                      # Specify laser events to remove
                                      # Events are listed as string in 'run:ls:event' format
                                      EventList = cms.untracked.vstring([]),
                                      # if verbose==true, run:ls:event for any event failing filter will be printed to cout
                                      verbose   = cms.untracked.bool(False),
                                      # Select a prefix to appear before run:ls:event when run info dumped to cout.  This makes searching for listed events a bit easier
                                      prefix    = cms.untracked.string(""),
                                      # If minrun or maxrun are >-1, then only a subsection of EventList corresponding to the given [minrun,maxrun] range are searched when looking to reject bad events.  This can speed up the code a bit when looking over a small section of data, since the bad EventList can be shortened considerably.  
                                      minrun    = cms.untracked.int32(-1),
                                      maxrun    = cms.untracked.int32(-1),
                                      WriteBadToFile = cms.untracked.bool(False), # if set to 'True', then the list of events failing the filter cut will be written to a text file 'badHcalLaserList_eventfilter.txt'.  Events in the file will not have any prefix added, but will be a simple list of run:ls:event.
                                      forceFilterTrue=cms.untracked.bool(False) # if specified, filter will always return 'True'.  You could use this along with the 'verbose' or 'WriteBadToFile' booleans in order to dump out bad event numbers without actually filtering them
                                      )
