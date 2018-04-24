from RecoMET.METFilters.hcallaserevent_cfi import * # from fillDescriptions

try:
    from hcalLaserBadEvents_2011 import badEvents
except:
    print "<hcalLaserEventFilter_cfi> WARNING!  No explicit list 'badEvents.py' containing bad HCAL laser run/events was found!"
    badEvents=[]

#hcalLaserEventFilter = cms.EDFilter("HcalLaserEventFilter",)
hcalLaserEventFilter = cms.EDFilter("HcalLaserEventFilter",
                                    BadRunEventNumbers=cms.vuint32(badEvents
                                    # badEvents file contains 218 bad events, as of 28 Nov. 2011
                                                                  ),
                                    )
