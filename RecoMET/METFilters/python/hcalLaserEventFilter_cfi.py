from RecoMET.METFilters.hcallaserevent_cfi import * # from fillDescriptions

try:
    from hcalLaserBadEvents_2011 import badEvents
except:
    print "<hcalLaserEventFilter_cfi> WARNING!  No explicit list 'badEvents.py' containing bad HCAL laser run/events was found!"
    badEvents=[]

hcalLaserEventFilter = hcallaserevent.clone(BadRunEventNumbers=cms.vuint32(badEvents) )
