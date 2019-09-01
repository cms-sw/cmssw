from __future__ import print_function
from __future__ import absolute_import
from RecoMET.METFilters.hcallaserevent_cfi import * # from fillDescriptions

try:
    from .hcalLaserBadEvents_2011 import badEvents
except:
    print("<hcalLaserEventFilter_cfi> WARNING!  No explicit list 'badEvents.py' containing bad HCAL laser run/events was found!")
    badEvents=[]

hcalLaserEventFilter = hcallaserevent.clone(BadRunEventNumbers=cms.vuint32(badEvents) )

from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
run2_HCAL_2018.toModify( hcalLaserEventFilter,
                           vetoByHBHEOccupancy=False,
                           minOccupiedHBHE=8000,
                       )

