import FWCore.ParameterSet.Config as cms

#
# Create sequence to separate HF composition samples.
# 
# These are exclusive priorities, so sample "i" will not overlap with "i+1".
# Note that the "dr" values below correspond to the dr between the
# matched genjet, and the sister genjet. 
#
# This filter will write out a single unsigned integer:
#
# 1) W+bb with >= 2 jets from the ME (dr > 0.5)
# 2) W+b or W+bb with 1 jet from the ME
# 3) W+cc from the ME (dr > 0.5)
# 4) W+c or W+cc with 1 jet from the ME
# 5) W+bb with 1 jet from the parton shower (dr == 0.0)
# 6) W+cc with 1 jet from the parton shower (dr == 0.0)
#
# These are the "trash bin" samples that we're throwing away:
#
# 7) W+bb with >= 2 partons but 1 jet from the ME (dr == 0.0)
# 8) W+cc with >= 2 partons but 1 jet from the ME (dr == 0.0)
# 9) W+bb with >= 2 partons but 2 jets from the PS (dr > 0.5)
# 10)W+cc with >= 2 partons but 2 jets from the PS (dr > 0.5)
#
# And here is the true "light flavor" sample:
#
# 11) Veto of all the previous (W+ light jets)
#
# The user can select one of these or another to filter on
# via the "pathToSelect" member, which can be 1-11. 

flavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                   bsrc = cms.InputTag("bFlavorHistoryProducer", "bPartonFlavorHistory"),
                                   csrc = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                   pathToSelect = cms.int32(-1),   # no path selected by default
                                   dr = cms.double(0.5),           # cutoff delta r
                                   verbose = cms.bool(False)
                                   )
