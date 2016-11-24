# Abstract all early deletion settings here

from RecoTracker.Configuration.customiseEarlyDeleteForSeeding import customiseEarlyDeleteForSeeding

def customiseEarlyDeleteForRECO(process):
    process = customiseEarlyDeleteForSeeding(process)
    return process
