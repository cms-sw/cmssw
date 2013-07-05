import FWCore.ParameterSet.Config as cms

from types import *

def dqmpset(d) :
    pset = dict()
    for key in d.iterkeys() :
        if isinstance(d[key], cms.PSet) :
            pset[key] = d[key]
        elif type(d[key]) is DictType :
            pset[key] = dqmpset(d[key])
        elif type(d[key]) is StringType :
            pset[key] = cms.untracked.string(d[key])
        elif type(d[key]) is IntType :
            pset[key] = cms.untracked.int32(d[key])
        elif type(d[key]) is FloatType :
            pset[key] = cms.untracked.double(d[key])
        elif type(d[key]) is BooleanType :
            pset[key] = cms.untracked.bool(d[key])
        elif type(d[key]) is ListType :
            if type(d[key][0]) is StringType :
                pset[key] = cms.untracked.vstring(*(d[key]))
            elif type(d[key][0]) is IntType :
                pset[key] = cms.untracked.vint32(*(d[key]))
            elif type(d[key][0]) is FloatType :
                pset[key] = cms.untracked.vdouble(*(d[key]))

    return cms.untracked.PSet(**pset)

def dqmpaths(prefix, dOfD) :
    pset = dict()
    for dKey in dOfD.iterkeys() :
        d = dOfD[dKey]
        psubset = dict()
        for key in d.iterkeys() :
            psubset[key] = cms.untracked.string(prefix + "/" + d[key])

        pset[dKey] = cms.untracked.PSet(**psubset)

    return cms.untracked.PSet(**pset)
