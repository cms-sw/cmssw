#!/usr/bin/env python
import FWCore.ParameterSet.Config as cms

def createElements():
    elements = list()
    for i in xrange(0,10):
        elements.append(cms.untracked.PSet(lowX   = cms.untracked.double(0),
                                           highX  = cms.untracked.double(11),
                                           nchX   = cms.untracked.int32(11),
                                           name   = cms.untracked.string("Foo"+str(i)),
                                           title  = cms.untracked.string("Foo"+str(i)),
                                           value  = cms.untracked.double(i),
                                           values = cms.untracked.vdouble()))
    elements.append(cms.untracked.PSet(lowX   = cms.untracked.double(0),
                                       highX  = cms.untracked.double(11),
                                       nchX   = cms.untracked.int32(11),
                                       name   = cms.untracked.string("Bar0"),
                                       title  = cms.untracked.string("Bar0"),
                                       value  = cms.untracked.double(-1),
                                       values = cms.untracked.vdouble([i for i in range(0,11)])))

    elements.append(cms.untracked.PSet(lowX   = cms.untracked.double(0),
                                       highX  = cms.untracked.double(11),
                                       nchX   = cms.untracked.int32(11),
                                       name   = cms.untracked.string("Bar1"),
                                       title  = cms.untracked.string("Bar1"),
                                       value  = cms.untracked.double(-1),
                                       values = cms.untracked.vdouble([10 - i for i in range(0,11)])))
    return elements

def createReadRunElements():
    readRunElements = list()
    for i in xrange(0,10):
        readRunElements.append(cms.untracked.PSet(name    = cms.untracked.string("Foo"+str(i)),
                                                  means   = cms.untracked.vdouble(i),
                                                  entries = cms.untracked.vdouble(1)
        ))
    readRunElements.append(cms.untracked.PSet(name    = cms.untracked.string("Bar0"),
                                              means   = cms.untracked.vdouble(7),
                                              entries = cms.untracked.vdouble(55)))
    readRunElements.append(cms.untracked.PSet(name    = cms.untracked.string("Bar1"),
                                              means   = cms.untracked.vdouble(3),
                                              entries = cms.untracked.vdouble(55)))
    return readRunElements

def createReadLumiElements():
    readLumiElements = list()
    for i in xrange(0,10):
        readLumiElements.append(cms.untracked.PSet(name=cms.untracked.string("Foo"+str(i)),
                                                   means = cms.untracked.vdouble([i for x in xrange(0,10)]),
                                                   entries=cms.untracked.vdouble([1 for x in xrange(0,10)])
                                                   ))
    readLumiElements.append(cms.untracked.PSet(name=cms.untracked.string("Bar0"),
                                               means = cms.untracked.vdouble([7 for x in xrange(0,10)]),
                                               entries=cms.untracked.vdouble([55 for x in xrange(0,10)])
                                               ))
    readLumiElements.append(cms.untracked.PSet(name=cms.untracked.string("Bar1"),
                                               means = cms.untracked.vdouble([3 for x in xrange(0,10)]),
                                               entries=cms.untracked.vdouble([55 for x in xrange(0,10)])
                                               ))
    return readLumiElements

####### MERGED FILES SECTION #######

def createReadRunElements_merged_file1_file2():
    readRunElements = list()
    for i in xrange(0,10):
        readRunElements.append(cms.untracked.PSet(name    = cms.untracked.string("Foo"+str(i)),
                                                  means   = cms.untracked.vdouble(i),
                                                  entries = cms.untracked.vdouble(2)
        ))
    readRunElements.append(cms.untracked.PSet(name    = cms.untracked.string("Bar0"),
                                              means   = cms.untracked.vdouble(7),
                                              entries = cms.untracked.vdouble(110)))
    readRunElements.append(cms.untracked.PSet(name    = cms.untracked.string("Bar1"),
                                              means   = cms.untracked.vdouble(3),
                                              entries = cms.untracked.vdouble(110)))
    return readRunElements

def createReadLumiElements_merged_file1_file2():
    readLumiElements = list()
    for i in xrange(0,10):
        readLumiElements.append(cms.untracked.PSet(name=cms.untracked.string("Foo"+str(i)),
                                                   means = cms.untracked.vdouble([i for x in xrange(0,20)]),
                                                   entries=cms.untracked.vdouble([1 for x in xrange(0,20)])
                                                   ))
    readLumiElements.append(cms.untracked.PSet(name=cms.untracked.string("Bar0"),
                                               means = cms.untracked.vdouble([7 for x in xrange(0,20)]),
                                               entries=cms.untracked.vdouble([55 for x in xrange(0,20)])
                                               ))
    readLumiElements.append(cms.untracked.PSet(name=cms.untracked.string("Bar1"),
                                               means = cms.untracked.vdouble([3 for x in xrange(0,20)]),
                                               entries=cms.untracked.vdouble([55 for x in xrange(0,20)])
                                               ))
    return readLumiElements
