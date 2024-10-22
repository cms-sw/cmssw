#!/usr/bin/env python3
"""
Configuration file to be used as input in unit tests of the utility hltFindDuplicates.

The configuration is made of modules labelled "d*" and "m*".

Details on the configuration.
 - For each group of modules (d* and m*),
   - modules are ordered in 3 levels (e.g. d1*, d2*, d3*), and
   - for every level, there are two versions (*x and *y) of the module (e.g. d1x, d1y).
 - The *x (*y) modules depend only on *x (*y) modules, and not on *y (*x) modules.
 - The *2* modules depend on *1* modules.
 - The *3* modules depend on *1* and *2* modules.
 - The m* modules are the counterparts of the d* modules.
   - The m* modules do not depend on d* modules (and viceversa).
   - A given m{1,2,3}{x,y} module may or may not be a duplicate of the corresponding d* module.

The --mode option determines how the ED modules are configured.

  - mode == 0:
     the m* modules are duplicates of the corresponding d* modules.

  - mode == 1:
     one parameter in m1y is changed compared to d1y
     and this makes all the m*y modules unique,
     while the m*x modules should ultimately
     be identified as duplicates of the d*x modules.
"""
import FWCore.ParameterSet.Config as cms

import os
import argparse

parser = argparse.ArgumentParser(
    prog = 'python3 '+os.path.basename(__file__),
    formatter_class = argparse.RawDescriptionHelpFormatter,
    description = __doc__,
    argument_default = argparse.SUPPRESS,
)

parser.add_argument("--mode",
    type = int,
    default = 0,
    choices = [0,1],
    help = "Choose how to configure the modules."
)

args,_ = parser.parse_known_args()

process = cms.Process('TEST')

### "d*" modules: the duplicates
###  - the *x (*y) modules depend only on *x (*y) modules, and not on *y (*x) modules
###  - the *2* modules depend on *1* modules
###  - the *3* modules depend on *1* and *2* modules
process.d1x = cms.EDProducer('P1',
    p1 = cms.InputTag('rawDataCollector'),
    p2 = cms.bool(False),
    p3 = cms.vbool(False, True),
    p4 = cms.uint32(1),
    p5 = cms.vuint32(1,2,3),
    p6 = cms.int32(-1),
    p7 = cms.vint32(-1,2,-3),
    p8 = cms.double(1.1),
    p9 = cms.vdouble(2.3, 4.5)
)

process.d1y = process.d1x.clone()

process.d2x = cms.EDFilter('F2',
    p1 = cms.vint32(1, 2, 3),
    p2 = cms.VInputTag('d1x'),
    p3 = cms.PSet(
        theStrings = cms.vstring('keyword1', 'keyword2')
    )
)

process.d2y = process.d2x.clone( p2 = ['d1y'] )

process.d3x = cms.EDAnalyzer('A3',
    p1 = cms.VPSet(
        cms.PSet(
            pset_a = cms.PSet(
                tag1 = cms.InputTag('d1x')
            ),
            pset_b = cms.PSet(
                tag2 = cms.InputTag('d2x')
            ),
        )
    ),
    p2 = cms.PSet(
        p_a = cms.PSet(
            p_b = cms.PSet(
                p_c = cms.VInputTag('d2x', 'd1x')
            )
        )
    )
)

process.d3y = process.d3x.clone()
process.d3y.p1[0].pset_a.tag1 = 'd1y'
process.d3y.p1[0].pset_b.tag2 = 'd2y'
process.d3y.p2.p_a.p_b.p_c = ['d2y', 'd1y']

### m* modules
###  - the m* modules are the counterparts of the d* modules
###  - m* modules do not depend on d* modules (and viceversa)
###  - if the mode "unique-m*y" is chosen,
###    one parameter in m1y is changed compared to d1y
###    and this makes all the m*y modules unique,
###    while the m*x modules should ultimately
###    be flagged as duplicates of the d*x modules
process.m1x = process.d1x.clone()

if args.mode == 0:
    process.m1y = process.d1y.clone()
elif args.mode == 1:
    process.m1y = process.d1y.clone( p2 = True )

process.m2x = process.d2x.clone( p2 = ['m1x'] )
process.m2y = process.d2y.clone( p2 = ['m1y'] )
process.m3x = process.d3x.clone()

process.m3x.p1[0].pset_a.tag1 = 'm1x'
process.m3x.p1[0].pset_b.tag2 = 'm2x'
process.m3x.p2.p_a.p_b.p_c = ['m2x', 'm1x']

process.m3y = process.d3y.clone()
process.m3y.p1[0].pset_a.tag1 = 'm1y'
process.m3y.p1[0].pset_b.tag2 = 'm2y'
process.m3y.p2.p_a.p_b.p_c = ['m2y', 'm1y']
