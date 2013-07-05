#!/usr/bin/env python
'''

Merge transforms

Merge decay mode specific Tanc transformations into a single transformation
file that can be read by RecoTauMVATransform

Author: Evan K. Friis (UC Davis)

'''

import FWCore.ParameterSet.Config as cms
import sys
import os

output_file = sys.argv[1]
input_files = sys.argv[2:]

# Append the python path
for input in input_files:
    path = os.path.dirname(input)
    sys.path.append(path)

input_transforms = {}
# Load the modules
for input in input_files:
    name = os.path.splitext(os.path.basename(input))[0]
    # Name is of the format transform_1prong1pi0_hpstanc
    nicename = name.split('_')[1]
    __import__(name)
    input_transforms[nicename] = sys.modules[name].transform

output_object = cms.VPSet()

# Figure out the cut to apply for the different purity discriminators
weighted_loose_cut = 0.0
weighted_medium_cut = 0.0
weighted_tight_cut = 0.0
weight_sum = 0.0

for name in input_transforms.keys():
    dm_weight = input_transforms[name].signalDecayModeWeight.value()
    weighted_loose_cut += input_transforms[name].looseCut.value()*dm_weight
    weighted_medium_cut += input_transforms[name].mediumCut.value()*dm_weight
    weighted_tight_cut += input_transforms[name].tightCut.value()*dm_weight
    weight_sum += dm_weight
    # format should be XprongYpi0  - this is sorta nasty
    nCharged = int(name[0])
    nPiZeros = int(name[6])
    output_object.append(
        cms.PSet(
            nCharged = cms.uint32(nCharged),
            nPiZeros = cms.uint32(nPiZeros),
            transform = input_transforms[name]
        )
    )

cuts = cms.PSet(
    looseCut = cms.double(weighted_loose_cut/weight_sum),
    mediumCut = cms.double(weighted_medium_cut/weight_sum),
    tightCut = cms.double(weighted_tight_cut/weight_sum),
)

output = open(output_file, 'w')
output.write('import FWCore.ParameterSet.Config as cms\n')
output.write('transforms = %s\n' % output_object.dumpPython())
output.write('cuts = %s\n' % cuts.dumpPython())
output.close()

print "Transform file %s created" % output_file
sys.exit(0)
