#import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use the VolumeBased magnetic
# field engine.
#Default is version 85l
#from MagneticField.Engine.volumeBasedMagneticField_85l_cfi import *

import sys
print "This file is obsolete and will soon be removed.";
print "please use Configuration/StandardSequences/python/MagneticField_cff.py or Configuration/StandardSequences/data/MagneticField.cff";
print "if still using old configuration files. Now exiting with -1 error. please remove inclusion of this file and try again.";
sys.exit(-1);
