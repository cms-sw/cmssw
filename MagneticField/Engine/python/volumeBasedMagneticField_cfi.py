#import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use the VolumeBased magnetic
# field engine.
#Default is version 85l
#from MagneticField.Engine.volumeBasedMagneticField_85l_cfi import *

import sys
print "MagneticField/Engine/data/volumeBasedMagneticField.cfi and \nMagneticField/Engine/python/volumeBasedMagneticField_cfi.py are obsolete and will soon be removed.";
print "please use Configuration/StandardSequences/python/MagneticField_cff.py \nor Configuration/StandardSequences/data/MagneticField.cff";
print "if still using old configuration files. Now exiting with -1 error. \nplease remove inclusion of this file and try again.";
print "starting with the next release including this file will cause an exception";
#sys.exit(-1);
