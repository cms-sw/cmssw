#! /usr/bin/env python

# import ROOT in batch mode
import sys
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
ROOT.gROOT.SetBatch(True)
sys.argv = oldargv

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.AutoLibraryLoader.enable()

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events
from RecoEgamma.ElectronIdentification import VIDElectronSelector

from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_CSA14_50ns_V0_cff import cutBasedElectronID_CSA14_50ns_V0_standalone_tight

#foo = VIDElectronSelector()
#foo.initialize(cutBasedElectronID_CSA14_50ns_V0_standalone_tight)

bar = VIDElectronSelector(cutBasedElectronID_CSA14_50ns_V0_standalone_tight)

#load versioned ID selector
#ROOT.gSystem.Load("libRecoEgammaElectronIdentification.so")



