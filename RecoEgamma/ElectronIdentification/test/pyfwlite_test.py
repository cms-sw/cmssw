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

#cms python data types
import FWCore.ParameterSet.Config as cms

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events
from RecoEgamma.ElectronIdentification import VIDElectronSelector

from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_CSA14_50ns_V0_cff import cutBasedElectronID_CSA14_50ns_V0_standalone_tight

#foo = VIDElectronSelector()
#foo.initialize(cutBasedElectronID_CSA14_50ns_V0_standalone_tight)

cutBasedElectronID_CSA14_50ns_V0_standalone_tight.cutFlow[6].vertexSrc = cms.InputTag('offlineSlimmedPrimaryVertices')
cutBasedElectronID_CSA14_50ns_V0_standalone_tight.cutFlow[7].vertexSrc = cms.InputTag('offlineSlimmedPrimaryVertices')
cutBasedElectronID_CSA14_50ns_V0_standalone_tight.cutFlow[10].conversionSrc = cms.InputTag('reducedEgamma:reducedConversions')

print 'trying to build VID selector'
selectElectron = VIDElectronSelector(cutBasedElectronID_CSA14_50ns_V0_standalone_tight)
print 'built VID selector'

# open file (you can use 'edmFileUtil -d /store/whatever.root' to get the physical file name)
events = Events("root://eoscms//eos/cms/store/cmst3/user/gpetrucc/miniAOD/74X/miniAOD-new_ZTT.root")

electrons, electronLabel = Handle("std::vector<pat::Electron>"), "slimmedElectrons"
test = ROOT.edm.Handle(ROOT.pat.Electron)()



for iev,event in enumerate(events):
    
    if iev > 10: break

    event.getByLabel(electronLabel, electrons)
    
    
    print "\nEvent %d: run %6d, lumi %4d, event %12d" % (iev,event.eventAuxiliary().run(), 
                                                         event.eventAuxiliary().luminosityBlock(),
                                                         event.eventAuxiliary().event())
    
    # Electrons
    for i,el in enumerate(electrons.product()):
        if el.pt() < 5: continue
        print "elec %2d: pt %4.1f, supercluster eta %+5.3f, sigmaIetaIeta %.3f (%.3f with full5x5 shower shapes), pass conv veto %d" % (
                    i, el.pt(), el.superCluster().eta(), el.sigmaIetaIeta(), el.full5x5_sigmaIetaIeta(), el.passConversionVeto())
        selectElectron(electrons.product(),i,event)
        print selectElectron




