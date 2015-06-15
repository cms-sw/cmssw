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
from RecoEgamma.ElectronIdentification.VIDElectronSelector import VIDElectronSelector


from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_PHYS14_PU20bx25_V1_cff import cutBasedElectronID_PHYS14_PU20bx25_V1_standalone_tight

selectElectron = VIDElectronSelector(cutBasedElectronID_PHYS14_PU20bx25_V1_standalone_tight)
print 'Initialized VID Selector for Electrons'
print selectElectron

from PhysicsTools.SelectorUtils.trivialCutFlow_cff import trivialCutFlow
testExprEval = VIDElectronSelector(trivialCutFlow)
print testExprEval


# try muons!
from RecoMuon.MuonIdentification.VIDMuonSelector import VIDMuonSelector
from RecoMuon.MuonIdentification.Identification.globalMuonPromptTight_V0_cff import globalMuonPromptTight_V0
from RecoMuon.MuonIdentification.Identification.cutBasedMuonId_MuonPOG_V0_cff import *

selectMuons = [VIDMuonSelector(globalMuonPromptTight_V0)]
for selectMuon in [cutBasedMuonId_MuonPOG_V0_loose, cutBasedMuonId_MuonPOG_V0_medium, cutBasedMuonId_MuonPOG_V0_tight,
                   cutBasedMuonId_MuonPOG_V0_soft, cutBasedMuonId_MuonPOG_V0_highpt]:
    for cf in selectMuon.cutFlow:
        cf.vertexSrc = "offlineSlimmedPrimaryVertices"
    selectMuons.append(VIDMuonSelector(selectMuon))
    print 'Initialized VID Selector for Muons'
    print selectMuon

# open file (you can use 'edmFileUtil -d /store/whatever.root' to get the physical file name)
#events = Events("root://eoscms//eos/cms/store/cmst3/user/gpetrucc/miniAOD/74X/miniAOD-new_ZTT.root")
events = Events("root://eoscms//eos/cms/store/relval/CMSSW_7_4_0_pre9_ROOT6/DoubleMu/MINIAOD/GR_R_74_V8A_RelVal_zMu2011A-v1/00000/06961B48-CFD1-E411-8B87-002618943971.root")

muons, muonLabel = Handle("std::vector<pat::Muon>"), "slimmedMuons"
electrons, electronLabel = Handle("std::vector<pat::Electron>"), "slimmedElectrons"

for iev,event in enumerate(events):
    
    if iev > 10: break
    event.getByLabel(muonLabel, muons)
    event.getByLabel(electronLabel, electrons)
    
    
    print "\nEvent %d: run %6d, lumi %4d, event %12d" % (iev,event.eventAuxiliary().run(), 
                                                         event.eventAuxiliary().luminosityBlock(),
                                                         event.eventAuxiliary().event())
    
    # Muons
    for i,mu in enumerate(muons.product()): 
        if mu.pt() < 5 or not mu.isLooseMuon(): continue
        print "muon %2d: pt %4.1f, POG loose id %d." % (
            i, mu.pt(), mu.isLooseMuon())
        for selectMuon in selectMuons:
            selectMuon(muons.product(),i,event)
            print selectMuon

    # Electrons
    for i,el in enumerate(electrons.product()):
        if el.pt() < 5: continue
        print "elec %2d: pt %4.1f, supercluster eta %+5.3f, sigmaIetaIeta %.3f (%.3f with full5x5 shower shapes), pass conv veto %d" % (
                    i, el.pt(), el.superCluster().eta(), el.sigmaIetaIeta(), el.full5x5_sigmaIetaIeta(), el.passConversionVeto())
        selectElectron(electrons.product(),i,event)       
        print selectElectron
        testExprEval(electrons.product(),i,event)
        print testExprEval

#test the validator framework

print 'test validation framework' 

selectElectronValid = VIDElectronSelector(cutBasedElectronID_PHYS14_PU20bx25_V1_standalone_tight)
selectMuonValid = VIDMuonSelector(globalMuonPromptTight_V0)
from  PhysicsTools.SelectorUtils.VIDSelectorValidator import VIDSelectorValidator
electron_validator = VIDSelectorValidator(selectElectronValid,'std::vector<pat::Electron>','slimmedElectrons')
muon_validator = VIDSelectorValidator(selectMuonValid,'std::vector<pat::Muon>','slimmedMuons')

signal_files     = []
background_files = []
mix_files        = ['root://eoscms//eos/cms/store/relval/CMSSW_7_4_0_pre9_ROOT6/DoubleMu/MINIAOD/GR_R_74_V8A_RelVal_zMu2011A-v1/00000/06961B48-CFD1-E411-8B87-002618943971.root']

electron_validator.setMixFiles(mix_files)
muon_validator.setMixFiles(mix_files)

electron_validator.runValidation()
muon_validator.runValidation()

