import ROOT
import pprint
import sys
from DataFormats.FWLite import Events, Handle
ROOT.gROOT.SetBatch()

events = Events('validate_ctag_pat.root')
jet_labels = ["selectedPatJets"]#, "selectedPatJetsAK4PF", "selectedPatJetsAK8PFCHSSoftDropSubjets"]
tested_discriminators = ['pfCombinedCvsLJetTags', 'pfCombinedCvsBJetTags']

evt = events.__iter__().next()
handle = Handle('std::vector<pat::Jet>')
for label in jet_labels:
   evt.getByLabel(label, handle)
   jets = handle.product()
   jet = jets.at(0)
   available = set([i.first for i in jet.getPairDiscri()])
   for test in tested_discriminators:
      print "%s in %s: %s" % (test, label, test in available)
