from __future__ import print_function
import ROOT
import pprint
import sys
from DataFormats.FWLite import Events, Handle
ROOT.gROOT.SetBatch()
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('file')
parser.add_argument('collections', default=['slimmedJets'], nargs='*')
args = parser.parse_args()

events = Events(args.file)
jet_labels = args.collections
tested_discriminators = ['pfCombinedCvsLJetTags', 'pfCombinedCvsBJetTags']

evt = next(events.__iter__())
handle = Handle('std::vector<pat::Jet>')
for label in jet_labels:
   evt.getByLabel(label, handle)
   jets = handle.product()
   jet = jets.at(0)
   available = set([i.first for i in jet.getPairDiscri()])
   for test in tested_discriminators:
      print("%s in %s: %s" % (test, label, test in available))
