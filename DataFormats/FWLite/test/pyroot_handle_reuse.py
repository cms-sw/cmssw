#! /usr/bin/env python3
import ROOT
from DataFormats.FWLite import Events, Handle

events = Events (['good_a.root'])

handleGP  = Handle ("edmtest::Thing")
labelGP = ("Thing")

for event in events:
    if event.getByLabel (labelGP, handleGP) :
        prod = handleGP.product()
