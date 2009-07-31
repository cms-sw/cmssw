import FWCore.ParameterSet.Config as cms

# $Id: RecoGenJetsAll_cff.py,v 1.2 2008/04/21 03:27:18 rpw Exp $

#
from RecoJets.JetProducers.RecoGenJets_cff import *

recoGenJetsAll = cms.Sequence(recoAllGenJets+recoAllGenJetsNoNu+recoAllGenJetsNoNuBSM)
