import FWCore.ParameterSet.Config as cms

# File: RecHits.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for ECAL and HCAL RecHits.
# Assumes ecalRecHit:EcalRecHitsEE, ecalRecHit:EcalRecHitsEB, hbhereco, horeco, and hfreco
# are in the event.
from DQMOffline.JetMET.RecHits_cfi import *
analyzeRecHits = cms.Sequence(ECALAnalyzer*HCALAnalyzer)

