import FWCore.ParameterSet.Config as cms

# Name:   RecoMET.cff
# Author: R.Cavanaugh
# Date:   05.11.2006
# Notes:  CaloMET.cfi assumes that a product with label "caloTowers" is 
#         already written into the event.
from RecoMET.METProducers.genMet_cfi import *
from RecoMET.METProducers.genMetNoNuBSM_cfi import *
from RecoMET.METProducers.genMetFromGenJets_cfi import *
#
# ShR 27 Mar 2007: genJetParticles from "RecoJets/Configuration/data/GenJetParticles.cff"
# must be executed before this sequence in order to work since genMetNoNugenMetNoNu
# needs genParticlesAllStableNoNu from that sequence
# can't append them here explicitly because of scheduling problem
recoGenMET = cms.Sequence(genMet+genMetNoNuBSM*genMetIC5GenJets)

